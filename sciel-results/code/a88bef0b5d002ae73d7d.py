import numpy as np
from scipy.optimize import minimize
import time

N = 50
m = 2 * N - 1  # 99
m_plus_1 = m + 1  # 100
w = 1.0 / m_plus_1  # spacing

def _evaluate(h):
    """Compute the objective value for a given set of heights h."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N or np.any(h < 0) or not np.all(np.isfinite(h)):
        return 0.0
    
    g = np.convolve(h, h)
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    y_sq = y ** 2
    cross = y * np.roll(y, -1)
    l2 = (w / 3.0) * np.sum(y_sq[:-1] + cross[:-1] + y_sq[1:])
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf < 1e-300:
        return 0.0
    
    return l2 / (norm_1 * norm_inf)

def _neg_value_smooth(h, beta=50.0):
    """Negative of the objective with smooth max surrogate."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N or not np.all(np.isfinite(h)):
        return 1e10
    
    g = np.convolve(h, h)
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    y_sq = y ** 2
    cross = y * np.roll(y, -1)
    l2 = (w / 3.0) * np.sum(y_sq[:-1] + cross[:-1] + y_sq[1:])
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    abs_g = np.abs(g)
    
    # Log-sum-exp for max
    scaled = beta * abs_g
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        norm_inf_approx = np.max(abs_g)
    else:
        norm_inf_approx = (1.0 / beta) * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))
    
    if norm_1 * norm_inf_approx < 1e-300:
        return 1e10
    
    return -l2 / (norm_1 * norm_inf_approx)

def _make_initializations():
    """Create diverse initializations."""
    inits = []
    
    # Uniform
    inits.append(np.ones(N))
    
    # Gaussian bump
    x = np.linspace(-1, 1, N)
    inits.append(np.exp(-x**2 / 0.3))
    
    # Triangle
    inits.append(1.0 - np.abs(np.linspace(-1, 1, N)))
    
    # Sine bump
    inits.append(np.maximum(0, np.sin(np.pi * np.linspace(0, 1, N))))
    
    # Two peaks
    x = np.linspace(-1, 1, N)
    inits.append(np.exp(-(x - 0.4)**2 / 0.15) + np.exp(-(x + 0.4)**2 / 0.15))
    
    # Plateau
    inits.append(np.where(np.abs(np.linspace(-1, 1, N)) < 0.5, 1.0, 0.0))
    
    # Random
    rng = np.random.default_rng(42)
    inits.append(rng.random(N) * 0.5 + 0.5)
    
    # Cosine
    inits.append(np.cos(np.pi * np.linspace(0, 1, N)) ** 2)
    
    # Narrow peak
    x = np.linspace(-1, 1, N)
    inits.append(np.exp(-x**2 / 0.1))
    
    # Wide bump
    x = np.linspace(-1, 1, N)
    inits.append(np.exp(-x**2 / 0.8))
    
    return inits

def construct():
    """Construct 50 nonnegative step heights to maximize the objective."""
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Start with uniform seed
    h0 = np.ones(N)
    best_h = h0.copy()
    best_val = _evaluate(h0)
    
    initializations = _make_initializations()
    
    # Try different beta values with different initializations
    beta_schedule = [
        (30.0, 800),
        (50.0, 600),
        (100.0, 400),
        (200.0, 300),
    ]
    
    for beta, maxiter in beta_schedule:
        if time.time() - start_time > deadline - 40:
            break
        
        for init in initializations:
            if time.time() - start_time > deadline - 30:
                break
            
            # Skip if this initialization already gave best
            if np.array_equal(init, best_h):
                continue
            
            try:
                result = minimize(
                    fun=lambda x: _neg_value_smooth(x, beta=beta),
                    x0=init.copy(),
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                
                if np.all(result.x >= 0) and np.all(np.isfinite(result.x)):
                    val = _evaluate(result.x)
                    if val > best_val:
                        best_val = val
                        best_h = result.x.copy()
            except Exception:
                pass
        
        # Also refine from current best
        if time.time() - start_time > deadline - 30:
            continue
        
        try:
            result = minimize(
                fun=lambda x: _neg_value_smooth(x, beta=beta),
                x0=best_h.copy(),
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            if np.all(result.x >= 0) and np.all(np.isfinite(result.x)):
                val = _evaluate(result.x)
                if val > best_val:
                    best_val = val
                    best_h = result.x.copy()
        except Exception:
            pass
    
    # Final high-beta refinement
    if time.time() - start_time < deadline - 20:
        for beta in [300.0, 500.0]:
            if time.time() - start_time > deadline - 15:
                break
            try:
                result = minimize(
                    fun=lambda x: _neg_value_smooth(x, beta=beta),
                    x0=best_h.copy(),
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 50}
                )
                
                if np.all(result.x >= 0) and np.all(np.isfinite(result.x)):
                    val = _evaluate(result.x)
                    if val > best_val:
                        best_val = val
                        best_h = result.x.copy()
            except Exception:
                pass
    
    # Local search with coordinate descent
    if time.time() - start_time < deadline - 10:
        h_local = best_h.copy()
        improved = True
        while improved and time.time() - start_time < deadline - 5:
            improved = False
            for i in range(N):
                for delta in [-0.01, 0.01, -0.001, 0.001, -0.1, 0.1]:
                    h_try = h_local.copy()
                    h_try[i] = max(0, h_try[i] + delta)
                    val = _evaluate(h_try)
                    if val > best_val:
                        best_val = val
                        best_h = h_try.copy()
                        h_local = h_try.copy()
                        improved = True
                        break
    
    # Ensure non-negativity and finiteness
    best_h = np.maximum(best_h, 0)
    best_h = np.where(np.isfinite(best_h), best_h, 0.0)
    
    return list(best_h)
