import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
DEADLINE = 145.0  # Safety margin under 150s

def _value_from_h(h):
    """
    Computes the value as per the contract.
    h: array of 50 non-negative heights.
    """
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)) or np.any(h < 0):
        return 0.0
    
    # Normalize to avoid overflow/underflow and handle scale invariance
    # We can fix max(h) = 1 if max > 0, otherwise return 0
    mx = np.max(h)
    if mx < 1e-12:
        return 0.0
    h = h / mx
    
    g = np.convolve(h, h)
    m = len(g)
    
    # y = [0, g[0], ..., g[m-1], 0]
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    w = 1.0 / (m + 1)
    
    # l2 = sum over i in 0..m of (w / 3) * (y[i]^2 + y[i] * y[i+1] + y[i+1]^2)
    a = y[:-1]
    b = y[1:]
    l2 = (w / 3.0) * np.sum(a * a + a * b + b * b)
    
    norm_1 = np.sum(g) / (m + 1)  # Since h >= 0, g >= 0, abs(g) = g
    norm_inf = np.max(g)
    
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
        
    return l2 / (norm_1 * norm_inf)

def _neg_value(h):
    """
    Negative value for minimization.
    h is assumed to be in [0, 1] due to bounds in DE/minimize.
    """
    try:
        # Ensure non-negative (bounds should handle this, but safe to clip)
        h = np.clip(h, 0.0, 1.0)
        return -_value_from_h(h)
    except Exception:
        return 1e6

def _generate_parametric_start(rng, n_params=10):
    """
    Generate a diverse initial guess using a parametric model.
    This helps DE or local optimizers start in good regions.
    """
    # Simple parametric model: sum of a few Gaussians
    # Params: centers, widths, amplitudes
    centers = rng.uniform(-0.8, 0.8, n_params)
    widths = rng.uniform(0.1, 0.5, n_params)
    amps = rng.uniform(0.5, 1.0, n_params)
    
    x = np.linspace(-1.0, 1.0, N)
    h = np.zeros(N)
    for c, w, a in zip(centers, widths, amps):
        h += a * np.exp(-((x - c) ** 2) / (2 * w ** 2))
    
    # Normalize to max 1
    mx = np.max(h)
    if mx > 0:
        h = h / mx
    return h

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_h = np.ones(N) / N
    best_val = _value_from_h(best_h)
    
    # Phase 1: Differential Evolution (Global Search)
    # Bounds: each height in [0, 1]
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    # Use a moderate number of iterations to stay within time
    # maxiter=50, popsize=15 -> 750 evaluations.
    # Each eval is fast (convolution of 50 elements).
    
    # Seed DE with some parametric starts to help it converge faster
    # DE doesn't directly take seeds in standard scipy, but we can
    # run a few local optimizations from parametric starts first.
    
    # Phase 1a: Local optimization from parametric starts
    parametric_starts = []
    for i in range(20):
        if time.time() - start_time > DEADLINE:
            break
        h0 = _generate_parametric_start(rng, n_params=3 + i % 4)
        try:
            res = minimize(_neg_value, h0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            h_res = np.clip(res.x, 0.0, 1.0)
            v_res = -res.fun
            if v_res > best_val:
                best_val = v_res
                best_h = h_res.copy()
            parametric_starts.append(h_res)
        except Exception:
            pass
        
        # Also try a "flat" start with noise
        h_noise = np.ones(N) + rng.normal(0, 0.1, N)
        h_noise = np.clip(h_noise, 0, 1)
        h_noise = h_noise / np.max(h_noise) if np.max(h_noise) > 0 else h_noise
        try:
            res = minimize(_neg_value, h_noise, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            h_res = np.clip(res.x, 0.0, 1.0)
            v_res = -res.fun
            if v_res > best_val:
                best_val = v_res
                best_h = h_res.copy()
            parametric_starts.append(h_res)
        except Exception:
            pass

    # Phase 2: Differential Evolution
    if time.time() - start_time < DEADLINE - 20: # Keep 20s for final polish
        try:
            # We can use the best parametric starts to initialize the population?
            # Standard DE in scipy doesn't easily allow custom population initialization
            # without subclassing. We'll just run standard DE.
            # To make it more effective, we can use a smaller popsize and more iterations
            # or just rely on the fact that the landscape is not too rugged.
            
            # Let's try a few DE runs with different random seeds if time permits, 
            # or one long run.
            # popsize=10, maxiter=30 -> 300 evals.
            
            res_de = differential_evolution(
                _neg_value, 
                bounds, 
                seed=123, 
                maxiter=40, 
                popsize=12, 
                tol=1e-6,
                polish=True  # Uses L-BFGS-B on the best result
            )
            
            if res_de.success:
                h_de = np.clip(res_de.x, 0.0, 1.0)
                v_de = -res_de.fun
                if v_de > best_val:
                    best_val = v_de
                    best_h = h_de.copy()
        except Exception:
            pass

    # Phase 3: Final Local Polish
    # Run L-BFGS-B on the current best and a few neighbors
    polish_candidates = [best_h]
    
    # Add some random perturbations of the best
    for i in range(5):
        if time.time() - start_time > DEADLINE - 5:
            break
        h_pert = best_h + rng.normal(0, 0.05, N)
        h_pert = np.clip(h_pert, 0.0, 1.0)
        # Re-normalize to max 1 to stay in bounds effectively
        mx = np.max(h_pert)
        if mx > 0:
            h_pert = h_pert / mx
        polish_candidates.append(h_pert)
        
        # Also try scaling the best
        h_scaled = best_h * rng.uniform(0.5, 1.5)
        h_scaled = np.clip(h_scaled, 0.0, 1.0)
        polish_candidates.append(h_scaled)

    for h0 in polish_candidates:
        if time.time() - start_time > DEADLINE - 2:
            break
        try:
            res = minimize(_neg_value, h0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            h_res = np.clip(res.x, 0.0, 1.0)
            v_res = -res.fun
            if v_res > best_val:
                best_val = v_res
                best_h = h_res.copy()
        except Exception:
            pass

    # Final check
    best_h = np.clip(best_h, 0.0, None)
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
        
    return list(best_h)

if __name__ == "__main__":
    # Quick self-test
    t0 = time.time()
    h = construct()
    t1 = time.time()
    v = _value_from_h(h)
    print(f"Time: {t1-t0:.2f}s, Value: {v:.6f}")
