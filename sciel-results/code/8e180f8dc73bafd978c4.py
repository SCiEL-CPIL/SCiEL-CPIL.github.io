"""Rewritten program for the second autocorrelation inequality task."""
import numpy as np
import time
import scipy.optimize as opt
from scipy.fftpack import dct, idct

N = 50

# Pre-compute constants for the objective
# The evaluator computes:
# g = np.convolve(h, h)  (length 2*N - 1 = 99)
# m = len(g) = 99
# y = [0, g[0], ..., g[m-1], 0]  (length m+2 = 101)
# points = np.linspace(-0.5, 0.5, m+2)
# w = 1.0 / (m+1)  (spacing)
# l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
# norm_1 = sum(abs(g)) / (m+1)
# norm_inf = max(abs(g))
# value = l2 / (norm_1 * norm_inf)

# Since h >= 0, g >= 0, so abs(g) = g, and max(abs(g)) = max(g).
# norm_1 = sum(g) / (m+1)
# Note: sum(g) = sum(h) * sum(h) = (sum(h))^2

M = 2 * N - 1  # length of g = 99
W = 1.0 / (M + 1)  # spacing

# Pre-compute the y vector structure
# y = [0, g[0], g[1], ..., g[M-1], 0]
# l2 = (W/3) * sum_{i=0}^{M} (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)

def compute_value(h):
    """Compute the objective value for given heights h."""
    h = np.asarray(h, dtype=np.float64)
    g = np.convolve(h, h)
    y = np.zeros(M + 2)
    y[1:M+1] = g
    
    # l2 = (W/3) * sum_{i=0}^{M} (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    l2_terms = y[:M+1]**2 + y[:M+1]*y[1:M+2] + y[1:M+2]**2
    l2 = (W / 3.0) * np.sum(l2_terms)
    
    norm_1 = np.sum(g) / (M + 1)
    norm_inf = np.max(g)
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    
    return l2 / (norm_1 * norm_inf)

def objective(params):
    """Objective function for optimization (negative because we maximize)."""
    # params: DCT coefficients
    h = np.abs(idct(params, type=2, norm='ortho'))
    h = np.clip(h, 0, None)
    return -compute_value(h)

def construct():
    """Construct 50 nonnegative step heights."""
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s limit
    
    # Start with uniform heights
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Use DCT parameterization
    # idct of a vector gives heights
    # Start with DCT coefficients that give uniform heights
    # For uniform h=1, the DCT coefficients are:
    # c[0] = sqrt(2/N) * sum(h) = sqrt(2/N) * N = sqrt(2*N)
    # c[k] = 0 for k > 0
    # Actually, let's just use the DCT of the uniform vector
    initial_params = dct(np.ones(N), type=2, norm='ortho')
    
    # Try multiple random restarts in DCT space
    rng = np.random.RandomState(42)
    
    # First, try optimizing from the uniform start
    try:
        result = opt.minimize(
            objective,
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if result.fun < -best_val:
            h_opt = np.abs(idct(result.x, type=2, norm='ortho'))
            h_opt = np.clip(h_opt, 0, None)
            val_opt = compute_value(h_opt)
            if val_opt > best_val:
                best_val = val_opt
                best_h = h_opt
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return list(best_h)
    
    # Try random restarts in DCT space
    for restart in range(20):
        if time.time() - start_time > deadline - 10:
            break
        
        # Random DCT coefficients
        params = rng.randn(N) * 0.5 + initial_params * 0.1
        
        try:
            result = opt.minimize(
                objective,
                params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < -best_val:
                h_opt = np.abs(idct(result.x, type=2, norm='ortho'))
                h_opt = np.clip(h_opt, 0, None)
                val_opt = compute_value(h_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt
        except Exception:
            pass
    
    # Also try a few structured patterns
    patterns = []
    
    # Triangular
    patterns.append(np.linspace(1, 0, N))
    patterns.append(np.linspace(0, 1, N))
    
    # Bimodal
    for center in range(5, N-5):
        patterns.append(np.abs(np.linspace(-1, 1, N)) ** 2)
        break
    
    # Symmetric bump
    x = np.linspace(-1, 1, N)
    patterns.append(np.exp(-x**2 * 2))
    patterns.append(np.exp(-x**2 * 0.5))
    patterns.append(1.0 / (1.0 + x**2))
    patterns.append(np.cos(x * np.pi / 2) ** 2)
    
    for p in patterns:
        if time.time() - start_time > deadline - 5:
            break
        p = np.clip(p, 0, None)
        val = compute_value(p)
        if val > best_val:
            best_val = val
            best_h = p
    
    return list(best_h)
