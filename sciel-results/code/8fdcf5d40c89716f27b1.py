import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 400

def _fft_size(n):
    # Next power of 2 >= n
    fsize = 1
    while fsize < n:
        fsize *= 2
    return fsize

def _autoconv_fft(h):
    n = len(h)
    size = _fft_size(2 * n - 1)
    H = np.fft.rfft(h, size)
    C = np.fft.irfft(H * H, size)
    return C[:2*n-1]

def _score(h):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return float('inf')
    c = _autoconv_fft(h)
    return np.max(np.abs(c)) / (s * s)

def _objective_de(h):
    # h is the raw vector from DE
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e6
    h_norm = h / s
    c = _autoconv_fft(h_norm)
    return np.max(np.abs(c))

def construct():
    start_time = time.time()
    deadline = start_time + 130.0 # 20s margin
    
    # 1. Global Search with Differential Evolution
    # Bounds: Since scale invariant, we can restrict to [-1, 1] or similar.
    # The normalization happens inside the objective.
    bounds = [(-1.0, 1.0) for _ in range(N)]
    
    best_de_result = None
    try:
        # DE is stochastic, seed it for reproducibility if needed, but random is fine too.
        # popsize=20, maxiter=100. 
        # Tolerance: we just want a good starting point for local refinement.
        res_de = differential_evolution(
            _objective_de, 
            bounds, 
            seed=42,
            maxiter=50,
            popsize=20,
            tol=1e-4,
            polish=False # We will do our own polish
        )
        if time.time() < deadline - 20:
            best_de_result = res_de.x
    except Exception as e:
        pass

    # 2. Local Refinement
    # Start from best DE result, or fallback to uniform if DE failed
    if best_de_result is not None:
        h_init = best_de_result
    else:
        h_init = np.ones(N)
        
    # Normalize initial guess
    s_init = np.sum(h_init)
    if abs(s_init) < 1e-10:
        h_init = np.ones(N)
        s_init = N
    else:
        h_init = h_init / s_init

    best_h = h_init.copy()
    best_score_val = _score(best_h)
    
    # Define a smooth objective for local optimizer
    # We use a soft-max to make it differentiable
    def _soft_max_abs_conv(h):
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e6
        h_norm = h / s
        c = _autoconv_fft(h_norm)
        abs_c = np.abs(c)
        # Log-sum-exp trick for stability
        max_c = np.max(abs_c)
        if max_c == 0:
            return 0.0
        beta = 100.0
        shifted = abs_c - max_c
        val = max_c + np.log(np.sum(np.exp(beta * shifted))) / beta
        return val

    def _objective_local(h):
        return _soft_max_abs_conv(h)

    def _constraint_sum(h):
        return np.sum(h) - 1.0

    # Try L-BFGS-B or SLSQP
    # SLSQP handles equality constraints better for small N, but L-BFGS-B is faster.
    # Since N=400, SLSQP might be slow. 
    # Alternative: Minimize _soft_max_abs_conv(h) without constraint, 
    # relying on the fact that the function is scale invariant? 
    # No, _soft_max_abs_conv(h) is scale invariant only if we normalize inside.
    # If we don't normalize inside, the gradient is tricky.
    # Let's stick to normalizing inside the objective for DE, 
    # and for local search, we can just minimize _soft_max_abs_conv(h) 
    # which effectively finds the shape, and the scale doesn't matter for the value, 
    # but the optimizer needs a consistent landscape.
    
    # Actually, if we minimize _soft_max_abs_conv(h) where h is normalized internally, 
    # the gradient with respect to h_raw is well-defined.
    # Let's use L-BFGS-B. It doesn't support equality constraints.
    # But since the function is flat along the scale direction (if we normalize inside), 
    # L-BFGS-B might wander in scale.
    # To fix this, we can just add a small penalty for sum(h) != 1 in the local objective.
    
    def _objective_local_penalized(h):
        base = _soft_max_abs_conv(h)
        penalty = (np.sum(h) - 1.0)**2 * 1e-4
        return base + penalty

    try:
        res_local = minimize(
            _objective_local_penalized,
            h_init,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-12, 'disp': False}
        )
        if time.time() < deadline - 5:
            h_res = res_local.x
            s_res = _score(h_res)
            if s_res < best_score_val:
                best_score_val = s_res
                best_h = h_res.copy()
    except Exception as e:
        pass

    # If time remains, try a few random restarts with local optimization
    # This helps escape local minima found by DE
    rng = np.random.default_rng(42)
    while time.time() < deadline - 10.0:
        # Random initialization
        h_rand = rng.normal(0, 1, N)
        s_rand = np.sum(h_rand)
        if abs(s_rand) < 1e-6:
            continue
        h_rand = h_rand / s_rand
        
        try:
            res_rand = minimize(
                _objective_local_penalized,
                h_rand,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-10, 'disp': False}
            )
            h_res = res_rand.x
            s_res = _score(h_res)
            if s_res < best_score_val:
                best_score_val = s_res
                best_h = h_res.copy()
        except:
            pass

    return best_h
