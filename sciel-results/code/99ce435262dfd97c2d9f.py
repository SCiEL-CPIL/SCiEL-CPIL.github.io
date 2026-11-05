"""Optimized program for third autocorrelation inequality using coordinate descent and soft-max smoothing."""
import numpy as np
import time

N = 400
DEADLINE = 140.0  # safety margin before 150s kill


def _autoconv_fft(h):
    """Compute full autoconvolution using FFT."""
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    return conv[:2 * n - 1]


def _objective(h):
    """Compute the objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = _autoconv_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _soft_max(conv, beta):
    """Compute soft-max approximation of max."""
    return (1.0 / beta) * np.log(np.sum(np.exp(beta * conv)))


def _soft_max_obj(h, beta):
    """Compute soft-max objective."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = _autoconv_fft(h)
    sm = _soft_max(conv, beta)
    return abs(2.0 * N * sm / (s * s))


def _random_init(rng, n_sparse=None):
    """Generate random initial h vector."""
    if n_sparse is not None and n_sparse < N:
        h = np.zeros(N)
        idx = rng.choice(N, size=n_sparse, replace=False)
        h[idx] = rng.randn(n_sparse)
        # Ensure non-zero sum
        if abs(np.sum(h)) < 1e-10:
            h[0] += 1.0
        return h
    else:
        h = rng.randn(N)
        if abs(np.sum(h)) < 1e-10:
            h[0] += 1.0
        return h


def _coordinate_descent(h, max_iters=5000, step_scale=0.1, deadline=None, start_time=None):
    """
    Coordinate descent / hill climbing on h.
    Tries small perturbations in each coordinate direction.
    """
    if start_time is None:
        start_time = time.time()
    if deadline is None:
        deadline = DEADLINE
    
    best_val = _objective(h)
    h = h.copy()
    s = np.sum(h)
    if abs(s) < 1e-12:
        h[0] += 1.0
        s = np.sum(h)
    
    # Normalize for stability
    h_norm = h / s
    best_val = _objective(h_norm)
    
    n_improved = 0
    no_improve_count = 0
    
    for it in range(max_iters):
        if time.time() - start_time > deadline - 5:
            break
        
        if no_improve_count > 200:
            break
        
        # Try random coordinate perturbations
        n_try = 20
        improved = False
        for _ in range(n_try):
            i = np.random.randint(N)
            # Try both positive and negative steps
            for sign in [1.0, -1.0]:
                step = sign * step_scale * np.random.rand()
                h_new = h_norm.copy()
                h_new[i] += step
                s_new = np.sum(h_new)
                if abs(s_new) < 1e-12:
                    continue
                val_new = _objective(h_new)
                if val_new < best_val:
                    h_norm = h_new
                    best_val = val_new
                    improved = True
                    n_improved += 1
                    break
        if not improved:
            no_improve_count += 1
            # Occasionally try larger steps
            if no_improve_count % 50 == 0:
                step_scale *= 0.5
        else:
            no_improve_count = 0
    
    return h_norm, best_val


def _full_vector_optimize(h, max_iters=100, deadline=None, start_time=None):
    """
    Try to refine using scipy L-BFGS-B with soft-max smoothing.
    """
    if start_time is None:
        start_time = time.time()
    if deadline is None:
        deadline = DEADLINE
    
    from scipy.optimize import minimize
    
    # Use soft-max with increasing beta
    best_h = h.copy()
    best_val = _objective(h)
    
    for beta in [10.0, 50.0, 100.0, 200.0]:
        if time.time() - start_time > deadline - 10:
            break
        
        def obj_soft(x):
            s = np.sum(x)
            if abs(s) < 1e-12:
                return 1e10
            conv = _autoconv_fft(x)
            sm = _soft_max(conv, beta)
            return abs(2.0 * N * sm / (s * s))
        
        try:
            res = minimize(obj_soft, best_h, method='L-BFGS-B',
                         options={'maxiter': max_iters, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_val:
                val_new = _objective(res.x)
                if val_new < best_val:
                    best_val = val_new
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val


def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    # Start with seed
    best_h = np.ones(N)
    best_val = _objective(best_h)  # should be 2.0
    
    # Strategy 1: Multiple random inits with coordinate descent
    n_random_inits = 10
    for i in range(n_random_inits):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Mix of sparse and dense initializations
        if i < 4:
            h_init = _random_init(rng, n_sparse=rng.choice([50, 100, 150, 200]))
        elif i < 8:
            h_init = _random_init(rng)
        else:
            # Near-constant with small perturbations
            h_init = np.ones(N) + rng.randn(N) * 0.1
        
        h_cd, val_cd = _coordinate_descent(h_init, max_iters=3000, step_scale=0.2,
                                           deadline=DEADLINE, start_time=start_time)
        
        if val_cd < best_val:
            best_val = val_cd
            best_h = h_cd.copy()
    
    # Strategy 2: Refine best with full-vector optimization
    if time.time() - start_time < DEADLINE - 15:
        best_h, best_val = _full_vector_optimize(best_h, max_iters=100,
                                                 deadline=DEADLINE, start_time=start_time)
    
    # Strategy 3: More coordinate descent rounds on best
    if time.time() - start_time < DEADLINE - 10:
        best_h, best_val = _coordinate_descent(best_h, max_iters=5000, step_scale=0.05,
                                               deadline=DEADLINE, start_time=start_time)
    
    # Final refinement
    if time.time() - start_time < DEADLINE - 5:
        best_h, best_val = _full_vector_optimize(best_h, max_iters=50,
                                                 deadline=DEADLINE, start_time=start_time)
    
    # Final verification
    final_val = _objective(best_h)
    if final_val < best_val:
        best_val = final_val
    
    return best_h


# Quick test
if __name__ == '__main__':
    h = construct()
    val = _objective(h)
    print(f"Value: {val}")
    print(f"Score: {2.0 / val}")
