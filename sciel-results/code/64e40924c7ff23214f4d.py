import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50
_TIME_LIMIT = 140.0  # safety margin below 150s kill
_start_time = time.time()

def _score(h):
    h = np.asarray(h, dtype=float)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 < 1e-12:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv < 1e-12:
        return 0.0
    return s2 / (s1 * max_conv)

def _gaussian_basis(center, width, n=N):
    """Create a Gaussian-shaped bump centered at 'center' with given 'width'."""
    idx = np.arange(n)
    return np.exp(-0.5 * ((idx - center) / width) ** 2)

def _build_from_params(params):
    """
    Build height vector from parameters: [amplitudes..., centers..., widths...]
    We use K Gaussian bumps.
    """
    K = len(params) // 3
    amps = params[:K]
    centers = params[K:2*K]
    widths = params[2*K:]
    
    h = np.zeros(N)
    for k in range(K):
        h += amps[k] * _gaussian_basis(centers[k], max(widths[k], 0.5))
    h = np.clip(h, 0, None)
    # Normalize so that sum is reasonable (scale invariant)
    s = np.sum(h)
    if s < 1e-12:
        return h
    return h

def _score_from_params(params):
    h = _build_from_params(params)
    return _score(h)

def construct():
    best_h = np.array([2.0/3.0, 1.0/3.0] + [0.0]*(N-2))
    best_score = _score(best_h)
    
    # Also try some simple spread patterns as seeds
    simple_seeds = [
        [1.0/2, 1.0/2] + [0.0]*(N-2),
        [1.0/3]*3 + [0.0]*(N-3),
        [0.2]*5 + [0.0]*(N-5),
        [0.15]*7 + [0.0]*(N-7),
        [0.1]*10 + [0.0]*(N-10),
    ]
    for seed in simple_seeds:
        s = _score(np.array(seed))
        if s > best_score:
            best_score = s
            best_h = np.array(seed)
    
    # Strategy: Use differential evolution over a parameterized family of
    # smooth bumps. This explores a fundamentally different shape space.
    
    K = 3  # number of Gaussian bumps
    bounds = [
        # amplitudes: [0, 2]
        [(0, 2)] * K,
        # centers: [0, N-1]
        [(0, N-1)] * K,
        # widths: [1, 10]
        [(1, 10)] * K,
    ]
    
    # Run differential evolution with limited evaluations
    rng_seed = 42
    de_kwargs = {
        'bounds': bounds,
        'strategy': 'best1bin',
        'maxiter': 30,
        'popsize': 15,
        'tol': 1e-6,
        'seed': rng_seed,
        'polish': False,
        'init': 'latinhypercube',
    }
    
    try:
        result = differential_evolution(
            lambda p: -_score_from_params(p),
            **de_kwargs
        )
        if result.success or result.fun < -best_score:
            h_opt = _build_from_params(result.x)
            s_opt = _score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
    except Exception:
        pass
    
    # Local refinement around best using Nelder-Mead on the full 50-dim space
    if time.time() - _start_time < _TIME_LIMIT - 20:
        def neg_score(x):
            x = np.clip(x, 0, None)
            return -_score(x)
        
        try:
            result2 = minimize(
                neg_score, best_h, method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
            )
            h_opt2 = np.clip(result2.x, 0, None)
            s_opt2 = _score(h_opt2)
            if s_opt2 > best_score:
                best_score = s_opt2
                best_h = h_opt2.copy()
        except Exception:
            pass
    
    # Additional random restarts in parameter space if time permits
    if time.time() - _start_time < _TIME_LIMIT - 30:
        rng = np.random.default_rng(123)
        for trial in range(5):
            if time.time() - _start_time > _TIME_LIMIT - 10:
                break
            # Random parameters
            amps = rng.uniform(0.1, 1.5, K)
            centers = rng.uniform(0, N-1, K)
            widths = rng.uniform(1, 8, K)
            params = np.concatenate([amps, centers, widths])
            h_trial = _build_from_params(params)
            s_trial = _score(h_trial)
            if s_trial > best_score:
                best_score = s_trial
                best_h = h_trial.copy()
            
            # Quick local polish from this trial
            if time.time() - _start_time < _TIME_LIMIT - 15:
                try:
                    result3 = minimize(
                        lambda x: -_score(np.clip(x, 0, None)),
                        best_h, method='Nelder-Mead',
                        options={'maxiter': 50, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
                    )
                    h_opt3 = np.clip(result3.x, 0, None)
                    s_opt3 = _score(h_opt3)
                    if s_opt3 > best_score:
                        best_score = s_opt3
                        best_h = h_opt3.copy()
                except Exception:
                    pass
    
    # Final tiny perturbations
    rng_final = np.random.default_rng(999)
    for _ in range(30):
        if time.time() - _start_time > _TIME_LIMIT - 5:
            break
        h_pert = best_h + rng_final.normal(0, 0.01, N)
        h_pert = np.clip(h_pert, 0, None)
        s_pert = _score(h_pert)
        if s_pert > best_score:
            best_score = s_pert
            best_h = h_pert
    
    return [float(x) for x in best_h]
