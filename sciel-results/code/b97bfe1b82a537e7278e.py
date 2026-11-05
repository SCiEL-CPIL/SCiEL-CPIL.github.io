"""Rewritten program for the third autocorrelation inequality task."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 400
DEADLINE = 135.0  # safety margin before 150s kill

def _value(h):
    """Compute the task value."""
    s = np.sum(h)
    if abs(s) < 1e-15:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s * s))

def _pulse_param_to_h(params):
    """
    Convert pulse parameters to step heights.
    params: [a1, p1, w1, a2, p2, w2, ..., aK, pK, wK]
    where a_i is amplitude, p_i is center position in [0, N), w_i is half-width.
    """
    K = len(params) // 3
    h = np.zeros(N)
    for i in range(K):
        a = params[3*i]
        p = params[3*i + 1]
        w = max(1.0, abs(params[3*i + 2]))
        # Integer positions for the pulse
        left = int(np.floor(p - w))
        right = int(np.ceil(p + w))
        left = max(0, left)
        right = min(N, right)
        if right > left:
            h[left:right] += a
    return h

def _pulse_objective(params):
    """Objective function for pulse-based parameterization."""
    h = _pulse_param_to_h(params)
    return _value(h)

def construct():
    t0 = time.time()
    
    # Strategy 1: Start with seed
    h_seed = np.ones(N)
    best_h = h_seed.copy()
    best_val = _value(best_h)
    
    # Strategy 2: Pulse-based optimization
    # Try different numbers of pulses
    for K in [2, 3, 4, 5]:
        if time.time() - t0 > DEADLINE - 20:
            break
        
        n_params = 3 * K
        # Bounds: amplitude [-10, 10], position [0, N), width [1, N/2]
        bounds = []
        for i in range(K):
            bounds.append((-10.0, 10.0))  # amplitude
            bounds.append((0.0, float(N)))  # position
            bounds.append((1.0, N / 2.0))  # half-width
        
        def pulse_obj(params):
            if time.time() - t0 > DEADLINE - 5:
                return best_val
            h = _pulse_param_to_h(params)
            s = np.sum(h)
            if abs(s) < 1e-10:
                return 1e10
            conv = np.convolve(h, h, mode='full')
            return abs(2.0 * N * np.max(conv) / (s * s))
        
        # Differential Evolution for pulse parameters
        try:
            de_result = differential_evolution(
                pulse_obj, bounds,
                maxiter=100, popsize=15,
                tol=1e-8, seed=42,
                polish=False,
                init='sobol'
            )
            if de_result.fun < best_val:
                h_candidate = _pulse_param_to_h(de_result.x)
                if np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-10:
                    v = _value(h_candidate)
                    if v < best_val:
                        best_h = h_candidate.copy()
                        best_val = v
        except Exception:
            pass
        
        # Local refinement from DE result
        if time.time() - t0 < DEADLINE - 10:
            try:
                res = minimize(
                    pulse_obj,
                    de_result.x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-9}
                )
                h_candidate = _pulse_param_to_h(res.x)
                if np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-10:
                    v = _value(h_candidate)
                    if v < best_val:
                        best_h = h_candidate.copy()
                        best_val = v
            except Exception:
                pass
    
    # Strategy 3: Direct optimization with smooth surrogate on full h vector
    if time.time() - t0 < DEADLINE - 15:
        def smooth_obj(h, beta=50.0):
            s = np.sum(h)
            if abs(s) < 1e-10:
                return 1e10
            conv = np.convolve(h, h, mode='full')
            m = np.max(conv)
            shifted = conv - m
            logsumexp = m + np.log(np.sum(np.exp(beta * shifted))) / beta
            val = 2.0 * N * logsumexp / (s * s)
            return abs(val)
        
        # Try a few initial points
        rng = np.random.default_rng(123)
        for init_type in ['ones', 'random', 'bimodal']:
            if time.time() - t0 > DEADLINE - 5:
                break
            
            if init_type == 'ones':
                h0 = np.ones(N)
            elif init_type == 'random':
                h0 = rng.normal(0, 1, N)
            else:
                # Bimodal: positive on left half, negative on right half
                h0 = np.ones(N)
                h0[N//2:] = -1.0
            
            try:
                res = minimize(
                    smooth_obj,
                    h0,
                    args=(50.0,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-9}
                )
                if np.all(np.isfinite(res.x)) and abs(np.sum(res.x)) > 1e-10:
                    v = _value(res.x)
                    if v < best_val:
                        best_h = res.x.copy()
                        best_val = v
            except Exception:
                pass
    
    # Strategy 4: Final polish with higher beta
    if time.time() - t0 < DEADLINE - 3:
        def smooth_obj_high(h, beta=200.0):
            s = np.sum(h)
            if abs(s) < 1e-10:
                return 1e10
            conv = np.convolve(h, h, mode='full')
            m = np.max(conv)
            shifted = conv - m
            logsumexp = m + np.log(np.sum(np.exp(beta * shifted))) / beta
            val = 2.0 * N * logsumexp / (s * s)
            return abs(val)
        
        try:
            res = minimize(
                smooth_obj_high,
                best_h,
                args=(200.0,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-11}
            )
            if np.all(np.isfinite(res.x)) and abs(np.sum(res.x)) > 1e-10:
                v = _value(res.x)
                if v < best_val:
                    best_h = res.x.copy()
                    best_val = v
        except Exception:
            pass
    
    # Final validation
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-15:
        return np.ones(N)
    
    return best_h
