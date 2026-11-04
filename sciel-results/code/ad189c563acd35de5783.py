"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.signal import fftconvolve

N = 400
DEADLINE = 115.0  # safety margin before 120s limit


def _autoconv_peak(h):
    """Compute the L-infinity norm of the autoconvolution of h."""
    # h is a sequence of N signed step heights
    # autoconvolution: c[k] = sum_i h[i] * h[i+k]
    # Using FFT for efficiency
    c = fftconvolve(h, h, mode='full')
    # The autoconvolution has length 2*N - 1
    # The peak is the maximum absolute value
    return np.max(np.abs(c))


def _score_from_heights(h):
    """Score is the upper bound on C3, which we interpret as the autoconvolution peak."""
    return _autoconv_peak(h)


def _random_signed_step():
    """Generate a random signed step function."""
    return np.random.choice([-1.0, 1.0], size=N)


def _optimize_heights(x0, time_limit=10.0):
    """Optimize heights using Nelder-Mead or similar local method."""
    start_time = time.time()
    
    def objective(h):
        if time.time() - start_time > time_limit:
            return float('inf')
        return _autoconv_peak(h)
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 10000, 'xatol': 1e-4, 'fatol': 1e-4})
    return result.fun, result.x


def _de_search(time_budget):
    """Global search using differential evolution."""
    start_time = time.time()
    
    def objective(h):
        if time.time() - start_time > time_budget:
            return float('inf')
        return _autoconv_peak(h)
    
    bounds = [(-1.0, 1.0)] * N
    result = differential_evolution(objective, bounds, maxiter=100, 
                                     popsize=15, tol=1e-3, seed=42,
                                     polish=False, init='latinhypercube')
    return result.fun, result.x


def construct():
    start_time = time.time()
    best_score = float('inf')
    best_h = _random_signed_step()
    
    # First, try a few random initializations
    for i in range(5):
        h = _random_signed_step()
        s = _score_from_heights(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
        if time.time() - start_time > DEADLINE - 60:
            break
    
    # Try local optimization from best random start
    if time.time() - start_time < DEADLINE - 30:
        local_time = min(20.0, DEADLINE - 30 - (time.time() - start_time))
        try:
            s, h_opt = _optimize_heights(best_h, time_limit=local_time)
            if s < best_score:
                best_score = s
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Try differential evolution if time permits
    if time.time() - start_time < DEADLINE - 15:
        de_time = DEADLINE - 15 - (time.time() - start_time)
        if de_time > 10:
            try:
                s, h_de = _de_search(de_time)
                if s < best_score:
                    best_score = s
                    best_h = h_de.copy()
            except Exception:
                pass
    
    # Final polish: try to improve with a few more local steps
    if time.time() - start_time < DEADLINE - 5:
        polish_time = DEADLINE - 5 - (time.time() - start_time)
        if polish_time > 2:
            try:
                s, h_polished = _optimize_heights(best_h, time_limit=polish_time)
                if s < best_score:
                    best_score = s
                    best_h = h_polished.copy()
            except Exception:
                pass
    
    # Return the best heights found
    # Ensure it's a sequence of signed step heights
    return best_h
# EVOLVE-BLOCK-END
