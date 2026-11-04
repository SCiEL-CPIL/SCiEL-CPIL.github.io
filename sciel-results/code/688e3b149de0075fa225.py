"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
START_TIME = time.time()
TIME_LIMIT = 110.0  # safety margin before 120s kill

def _overlap_at_shifts(h, shifts):
    """Compute overlap integral at each shift. h is array of N steps on [0,2].
    overlap(s) = integral_0^2 h(x) * (1 - h(x+s)) dx with wrap-around or zero extension.
    For step functions, we compute this discretely.
    """
    # h has N steps each of width 2/N on [0,2]
    # For shift s (in units of step widths), compute overlap
    # Using discrete approximation: overlap_k = sum_i h[i] * (1 - h[(i+k) % N]) * (2/N)
    # But we need to be careful about the shift interpretation.
    # Actually for the Erdos problem, the shift is continuous. Let's use the discrete shifts.
    
    # For discrete shift k (0 to N-1):
    # overlap_k = (2/N) * sum_{i=0}^{N-1} h[i] * (1 - h[(i+k) % N])
    # This is the integral approximation.
    
    # More carefully: h is defined on [0,2] with N equal steps.
    # h(x) = h[i] for x in [2i/N, 2(i+1)/N)
    # Shift by s: (1-h)(x+s) means 1 - h(x+s)
    # Overlap(s) = integral_0^2 h(x) * (1 - h(x+s)) dx
    
    # For discrete shifts that are multiples of 2/N:
    # shift k means s = 2k/N
    # Then h(x + 2k/N) is h[(i+k) % N] for x in step i (with wrap)
    
    # Let's compute for all N discrete shifts
    # overlap_k = (2/N) * sum_i h[i] * (1 - h[(i+k) % N])
    
    h_ext = np.tile(h, 2)  # for wrap-around indexing
    # For shift k: h_shifted[i] = h[(i+k) % N] = h_ext[i+k]
    # overlap_k = (2/N) * dot(h, 1 - h_shifted)
    
    # Compute all overlaps at once using FFT or direct
    # overlap_k = (2/N) * (sum_i h[i] - sum_i h[i]*h[(i+k)%N])
    # sum_i h[i] = N/2 (constraint)
    # So overlap_k = (2/N) * (N/2 - sum_i h[i]*h[(i+k)%N]) = 1 - (2/N)*sum_i h[i]*h[(i+k)%N]
    
    # Compute autocorrelation: c[k] = sum_i h[i]*h[(i+k)%N]
    # This is circular cross-correlation
    # Use FFT
    H = np.fft.fft(h, 2*N)
    # Circular autocorrelation of length N
    # c[k] = sum_i h[i] * h[(i+k) % N]
    # This is the circular cross-correlation of h with itself, shifted
    # Using FFT: c = ifft(fft(h) * conj(fft(h))) gives c[k] = sum_i h[i+k]*h[i] (need to be careful)
    
    # Let me just do it directly for N=100 (100*100 = 10000 ops, fast)
    overlaps = np.zeros(N)
    for k in range(N):
        h_shifted = np.roll(h, -k)
        overlaps[k] = (2.0/N) * np.sum(h * (1.0 - h_shifted))
    
    return overlaps

def _objective_with_t(x, n_shifts_sample=50):
    """Epigraph: x = [h_0, ..., h_{N-1}, t]
    Minimize t + penalty for constraints
    """
    h = x[:N]
    t = x[N]
    
    # Compute overlaps for a subset of shifts (or all N)
    overlaps = _overlap_at_shifts(h, None)
    
    # Soft max: use log-sum-exp for smoothness
    # max(overlaps) approximated by log-sum-exp
    # But we also need to enforce h in [0,1] and sum(h) = N/2
    
    # Use a smooth approximation: soft_max = (1/beta) * log(sum(exp(beta * overlaps)))
    beta = 20.0
    soft_max = (1.0/beta) * np.log(np.sum(np.exp(beta * overlaps)))
    
    # Penalty for constraints
    # h in [0,1]: penalty if outside
    h_pen = np.sum(np.maximum(0, h - 1)**2) + np.sum(np.maximum(0, -h)**2)
    # sum(h) = N/2
    sum_pen = (np.sum(h) - N/2)**2
    
    return t + 0.1 * soft_max + 10.0 * h_pen + 10.0 * sum_pen

def _objective_direct(h):
    """Direct objective: max overlap (with smooth approximation)"""
    overlaps = _overlap_at_shifts(h, None)
    beta = 20.0
    soft_max = (1.0/beta) * np.log(np.sum(np.exp(beta * overlaps)))
    return soft_max

def _project(h):
    """Project h to satisfy sum(h) = N/2 and 0 <= h <= 1"""
    h = h.copy()
    # First clip to [0,1]
    h = np.clip(h, 0, 1)
    # Then adjust to make sum = N/2
    # Simple: shift all values
    current_sum = np.sum(h)
    target_sum = N / 2.0
    if current_sum != 0:
        h = h + (target_sum - current_sum) / N
        h = np.clip(h, 0, 1)
        # Iterate a few times
        for _ in range(20):
            current_sum = np.sum(h)
            diff = target_sum - current_sum
            if abs(diff) < 1e-10:
                break
            # Only adjust elements that aren't at bounds
            can_increase = h < 1.0
            can_decrease = h > 0.0
            if diff > 0:
                mask = can_increase
                if np.sum(mask) > 0:
                    h[mask] += diff / np.sum(mask)
            else:
                mask = can_decrease
                if np.sum(mask) > 0:
                    h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0, 1)
    return h

def _random_init(seed):
    """Generate a random valid initial h"""
    rng = np.random.RandomState(seed)
    # Start with random values and project
    h = rng.uniform(0, 1, N)
    h = _project(h)
    return h

def _try_optimize(h0, max_iter=200, method='L-BFGS-B'):
    """Optimize from initial point h0"""
    def obj(h):
        if time.time() - START_TIME > TIME_LIMIT:
            return 1e10
        h_proj = _project(h)
        overlaps = _overlap_at_shifts(h_proj, None)
        beta = 20.0
        soft_max = (1.0/beta) * np.log(np.sum(np.exp(beta * overlaps)))
        # Add small penalty to keep in bounds
        pen = np.sum(np.maximum(0, h - 1)**2) + np.sum(np.maximum(0, -h)**2)
        return soft_max + 1.0 * pen
    
    def grad(h):
        if time.time() - START_TIME > TIME_LIMIT:
            return np.zeros(N)
        # Numerical gradient
        eps = 1e-6
        g = np.zeros(N)
        for i in range(N):
            h_plus = h.copy()
            h_plus[i] += eps
            h_minus = h.copy()
            h_minus[i] -= eps
            g[i] = (obj(h_plus) - obj(h_minus)) / (2*eps)
        return g
    
    try:
        result = minimize(obj, h0, method='L-BFGS-B', jac=grad, 
                         options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
        h_opt = _project(result.x)
        return h_opt, _objective_direct(h_opt)
    except Exception:
        return h0, _objective_direct(h0)

def construct():
    best_h = np.full(N, 0.5)
    best_score = _objective_direct(best_h)
    
    # Try multiple random restarts
    n_restarts = 20
    for i in range(n_restarts):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Diverse initializations
        if i == 0:
            h0 = np.full(N, 0.5)
        elif i == 1:
            # Block pattern
            h0 = np.zeros(N)
            h0[::2] = 1.0
            h0 = _project(h0)
        elif i == 2:
            # Another block pattern
            h0 = np.zeros(N)
            h0[::3] = 1.5
            h0 = _project(h0)
        elif i == 3:
            # Sinusoidal
            h0 = 0.5 + 0.5 * np.sin(2*np.pi*3*np.arange(N)/N)
            h0 = _project(h0)
        elif i == 4:
            # Another sinusoidal
            h0 = 0.5 + 0.5 * np.sin(2*np.pi*5*np.arange(N)/N)
            h0 = _project(h0)
        else:
            h0 = _random_init(i * 1000 + 42)
        
        h_opt, score = _try_optimize(h0, max_iter=100)
        
        if score < best_score:
            best_score = score
            best_h = h_opt
    
    # Final polish with tighter tolerance
    if time.time() - START_TIME < TIME_LIMIT - 5:
        def obj_polish(h):
            h_proj = _project(h)
            overlaps = _overlap_at_shifts(h_proj, None)
            beta = 50.0  # tighter
            soft_max = (1.0/beta) * np.log(np.sum(np.exp(beta * overlaps)))
            pen = np.sum(np.maximum(0, h - 1)**2) + np.sum(np.maximum(0, -h)**2)
            return soft_max + 1.0 * pen
        
        def grad_polish(h):
            eps = 1e-7
            g = np.zeros(N)
            for i in range(N):
                h_plus = h.copy()
                h_plus[i] += eps
                h_minus = h.copy()
                h_minus[i] -= eps
                g[i] = (obj_polish(h_plus) - obj_polish(h_minus)) / (2*eps)
            return g
        
        try:
            result = minimize(obj_polish, best_h, method='L-BFGS-B', jac=grad_polish,
                             options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            h_polished = _project(result.x)
            score_polished = _objective_direct(h_polished)
            if score_polished < best_score:
                best_h = h_polished
                best_score = score_polished
        except Exception:
            pass
    
    # Ensure exact sum
    best_h = _project(best_h)
    return best_h

# EVOLVE-BLOCK-END
