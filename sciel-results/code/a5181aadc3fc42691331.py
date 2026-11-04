"""Convex spectral magnitude minimization: minimize L1(autoconv) subject to fixed DC component."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_max_abs(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))

def _integral(x):
    return float(np.sum(x))

def _metric(x):
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _autoconv_values(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return conv[:2 * N - 1]

def _soft_max_abs(conv, beta=1000.0):
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    shifted = abs_conv - max_val
    return max_val + np.log(np.sum(np.exp(beta * shifted))) / beta

def _smooth_metric(x, beta=500.0):
    conv = _autoconv_values(x)
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _soft_max_abs(conv, beta) / (s * s)

def _solve_convex_spectral(deadline):
    """
    Solve the convex problem:
    Minimize max_k |IFFT(m^2)_k|
    Subject to:
      m[0] = 1
      m[k] >= 0
    
    Using an epigraph formulation with an auxiliary variable t.
    Variables: m[1..K-1], t
    Objective: t
    Constraints:
      -t <= IFFT(m^2)_k <= t for all k
      m[k] >= 0
    
    Since scipy.optimize.minimize doesn't support constraints with t directly in a simple way for L-BFGS-B,
    we use a penalty method or SLSQP with constraints.
    However, L-BFGS-B is faster. Let's stick to the direct minimization of the soft-max or max.
    The previous approach used L-BFGS-B on the max function directly, which worked.
    Let's try to improve the convergence by using a better initial guess or multiple starts.
    """
    K = _FFT_SIZE // 2 + 1  # Number of points in rfft
    
    def objective(m_vars):
        m = np.zeros(K)
        m[0] = 1.0
        m[1:] = m_vars
        
        power = m * m
        conv = np.fft.irfft(power, _FFT_SIZE)
        max_abs = np.max(np.abs(conv[:2 * N - 1]))
        return max_abs

    def gradient(m_vars):
        # Finite difference for simplicity
        eps = 1e-7
        grad = np.zeros_like(m_vars)
        for i in range(len(m_vars)):
            m_plus = m_vars.copy()
            m_plus[i] += eps
            val_plus = objective(m_plus)
            
            m_minus = m_vars.copy()
            m_minus[i] -= eps
            val_minus = objective(m_minus)
            
            grad[i] = (val_plus - val_minus) / (2 * eps)
        
        return grad

    # Initial guess: flat spectrum
    m_init = np.ones(K - 1) * 0.1
    
    # Bounds: m[k] >= 0
    bounds = [(0, None)] * (K - 1)
    
    try:
        result = minimize(
            objective,
            m_init,
            jac=gradient,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 10000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 50000}
        )
        
        m_opt = np.zeros(K)
        m_opt[0] = 1.0
        m_opt[1:] = result.x
        
        # Construct X
        X = m_opt.astype(complex)
        
        # Construct x
        x = np.fft.irfft(X, _FFT_SIZE)[:N]
        
        return x, _metric(x)
        
    except Exception:
        return None, float('inf')

def _try_local_search_fast(x, deadline, rng, max_rounds=100000):
    """Fast local search with in-place modifications."""
    best_x = x.copy()
    best_val = _metric(best_x)
    no_improve = 0
    step_scale = 0.2
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        if no_improve > 150:
            step_scale *= 0.9
            no_improve = 0
            if step_scale < 0.001:
                break
        
        k = int(rng.integers(1, 5))
        indices = rng.choice(N, size=k, replace=False)
        
        best_delta = 0
        best_idx = None
        for trial in range(2):
            x_test = best_x.copy()
            for idx in indices:
                x_test[idx] += rng.normal(0, step_scale)
            v = _metric(x_test)
            if v < best_val + best_delta:
                best_delta = v - best_val
                best_idx = x_test
        
        if best_idx is not None and best_delta < 0:
            best_val += best_delta
            best_x = best_idx
            no_improve = 0
        else:
            no_improve += 1
    
    return best_x, best_val

def _final_polish(x, deadline, rng):
    """High-precision final polish with coordinate descent."""
    best_x = x.copy()
    best_val = _metric(best_x)
    
    for sweep in range(30):
        if time.time() > deadline:
            break
        
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            
            lo, hi = -0.05, 0.05
            best_delta = 0.0
            best_v = best_val
            
            for _ in range(10):
                if time.time() > deadline:
                    break
                m1 = lo + (hi - lo) / 3
                m2 = hi - (hi - lo) / 3
                
                x1 = best_x.copy()
                x1[idx] += m1
                v1 = _metric(x1)
                
                x2 = best_x.copy()
                x2[idx] += m2
                v2 = _metric(x2)
                
                if v1 < v2:
                    hi = m2
                else:
                    lo = m1
                
                if v1 < best_v:
                    best_v = v1
                    best_delta = m1
                if v2 < best_v:
                    best_v = v2
                    best_delta = m2
            
            if best_delta != 0.0 and best_v < best_val:
                best_x[idx] += best_delta
                best_val = best_v
                improved = True
        
        if not improved:
            break
    
    return best_x, best_val

def construct():
    deadline = time.time() + 145.0
    
    rng = np.random.default_rng(2024)
    
    # Phase 1: Convex Spectral Optimization
    # This finds the theoretical lower bound for the max|autoconv| given the sum.
    x_spec, val_spec = _solve_convex_spectral(deadline - 10)
    
    if x_spec is not None and val_spec < float('inf'):
        best_x = x_spec
        best_val = val_spec
    else:
        # Fallback to ones
        best_x = np.ones(N)
        best_val = _metric(best_x)
    
    # Phase 2: Local Search
    # The convex solution is symmetric (even). Local search can break symmetry if it helps?
    # No, autoconv is determined by power spectrum. Breaking symmetry in time domain
    # changes the power spectrum if we don't adjust phases.
    # But wait, local search changes x directly.
    # Changing x changes |X|^2.
    # So local search can potentially find a better |X|^2 if the convex solution was not global?
    # But the convex solution IS the global minimum for the magnitude spectrum.
    # So local search should not improve the autoconvolution magnitude, unless there's a numerical issue
    # or the "real signal" constraint forces a suboptimal magnitude spectrum?
    # No, any magnitude spectrum m can be realized by a real signal (with appropriate phases).
    # So the convex solution is globally optimal for the metric.
    
    # Why would local search help?
    # Maybe the convex solver didn't converge fully?
    # Or maybe there are multiple local minima in the phase space that affect the "realness"?
    # No, we constructed x from m directly.
    
    # Let's run local search just in case, but expect little improvement.
    if time.time() < deadline - 15:
        best_x, best_val = _try_local_search_fast(best_x, deadline - 15, rng, max_rounds=50000)
    
    # Phase 3: Final polish
    if time.time() < deadline - 2:
        best_x, best_val = _final_polish(best_x, deadline - 2, rng)
    
    return best_x
# EVOLVE-BLOCK-END
