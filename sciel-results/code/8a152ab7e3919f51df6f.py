"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START

def _autoconv(x):
    """Compute the full linear autoconvolution of x via FFT."""
    n = len(x)
    size = 2 * n - 1
    # Pad to next power of 2 for FFT efficiency
    fft_size = 1 << (size - 1).bit_length()
    X = np.fft.rfft(x, fft_size)
    ac = np.fft.irfft(X * X, fft_size)[:size]
    return ac

def _objective(x):
    """
    Objective: maximize the squared integral (L2 norm squared) subject to
    the constraint that max|autoconv| <= 1.
    We minimize -||x||^2 + penalty * max(0, max|ac| - 1)
    """
    x = np.asarray(x, dtype=np.float64)
    ac = _autoconv(x)
    max_ac = np.max(np.abs(ac))
    penalty = 1e4
    obj = -np.dot(x, x) + penalty * max(0.0, max_ac - 1.0)
    return obj

def _objective_grad(x):
    """
    Gradient of the objective.
    obj = -||x||^2 + penalty * max(0, max|ac| - 1)
    grad = -2*x + penalty * grad(max(0, max|ac|-1))
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    size = 2 * n - 1
    fft_size = 1 << (size - 1).bit_length()
    
    X = np.fft.rfft(x, fft_size)
    ac_full = np.fft.irfft(X * X, fft_size)[:size]
    
    max_ac = np.max(np.abs(ac_full))
    if max_ac <= 1.0:
        return -2.0 * x
    
    # Find argmax of |ac_full|
    idx = np.argmax(np.abs(ac_full))
    sign_ac = np.sign(ac_full[idx])
    
    # Gradient of ac[k] with respect to x[j]:
    # ac[k] = sum_j x[j] * x[k-j] (with appropriate bounds)
    # d(ac[k])/dx[j] = x[k-j] + x[j-k] (if indices valid)
    # So grad of |ac[k]| w.r.t x[j] = sign_ac * (x[k-j] + x[j-k])
    
    j = np.arange(n)
    # x[k-j] is valid when 0 <= k-j < n
    mask1 = (idx - j >= 0) & (idx - j < n)
    # x[j-k] is valid when 0 <= j-idx < n  => j >= idx and j - idx < n
    mask2 = (j - idx >= 0) & (j - idx < n)
    
    grad_ac = np.zeros(n)
    grad_ac[mask1] += x[idx - j[mask1]]
    grad_ac[mask2] += x[j[mask2] - idx]
    
    penalty = 1e4
    return -2.0 * x + penalty * sign_ac * grad_ac

def _score(x):
    """
    Score is the upper bound on C3': max|autoconv| / (integral)^2
    where integral = sum(x) (or L1? The task says "squared integral").
    Let's interpret "integral" as the sum of the step heights (L1 norm of the discrete signal).
    Actually, re-reading: "minimizing the maximum absolute autoconvolution divided by squared integral"
    C3' = max|ac| / (sum(x))^2 ? Or max|ac| / (||x||_1^2)?
    Given the context of C3 (autocorrelation), it's likely max|ac| / (sum(x))^2.
    Let's assume integral = sum(x).
    """
    x = np.asarray(x, dtype=np.float64)
    s = np.sum(x)
    if np.abs(s) < 1e-12:
        return np.inf
    ac = _autoconv(x)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial random initialization
    np.random.seed(42)
    x0 = np.random.randn(N)
    
    # Normalize initial guess
    s0 = np.sum(x0)
    if np.abs(s0) > 1e-12:
        x0 = x0 / s0  # Make integral 1
    
    best_x = x0.copy()
    best_score = _score(best_x)
    
    # Optimization loop
    # We want to minimize max|ac| / (sum(x))^2
    # This is equivalent to minimizing max|ac| subject to sum(x) = 1
    # Or maximizing (sum(x))^2 / max|ac|
    
    # Let's use a penalty method to enforce sum(x) = 1
    # Objective: max|ac| + lambda * (sum(x) - 1)^2
    
    def obj_penalty(x):
        ac = _autoconv(x)
        max_ac = np.max(np.abs(ac))
        s = np.sum(x)
        return max_ac + 100.0 * (s - 1.0)**2
    
    def grad_penalty(x):
        ac = _autoconv(x)
        max_ac = np.max(np.abs(ac))
        idx = np.argmax(np.abs(ac))
        sign_ac = np.sign(ac[idx])
        
        j = np.arange(N)
        mask1 = (idx - j >= 0) & (idx - j < N)
        mask2 = (j - idx >= 0) & (j - idx < N)
        
        grad_ac = np.zeros(N)
        grad_ac[mask1] += x[idx - j[mask1]]
        grad_ac[mask2] += x[j[mask2] - idx]
        
        s = np.sum(x)
        return sign_ac * grad_ac + 200.0 * (s - 1.0) * np.ones(N)
    
    # Start with a few random restarts
    for restart in range(5):
        if time.time() > deadline - 20.0:
            break
            
        if restart == 0:
            x_current = x0.copy()
        else:
            x_current = np.random.randn(N)
            s = np.sum(x_current)
            if np.abs(s) > 1e-12:
                x_current = x_current / s
        
        # Run L-BFGS-B
        try:
            res = minimize(
                obj_penalty,
                x_current,
                jac=grad_penalty,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < 1e10:
                x_new = res.x
                score_new = _score(x_new)
                if score_new < best_score:
                    best_score = score_new
                    best_x = x_new.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 10.0:
            break
    
    return best_x

# EVOLVE-BLOCK-END
