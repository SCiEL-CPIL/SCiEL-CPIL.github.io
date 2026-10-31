import numpy as np
import time
from scipy.optimize import minimize

N = 400
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0
DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN

def _autoconv(f):
    return np.convolve(f, f, mode='full')

def _objective(f):
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv(f)
    return np.max(np.abs(C)) / (S * S)

def _smooth_objective(f, t=0.1):
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv(f)
    absC = np.abs(C)
    max_val = np.max(absC)
    lse = max_val + np.log(np.sum(np.exp(t * (absC - max_val))))
    return (lse / t) / (S * S)

def _gradient_smooth(f, t=0.1):
    S = np.sum(f)
    if abs(S) < 1e-12:
        return np.zeros_like(f)
    C = _autoconv(f)
    absC = np.abs(C)
    signC = np.sign(C)
    max_val = np.max(absC)
    exp_term = np.exp(t * (absC - max_val))
    sum_exp = np.sum(exp_term)
    weights = exp_term / sum_exp
    A = weights * signC
    
    grad = np.zeros_like(f)
    for i in range(N):
        k_start = i
        k_end = i + N
        if k_start < len(A) and k_end <= len(A):
            grad[i] = 2.0 * np.dot(A[k_start:k_end], f)
        elif k_start < len(A):
            grad[i] = 2.0 * np.dot(A[k_start:], f[:len(A)-k_start])
            
    LSE_val = max_val + np.log(np.sum(np.exp(t * (absC - max_val))))
    G_val = LSE_val / t
    return (1.0 / (S * S)) * grad - (G_val / (S * S * S)) * 2.0 * S

def _van_der_corput(n, base=2):
    """Generate Van der Corput sequence (low discrepancy)."""
    seq = np.zeros(n)
    for i in range(n):
        f = 1.0
        x = i + 1
        val = 0.0
        while x > 0:
            x, r = divmod(x, base)
            f /= base
            val += r * f
        seq[i] = val
    return seq

def construct():
    start_time = time.time()
    
    # Baseline: All ones
    best_f = np.ones(N)
    best_score = _objective(best_f)
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: Optimizing from Low Discrepancy starts
    # Map [0,1) to [-1,1]
    vdc = _van_der_corput(N)
    f_vdc = 2.0 * vdc - 1.0
    
    # Strategy 2: Random Gaussian
    f_gauss = rng.standard_normal(N)
    
    # Strategy 3: Random Uniform
    f_uni = rng.uniform(-1, 1, N)
    
    initializations = [f_vdc, f_gauss, f_uni, rng.choice([-1.0, 1.0], size=N)]
    
    for f0 in initializations:
        if time.time() > DEADLINE:
            break
        
        # Normalize to avoid scale issues, though objective is scale invariant
        S0 = np.sum(f0)
        if abs(S0) < 1e-6:
            continue
            
        # Anneal the temperature for the smooth max
        for t in [0.5, 0.1, 0.05, 0.01]:
            if time.time() > DEADLINE:
                break
            
            try:
                res = minimize(
                    _smooth_objective,
                    f0,
                    args=(t,),
                    method='L-BFGS-B',
                    jac=_gradient_smooth,
                    options={'maxiter': 50, 'maxls': 50, 'ftol': 1e-9, 'gtol': 1e-7}
                )
                
                true_score = _objective(res.x)
                if true_score < best_score:
                    best_score = true_score
                    best_f = res.x.copy()
                    
                # Update f0 for next iteration (cooling)
                f0 = res.x
            except Exception:
                pass

    return best_f
