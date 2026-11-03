"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _autoconv_abs_max(x):
    """
    Compute the maximum absolute value of the linear autoconvolution of x.
    x is a 1D array of length N.
    """
    n = len(x)
    # Next power of 2 for FFT efficiency
    size = 1
    while size < 2 * n - 1:
        size *= 2
    X = np.fft.rfft(x, n=size)
    C = np.fft.irfft(X * X, n=size)
    c = C[:2*n-1]
    return np.max(np.abs(c))

def _integral_sq(x):
    """
    Squared integral of the step function.
    Assuming unit width steps, integral = sum(x).
    """
    return np.sum(x) ** 2

def _score(x):
    """
    Score is the ratio: max_abs_autoconv / integral_sq.
    Lower is better.
    """
    num = _autoconv_abs_max(x)
    den = _integral_sq(x)
    if den < 1e-12:
        return float('inf')
    return num / den

def _spectral_objective(h):
    """
    Objective for spectral optimization.
    We want to minimize max_k |C_k| where C_k = sum_i h_i h_{i-k} (linear autoconv).
    However, the ratio is max |C_k| / (sum h)^2.
    
    Since the problem is homogeneous, we can fix sum(h) = 1 and minimize max |C_k|.
    But sum(h)=1 is a constraint.
    
    Alternatively, we can use a soft-max approximation of the max to make it differentiable.
    Let f(h) = max_k |C_k(k)|.
    We can approximate f(h) ~ log(sum exp(beta * |C_k|)) / beta.
    
    However, C_k depends on h in a quadratic way.
    Let's try a different approach:
    Minimize (max_k |C_k|) subject to sum(h) = 1.
    
    Actually, let's just minimize the soft max of the autoconvolution magnitudes.
    And add a penalty if sum(h) is not close to 1.
    
    Wait, the ratio is scale invariant. So we can fix sum(h)=1.
    Then we minimize max_k |C_k|.
    
    Let's use the soft max:
    obj = (1/beta) * log( sum_k exp(beta * |C_k|) )
    where C_k is the linear autoconvolution.
    
    We can compute C via FFT.
    But |C_k| is not differentiable at 0. We can use sqrt(C_k^2 + eps).
    
    Let's define:
    C = IFFT(|FFT(h)|^2)
    obj = (1/beta) * log( sum_k exp(beta * sqrt(C_k^2 + eps)) )
    
    And we need to enforce sum(h) = 1.
    We can use a penalty term: lambda * (sum(h) - 1)^2.
    
    Or we can use a constrained optimization.
    scipy.optimize.minimize with method='SLSQP' supports constraints.
    """
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    H = np.fft.rfft(h, n=size)
    C = np.fft.irfft(H * H, n=size)
    C = C[:2*n-1]
    
    # Soft max of |C_k|
    beta = 10.0
    eps = 1e-12
    abs_C = np.sqrt(C**2 + eps)
    # Avoid overflow in exp
    max_abs_C = np.max(abs_C)
    if max_abs_C > 0:
        obj = (1.0/beta) * (max_abs_C + np.log(np.sum(np.exp(beta * (abs_C - max_abs_C)))) / n)
    else:
        obj = 0.0
    
    # Penalty for sum(h) != 1
    sum_h = np.sum(h)
    penalty = 100.0 * (sum_h - 1.0)**2
    
    return obj + penalty

def _spectral_grad(h):
    """
    Numerical gradient of the spectral objective.
    """
    eps = 1e-6
    grad = np.zeros_like(h)
    base = _spectral_objective(h)
    for i in range(len(h)):
        h_plus = h.copy()
        h_plus[i] += eps
        h_minus = h.copy()
        h_minus[i] -= eps
        grad[i] = (_spectral_objective(h_plus) - _spectral_objective(h_minus)) / (2 * eps)
    return grad

def _generate_candidate(seed):
    """
    Generate a random candidate vector with sum 1.
    """
    rng = np.random.default_rng(seed)
    h = rng.standard_normal(N)
    # Ensure sum is 1
    h = h / np.sum(h)
    return h

def _local_search_refine(h, deadline):
    """
    Local search: try flipping signs or small perturbations.
    """
    best_h = h.copy()
    best_score = _score(best_h)
    
    # Try flipping signs of individual elements
    for _ in range(50):  # Limit iterations
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            new_h = best_h.copy()
            new_h[i] *= -1.0
            s = _score(new_h)
            if s < best_score:
                best_score = s
                best_h = new_h
                improved = True
        if not improved:
            break
    
    return best_h, best_score

def _spectral_optimize(h_init, deadline):
    """
    Use scipy.optimize.minimize to refine the vector in continuous space.
    """
    if time.time() > deadline:
        return h_init
    
    try:
        res = minimize(
            _spectral_objective,
            h_init,
            method='SLSQP',
            constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}],
            options={'maxiter': 50, 'ftol': 1e-6}
        )
        if time.time() > deadline:
            return h_init
        return res.x
    except:
        return h_init

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    best_h = None
    best_score = float('inf')
    
    # 1. Try a few random starts with spectral optimization
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        seed = 42 + i
        h_init = _generate_candidate(seed)
        
        # Spectral optimization
        h_opt = _spectral_optimize(h_init, deadline)
        
        # Local search refinement
        h_refined, s = _local_search_refine(h_opt, deadline)
        
        if s < best_score:
            best_score = s
            best_h = h_refined.copy()
    
    # 2. Try a few structured patterns
    # Pattern: Alternating
    h = np.ones(N)
    h[1::2] = -1.0
    h = h / np.sum(h)
    s = _score(h)
    if s < best_score:
        best_score = s
        best_h = h.copy()
    
    # Pattern: Blocks
    for bs in [2, 4, 8, 16]:
        if time.time() > deadline:
            break
        h = np.zeros(N)
        for i in range(0, N, bs):
            sign = 1.0 if (i // bs) % 2 == 0 else -1.0
            h[i:i+bs] = sign
        h = h / np.sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N) / N
    
    # Ensure the output is a sequence of signed step heights
    # The task says "sequence of signed step heights"
    # The score is the ratio.
    # The evaluator measures the raw value (lower is better).
    # The return value is the sequence.
    
    return best_h
# EVOLVE-BLOCK-END
