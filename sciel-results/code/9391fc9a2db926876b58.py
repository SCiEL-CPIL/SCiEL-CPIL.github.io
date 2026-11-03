"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Approach: FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) with a smoothed
max objective. The objective is a log-sum-exp of the overlaps, which approximates
the max overlap. Gradients are computed via convolutions. Projection onto the
constraint set (sum=N/2, 0<=h<=1) is handled via bisection on the Lagrange multiplier.
"""
import time
import numpy as np

N = 100
_DEADLINE = time.time() + 115


def _compute_overlaps(h):
    """Compute C_k = (2/N)*(N/2 - r_k) for k=0..N-1."""
    n = len(h)
    r = np.zeros(n)
    for k in range(n):
        r[k] = np.dot(h[k:], h[:n-k])
    return (2.0 / n) * (n / 2.0 - r)


def _score(h):
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _compute_overlaps(h)
    return float(np.max(cs))


def _smooth_obj(y, beta):
    """Compute the smooth log-sum-exp objective."""
    cs = _compute_overlaps(y)
    cs_max = np.max(cs)
    return (1.0 / beta) * np.log(np.sum(np.exp(beta * (cs - cs_max)))) + cs_max


def _project(h, target_sum):
    """
    Project h onto the set {x : sum(x) = target_sum, 0 <= x_i <= 1}.
    Uses bisection on lambda to find x_i = clip(h_i - lambda, 0, 1) s.t. sum(x) = target_sum.
    """
    # Bisection bounds for lambda
    lo = np.min(h) - 1.0
    hi = np.max(h) + 1.0
    
    def sum_at_lambda(lam):
        return np.sum(np.clip(h - lam, 0.0, 1.0))
    
    # We want sum_at_lambda(lam) = target_sum
    # sum_at_lambda is decreasing in lam.
    
    if target_sum <= 0:
        return np.zeros(len(h))
    if target_sum >= len(h):
        return np.ones(len(h))
        
    # Standard bisection
    for _ in range(50):
        mid = (lo + hi) / 2.0
        s_mid = sum_at_lambda(mid)
        if s_mid > target_sum:
            lo = mid
        else:
            hi = mid
            
    lam = (lo + hi) / 2.0
    return np.clip(h - lam, 0.0, 1.0)


def construct():
    """
    Construct step heights minimizing maximum overlap with shifted complement.
    Uses FISTA with smoothed max objective.
    """
    t0 = time.time()
    
    # Parameters
    beta = 20.0  # Smoothing parameter for log-sum-exp
    max_iter = 5000
    
    # Initialize
    h = np.full(N, 0.5)
    y = h.copy()
    t_k = 1.0
    
    best_h = h.copy()
    best_score = _score(best_h)
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
            
        # 1. Compute overlaps and smooth objective at y
        cs = _compute_overlaps(y)
        f_y = _smooth_obj(y, beta)
        
        # 2. Compute softmax probabilities p_k
        cs_max = np.max(cs)
        exp_cs = np.exp(beta * (cs - cs_max))
        Z = np.sum(exp_cs)
        p = exp_cs / Z
        
        # 3. Compute gradient of smoothed objective L(y)
        # grad L = beta * sum p_k grad C_k
        # grad C_k(i) = -2/N * (y[i-k] + y[i+k])
        # So grad L = -2/N * sum_k p_k * (y[i-k] + y[i+k])
        
        S = np.zeros(N)
        for k in range(N):
            pk = p[k]
            if pk < 1e-6:
                continue
            # Add pk * y[i+k] for valid i
            S[:N-k] += pk * y[k:]
            # Add pk * y[i-k] for valid i
            S[k:] += pk * y[:N-k]
            
        grad = -2.0 / N * S
        
        # 4. Backtracking line search
        alpha = 1.0
        grad_norm_sq = np.dot(grad, grad)
        if grad_norm_sq < 1e-12:
            break
            
        for _ in range(20):
            z = y - alpha * grad
            y_proj = _project(z, N / 2.0)
            f_z = _smooth_obj(y_proj, beta)
            if f_z <= f_y - 0.5 * alpha * grad_norm_sq:
                break
            alpha *= 0.5
        y_new = y_proj
        
        # 5. FISTA momentum
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        y = y_new + ((t_k - 1.0) / t_next) * (y_new - y)
        y = _project(y, N / 2.0)
        
        t_k = t_next
        
        # Check convergence
        score = _score(y)
        if score < best_score:
            best_score = score
            best_h = y.copy()
            
    # Ensure valid output
    best_h = np.clip(best_h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-12:
        best_h = _project(best_h, N / 2.0)
        
    return best_h
