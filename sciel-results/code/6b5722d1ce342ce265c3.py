import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """Compute the score for a given height vector h."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution peak
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # The objective is R(h) = ||h||_2^2 / (||h||_1 * max_k (h*h)_k)
    # By homogeneity, we can fix ||h||_1 = 1 and maximize ||h||_2^2 / max_k (h*h)_k
    # Equivalently, minimize max_k (h*h)_k subject to ||h||_1 = 1 and ||h||_2^2 = constant
    # But actually, we want to maximize the ratio.
    #
    # Key insight: The autoconvolution peak is a convex function of h (since (h*h)_k is
    # quadratic in h, and max of convex functions is convex). So minimizing the peak
    # autoconvolution subject to ||h||_1 = 1 and h >= 0 is a convex problem.
    # But we want to maximize ||h||_2^2 / peak_autoconv, which is not convex.
    #
    # Alternative: Fix ||h||_1 = 1. Then maximize ||h||_2^2 / peak_autoconv.
    # This is equivalent to: for a fixed value of peak_autoconv = t, maximize ||h||_2^2
    # subject to ||h||_1 = 1, h >= 0, and (h*h)_k <= t for all k.
    # The constraint (h*h)_k <= t is convex (quadratic inequality). So for fixed t,
    # this is a convex feasibility/optimization problem.
    #
    # We can do a bisection on t: find the minimum t such that there exists h with
    # ||h||_1 = 1, h >= 0, (h*h)_k <= t for all k, and ||h||_2^2 is maximized.
    # Actually, for fixed t, the maximum of ||h||_2^2 subject to the constraints is
    # achieved at some point. We want the largest ratio, so we want small t and large ||h||_2^2.
    #
    # Let's use a different approach: directly optimize with SLSQP using analytic gradients.
    
    x_idx = np.arange(N)
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # Compute autoconvolution peak and its gradient
    def _autoconv_peak_and_grad(h):
        """Return (peak_value, gradient_of_peak) for autoconvolution peak."""
        h = np.asarray(h, dtype=np.float64)
        # Full convolution
        conv = np.convolve(h, h, mode='full')
        peak_idx = np.argmax(conv)
        peak_val = conv[peak_idx]
        
        # Gradient of conv[peak_idx] with respect to h
        # conv[k] = sum_i h[i] * h[k-i] (with appropriate bounds)
        # For mode='full', conv has length 2N-1, conv[k] = sum_{i=max(0,k-(N-1))}^{min(N-1,k)} h[i]*h[k-i]
        # d(conv[k])/dh[j] = h[k-j] + h[j-k] (if indices valid)
        
        k = peak_idx
        grad = np.zeros(N)
        for j in range(N):
            # Term 1: h[j] * h[k-j]
            if 0 <= k - j < N:
                grad[j] += h[k - j]
            # Term 2: h[k-j'] * h[j'] where j' = j, so h[k-j] * h[j]
            # Actually, conv[k] = sum_i h[i]*h[k-i]
            # d/dh[j] = h[k-j] (from i=j term) + sum_{i where k-i=j, i.e., i=k-j} h[i]*1
            # = h[k-j] + h[k-j] = 2*h[k-j] if k-j is valid, else h[k-j] if only one term
            # More carefully:
            # conv[k] = sum_{i=0}^{N-1} h[i] * h[k-i] where h[m]=0 if m not in [0,N-1]
            # d(conv[k])/dh[j] = sum_i [delta_{i,j} * h[k-i] + h[i] * delta_{k-i,j}]
            # = h[k-j] + h[k-j] = 2*h[k-j] (if k-j in [0,N-1])
            # But if k-j is out of bounds, one of the terms vanishes.
            # Actually both terms give the same contribution when valid.
            pass
        
        # Let me redo this more carefully
        # conv[k] = sum_{i=0}^{N-1} h[i] * h[k-i] where h[m] = 0 if m < 0 or m >= N
        # d(conv[k])/dh[j] = h[k-j] + h[j-k]  (but h[j-k] is only nonzero if j-k in [0,N-1])
        # Wait: d/dh[j] of h[i]*h[k-i]:
        # If i=j: contributes h[k-j]
        # If k-i=j, i.e., i=k-j: contributes h[k-j]
        # These are the same term when j = k-j, i.e., j = k/2.
        # So: d(conv[k])/dh[j] = h[k-j] + h[j-k] where we interpret h[m]=0 if m out of [0,N-1]
        # But h[j-k] is nonzero only if j-k >= 0 and j-k < N, i.e., j >= k and j-k < N
        # And h[k-j] is nonzero only if k-j >= 0 and k-j < N, i.e., j <= k and k-j < N
        
        grad = np.zeros(N)
        for j in range(N):
            val = 0.0
            if 0 <= k - j < N:
                val += h[k - j]
            if 0 <= j - k < N:
                val += h[j - k]
            grad[j] = val
        
        return peak_val, grad
    
    # Objective: minimize peak autoconvolution (convex)
    # Constraint: sum(h) = 1, h >= 0
    def _objective(h):
        peak_val, _ = _autoconv_peak_and_grad(h)
        return peak_val
    
    def _gradient(h):
        _, grad = _autoconv_peak_and_grad(h)
        return grad
    
    # Strategy: Minimize peak autoconvolution subject to L1=1
    # Then the score is L2^2 / (L1 * peak * DELTA) = L2^2 / (peak * DELTA) since L1=1
    # Wait, the score is L2^2 / (L1 * autoconv_max) where autoconv_max = DELTA * peak_conv
    # So score = L2^2 / (L1 * DELTA * peak_conv)
    # With L1=1: score = L2^2 / (DELTA * peak_conv)
    # To maximize score, we want to minimize peak_conv for a given L2^2, or maximize L2^2 for given peak_conv.
    
    # Let's try: minimize peak_conv subject to L1=1 and L2^2 >= some threshold
    # Or just minimize peak_conv subject to L1=1, then compute the score.
    
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}]
    bounds = [(0.0, 1.0)] * N
    
    # Multiple starting points
    starts = []
    
    # Uniform
    starts.append(np.ones(N) / N)
    
    # Sparse peaks
    for sparsity in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        mask = np.zeros(N)
        num_on = max(1, int(N * sparsity))
        idx = np.random.choice(N, num_on, replace=False)
        mask[idx] = 1.0
        h_start = mask / np.sum(mask)
        starts.append(h_start)
    
    # Clustered
    for n_clusters in [2, 3, 4, 5]:
        h_cl = np.zeros(N)
        for i in range(n_clusters):
            center = int((i + 0.5) * N / n_clusters)
            width = max(1, N // (n_clusters * 3))
            start = max(0, center - width)
            end = min(N, center + width)
            h_cl[start:end] = 1.0
        if np.sum(h_cl) > 0:
            h_cl /= np.sum(h_cl)
            starts.append(h_cl)
    
    # Gaussian-like
    for center in [N//4, N//2, 3*N//4]:
        for width in [3, 5, 8, 12]:
            h_g = np.exp(-0.5 * ((x_idx - center) / width)**2)
            if np.sum(h_g) > 0:
                h_g /= np.sum(h_g)
                starts.append(h_g)
    
    # Periodic
    for period in [2, 3, 4, 5, 7]:
        h_p = np.zeros(N)
        h_p[::period] = 1.0
        if np.sum(h_p) > 0:
            h_p /= np.sum(h_p)
            starts.append(h_p)
    
    # Optimize from each starting point
    for h_init in starts:
        if time.time() > deadline - 15:
            break
        
        h_init = np.asarray(h_init, dtype=np.float64)
        l1 = np.sum(h_init)
        if l1 == 0:
            continue
        h_start = h_init / l1
        
        try:
            res = minimize(_objective, h_start, jac=_gradient, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 800, 'ftol': 1e-12})
            if res.success:
                h_opt = np.maximum(res.x, 0)
                l1_opt = np.sum(h_opt)
                if l1_opt > 0:
                    h_opt /= l1_opt
                    s = _score_raw(h_opt)
                    if s > best_score:
                        best_score = s
                        best_h = h_opt.copy()
        except Exception:
            pass
    
    # Random restarts
    n_random = 40
    for i in range(n_random):
        if time.time() > deadline - 10:
            break
        
        h_rand = np.random.rand(N)
        if np.random.rand() < 0.4:
            mask = np.random.rand(N) < 0.5
            h_rand[mask] = np.random.rand(np.sum(mask))
            h_rand[~mask] = 0
        
        l1 = np.sum(h_rand)
        if l1 == 0:
            continue
        h_start = h_rand / l1
        
        try:
            res = minimize(_objective, h_start, jac=_gradient, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 500, 'ftol': 1e-12})
            if res.success:
                h_opt = np.maximum(res.x, 0)
                l1_opt = np.sum(h_opt)
                if l1_opt > 0:
                    h_opt /= l1_opt
                    s = _score_raw(h_opt)
                    if s > best_score:
                        best_score = s
                        best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 5:
        try:
            res = minimize(_objective, best_h, jac=_gradient, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 1000, 'ftol': 1e-14})
            if res.success:
                h_opt = np.maximum(res.x, 0)
                l1_opt = np.sum(h_opt)
                if l1_opt > 0:
                    h_opt /= l1_opt
                    s = _score_raw(h_opt)
                    if s > best_score:
                        best_score = s
                        best_h = h_opt.copy()
        except Exception:
            pass
    
    return list(best_h)

# EVOLVE-BLOCK-END
