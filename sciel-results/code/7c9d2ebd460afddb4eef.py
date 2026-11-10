# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_area(x):
    """Compute the area of the union of 64 triangles given base positions x."""
    n = 64
    a = (np.arange(n) + 1) / n
    m = np.arange(20000)
    y = (m + 0.5) / 20000.0
    
    L = x[None, :] + a[None, :] * y[:, None]
    w = (1.0 - y) / 64.0
    
    L_sorted = np.sort(L, axis=1)
    d = np.diff(L_sorted, axis=1)
    d_clipped = np.minimum(d, w[:, None])
    union_lengths = w + d_clipped.sum(axis=1)
    area = union_lengths.mean()
    
    return area


def _compute_area_and_grad(x):
    """Compute area and gradient of area with respect to x using central differences."""
    n = 64
    eps = 1e-8
    grad = np.zeros(n)
    base_area = _compute_area(x)
    
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (_compute_area(x_plus) - _compute_area(x_minus)) / (2 * eps)
    
    return base_area, grad


def _softmin(x, beta):
    """Soft minimum using log-sum-exp trick."""
    x_min = np.min(x, axis=1, keepdims=True)
    return x_min + np.log(np.exp(beta * (x - x_min)).mean(axis=1, keepdims=True)) / beta


def _compute_softmin_area(x, beta):
    """Compute a smooth surrogate for the area using soft-min instead of hard min."""
    n = 64
    a = (np.arange(n) + 1) / n
    m = np.arange(20000)
    y = (m + 0.5) / 20000.0
    
    L = x[None, :] + a[None, :] * y[:, None]
    w = (1.0 - y) / 64.0
    
    L_sorted = np.sort(L, axis=1)
    d = np.diff(L_sorted, axis=1)
    
    # Use soft-min: min(d_j, w) ≈ -logsumexp(-beta*d_j, -beta*w)/beta
    # For each row m, we have 63 values d[m, :] and scalar w[m]
    # softmin(d, w) = -logsumexp(-beta*d, -beta*w) / beta
    # We need to be careful with the shape
    
    # d has shape (20000, 63), w has shape (20000,)
    # We want element-wise softmin of d and w (broadcasting w to match d)
    # softmin(d_j, w) for each j
    
    # -beta * d has shape (20000, 63)
    # -beta * w has shape (20000,) -> broadcast to (20000, 63)
    neg_d = -beta * d  # (20000, 63)
    neg_w = -beta * w[:, None]  # (20000, 1) -> broadcast
    
    # logsumexp along axis=1 of [neg_d, neg_w]
    # We need logsumexp of 64 values per row: 63 from neg_d and 1 from neg_w
    # logsumexp([a1,...,a63, b]) = log(sum(exp(ai)) + exp(b))
    
    # Use the max trick for numerical stability
    # max_val = max(max(neg_d, axis=1), neg_w)
    max_d = np.max(neg_d, axis=1, keepdims=True)  # (20000, 1)
    max_val = np.maximum(max_d, neg_w)  # (20000, 1)
    
    # sum_exp = sum(exp(neg_d - max_val), axis=1) + exp(neg_w - max_val)
    exp_d = np.exp(neg_d - max_val)  # (20000, 63)
    exp_w = np.exp(neg_w - max_val)  # (20000, 1)
    
    sum_exp = exp_d.sum(axis=1) + exp_w.squeeze(axis=1)  # (20000,)
    
    # softmin = -(max_val + log(sum_exp)) / beta
    # max_val has shape (20000, 1), squeeze to (20000,)
    max_val_squeezed = max_val.squeeze(axis=1)
    softmin_val = -(max_val_squeezed + np.log(sum_exp)) / beta  # (20000,)
    
    # union_lengths = w + sum(softmin_val)
    union_lengths = w + softmin_val.sum()
    
    # Wait, that's not right. The union length is w + sum over consecutive pairs of min(d_j, w)
    # So it's w + sum_j min(d_j, w)
    # softmin_val is the softmin of each d_j with w, shape (20000, 63)
    # We need to sum over axis=1
    
    # Let me redo this
    # softmin_val should have shape (20000, 63)
    # softmin(d_j, w) for each j
    
    # Actually, let me reconsider. The formula is:
    # union_length = w + sum_{j=0}^{62} min(d_j, w)
    
    # So we need softmin for each of the 63 differences
    # softmin(d_j, w) = -logsumexp(-beta*d_j, -beta*w) / beta
    
    # Let me recompute
    # neg_d has shape (20000, 63)
    # neg_w has shape (20000, 1)
    
    # For each row m and column j:
    # softmin_val[m, j] = -logsumexp(-beta*d[m,j], -beta*w[m]) / beta
    
    # logsumexp(a, b) = log(exp(a) + exp(b))
    # = max(a,b) + log(exp(a-max) + exp(b-max))
    
    max_ab = np.maximum(neg_d, neg_w)  # (20000, 63) via broadcasting
    
    exp_a = np.exp(neg_d - max_ab)  # (20000, 63)
    exp_b = np.exp(neg_w - max_ab)  # (20000, 63) via broadcasting
    
    sum_exp = exp_a + exp_b  # (20000, 63)
    
    logsumexp_val = max_ab + np.log(sum_exp)  # (20000, 63)
    
    softmin_val = -logsumexp_val / beta  # (20000, 63)
    
    union_lengths = w + softmin_val.sum(axis=1)  # (20000,)
    
    area = union_lengths.mean()
    
    return area


def _compute_softmin_grad(x, beta):
    """Compute gradient of softmin area with respect to x using central differences."""
    n = 64
    eps = 1e-8
    grad = np.zeros(n)
    
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (_compute_softmin_area(x_plus, beta) - _compute_softmin_area(x_minus, beta)) / (2 * eps)
    
    return grad


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    Uses L-BFGS-B optimization with smooth surrogate and coordinate descent refinement.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    n = 64
    
    # Initial Keich construction
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x0 = np.array(positions, dtype=np.float64)
    
    best_x = x0.copy()
    best_area = _compute_area(x0)
    
    # Phase 1: Optimize using smooth surrogate with increasing beta
    betas = [5.0, 20.0, 100.0, 500.0]
    
    for beta in betas:
        elapsed = time.time() - start_time
        if elapsed > deadline - 60:
            break
        
        try:
            result = minimize(
                lambda x: _compute_softmin_area(x, beta),
                best_x,
                jac=lambda x: _compute_softmin_grad(x, beta),
                method='L-BFGS-B',
                options={
                    'maxiter': 300,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxfun': 5000,
                    'maxls': 50
                }
            )
            
            if result.fun < _compute_softmin_area(best_x, beta):
                candidate_area = _compute_area(result.x)
                if candidate_area < best_area:
                    best_area = candidate_area
                    best_x = result.x.copy()
                    
        except Exception:
            pass
    
    # Phase 2: Optimize using the true (nonsmooth) objective
    elapsed = time.time() - start_time
    if elapsed < deadline - 120:
        try:
            result = minimize(
                _compute_area,
                best_x,
                jac=lambda x: _compute_area_and_grad(x)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                    'maxfun': 10000,
                    'maxls': 50
                }
            )
            
            if result.fun < best_area:
                best_area = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
    
    # Phase 3: Coordinate descent refinement
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        # Simple coordinate descent: iterate over each coordinate, do a 1D line search
        for iteration in range(5):
            elapsed = time.time() - start_time
            if elapsed > deadline - 30:
                break
            
            improved = False
            for i in range(n):
                # Try small perturbations in coordinate i
                best_val = best_area
                best_xi = best_x[i]
                
                # Try a few steps
                for delta in [-0.01, -0.005, -0.001, 0.001, 0.005, 0.01]:
                    x_trial = best_x.copy()
                    x_trial[i] = best_xi + delta
                    area_trial = _compute_area(x_trial)
                    
                    if area_trial < best_val - 1e-12:
                        best_val = area_trial
                        best_xi = x_trial[i]
                        improved = True
                
                best_x[i] = best_xi
            
            if best_val < best_area:
                best_area = best_val
            
            if not improved:
                break
    
    # Final polish with tighter tolerance
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        try:
            result = minimize(
                _compute_area,
                best_x,
                method='L-BFGS-B',
                options={
                    'maxiter': 200,
                    'ftol': 1e-16,
                    'gtol': 1e-12,
                    'maxfun': 3000,
                    'maxls': 50
                }
            )
            
            if result.fun < best_area:
                best_area = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
    
    return best_x


# EVOLVE-BLOCK-END
