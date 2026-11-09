# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from collections import defaultdict

def _entropy_and_grad_from_values(values, weights, n_unique):
    """
    Compute Shannon entropy in bits and its gradient w.r.t. weights.
    values: array of pooled values (indices into bins)
    weights: array of weights per cell
    n_unique: number of unique bins
    Returns (entropy, grad) where grad[i] = dH/dw[i]
    """
    # Pool weights by value
    bin_weights = np.zeros(n_unique)
    np.add.at(bin_weights, values, weights)
    total = bin_weights.sum()
    if total <= 1e-15:
        return 0.0, np.zeros_like(weights)
    
    # Nonzero bins
    mask = bin_weights > 1e-15
    probs = bin_weights[mask] / total
    if len(probs) == 0:
        return 0.0, np.zeros_like(weights)
    
    # Entropy: H = -sum p log2 p = -sum (w_i/total) * log2(w_i/total)
    # = -sum w_i * (log2(w_i) - log2(total)) / total
    # = (-sum w_i * log2(w_i) + log2(total) * sum w_i) / total
    # = (-sum w_i * log2(w_i) + log2(total) * total) / total
    # = -sum w_i * log2(w_i) / total + log2(total)
    
    # dH/dw_i for a bin j with weight w_j:
    # H = -sum_k w_k log2(w_k) / total + log2(total)
    # dH/dw_i = -[log2(w_i) * total - w_i * (1/total) * total * ln(2)^{-1}... ]
    # Let me redo this more carefully.
    # H = -sum_{k in mask} (w_k / total) * log2(w_k / total)
    #   = -sum_k w_k * log2(w_k) / total + sum_k w_k * log2(total) / total
    #   = -sum_k w_k * log2(w_k) / total + log2(total)
    # 
    # dH/dw_i = -[log2(w_i) * total - w_i * (1/(total * ln2)) * total] / total^2 + (1/(total * ln2))
    # Wait, let me use natural log and convert.
    # H_bits = H_nats / ln(2)
    # H_nats = -sum_k p_k ln p_k where p_k = w_k / total
    # dH_nats/dw_i = -[ln(w_i/total) * (1/total) + (w_i/total) * (-1/(w_i)) * (1/total)]
    #              = -[ln(w_i) - ln(total)] / total + 1 / total
    #              = -ln(w_i/total) / total + 1/total
    #              = (1 - ln(w_i/total)) / total
    # dH_bits/dw_i = (1 - ln(w_i/total)) / (total * ln(2))
    
    # For each cell i, it contributes to bin values[i]
    # dH/dw_i = (1 - ln(w_{bin(i)}/total)) / (total * ln(2)) if w_{bin(i)} > 0
    
    grad = np.zeros_like(weights)
    ln_probs = np.log(probs)  # ln(w_k/total) for nonzero bins
    # Map back: for each cell, find its bin's contribution
    # dH/dw_i = (1 - ln(w_{bin(i)}/total)) / (total * ln2)
    bin_grad = np.zeros(n_unique)
    bin_grad[mask] = (1.0 - ln_probs) / (total * np.log(2))
    
    grad = bin_grad[values]
    
    entropy = -np.sum(probs * np.log2(probs))
    return entropy, grad

def _compute_ratio_and_grad(P):
    """Compute D/M ratio and its gradient w.r.t. P (unnormalized)."""
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-15:
        return 0.0, np.zeros_like(P)
    Pn = P / s
    
    y_idx, x_idx = np.indices((10, 10))
    x_flat = x_idx.flatten()
    y_flat = y_idx.flatten()
    w = Pn.flatten()
    
    # Precompute bin assignments
    d_vals = x_flat - y_flat  # range -9 to 9, 19 bins
    x_vals = x_flat           # range 0 to 9, 10 bins
    xy_vals = x_flat + y_flat # range 0 to 18, 19 bins
    x2y_vals = x_flat + 2 * y_flat  # range 0 to 27, 28 bins
    
    # Shift to non-negative indices
    d_idx = d_vals + 9       # 0..18
    x_idx_f = x_vals         # 0..9
    xy_idx = xy_vals         # 0..18
    x2y_idx = x2y_vals       # 0..27
    
    n_d = 19
    n_x = 10
    n_xy = 19
    n_x2y = 28
    
    D, grad_D = _entropy_and_grad_from_values(d_idx, w, n_d)
    H0, grad_H0 = _entropy_and_grad_from_values(x_idx_f, w, n_x)
    H1, grad_H1 = _entropy_and_grad_from_values(xy_idx, w, n_xy)
    H2, grad_H2 = _entropy_and_grad_from_values(x2y_idx, w, n_x2y)
    
    M = max(H0, H1, H2)
    if M <= 1e-15:
        return 0.0, np.zeros_like(P)
    
    ratio = D / M
    
    # d(ratio)/dw = (grad_D * M - D * grad_M) / M^2
    # grad_M = grad of the max component
    if H0 >= H1 and H0 >= H2:
        grad_M = grad_H0
    elif H1 >= H0 and H1 >= H2:
        grad_M = grad_H1
    else:
        grad_M = grad_H2
    
    grad_ratio = (grad_D * M - D * grad_M) / (M * M)
    
    # Now convert from gradient w.r.t. w (normalized P) to gradient w.r.t. P (unnormalized)
    # w = P / s, so dw/dP = (1/s) * I - (P/s^2) * 1^T
    # d(ratio)/dP = (dw/dP)^T * d(ratio)/dw
    # = (1/s) * grad_ratio - (1/s^2) * (grad_ratio @ P) * ones
    # But we want gradient w.r.t. the log-parameterization or just P directly
    # For L-BFGS-B on P directly with bounds [0, inf], we can use:
    # d(ratio)/dP_i = (1/s) * grad_ratio_i - (1/s^2) * (sum_j grad_ratio_j * P_j)
    
    s_val = s
    dot = np.dot(grad_ratio, P)
    grad_P = grad_ratio / s_val - (dot / (s_val * s_val)) * np.ones_like(P)
    
    # Actually, for optimization with L-BFGS-B on P directly (unconstrained except bounds),
    # we should provide the gradient of the objective w.r.t. P.
    # The objective is ratio(P) where P is the input (unnormalized).
    # ratio(P) = D(P/s) / M(P/s)
    # Since D and M are homogeneous of degree 0 in w, ratio is invariant to scaling of P.
    # So d(ratio)/dP_i = sum_j d(ratio)/dw_j * dw_j/dP_i
    # dw_j/dP_i = delta_{ij}/s - P_j/s^2
    # d(ratio)/dP_i = (1/s) * grad_ratio_i - (1/s^2) * sum_j grad_ratio_j * P_j
    # = (1/s) * grad_ratio_i - (dot / s^2)
    
    return ratio, grad_P

def _optimize_P():
    """Optimize P to maximize D/M using gradient-based methods."""
    deadline = time.time() + 580  # safety margin
    
    n = 10
    best_P = np.ones((n, n)) / (n * n)
    best_score = _compute_ratio_and_grad(best_P)[0]
    
    def neg_ratio_with_grad(P_flat):
        P = P_flat.reshape(10, 10)
        score, grad = _compute_ratio_and_grad(P)
        if score > best_score_holder[0]:
            best_score_holder[0] = score
            best_P_holder[:] = P
        return -score, -grad
    
    best_score_holder = [best_score]
    best_P_holder = best_P.copy()
    
    # Multiple initializations
    initializations = []
    
    # 1. Uniform
    initializations.append(np.ones((10, 10)) / 100.0)
    
    # 2. Diagonal concentration
    P_diag = np.zeros((10, 10))
    for i in range(10):
        P_diag[i, i] = 1.0 / 10
    initializations.append(P_diag)
    
    # 3. Anti-diagonal (x - y = constant)
    P_anti = np.zeros((10, 10))
    for i in range(10):
        P_anti[i, 9-i] = 1.0 / 10
    initializations.append(P_anti)
    
    # 4. Corner concentration
    P_corner = np.zeros((10, 10))
    P_corner[0, 0] = 0.4
    P_corner[9, 9] = 0.4
    P_corner[0, 9] = 0.1
    P_corner[9, 0] = 0.1
    initializations.append(P_corner)
    
    # 5. Row concentration (varying y)
    P_row = np.zeros((10, 10))
    P_row[5, :] = 1.0 / 10
    initializations.append(P_row)
    
    # 6. Column concentration (varying x)
    P_col = np.zeros((10, 10))
    P_col[:, 5] = 1.0 / 10
    initializations.append(P_col)
    
    # 7. Random distributions
    np.random.seed(42)
    for _ in range(15):
        P_rand = np.random.dirichlet(np.ones(100))
        initializations.append(P_rand.reshape(10, 10))
    
    # 8. Sparse patterns - concentrate on few cells
    for _ in range(10):
        P_sparse = np.zeros((10, 10))
        k = np.random.randint(3, 15)
        indices = np.random.choice(100, k, replace=False)
        P_sparse.flat[indices] = 1.0 / k
        initializations.append(P_sparse)
    
    # 9. Bimodal distributions
    for _ in range(5):
        P_bi = np.zeros((10, 10))
        cx1, cy1 = np.random.randint(0, 10), np.random.randint(0, 10)
        cx2, cy2 = np.random.randint(0, 10), np.random.randint(0, 10)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                y1, x1 = cy1 + dy, cx1 + dx
                y2, x2 = cy2 + dy, cx2 + dx
                if 0 <= y1 < 10 and 0 <= x1 < 10:
                    P_bi[y1, x1] += 1.0
                if 0 <= y2 < 10 and 0 <= x2 < 10:
                    P_bi[y2, x2] += 1.0
        P_bi /= P_bi.sum()
        initializations.append(P_bi)
    
    # Optimize from each initialization
    for init_idx, P_init in enumerate(initializations):
        if time.time() > deadline:
            break
        
        P_flat = P_init.flatten()
        
        try:
            result = minimize(
                neg_ratio_with_grad,
                P_flat,
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, None)] * 100,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            P_opt = result.x.reshape(10, 10)
            P_opt = np.clip(P_opt, 0, None)
            s = P_opt.sum()
            if s > 1e-15:
                P_opt = P_opt / s
                score = _compute_ratio_and_grad(P_opt)[0]
                if score > best_score_holder[0]:
                    best_score_holder[0] = score
                    best_P_holder[:] = P_opt
        except Exception:
            pass
    
    # Additional refinement: use the best found as starting point for more iterations
    if time.time() < deadline - 30:
        try:
            P_flat = best_P_holder.flatten()
            result = minimize(
                neg_ratio_with_grad,
                P_flat,
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, None)] * 100,
                options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
            )
            P_opt = result.x.reshape(10, 10)
            P_opt = np.clip(P_opt, 0, None)
            s = P_opt.sum()
            if s > 1e-15:
                P_opt = P_opt / s
                score = _compute_ratio_and_grad(P_opt)[0]
                if score > best_score_holder[0]:
                    best_score_holder[0] = score
                    best_P_holder[:] = P_opt
        except Exception:
            pass
    
    return best_P_holder

def _compute_ratio(P):
    """Compute D/M ratio for a 10x10 distribution P[y,x]."""
    return _compute_ratio_and_grad(P)[0]

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    start_time = time.time()
    
    P = _optimize_P()
    
    # Final safety checks
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-9:
        P = np.ones((10, 10)) / 100.0
    else:
        P = P / s
    
    # Ensure finite
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Final validation
    if P.shape != (10, 10):
        P = np.ones((10, 10)) / 100.0
    if np.any(~np.isfinite(P)):
        P = np.ones((10, 10)) / 100.0
    if P.sum() <= 1e-9:
        P = np.ones((10, 10)) / 100.0
    P = np.clip(P, 0, None)
    P = P / P.sum()
    
    return P

# EVOLVE-BLOCK-END
