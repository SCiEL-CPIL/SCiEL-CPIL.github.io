"""Sphere packing in a unit cube using Fibonacci lattice init + soft-min L-BFGS-B."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # safety margin before 150s kill


def _true_score(pts):
    """Compute the true packing radius: min(half pairwise dist, min boundary dist)."""
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pairwise = np.sqrt(np.min(dists_sq))
    # Boundary
    dist_to_face = 0.5 - np.abs(pts)
    min_boundary = np.min(dist_to_face)
    return min(min_pairwise / 2.0, min_boundary)


def _soft_min_with_grad(pts, beta):
    """Compute soft-min surrogate value and gradient analytically."""
    n = len(pts)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(np.maximum(dists_sq, 1e-30))
    half_dists = dists / 2.0
    
    # Boundary constraints: 6 per point
    # c_ik^+ = pts[i,k] + 0.5, c_ik^- = 0.5 - pts[i,k]
    b_plus = pts + 0.5   # (n, 3)
    b_minus = 0.5 - pts  # (n, 3)
    
    # Build arrays for soft-min
    # We'll compute: -1/beta * log( sum_{pairs} exp(-beta * d_ij/2) + sum_{i,k} exp(-beta*(pts[i,k]+0.5)) + sum_{i,k} exp(-beta*(0.5-pts[i,k])) )
    
    # For numerical stability, find max constraint value
    max_pair = np.max(half_dists[np.triu_indices(n, k=1)])
    max_bplus = np.max(b_plus)
    max_bminus = np.max(b_minus)
    max_c = max(max_pair, max_bplus, max_bminus)
    
    # Compute log-sum-exp for each group
    # Pairwise
    half_dists_triu = half_dists[np.triu_indices(n, k=1)]
    exp_pairs = np.exp(-beta * (half_dists_triu - max_c))
    log_sum_pairs = np.log(np.sum(exp_pairs)) + beta * max_c
    
    # Boundary plus
    exp_bplus = np.exp(-beta * (b_plus - max_c))
    log_sum_bplus = np.log(np.sum(exp_bplus)) + beta * max_c
    
    # Boundary minus
    exp_bminus = np.exp(-beta * (b_minus - max_c))
    log_sum_bminus = np.log(np.sum(exp_bminus)) + beta * max_c
    
    # Total log-sum-exp
    # log(exp(log_sum_pairs) + exp(log_sum_bplus) + exp(log_sum_bminus))
    max_log = max(log_sum_pairs, log_sum_bplus, log_sum_bminus)
    total_log = max_log + np.log(np.exp(log_sum_pairs - max_log) + 
                                  np.exp(log_sum_bplus - max_log) + 
                                  np.exp(log_sum_bminus - max_log))
    
    soft_min_val = -total_log / beta
    
    # Gradient: d(soft_min)/d(pts) = -1/beta * d(total_log)/d(pts)
    # total_log = log( S_pairs + S_bplus + S_bminus ) where S = sum(exp(-beta*x))
    # d(total_log)/d(pts) = (1/(S_pairs+S_bplus+S_bminus)) * (dS_pairs + dS_bplus + dS_bminus)
    
    S_pairs = np.sum(exp_pairs) * np.exp(beta * max_c)  # not quite, let me redo
    
    # Let me use a cleaner approach with weights
    # w_j = exp(-beta * x_j) / sum(exp(-beta * x_k))
    # gradient of soft_min w.r.t. x_j = -w_j
    # Then chain rule for x_j w.r.t. pts
    
    # Recompute with normalized weights
    all_x = np.concatenate([half_dists_triu, b_plus.flatten(), b_minus.flatten()])
    max_all = np.max(all_x)
    exp_all = np.exp(-beta * (all_x - max_all))
    sum_exp = np.sum(exp_all)
    weights = exp_all / sum_exp  # sum to 1
    
    # soft_min = -1/beta * log(sum_exp) + max_all = max_all - log(sum_exp)/beta
    # (since log(sum(exp(-beta*(x-max)))) = log(sum_exp) when shifted)
    # Actually: soft_min = max_all - (1/beta)*log(sum_exp)
    
    # Gradient w.r.t. pts:
    # d(soft_min)/d(pts) = -sum_j w_j * dx_j/d(pts)
    # For pairwise: x_ij = |p_i - p_j|/2, dx/dp_i = (p_i - p_j)/|p_i - p_j|, dx/dp_j = (p_j - p_i)/|p_i - p_j|
    # For b_plus[i,k] = p[i,k] + 0.5: d/dp[i,k] = 1
    # For b_minus[i,k] = 0.5 - p[i,k]: d/dp[i,k] = -1
    
    grad = np.zeros((n, 3))
    
    # Pairwise contribution
    n_pairs = len(half_dists_triu)
    pair_weights = weights[:n_pairs]
    
    # Get pair indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    for idx in range(n_pairs):
        w = pair_weights[idx]
        i, j = i_idx[idx], j_idx[idx]
        d = pts[i] - pts[j]
        dist = np.sqrt(np.sum(d * d))
        if dist > 1e-12:
            # dx/dp_i = d/dist, dx/dp_j = -d/dist
            # contribution to grad: -w * dx/dp
            grad[i] -= w * d / dist
            grad[j] += w * d / dist
    
    # Boundary plus contribution
    bplus_weights = weights[n_pairs:n_pairs + n * 3]
    bplus_w = bplus_weights.reshape(n, 3)
    # d(b_plus[i,k])/d(pts[i,k]) = 1, so grad += -w * 1
    grad -= bplus_w
    
    # Boundary minus contribution
    bminus_weights = weights[n_pairs + n * 3:]
    bminus_w = bminus_weights.reshape(n, 3)
    # d(b_minus[i,k])/d(pts[i,k]) = -1, so grad += -w * (-1) = +w
    grad += bminus_w
    
    return soft_min_val, grad


def _fibonacci_lattice(n=50):
    """Create a deterministic Fibonacci lattice initialization."""
    # 3D Fibonacci lattice (spherical spiral)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    indices = np.arange(n)
    z = 1 - 2 * (indices + 0.5) / n
    radius = np.sqrt(1 - z*z)
    theta = golden_angle * indices
    
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    
    # Scale to fit in cube [-0.5, 0.5]^3
    # The points are on the unit sphere, so scale by 0.5
    pts = np.column_stack([x, y, z]) * 0.5
    
    return pts


def _optimize_with_lbfgsb(pts, beta, deadline):
    """Run L-BFGS-B on the soft-min surrogate."""
    x0 = pts.flatten()
    
    def objective(x):
        pts_x = x.reshape(-1, 3)
        val, grad = _soft_min_with_grad(pts_x, beta)
        # We want to MAXIMIZE soft_min, so minimize negative
        return -val, -grad
    
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    result = minimize(
        lambda x: objective(x)[0],
        x0,
        jac=lambda x: objective(x)[1],
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
    )
    
    return result.x.reshape(-1, 3)


def _local_polish(pts, deadline):
    """Fast local optimization: iteratively push apart closest pairs."""
    n = len(pts)
    current_pts = pts.copy()
    current_score = _true_score(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    for it in range(2000):
        if time.time() > deadline:
            break
        
        # Find closest pair
        diff = current_pts[:, np.newaxis, :] - current_pts[np.newaxis, :]
        dists_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        
        # Find minimum
        min_idx = np.unravel_index(np.argmin(dists_sq), dists_sq.shape)
        i, j = min_idx
        if i > j:
            i, j = j, i
        
        # Also check boundary
        dist_to_face = 0.5 - np.abs(current_pts)
        min_boundary_idx = np.unravel_index(np.argmin(dist_to_face), dist_to_face.shape)
        boundary_val = dist_to_face[min_boundary_idx]
        pair_val = np.sqrt(dists_sq[i, j]) / 2.0
        
        if pair_val < boundary_val:
            # Push pair apart
            d = current_pts[i] - current_pts[j]
            dist = np.sqrt(np.sum(d * d))
            if dist > 1e-12:
                push = d / dist * 0.002
                current_pts[i] += push
                current_pts[j] -= push
        else:
            # Push boundary point inward
            bi, bk = min_boundary_idx
            # Determine which face
            if current_pts[bi, bk] > 0:
                current_pts[bi, bk] -= 0.002
            else:
                current_pts[bi, bk] += 0.002
        
        # Clamp
        current_pts = np.clip(current_pts, -0.5, 0.5)
        
        new_score = _true_score(current_pts)
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = current_pts.copy()
        else:
            # Revert
            current_pts = best_pts.copy()
            current_score = best_score
    
    return best_pts, best_score


def construct():
    """Construct the sphere packing configuration."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with Fibonacci lattice
    pts = _fibonacci_lattice(N)
    best_score = _true_score(pts)
    best_pts = pts.copy()
    
    # Annealing schedule for beta
    # Start with low beta (smooth), end with high beta (sharp)
    beta_schedule = [10, 30, 100, 300, 1000]
    
    for beta in beta_schedule:
        if time.time() > deadline - 15:
            break
        
        # Run L-BFGS-B on the soft-min surrogate
        try:
            new_pts = _optimize_with_lbfgsb(pts, beta, deadline)
            # Ensure validity
            new_pts = np.clip(new_pts, -0.5, 0.5)
            new_score = _true_score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                pts = new_pts.copy()
            else:
                # Keep previous pts for next iteration
                pass
        except Exception:
            # If optimization fails, continue with current pts
            pass
    
    # Final local polish
    if time.time() < deadline - 5:
        polished_pts, polished_score = _local_polish(pts, deadline)
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished_pts
    
    # Verify validity
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_lattice(N)
    
    return best_pts
