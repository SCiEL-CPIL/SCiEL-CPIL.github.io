"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from math import cos, sin, pi

def _score(points):
    """Compute Delta / n^n for 8 points, where Delta is product of ordered pair distances
    after scaling to diameter 2."""
    n = len(points)
    pts = np.array(points, dtype=float)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Diameter = max distance
    diam = np.max(dists)
    if diam < 1e-15:
        return 0.0
    # Scale to diameter 2
    scaled_dists = dists / diam * 2.0
    # Product of ordered pair distances (exclude self-distances which are 0)
    mask = ~np.eye(n, dtype=bool)
    ordered_dists = scaled_dists[mask]
    # Delta = product of all ordered pair distances
    # Use log to avoid underflow
    log_delta = np.sum(np.log(ordered_dists))
    log_score = log_delta - n * np.log(n)
    return np.exp(log_score)

def _neg_score(points):
    return -_score(points)

def _neg_log_score_and_grad(points):
    """Compute negative log-score and its analytical gradient.
    Objective: Maximize sum(log(d_ij)) - n(n-1)log(D)
    This is equivalent to maximizing the score, up to constants.
    """
    pts = np.array(points, dtype=float)
    n = len(pts)
    
    # Pairwise differences and distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid log(0)
    mask = ~np.eye(n, dtype=bool)
    dists_masked = dists[mask]
    if np.min(dists_masked) < 1e-15:
        return 0.0, np.zeros_like(pts)
        
    # Log of product of distances (ordered)
    # sum_{i != j} log(d_ij) = 2 * sum_{i < j} log(d_ij)
    log_prod = 2.0 * np.sum(np.log(dists_masked))
    
    # Soft max for diameter to make it differentiable
    # D_soft = (1/beta) * log(sum(exp(beta * d_ij)))
    beta = 50.0
    # We only care about i < j for max, but sum over all i!=j is fine if we adjust
    # Actually, max distance is over all pairs.
    # Let's use the max of the upper triangle
    triu_indices = np.triu_indices(n, k=1)
    d_pairs = dists[triu_indices]
    
    # Soft max
    max_d = np.max(d_pairs)
    exp_d = np.exp(beta * (d_pairs - max_d)) # Stability
    log_sum_exp = max_d + np.log(np.sum(exp_d)) / beta
    D_soft = log_sum_exp
    
    # Objective: log_prod - n(n-1) * log(D_soft)
    # Note: score = exp(log_prod - n(n-1)log(D) + const)
    # We want to maximize this.
    # Let f = log_prod - n(n-1) * log(D_soft)
    
    val = log_prod - n * (n - 1) * np.log(D_soft)
    
    # Gradient
    # grad log_prod = 2 * sum_{j != i} (p_i - p_j) / ||p_i - p_j||^2
    # grad D_soft = (1/D_soft) * (1/beta) * sum_{k} exp(beta d_k) * grad d_k
    # Actually, d/df log(D_soft) = (1/D_soft) * dD_soft
    # dD_soft = (1/beta) * (1/sum(exp)) * sum(exp * grad d)
    # Let w_k = exp(beta d_k) / sum(exp(beta d_j))
    # dD_soft = sum(w_k * grad d_k)
    
    # Compute weights for soft max
    w = exp_d / np.sum(exp_d)
    
    # grad d_k with respect to p_i
    # d_k is distance between p_a and p_b
    # grad_{p_a} d_k = (p_a - p_b) / d_k
    # grad_{p_b} d_k = (p_b - p_a) / d_k
    
    grad_val = np.zeros_like(pts)
    
    # Part 1: Gradient of log_prod
    # For each i, sum over j != i of (p_i - p_j) / d_ij^2
    # We can vectorize this
    # diff is (n, n, 2). dists is (n, n).
    # term_ij = (p_i - p_j) / d_ij^2
    # We want sum_j term_ij for each i.
    # Note: term_ji = -term_ij.
    
    # Create a matrix of gradients
    # dists_inv_sq = 1 / d^2, with 0 on diagonal
    dists_inv_sq = np.zeros_like(dists)
    np.fill_diagonal(dists, 1.0) # Avoid div by 0
    dists_inv_sq = 1.0 / (dists * dists)
    np.fill_diagonal(dists_inv_sq, 0.0)
    
    # grad_log_prod[i] = sum_j (p_i - p_j) * dists_inv_sq[i, j]
    # diff is p_i - p_j
    # We can do: grad_log_prod = np.einsum('ij,ijk->ik', dists_inv_sq, diff)
    grad_log_prod = np.einsum('ij,ijk->ik', dists_inv_sq, diff)
    # Multiply by 2 because of ordered pairs (sum_{i!=j} = 2 * sum_{i<j})
    # Wait, my einsum sums over j for each i.
    # sum_{j != i} (p_i - p_j) / d^2. This is exactly the gradient of sum_{j != i} log(d_ij).
    # Since log_prod = 2 * sum_{i<j} log(d_ij), the gradient is 2 * sum_{j != i} ...
    # Actually, d/dp_i sum_{j != i} log(d_ij) = sum_{j != i} (p_i - p_j)/d^2.
    # So grad log_prod = 2 * grad_log_prod_einsum?
    # Let's check: L = sum_{i<j} 2 log d_ij.
    # dL/dp_i = 2 * sum_{j > i} (p_i - p_j)/d^2 + 2 * sum_{j < i} (p_i - p_j)/d^2
    # = 2 * sum_{j != i} (p_i - p_j)/d^2.
    # Yes, factor of 2.
    grad_log_prod *= 2.0
    
    # Part 2: Gradient of -n(n-1) log(D_soft)
    # = -n(n-1) * (1/D_soft) * grad D_soft
    # grad D_soft = sum_k w_k * grad d_k
    # d_k is dists[a, b]
    
    grad_D = np.zeros_like(pts)
    # We need to scatter the gradients
    # For each pair (a, b) with weight w_k:
    # grad_D[a] += w_k * (p_a - p_b) / d_k
    # grad_D[b] += w_k * (p_b - p_a) / d_k
    
    # We have w (vector of size n(n-1)/2)
    # We need to map indices
    for k in range(len(w)):
        a, b = triu_indices[0][k], triu_indices[1][k]
        d_k = dists[a, b]
        if d_k < 1e-15: continue
        vec = (pts[a] - pts[b]) / d_k
        grad_D[a] += w[k] * vec
        grad_D[b] += w[k] * (-vec)
        
    grad_penalty = -n * (n - 1) * (1.0 / D_soft) * grad_D
    
    total_grad = grad_log_prod + grad_penalty
    
    # We are minimizing -val
    return -val, -total_grad

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    # Start with regular octagon
    octagon = [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
    best_points = np.array(octagon, dtype=float)
    best_score = _score(best_points)
    
    rng = np.random.RandomState(42)
    
    # Generate diverse starting points
    def generate_starts():
        starts = []
        # Regular octagon
        starts.append(np.array(octagon, dtype=float))
        
        # Perturbed octagons with various patterns
        for trial in range(30):
            angles = 2 * pi * np.arange(8) / 8
            # Different perturbation strategies
            strategy = trial % 6
            
            if strategy == 0:
                # Random radius perturbations
                radii = 1.0 + 0.3 * rng.randn(8)
                radii = np.clip(radii, 0.5, 1.5)
                angles_pert = angles + 0.1 * rng.randn(8)
            elif strategy == 1:
                # Alternating radii (two concentric circles)
                base_r = 1.0 + 0.2 * rng.randn()
                alt_r = 1.0 - 0.2 * rng.randn()
                radii = np.where(np.arange(8) % 2 == 0, base_r, alt_r)
                angles_pert = angles + 0.1 * rng.randn(8)
            elif strategy == 2:
                # Grouped perturbations
                radii = 1.0 + 0.2 * rng.randn(4)
                radii = np.tile(radii, 2)
                angles_pert = angles + 0.1 * rng.randn(8)
            elif strategy == 3:
                # Sine wave perturbation
                phase = rng.randn()
                radii = 1.0 + 0.2 * np.sin(2 * angles + phase)
                angles_pert = angles + 0.1 * rng.randn(8)
            elif strategy == 4:
                # Larger random perturbations
                radii = 1.0 + 0.4 * rng.randn(8)
                radii = np.clip(radii, 0.3, 1.7)
                angles_pert = angles + 0.2 * rng.randn(8)
            else:
                # Random points in disk
                radii = np.sqrt(rng.rand(8))
                angles_pert = rng.rand(8) * 2 * pi
            
            pts = np.column_stack([radii * np.cos(angles_pert), radii * np.sin(angles_pert)])
            starts.append(pts)
        
        # Square with vertices at corners and midpoints
        # Try configurations with 4-fold symmetry
        for r_inner in [0.5, 0.7, 0.8, 0.9]:
            angles_outer = np.array([0, pi/2, pi, 3*pi/2])
            angles_inner = angles_outer + pi/4
            pts_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
            pts_inner = r_inner * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
            pts = np.vstack([pts_outer, pts_inner])
            starts.append(pts)
        
        # Try configurations with 2-fold symmetry
        for dx in [0.1, 0.2, 0.3]:
            base = np.array(octagon, dtype=float)
            # Shift alternating points
            for i in range(8):
                if i % 2 == 0:
                    base[i, 0] += dx
                else:
                    base[i, 0] -= dx
            starts.append(base.copy())
            
        # Add some random configurations
        for _ in range(10):
            pts = rng.rand(8, 2) * 2 - 1
            starts.append(pts)
        
        return starts
    
    starts = generate_starts()
    
    # Optimize each starting point
    for idx, x0 in enumerate(starts):
        if time.time() > deadline - 5.0:
            break
        
        x0_flat = x0.flatten()
        
        # Try L-BFGS-B with analytical gradient
        try:
            if time.time() > deadline - 3.0:
                break
            result = minimize(_neg_log_score_and_grad, x0_flat, method='L-BFGS-B',
                            options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12})
            
            if time.time() > deadline - 3.0:
                break
            
            candidate = result.x.reshape(8, 2)
            # Check distinctness
            diff = candidate[:, np.newaxis, :] - candidate[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) < 1e-8:
                continue
            
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate
                # Refine with Nelder-Mead on the original score (more robust to scaling issues?)
                if time.time() < deadline - 10.0:
                    result2 = minimize(_neg_score, best_points.flatten(), method='Nelder-Mead',
                                      options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14})
                    candidate2 = result2.x.reshape(8, 2)
                    diff2 = candidate2[:, np.newaxis, :] - candidate2[np.newaxis, :, :]
                    dists2 = np.linalg.norm(diff2, axis=2)
                    np.fill_diagonal(dists2, np.inf)
                    if np.min(dists2) >= 1e-8:
                        cand_score2 = _score(candidate2)
                        if cand_score2 > best_score:
                            best_score = cand_score2
                            best_points = candidate2
                            
        except Exception:
            pass
    
    # Final local refinement with multiple small perturbations
    if time.time() < deadline - 5.0:
        for pert_trial in range(20):
            if time.time() > deadline - 3.0:
                break
            x0 = best_points.flatten() + 0.01 * rng.randn(16)
            try:
                result = minimize(_neg_log_score_and_grad, x0, method='L-BFGS-B',
                                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
                candidate = result.x.reshape(8, 2)
                diff = candidate[:, np.newaxis, :] - candidate[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                np.fill_diagonal(dists, np.inf)
                if np.min(dists) < 1e-8:
                    continue
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = candidate
            except Exception:
                pass
    
    # Return the best points found
    return [[float(p[0]), float(p[1])] for p in best_points]
# EVOLVE-BLOCK-END
