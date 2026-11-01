"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

DEADLINE_MARGIN = 5.0  # seconds of safety margin before the 150s kill

def _compute_radius(points):
    """Compute the maximum common radius for given centers."""
    if len(points) < 2:
        return 0.0
    # Distance to boundary
    dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dists)
    # Pairwise distances
    # For N=300, 300*299/2 = 44850 pairs, manageable
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    # Set diagonal to inf so it doesn't affect min
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = np.min(dist_matrix) / 2.0
    return min(min_boundary, min_pair)

def _local_optimize(points, max_iter=500, tol=1e-12):
    """
    Local optimization to maximize the minimum distance (radius).
    Uses a smooth surrogate: minimize negative of soft-min of all constraints.
    Actually, let's use a standard approach: maximize r subject to constraints.
    Since scipy's minimize is for minimization, we minimize -r.
    But r is nonsmooth.
    
    Alternative: Use a penalty method or barrier method.
    Let's try a simpler approach: Gradient ascent on the soft-min of distances.
    Or better: use a standard solver like L-BFGS-B with a smooth approximation.
    
    Actually, a very effective method for circle packing is to use the "repulsion" force.
    Each circle is repelled by every other circle and the boundary.
    We can simulate this with gradient descent on a potential energy function.
    
    Let's define the objective as:
    F(x) = sum_{i<j} max(0, 2r - dist(i,j))^2 + sum_i max(0, r - (1 - dist(i, 0)))^2
    But r is a variable too.
    
    Let's fix r and optimize positions, then adjust r? No, that's complex.
    
    Let's use a different strategy:
    1. Start with a good initial guess (hexagonal lattice).
    2. Use a local optimizer that maximizes the minimum distance.
    
    We can use the "soft-min" trick:
    soft_min(a, b) = -log(exp(-a) + exp(-b)) / log(2) ... no, that's for min.
    min(a, b) ≈ -log(exp(-a) + exp(-b)) for large negative a,b.
    
    Let's define:
    d_min = min( min_{i<j} dist(i,j), min_i (1 - |c_i|) )
    We want to maximize d_min.
    
    Let's use a penalty method:
    Minimize: -d_min_soft + lambda * violation
    where d_min_soft is a smooth approximation of d_min.
    
    Actually, a common approach is to use the "log-sum-exp" for the max (or -log-sum-exp for min).
    min(x) ≈ -log(sum(exp(-x_i))) / log(N) ... wait.
    min(x_i) = - max(-x_i) = -log(sum(exp(-x_i))) + log(N) (if x_i are close).
    
    Let's define the objective function for minimization:
    f(x) = - soft_min(distances)
    where distances are all pairwise distances and boundary distances.
    
    soft_min(x_1, ..., x_m) = - (1/t) * log( sum( exp(-t * x_i) ) )
    As t -> infinity, this approaches min(x_i).
    
    So we minimize: (1/t) * log( sum( exp(-t * d_i) ) )
    where d_i are the distances (pairwise and boundary).
    
    This is smooth and differentiable.
    Gradient: 
    Let S = sum(exp(-t * d_i))
    f = (1/t) log(S)
    df/dx = (1/t) * (1/S) * sum( exp(-t * d_i) * (-t * dd_i/dx) )
    df/dx = - (1/S) * sum( exp(-t * d_i) * dd_i/dx )
    
    This is a weighted average of the gradients of the individual distances, weighted by exp(-t * d_i).
    The smallest distances get the most weight.
    
    This is a good approach. Let's implement it.
    """
    n = len(points)
    points = points.copy()
    
    # Precompute indices for efficiency
    # We'll use a simple gradient descent with adaptive step size.
    
    t = 50.0  # Temperature parameter, larger is sharper
    lr = 0.01
    step = 0
    
    start_time = time.time()
    
    # Helper to compute distances and gradient
    def compute_grad(pos):
        pos = pos.reshape(n, 2)
        # Pairwise distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Boundary distances
        boundary_dists = 1.0 - np.linalg.norm(pos, axis=1)
        
        # Combine all distances
        # We need to be careful with the size. 300*299/2 + 300 ~ 45k distances.
        # This is fine.
        
        # For the soft-min, we only care about the smallest distances.
        # To save time, we can truncate the sum to the k smallest distances.
        # But for now, let's use all.
        
        # Flatten pairwise distances (upper triangle)
        iu = np.triu_indices(n, k=1)
        pair_dists = dist_matrix[iu]
        
        # All distances
        all_dists = np.concatenate([pair_dists, boundary_dists])
        
        # Compute weights
        # exp(-t * d) might underflow for large d.
        # We can shift by the min distance to avoid underflow.
        d_min = np.min(all_dists)
        exp_terms = np.exp(-t * (all_dists - d_min))
        S = np.sum(exp_terms)
        
        # Gradient of the soft-min
        # f = (1/t) log(S) + d_min
        # df/dx = - (1/S) * sum( exp_terms * d(all_dists)/dx )
        
        grad = np.zeros_like(pos)
        
        # Pairwise gradient
        # d(dist_ij)/dx_i = (x_i - x_j) / dist_ij
        # d(dist_ij)/dx_j = (x_j - x_i) / dist_ij
        # 
        # We need to accumulate: - (1/S) * sum_k w_k * d(d_k)/dx
        # 
        # Let's compute the gradient for each pair.
        # This might be slow in Python. Let's vectorize.
        
        # For each pair (i, j), the contribution to grad[i] is -w_ij * (x_i - x_j) / d_ij
        # and to grad[j] is -w_ij * (x_j - x_i) / d_ij
        
        # w_ij = exp(-t * d_ij) / S
        
        # We have pair_dists and exp_terms for pairs.
        pair_weights = exp_terms[:len(pair_dists)] / S
        
        # diff for pairs
        diff_pairs = diff[iu]
        # Normalize
        norm_pairs = pair_dists[:, np.newaxis]
        # Avoid division by zero
        norm_pairs[norm_pairs == 0] = 1e-12
        
        unit_vecs = diff_pairs / norm_pairs
        
        # Accumulate gradient
        # For each pair, add -w * unit_vec to grad[i] and -w * (-unit_vec) to grad[j]
        # i.e., -w * unit_vec to grad[i] and w * unit_vec to grad[j]
        
        i_idx, j_idx = iu
        np.add.at(grad, i_idx, -pair_weights[:, np.newaxis] * unit_vecs)
        np.add.at(grad, j_idx, pair_weights[:, np.newaxis] * unit_vecs)
        
        # Boundary gradient
        # d(1 - |x_i|)/dx_i = -x_i / |x_i|
        boundary_weights = exp_terms[len(pair_dists):] / S
        norms = np.linalg.norm(pos, axis=1)[:, np.newaxis]
        norms[norms == 0] = 1e-12
        unit_pos = pos / norms
        # Contribution: -w * (-unit_pos) = w * unit_pos
        grad += boundary_weights[:, np.newaxis] * unit_pos
        
        return grad, d_min
    
    # Gradient descent
    for step in range(max_iter):
        if time.time() - start_time > DEADLINE_MARGIN - 1.0:
            break
            
        grad, d_min = compute_grad(points)
        
        # Adaptive step size
        # Simple: if the objective increases, reduce step size.
        # But we need to compute the objective.
        # Let's just use a fixed small step size for now.
        # Or use a backtracking line search.
        
        # Let's use a fixed step size for simplicity.
        points_new = points - lr * grad
        
        # Check if it improved
        # Compute d_min for new points
        # This is expensive. Let's just accept the step.
        # Or, we can check if the min distance increased.
        
        # For now, let's just update.
        points = points_new
        
        # Decay learning rate
        lr *= 0.999
        
        if lr < 1e-8:
            break
            
    return points

def construct():
    start_time = time.time()
    
    # Initial guess: Hexagonal lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            points = points[np.argsort(radius, kind="stable")[:N]]
            break
        spacing *= 0.98
    
    # Local optimization
    if time.time() - start_time < DEADLINE_MARGIN:
        points = _local_optimize(points)
    
    return points
# EVOLVE-BLOCK-END
