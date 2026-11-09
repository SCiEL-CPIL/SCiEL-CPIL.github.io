"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def _compute_radius(points):
    """Compute the feasible radius for a set of points."""
    if points is None or len(points) == 0:
        return 0.0
    # Distance to boundary
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dist_boundary)
    
    # Pairwise distances
    d = pdist(points)
    min_dist = np.min(d)
    min_pair = min_dist / 2.0
    
    return min(min_boundary, min_pair)

def _initial_hexagonal():
    """Generate a hexagonal lattice packing as starting point."""
    # Estimate spacing for N circles in unit disk
    # Density ~ 0.9069, so r ~ sqrt(0.9069/N)
    est_r = np.sqrt(0.9069 / N)
    est_s = 2 * est_r
    
    best_points = None
    best_r = 0.0
    
    for spacing in np.linspace(est_s * 0.95, est_s * 1.05, 10):
        a = spacing
        h = a * np.sqrt(3) / 2
        
        points = []
        max_range = int(np.ceil(1.0 / h)) + 2
        for j in range(-max_range, max_range + 1):
            for i in range(-max_range, max_range + 1):
                x = (i + 0.5 * (j % 2)) * a
                y = j * h
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < N:
            continue
        
        # Select N points closest to center
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        selected = points[idx]
        
        r = _compute_radius(selected)
        if r > best_r:
            best_r = r
            best_points = selected.copy()
    
    if best_points is None:
        # Fallback: concentric rings
        best_points = np.zeros((N, 2))
        best_r = 0.01
    
    return best_points, best_r

def _soft_min_objective(x, N, sharpness=50.0):
    """
    Differentiable approximation of the packing radius using soft-min.
    x is flattened array of 2*N coordinates.
    We want to maximize r = min(min_pair_dist/2, min_boundary_dist)
    This is equivalent to minimizing -r.
    
    Using log-sum-exp for soft-min:
    min(a, b) ~ -1/s * log(exp(-s*a) + exp(-s*b))
    """
    points = x.reshape(N, 2)
    
    # Boundary distances
    norms = np.linalg.norm(points, axis=1)
    boundary_dists = 1.0 - norms
    
    # Pairwise distances
    d = pdist(points)
    pair_dists = d / 2.0  # radius contribution from pairs
    
    # Combine all constraints into one array for soft-min
    # We want the minimum of all these values
    all_vals = np.concatenate([boundary_dists, pair_dists])
    
    # Soft-min approximation
    # To avoid overflow, subtract max
    max_val = np.max(all_vals)
    log_sum = np.log(np.sum(np.exp(-sharpness * (all_vals - max_val))))
    soft_min_val = max_val - np.log(log_sum) / sharpness
    
    # We want to maximize this, so minimize negative
    return -soft_min_val

def _soft_min_gradient(x, N, sharpness=50.0):
    """
    Gradient of the soft-min objective.
    """
    points = x.reshape(N, 2)
    
    # Boundary distances
    norms = np.linalg.norm(points, axis=1)
    boundary_dists = 1.0 - norms
    
    # Pairwise distances
    d = pdist(points)
    pair_dists = d / 2.0
    
    all_vals = np.concatenate([boundary_dists, pair_dists])
    
    # Soft-min value
    max_val = np.max(all_vals)
    exp_terms = np.exp(-sharpness * (all_vals - max_val))
    log_sum = np.log(np.sum(exp_terms))
    
    # Gradient of soft-min w.r.t. each constraint value
    # d(soft_min)/d(val_i) = -sharpness * exp(-sharpness*(val_i - max)) / sum(exp(...))
    # Actually, let's derive properly.
    # f = max_val - (1/s) * log(sum(exp(-s*(v_i - max_val))))
    # Let u_i = -s*(v_i - max_val)
    # f = max_val - (1/s) * log(sum(exp(u_i)))
    # df/dv_i = df/du_i * du_i/dv_i
    # df/du_i = -(1/s) * exp(u_i) / sum(exp(u_j))
    # du_i/dv_i = -s
    # So df/dv_i = -(1/s) * exp(u_i) / sum(exp(u_j)) * (-s) = exp(u_i) / sum(exp(u_j))
    
    weights = exp_terms / np.sum(exp_terms)
    
    # Gradient w.r.t. points
    grad = np.zeros((N, 2))
    
    # Boundary part: d(boundary_dist_i)/d(point_i) = -point_i / norm_i
    for i in range(N):
        if norms[i] > 1e-10:
            grad[i] -= weights[i] * points[i] / norms[i]
        else:
            grad[i] -= weights[i] * 0.0  # gradient undefined at origin, use 0
    
    # Pair part: d(pair_dist_k)/d(point_i) and d(point_j)
    # pair_dist_k = ||p_i - p_j|| / 2
    # d/dp_i = (p_i - p_j) / (2 * ||p_i - p_j||)
    # d/dp_j = (p_j - p_i) / (2 * ||p_i - p_j||)
    
    # We need to map from pdist index to (i, j)
    # pdist uses condensed form. Let's just compute pairwise directly for gradient
    # This is O(N^2) but N=300 is manageable for gradient computation
    
    for k in range(N * (N - 1) // 2):
        # Find i, j for condensed index k
        # k = i*(2*N - i - 3)//2 + j - i - 1
        # This is complex, let's use a different approach
        pass
    
    # Alternative: compute gradient numerically or use a simpler formulation
    # For now, let's use finite differences or a simpler gradient estimate
    # Actually, let's just use the boundary gradient and approximate pair gradient
    
    # For pair distances, the gradient contribution to point i is:
    # sum over j != i of weights_pair_{ij} * (p_i - p_j) / (2 * ||p_i - p_j||)
    
    # Let's compute this properly
    pair_weights = weights[N:]  # weights for pair distances
    
    # We need to map pair_weights back to (i, j) pairs
    # Let's just iterate through all pairs
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            w = pair_weights[idx]
            diff = points[i] - points[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-10:
                grad[i] += w * diff / (2 * dist)
                grad[j] -= w * diff / (2 * dist)
            idx += 1
    
    # The objective is -soft_min, so gradient is -grad
    return -grad.flatten()

def construct():
    """Construct the optimal packing."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # 1. Get initial hexagonal lattice
    points, r_init = _initial_hexagonal()
    best_points = points.copy()
    best_r = r_init
    
    # 2. Optimize using L-BFGS-B with soft-min objective
    if time.time() < deadline - 10:
        x0 = points.flatten()
        
        # Try different sharpness values
        for sharpness in [30.0, 50.0, 100.0]:
            if time.time() >= deadline - 10:
                break
            
            try:
                result = minimize(
                    _soft_min_objective,
                    x0,
                    method='L-BFGS-B',
                    args=(N, sharpness),
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 10000}
                )
                
                if time.time() >= deadline - 10:
                    break
                
                # Check if result improved
                new_points = result.x.reshape(N, 2)
                new_r = _compute_radius(new_points)
                
                if new_r > best_r + 1e-9:
                    best_r = new_r
                    best_points = new_points.copy()
                    x0 = result.x  # Continue from here if improved
                    
            except Exception as e:
                pass
    
    # 3. Local refinement with small perturbations
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(N):
                if time.time() >= deadline - 5:
                    break
                # Try small perturbations in 4 directions
                for dx, dy in [(-0.001, 0), (0.001, 0), (0, -0.001), (0, 0.001)]:
                    new_points = best_points.copy()
                    new_points[i] = [best_points[i][0] + dx, best_points[i][1] + dy]
                    
                    # Quick boundary check
                    if np.linalg.norm(new_points[i]) > 1.0:
                        continue
                    
                    r_new = _compute_radius(new_points)
                    if r_new > best_r + 1e-9:
                        best_r = r_new
                        best_points = new_points
                        improved = True
    
    return best_points
# EVOLVE-BLOCK-END
