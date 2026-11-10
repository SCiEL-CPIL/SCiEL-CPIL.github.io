import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _hex_lattice_init(spacing, offset_x=0.0, offset_y=0.0):
    """Generate points on a hexagonal lattice inside the unit circle."""
    points = []
    # Estimate range
    k = int(1.0 / spacing) + 3
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing + offset_x
            y = j * spacing * np.sqrt(3.0) / 2.0 + offset_y
            if x*x + y*y <= 1.0:
                points.append([x, y])
    points = np.array(points)
    if len(points) < N:
        # If not enough, reduce spacing and retry or just pad (should not happen with good spacing)
        return None
    # Select the N points closest to the origin to form a compact cluster
    radii = np.hypot(points[:, 0], points[:, 1])
    indices = np.argsort(radii)[:N]
    return points[indices]

def _calc_radius(points):
    """Calculate the exact packing radius for a given set of points."""
    # Boundary constraint
    boundary_r = 1.0 - np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(boundary_r)
    
    # Pairwise constraint
    # Use broadcasting for efficiency, O(N^2) memory but N=50 is small
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    # Set diagonal to inf to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)

def _softmin_objective(params, beta):
    """
    Objective function for optimization.
    Minimizes -softmin(constraints).
    """
    points = params.reshape(-1, 2)
    
    # 1. Boundary constraints: 1 - ||p_i||
    norms = np.hypot(points[:, 0], points[:, 1])
    boundary_vals = 1.0 - norms
    
    # 2. Pairwise constraints: ||p_i - p_j|| / 2
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    # We only need the upper triangle to avoid duplicates, but for softmin sum it doesn't matter much if we are consistent.
    # Actually, summing over all i!=j is fine.
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    pair_vals = dists[mask] / 2.0
    
    # Combine all constraints
    all_vals = np.concatenate([boundary_vals, pair_vals])
    
    # Soft min calculation
    # To avoid overflow/underflow in exp, shift by min
    min_val = np.min(all_vals)
    exp_vals = np.exp(-beta * (all_vals - min_val))
    sum_exp = np.sum(exp_vals)
    
    soft_min = min_val - np.log(sum_exp) / beta
    
    # We want to MAXIMIZE soft_min, so we MINIMIZE -soft_min
    return -soft_min

def _softmin_gradient(params, beta):
    """
    Gradient of the softmin objective.
    """
    points = params.reshape(-1, 2)
    
    # 1. Boundary constraints
    norms = np.hypot(points[:, 0], points[:, 1])
    boundary_vals = 1.0 - norms
    
    # 2. Pairwise constraints
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    mask = ~np.eye(N, dtype=bool)
    pair_vals = dists[mask] / 2.0
    
    all_vals = np.concatenate([boundary_vals, pair_vals])
    
    min_val = np.min(all_vals)
    exp_vals = np.exp(-beta * (all_vals - min_val))
    sum_exp = np.sum(exp_vals)
    
    # Weights for each constraint
    weights = exp_vals / sum_exp
    
    # Gradient accumulation
    grad = np.zeros_like(points)
    
    # Boundary part
    # d/dp_i (1 - ||p_i||) = -p_i / ||p_i||
    # Contribution to grad: -weight * (d softmin/d val) * (d val/d p)
    # d softmin/d val_k = weight_k
    # Objective is -softmin, so grad_obj = - sum(weight_k * grad_val_k)
    
    # Boundary vals are the first N elements of all_vals
    w_boundary = weights[:N]
    
    # Avoid division by zero if point is at origin
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    boundary_grad = -points / safe_norms[:, np.newaxis]
    
    grad += -w_boundary[:, np.newaxis] * boundary_grad
    
    # Pairwise part
    # The pair_vals array is flattened from the masked matrix.
    # We need to map weights back to the (i,j) pairs.
    # This is tricky with the flattened array. 
    # Alternative: Compute gradient analytically for pairs directly.
    
    # Let's re-calculate gradient for pairs more directly to avoid mapping issues.
    # For each pair (i, j), val = ||p_i - p_j|| / 2.
    # grad_val wrt p_i = (p_i - p_j) / ||p_i - p_j||
    # grad_val wrt p_j = (p_j - p_i) / ||p_i - p_j||
    
    # We have weights for all pairs in `weights[N:]` corresponding to `pair_vals`.
    # We need to reconstruct which weight goes with which (i,j).
    # The mask `mask` is True for i != j.
    # The order of `pair_vals` is row-major order of the masked matrix.
    
    # Let's create a weight matrix
    w_pair_matrix = np.zeros((N, N))
    # Flatten the mask to get indices
    idx_i, idx_j = np.where(mask)
    w_pair_vals = weights[N:]
    
    w_pair_matrix[idx_i, idx_j] = w_pair_vals
    
    # Now sum up gradients
    # For point i, it is involved in pairs (i, j) for all j != i.
    # Gradient contribution from pair (i, j) to p_i:
    # - w_ij * (p_i - p_j) / ||p_i - p_j||
    
    # Vectorized approach:
    # diff = p_i - p_j for all i,j
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_matrix = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero
    safe_dists = np.where(dists_matrix > 1e-10, dists_matrix, 1.0)
    
    # Unit vectors
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    # We want sum over j of - w_ij * unit_vec_ij
    # Note: w_pair_matrix is symmetric? 
    # dists is symmetric, so pair_vals is symmetric. Weights are symmetric.
    # w_pair_matrix[i, j] = w_pair_matrix[j, i].
    
    # grad_i = - sum_j w_ij * unit_vec_ij
    # We can do this by matrix multiplication if we treat weights as a matrix.
    # grad = - (w_pair_matrix * unit_vecs).sum(axis=1) ? No, unit_vecs is 3D.
    
    # Let's loop or use einsum
    # w_pair_matrix is (N, N)
    # unit_vecs is (N, N, 2)
    # We want to sum over j (axis 1) for each i.
    
    # Multiply weight matrix (broadcast to N, N, 1) with unit vectors
    weighted_vecs = w_pair_matrix[:, :, np.newaxis] * unit_vecs
    pair_grad = -np.sum(weighted_vecs, axis=1)
    
    grad += pair_grad
    
    return grad.flatten()

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    best_radius = 0.0
    best_points = None
    
    # Try a few different initializations
    # 1. Standard hex lattice
    # 2. Slightly perturbed hex lattice
    # 3. Different spacing
    
    init_strategies = []
    
    # Strategy 1: Hex lattice with spacing tuned to fit ~50 points
    # Area of unit circle ~ 3.14. Area per circle ~ 3.14/50 ~ 0.062.
    # Hex packing density ~ 0.9.
    # Effective area per circle ~ 0.062 / 0.9 ~ 0.069.
    # Hex cell area ~ s^2 * sqrt(3)/2.
    # s^2 ~ 0.069 * 2 / 1.732 ~ 0.079.
    # s ~ 0.28.
    # Radius r ~ s/2 ~ 0.14? No, r is the circle radius.
    # If s is distance between centers, min dist is s. r = s/2.
    # So s ~ 0.25 -> r ~ 0.125.
    
    spacings = [0.25, 0.26, 0.24, 0.27]
    offsets = [(0,0), (0.01, 0.01), (-0.01, 0.01), (0.01, -0.01)]
    
    for s in spacings:
        for ox, oy in offsets:
            pts = _hex_lattice_init(s, ox, oy)
            if pts is not None and len(pts) == N:
                init_strategies.append(pts)
    
    # If we didn't get enough inits, add a random one
    if not init_strategies:
        pts = np.random.rand(N, 2) * 2 - 1
        pts = pts[np.hypot(pts[:,0], pts[:,1]) <= 1]
        if len(pts) < N:
             # Resample
             while len(pts) < N:
                 new_pts = np.random.rand(N, 2) * 2 - 1
                 pts = np.vstack([pts, new_pts])
        init_strategies.append(pts[:N])

    # Optimization loop
    # We will iterate through strategies, and for each, do an annealing schedule.
    
    for init_pts in init_strategies:
        if time.time() - start_time > time_limit:
            break
            
        current_pts = init_pts.copy()
        current_params = current_pts.flatten()
        
        # Annealing schedule for beta
        betas = [10, 50, 200, 1000, 5000]
        
        for beta in betas:
            if time.time() - start_time > time_limit:
                break
                
            try:
                res = minimize(
                    _softmin_objective,
                    current_params,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=_softmin_gradient,
                    options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                # Check if optimization succeeded
                if res.success or res.status in [0, 1, 2, 3, 4, 5, 6]:
                    new_pts = res.x.reshape(-1, 2)
                    
                    # Validate: ensure points are inside circle
                    # If a point went outside, we should reject or clamp
                    # The objective penalizes going outside, but L-BFGS-B might overshoot if bounds aren't set.
                    # We didn't set bounds in minimize. We should add bounds: [-1, 1] for each coord.
                    # Let's restart with bounds if we haven't been doing it.
                    # For now, let's just check if it's valid.
                    
                    # Check if all points are roughly inside
                    if np.all(np.hypot(new_pts[:, 0], new_pts[:, 1]) < 1.0 + 1e-6):
                         current_params = res.x
                         current_pts = new_pts
            except Exception:
                pass
        
        # Calculate final radius for this strategy
        final_radius = _calc_radius(current_pts)
        
        if final_radius > best_radius:
            best_radius = final_radius
            best_points = current_pts.copy()
            
        # Optional: Local polish with very high beta
        # Already covered by the last beta in the schedule.
        
    # Final check
    if best_points is None:
        # Fallback to initial lattice
        pts = _hex_lattice_init(0.25)
        if pts is not None:
            best_points = pts
            best_radius = _calc_radius(pts)
        else:
            # Absolute fallback
            best_points = np.zeros((N, 2))
            best_radius = 0.0

    return best_points
# EVOLVE-BLOCK-END
