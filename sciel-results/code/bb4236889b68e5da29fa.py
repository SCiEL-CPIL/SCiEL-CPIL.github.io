"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

_T0 = time.time()
_DEADLINE = _T0 + 140.0

def _compute_radius(pts):
    """Compute the exact packing radius."""
    if len(pts) < 2:
        return 0.0
    # Boundary clearance
    boundary = 0.5 - np.max(np.abs(pts), axis=1)
    min_boundary = np.min(boundary)
    if min_boundary <= 0:
        return 0.0
    # Pairwise distances
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    min_pair = np.min(dists[:, 1])
    return min(min_boundary, min_pair / 2.0)

def _hex_initial(n):
    """Generate a hexagonal grid initialization."""
    # Generate a large hex grid
    # dx = 1, dy = sqrt(3)/2
    # We want to fit in [-0.5, 0.5]
    # Scale factor s such that the grid fits.
    # Let's generate points in a larger area and select the best n.
    
    # Hex lattice vectors
    a = 1.0
    b = a * np.sqrt(3) / 2
    
    # Generate grid indices
    # Estimate size: area of square is 1. Area of hex cell is a*b = sqrt(3)/2 ~ 0.866.
    # So we need roughly 1/0.866 ~ 1.15 cells per unit area? No, 1 cell covers area a*b.
    # Number of points ~ Area / (a*b) * 2 (for packing density?). 
    # Actually, just generate a 20x20 grid and center it.
    
    points = []
    for j in range(-10, 10):
        for i in range(-10, 10):
            x = i * a + (j % 2) * a / 2
            y = j * b
            points.append([x, y])
            
    points = np.array(points)
    points -= points.mean(axis=0)
    
    # Scale to fit within [-0.5, 0.5] with a margin
    # We want the points to be inside, so scale down slightly
    max_coord = np.max(np.abs(points))
    scale = 0.45 / max_coord if max_coord > 0 else 1.0
    points *= scale
    
    # We have ~400 points. We need 200.
    # Select the 200 points closest to the center? 
    # This tends to create a blob. 
    # Better: select points that are well-distributed.
    # For now, closest to center is a decent start for relaxation.
    dists = np.sum(points**2, axis=1)
    indices = np.argsort(dists)[:n]
    return points[indices]

def _soft_min_obj_and_grad(x, alpha):
    """
    Compute the negative soft-min radius and its gradient.
    x: flattened array of shape (2*N,)
    alpha: sharpness parameter
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # 1. Compute pairwise distances
    # Using cKDTree is faster for large N, but for N=200, full matrix is fine and easier for gradients
    # However, cKDTree is O(N log N). Let's use it for the value, but we need gradients.
    # For gradients, we need to know which pairs contribute.
    # With soft min, ALL pairs contribute, but distant ones have negligible weight.
    # We can use a threshold to ignore very distant pairs to speed up gradient calc?
    # Or just compute all. 200*200 = 40000.
    
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Zero out self-distances
    np.fill_diagonal(dists, np.inf)
    
    # Compute boundary distances
    boundary_dists = 0.5 - np.max(np.abs(pts), axis=1)
    
    # Combine all distances into a list of (distance, gradient_info)
    # We need to sum exp(-alpha * d) * grad(d)
    
    # Weights for pairs
    # To avoid underflow/overflow, we can subtract the min distance? 
    # Soft min is -1/alpha * log(sum(exp(-alpha * d)))
    # Let d_min be the min distance.
    # sum(exp(-alpha * (d - d_min))) * exp(-alpha * d_min)
    # log(sum) + log(exp(-alpha * d_min)) = log(sum) - alpha * d_min
    # This is numerically stable.
    
    # We need gradients for each point.
    # grad_obj = - 1/alpha * ( 1/S * sum_k (-alpha * exp(-alpha * d_k) * grad d_k) )
    #           = 1/S * sum_k exp(-alpha * d_k) * grad d_k
    # We are minimizing -soft_min, so we return -soft_min and its gradient.
    # Let J = -soft_min. grad J = - grad soft_min = - (1/S * sum ... )
    
    # Let's compute S = sum(exp(-alpha * d_k))
    # And GradSum = sum(exp(-alpha * d_k) * grad d_k) for each coordinate
    
    # Initialize gradient accumulator
    grad_acc = np.zeros_like(pts)
    
    # Process pairs
    # We only need i < j
    # Create indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    # Distances for these pairs
    d_pairs = dists[i_idx, j_idx]
    
    # Weights
    # Use stable log-sum-exp trick?
    # Actually, for alpha ~ 10-100, exp(-alpha * d) might underflow for large d.
    # But d is at least ~0.01. alpha=100 -> exp(-1) ~ 0.3.
    # If d is large, weight is 0.
    
    # To be safe and fast, we can filter pairs where d < some threshold?
    # No, soft min uses all. But weights decay fast.
    # Let's just compute all.
    
    weights = np.exp(-alpha * d_pairs)
    
    # Sum of weights for S
    S = np.sum(weights)
    
    # If S is 0, return 0 gradient (shouldn't happen)
    if S == 0:
        return 0.0, np.zeros_like(x)
    
    # Gradients of distances for pairs
    # d_ij = ||p_i - p_j||
    # grad_p_i d_ij = (p_i - p_j) / d_ij
    # grad_p_j d_ij = (p_j - p_i) / d_ij
    
    # diff_ij = p_i - p_j
    diff_pairs = diff[i_idx, j_idx]
    
    # Avoid division by zero (shouldn't happen for i!=j)
    d_pairs_safe = np.where(d_pairs > 1e-10, d_pairs, 1.0)
    
    # Unit vectors
    unit_vecs = diff_pairs / d_pairs_safe[:, np.newaxis]
    
    # Contribution to gradient of soft min for each point
    # grad soft_min = 1/S * sum_k w_k * grad d_k
    
    # For point i, it appears in pairs (i, j) where i < j and (k, i) where k < i.
    # This is complex to accumulate with vectorization if we don't use scatter.
    # Let's use a loop for accumulation? N=200 is small.
    # Or use np.add.at
    
    # We have weights w_k for each pair k.
    # For each pair (i, j):
    # grad_i += w_k * unit_vec_ij
    # grad_j += w_k * (-unit_vec_ij)
    
    # np.add.at is efficient enough for 200 points.
    np.add.at(grad_acc, i_idx, weights[:, np.newaxis] * unit_vecs)
    np.add.at(grad_acc, j_idx, -weights[:, np.newaxis] * unit_vecs)
    
    # Process boundaries
    # d_bound_i = 0.5 - max(|x_i|, |y_i|)
    # grad d_bound_i = - grad max(|x_i|, |y_i|)
    
    # Determine which coordinate is max
    abs_x = np.abs(pts[:, 0])
    abs_y = np.abs(pts[:, 1])
    
    # Masks
    mask_x = abs_x > abs_y
    mask_y = ~mask_x
    
    # Gradients for boundary terms
    # If x is max: grad d = (-sign(x), 0)
    # If y is max: grad d = (0, -sign(y))
    
    w_bound = np.exp(-alpha * boundary_dists)
    
    # Accumulate boundary gradients
    # x-component
    grad_acc[mask_x, 0] += w_bound[mask_x] * (-np.sign(pts[mask_x, 0]))
    grad_acc[mask_y, 1] += w_bound[mask_y] * (-np.sign(pts[mask_y, 1]))
    
    # Total S includes boundary terms
    S_total = S + np.sum(w_bound)
    
    # Total gradient of soft min
    grad_soft_min = grad_acc / S_total
    
    # Objective is -soft_min
    # Value: -(-1/alpha * log(S_total)) = 1/alpha * log(S_total) ??
    # Wait. soft_min = -1/alpha * log(S_total).
    # We want to MAXIMIZE soft_min.
    # So we MINIMIZE -soft_min = 1/alpha * log(S_total).
    
    obj_val = (1.0 / alpha) * np.log(S_total)
    
    # Gradient of obj_val
    # d/dx (1/alpha * log(S_total)) = 1/alpha * 1/S_total * dS_total/dx
    # dS_total/dx = sum_k (-alpha * exp(-alpha * d_k) * grad d_k) = -alpha * (S_total * grad_soft_min)
    # So grad obj = 1/alpha * 1/S_total * (-alpha * S_total * grad_soft_min) = -grad_soft_min
    
    grad_obj = -grad_soft_min
    
    return obj_val, grad_obj.flatten()

def construct():
    start_time = time.time()
    
    # Initial points
    pts = _hex_initial(N)
    x0 = pts.flatten()
    
    best_x = x0.copy()
    best_radius = _compute_radius(pts)
    
    # Annealing schedule for alpha
    # Start with smooth (small alpha), end with sharp (large alpha)
    alphas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    for alpha in alphas:
        if time.time() > _DEADLINE - 10:
            break
            
        def obj_and_grad(x):
            return _soft_min_obj_and_grad(x, alpha)
        
        try:
            res = minimize(
                obj_and_grad,
                best_x,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 200,
                    'maxls': 20,
                    'ftol': 1e-12,
                    'gtol': 1e-8
                }
            )
            
            if res.success or res.fun < 0: # res.fun is the obj value
                new_pts = res.x.reshape(-1, 2)
                # Check validity
                if np.all(np.isfinite(new_pts)):
                    new_radius = _compute_radius(new_pts)
                    if new_radius > best_radius:
                        best_radius = new_radius
                        best_x = res.x.copy()
                        best_pts = new_pts.copy()
        except Exception:
            continue
            
    # Final polish with a local search if time permits?
    # L-BFGS-B should be good enough.
    
    # Ensure points are strictly inside
    best_pts = np.clip(best_x.reshape(-1, 2), -0.499999, 0.499999)
    
    # Final radius check
    final_radius = _compute_radius(best_pts)
    if final_radius <= 0:
        # Fallback
        return _hex_initial(N)
        
    return best_pts

# EVOLVE-BLOCK-END
