"""Improved packer: hex-lattice seed + soft-min L-BFGS-B optimization with time guard."""
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 300
SEED = 12345
T_START = time.time()
DEADLINE = 145.0  # safety margin before the 150 s wall-clock limit


def _score(points):
    """Return the feasible common radius (0.0 if invalid)."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # distance to container boundary
    norm = np.linalg.norm(points, axis=1)
    r_bound = 1.0 - norm
    if np.min(r_bound) < 0.0:
        return 0.0
    # minimum pairwise half-distance using KD-tree for efficiency
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    r_pair = 0.5 * np.min(dists[:, 1])
    r = min(r_bound.min(), r_pair)
    return float(r) if r > 0.0 else 0.0


def _min_dist(points):
    """Efficiently compute the minimum pairwise distance."""
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    return float(np.min(dists[:, 1]))


def _lattice_seed(spacing, rng):
    """Generate a hexagonal lattice and pick N points inside the disk."""
    # Add slight random jitter to break symmetry
    while True:
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                # Add small noise
                x += rng.normal(0, spacing * 0.05)
                y += rng.normal(0, spacing * 0.05)
                rows.append((x, y))
        pts = np.array(rows)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        keep = rad <= 1.0 - spacing / 2.0
        pts = pts[keep]
        if len(pts) >= N:
            # prefer inner points (smaller radius) to maximise margin
            order = np.argsort(np.hypot(pts[:, 0], pts[:, 1]), kind="stable")
            return pts[order[:N]].copy()
        spacing *= 0.97


def _soft_min_obj(x, beta):
    """
    Objective function for L-BFGS-B.
    We want to maximize the minimum distance.
    L-BFGS-B minimizes, so we minimize -soft_min(distances).
    
    Distances include:
    1. Pairwise distances between centers.
    2. Distances to the boundary (1 - ||x_i||).
    
    We use a log-sum-exp based soft-min to smooth the min function.
    """
    pts = x.reshape(N, 2)
    
    # Check bounds (should be handled by constraints, but safe to check)
    norms = np.linalg.norm(pts, axis=1)
    if np.any(norms > 1.0):
        return 1e10 # Penalty for being outside

    # Pairwise distances
    # To avoid O(N^2) memory blowup for large N in Python loops, 
    # we rely on the fact that L-BFGS-B calls this many times.
    # For N=300, N^2 = 90,000 distances. This is manageable in NumPy.
    # However, constructing the full distance matrix every time is slow.
    # Let's use a more efficient approach or accept the cost if optimized.
    
    # Efficient distance calculation using broadcasting
    # d_ij = ||x_i - x_j||
    # We need min_{i<j} d_ij and min_i (1 - ||x_i||)
    
    # Boundary terms
    boundary_terms = 1.0 - norms
    
    # Pairwise terms
    # Calculate all pairwise distances
    # diff = x_i - x_j
    # Using broadcasting: (N, 1, 2) - (1, N, 2) -> (N, N, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask out self-distances (diagonal)
    mask = ~np.eye(N, dtype=bool)
    pair_dists = dists[mask]
    
    # Combine all terms: pairwise distances and boundary margins
    # We want to maximize the minimum of these.
    # Let's collect all relevant distances into a single array
    all_terms = np.concatenate([pair_dists, boundary_terms])
    
    # Soft min: min(x) approx -1/beta * log(sum(exp(-beta*x)))
    # To maximize min, we minimize -soft_min = 1/beta * log(sum(exp(-beta*x)))
    
    # Numerical stability: shift by max value
    max_term = np.max(all_terms)
    exp_terms = np.exp(-beta * (all_terms - max_term))
    log_sum_exp = np.log(np.sum(exp_terms)) + beta * max_term
    soft_min_val = -log_sum_exp / beta
    
    # Return negative to minimize
    return -soft_min_val


def _grad_soft_min_obj(x, beta):
    """
    Gradient of the soft-min objective.
    d/dx_i (-soft_min) = sum_j (w_j * d(d_j)/dx_i)
    where w_j = exp(-beta*d_j) / sum_k(exp(-beta*d_k))
    """
    pts = x.reshape(N, 2)
    norms = np.linalg.norm(pts, axis=1)
    
    # Boundary gradients
    # d(1 - ||x_i||)/dx_i = -x_i / ||x_i||
    boundary_grads = -pts / norms[:, np.newaxis]
    boundary_terms = 1.0 - norms
    
    # Pairwise gradients
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero for self
    dists[np.eye(N, dtype=bool)] = 1.0
    
    # Weights for soft-min
    # w_j = exp(-beta * d_j) / Z
    # Note: We only consider unique pairs i<j and boundaries.
    # This is tricky to vectorize perfectly with the mask.
    
    # Let's compute the gradient numerically if analytical is too complex/slow to implement correctly in one go?
    # No, analytical is better for L-BFGS-B.
    
    # Let's re-formulate.
    # Let D be the vector of all distances (pairs and boundaries).
    # Obj = -1/beta * log(sum(exp(-beta * D)))
    # Grad = 1/beta * sum_j (exp(-beta * D_j) / sum_k(exp(-beta * D_k))) * d(D_j)/dx
    
    # We need to accumulate gradients for each point.
    
    # 1. Boundary part
    # For each point i, term is 1 - ||x_i||
    # d/dx_i = -x_i/||x_i||
    # Weight w_i_bound = exp(-beta * (1 - ||x_i||)) / Z_total
    
    # 2. Pairwise part
    # For each pair (i, j), term is ||x_i - x_j||
    # d/dx_i = (x_i - x_j) / ||x_i - x_j||
    # d/dx_j = (x_j - x_i) / ||x_i - x_j||
    # Weight w_ij = exp(-beta * ||x_i - x_j||) / Z_total
    
    # Z_total = sum_boundaries(exp(-beta * b_i)) + sum_pairs(exp(-beta * d_ij))
    
    # Calculate exp terms
    exp_bound = np.exp(-beta * boundary_terms)
    
    # For pairs, we need the upper triangle
    iu = np.triu_indices(N, 1)
    pair_dists = dists[iu]
    exp_pairs = np.exp(-beta * pair_dists)
    
    Z_total = np.sum(exp_bound) + np.sum(exp_pairs)
    
    if Z_total == 0:
        return np.zeros_like(x)
        
    grad = np.zeros_like(pts)
    
    # Boundary contributions
    w_bound = exp_bound / Z_total
    grad += np.sum(w_bound[:, np.newaxis] * boundary_grads, axis=0) # Wait, sum is over points? No.
    # The objective is a scalar. The gradient is a vector.
    # d(Obj)/dx_i gets contribution from its own boundary term and all pairs involving i.
    
    # Correction:
    # Grad_i = w_i_bound * (-x_i/||x_i||) + sum_{j != i} w_ij * (x_i - x_j)/d_ij
    
    # Let's compute pairwise gradients properly.
    # We have vectors for iu[0] (i indices) and iu[1] (j indices).
    # diff_ij = x_i - x_j
    # unit_ij = diff_ij / d_ij
    
    diff_pairs = diff[iu] # Shape (M, 2)
    unit_pairs = diff_pairs / pair_dists[:, np.newaxis]
    
    w_pairs = exp_pairs / Z_total
    
    # Scatter add to grad
    # For each pair (i, j):
    # grad[i] += w * unit
    # grad[j] += -w * unit
    
    np.add.at(grad, iu[0], w_pairs[:, np.newaxis] * unit_pairs)
    np.add.at(grad, iu[1], -w_pairs[:, np.newaxis] * unit_pairs)
    
    # Boundary part
    # grad[i] += w_bound[i] * (-x_i/||x_i||)
    grad += w_bound[:, np.newaxis] * boundary_grads
    
    return grad.flatten()


def _optimize(points, rng):
    """Run L-BFGS-B optimization starting from points."""
    start = time.time()
    if time.time() - start > 5: # Safety check
        return points, _score(points)
        
    x0 = points.flatten()
    
    # Adaptive beta schedule
    # Start with low beta (smooth) and increase (sharper)
    betas = [50, 100, 200, 500, 1000]
    
    best_x = x0.copy()
    best_r = _score(points)
    
    for beta in betas:
        if time.time() - start > 10: # Time budget per start
            break
            
        # Define objective and gradient for this beta
        def obj(x):
            return _soft_min_obj(x, beta)
            
        def grad(x):
            return _grad_soft_min_obj(x, beta)
            
        try:
            res = minimize(obj, x0, jac=grad, method='L-BFGS-B', options={'maxiter': 200, 'maxfun': 200})
            if res.success or res.fun < obj(x0):
                new_pts = res.x.reshape(N, 2)
                # Validate and project inside if necessary (L-BFGS-B doesn't strictly enforce hard bounds if not specified, 
                # but we can add bounds)
                
                # Let's add bounds to L-BFGS-B to keep points in [-1, 1] roughly
                # Actually, the penalty in obj handles it, but bounds are better.
                # Bounds: -1 <= x <= 1
                bounds = [(-1.0, 1.0)] * (2 * N)
                
                # Re-run with bounds? Or just accept if valid.
                # Let's just check validity.
                r = _score(new_pts)
                if r > best_r:
                    best_r = r
                    best_x = res.x.copy()
                    x0 = res.x.copy()
        except Exception:
            pass
            
    return best_x.reshape(N, 2), best_r


def construct():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # Initial hex-lattice seed
    # Estimate spacing for 300 points in unit circle
    # Area ~ pi * r^2. N * pi * (r/2)^2 ~ pi * 1^2 -> r ~ 2/sqrt(N)
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    
    # Generate initial seed
    pts = _lattice_seed(spacing, rng)
    
    best_pts = pts.copy()
    best_r = _score(best_pts)
    
    # Multi-start optimization
    # We have ~140s budget.
    # Each optimization run takes ~5-10s depending on convergence.
    # We can afford ~10-20 restarts.
    
    num_restarts = 15
    
    for i in range(num_restarts):
        if time.time() - t0 > DEADLINE - 10:
            break
            
        # Perturb the current best or generate a new seed
        if i == 0:
            start_pts = best_pts
        else:
            # Mix of best and random perturbation
            if rng.random() < 0.5:
                start_pts = best_pts + rng.normal(0, 0.05, (N, 2))
            else:
                start_pts = _lattice_seed(spacing, rng)
                
        # Clip to unit circle
        norms = np.linalg.norm(start_pts, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            start_pts[mask] = start_pts[mask] / norms[mask, None] * (1.0 - 1e-6)
            
        # Optimize
        opt_pts, opt_r = _optimize(start_pts, rng)
        
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts
            
        # Check time
        if time.time() - t0 > DEADLINE - 5:
            break
            
    # Final check
    final_r = _score(best_pts)
    if final_r <= 0.0:
        # Fallback
        best_pts = _lattice_seed(spacing, rng)
        
    return best_pts
