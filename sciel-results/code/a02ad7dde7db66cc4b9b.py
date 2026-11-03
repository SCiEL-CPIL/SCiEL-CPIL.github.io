import numpy as np
import time

N = 300

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice initialization."""
    rng = np.random.default_rng(seed)
    # Optimal spacing for 300 circles is roughly 2/sqrt(300) ~ 0.115
    # But we need to fit in unit circle. Area of unit circle is pi.
    # Area per circle ~ pi/300. Radius ~ sqrt(pi/300/3.14) ~ 1/sqrt(300) ~ 0.057.
    # Distance ~ 2r ~ 0.115.
    
    best_pts = None
    best_r = 0.0
    
    # Try a few different lattice spacings to find a good start
    for attempt in range(5):
        # Start with a slightly looser lattice to ensure validity, then tighten
        spacing = 2.0 / (np.sqrt(n_points) * (0.8 + 0.05 * attempt))
        k = int(1.0 / spacing) + 2
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing/2: # Ensure initial validity roughly
                    points.append([x, y])
        
        if len(points) < n_points:
            continue
            
        pts = np.array(points)
        # Select the n_points closest to the center
        dists = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(dists)
        pts = pts[idx[:n_points]]
        
        # Calculate initial radius
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
            
    if best_pts is None:
        # Fallback: uniform random
        best_pts = rng.random((n_points, 2)) * 0.8
        best_pts -= 0.4
        # Scale to fit
        d = np.hypot(best_pts[:, 0], best_pts[:, 1])
        max_d = np.max(d)
        if max_d > 0:
            best_pts *= 0.9 / max_d
        best_r = _compute_radius(best_pts)
            
    return best_pts, best_r

def _compute_radius(pts):
    """Compute the maximum valid radius for the given centers."""
    n = len(pts)
    # Container constraint
    r_cont = 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1]))
    
    # Pairwise constraint
    # For N=300, O(N^2) is 90k ops, which is fast in numpy
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    
    return min(r_cont, r_pair)

def _soft_min_logsumexp(values, beta):
    """
    Compute -1/beta * log(sum(exp(-beta * values)))
    This approximates min(values) as beta -> infinity.
    """
    # Shift for numerical stability
    min_val = np.min(values)
    shifted = values - min_val
    exp_vals = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_vals))
    return min_val - (1.0 / beta) * log_sum

def _objective_and_grad(pts, beta):
    """
    Compute the soft-min objective and its gradient.
    Objective: min( min_ij dist_ij/2, min_i (1 - dist_i) )
    We combine them into a single list of constraints.
    """
    n = len(pts)
    
    # 1. Container constraints: c_i = 1 - |p_i|
    dists_center = np.hypot(pts[:, 0], pts[:, 1])
    c_cont = 1.0 - dists_center
    
    # 2. Pairwise constraints: c_ij = |p_i - p_j| / 2
    # We need the gradient of these.
    # Gradient of |p_i - p_j| wrt p_i is (p_i - p_j) / |p_i - p_j|
    # Gradient of c_ij wrt p_i is 0.5 * (p_i - p_j) / |p_i - p_j|
    
    # To save memory and time, we can compute the gradient directly in the loop
    # or use vectorization.
    
    # Let's build the list of all constraints and their gradients.
    # This is O(N^2) which is acceptable for N=300.
    
    # We will compute the weights w_k = exp(-beta * c_k) / sum(exp(-beta * c_j))
    # Then grad = sum w_k * grad(c_k)
    
    # For efficiency, we'll compute the "energy" E = sum exp(-beta * c_k)
    # and the gradient.
    
    # We process container constraints first
    # c_i = 1 - dist_i
    # grad c_i = - p_i / dist_i
    
    # Pairwise constraints
    # c_ij = 0.5 * dist_ij
    # grad c_ij wrt p_i = 0.5 * (p_i - p_j) / dist_ij
    # grad c_ij wrt p_j = 0.5 * (p_j - p_i) / dist_ij
    
    # We'll accumulate the sum of exp(-beta*c) and the weighted gradients.
    
    # To avoid storing all N^2 constraints, we can compute the weights on the fly
    # but we need the normalization factor (sum of all exps).
    # So we do two passes or store the values.
    
    # Pass 1: Compute all constraint values and exps
    # Container
    exp_cont = np.exp(-beta * c_cont)
    sum_exp_cont = np.sum(exp_cont)
    
    # Pairwise
    # We need to be careful with memory. 300x300 is 90k floats. Fine.
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    c_pair = 0.5 * dists
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    
    # Compute exps for pairs
    # To avoid underflow/overflow issues with large beta, we already shifted in _soft_min, 
    # but here we are computing raw exps for weights.
    # We should shift by the global min to be safe.
    min_c = np.min(np.concatenate([c_cont, c_pair[mask]]))
    
    # Recalculate exps with shift
    exp_cont_shifted = np.exp(-beta * (c_cont - min_c))
    exp_pair_shifted = np.exp(-beta * (c_pair - min_c))
    exp_pair_shifted[~mask] = 0.0
    
    total_exp = np.sum(exp_cont_shifted) + np.sum(exp_pair_shifted)
    
    if total_exp == 0:
        return 0, np.zeros_like(pts)
    
    # Objective value (reversing the shift)
    obj_val = min_c - (1.0 / beta) * np.log(total_exp)
    
    # Gradients
    grad = np.zeros_like(pts)
    
    # Container gradient contribution
    # w_i = exp_cont_shifted[i] / total_exp
    # grad c_i = - p_i / dist_i
    weights_cont = exp_cont_shifted / total_exp
    # Handle dist=0
    safe_dist = np.where(dists_center < 1e-8, 1.0, dists_center)
    grad_cont = -pts / safe_dist[:, np.newaxis]
    grad += np.sum(weights_cont[:, np.newaxis] * grad_cont, axis=0)
    
    # Pairwise gradient contribution
    # For each pair (i,j), weight w_ij = exp_pair_shifted[i,j] / total_exp
    # grad wrt p_i: 0.5 * (p_i - p_j) / dist_ij
    # grad wrt p_j: 0.5 * (p_j - p_i) / dist_ij
    
    # Vectorized:
    # We have diff[i,j] = p_i - p_j
    # dists[i,j]
    # unit_vec[i,j] = diff[i,j] / dists[i,j]
    
    safe_dists = np.where(dists < 1e-8, 1.0, dists)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    # weights_pair[i,j] is the weight for the constraint between i and j
    weights_pair = exp_pair_shifted / total_exp
    
    # grad_i += sum_j w_ij * 0.5 * unit_vec[i,j]
    # grad_j += sum_i w_ij * 0.5 * (-unit_vec[i,j])
    
    # Using matrix multiplication:
    # grad_pair_i = 0.5 * (weights_pair @ unit_vecs)
    # Note: weights_pair is symmetric.
    
    grad_pair_i = 0.5 * (weights_pair @ unit_vecs)
    
    # The total gradient is the sum of contributions from i being the first index and j being the second.
    # Since the matrix is symmetric, the contribution to point i from pair (i,j) is 0.5 * w_ij * u_ij.
    # The contribution to point i from pair (j,i) is 0.5 * w_ji * u_ji = 0.5 * w_ij * (-u_ij).
    # Wait, the constraint c_ij is the same as c_ji.
    # In our sum, we iterate over all i,j.
    # The term for point i in the gradient of the objective is:
    # sum_{j != i} w_ij * grad_{p_i} c_ij
    # = sum_{j != i} w_ij * 0.5 * (p_i - p_j)/dist_ij
    
    # So we just need to sum the weighted unit vectors for each row i.
    grad += grad_pair_i
    
    return obj_val, grad

def _local_optimize(pts, max_iters=2000, beta_start=10.0, beta_end=1000.0, lr_init=0.01):
    """
    Gradient ascent on the soft-min objective.
    """
    n = len(pts)
    pts = pts.copy()
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    # Adaptive learning rate and beta
    for it in range(max_iters):
        # Increase beta to focus on active set
        progress = it / max_iters
        beta = beta_start * (beta_end / beta_start) ** progress
        
        # Get objective and gradient
        _, grad = _objective_and_grad(pts, beta)
        
        # Normalize gradient to prevent huge steps
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-8:
            grad = grad / grad_norm
            
        # Adaptive learning rate: decay over time
        lr = lr_init * (0.5 ** (progress * 5))
        
        # Update
        pts_new = pts + lr * grad
        
        # Project to unit disk
        dists = np.hypot(pts_new[:, 0], pts_new[:, 1])
        mask = dists > 0.999
        if np.any(mask):
            pts_new[mask] *= 0.999 / dists[mask, np.newaxis]
            
        # Check improvement
        r_new = _compute_radius(pts_new)
        if r_new > best_r:
            best_r = r_new
            best_pts = pts_new.copy()
            pts = pts_new
        else:
            # If no improvement, try a smaller step or kick
            # For now, just keep old pts, but we might get stuck.
            # Let's add a small random kick if stuck for a while
            if it % 100 == 0 and it > 0:
                # Simple kick
                kick = np.random.randn(n, 2) * 0.001
                pts_kick = pts + kick
                d = np.hypot(pts_kick[:, 0], pts_kick[:, 1])
                pts_kick[d > 0.999] *= 0.999 / d[d > 0.999, np.newaxis]
                r_kick = _compute_radius(pts_kick)
                if r_kick > best_r:
                    best_r = r_kick
                    best_pts = pts_kick.copy()
                    pts = pts_kick
    
    return best_pts, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    # 1. Initialization
    pts, r_init = _hex_init(N, seed=42)
    best_pts = pts.copy()
    best_r = r_init
    
    # 2. Local Optimization Phase 1 (Medium Beta)
    pts, r = _local_optimize(pts, max_iters=1000, beta_start=5.0, beta_end=100.0, lr_init=0.02)
    if r > best_r:
        best_r = r
        best_pts = pts.copy()
        
    if time.time() > deadline - 30:
        return best_pts
        
    # 3. Local Optimization Phase 2 (High Beta, Focus on Active Set)
    pts, r = _local_optimize(pts, max_iters=1000, beta_start=100.0, beta_end=2000.0, lr_init=0.005)
    if r > best_r:
        best_r = r
        best_pts = pts.copy()
        
    if time.time() > deadline - 15:
        return best_pts
        
    # 4. Final Polish (Very High Beta, Tiny Steps)
    pts, r = _local_optimize(pts, max_iters=500, beta_start=2000.0, beta_end=5000.0, lr_init=0.001)
    if r > best_r:
        best_r = r
        best_pts = pts.copy()
        
    return best_pts
