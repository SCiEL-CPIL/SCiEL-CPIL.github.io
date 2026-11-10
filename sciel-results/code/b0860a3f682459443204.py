"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def _smooth_min(x, beta=50.0):
    """Smooth approximation of min using log-sum-exp."""
    # min(x) = -max(-x) = -(1/beta) * log(sum(exp(-beta*x_i)))
    neg_beta_x = -beta * x
    max_neg = np.max(neg_beta_x)
    if max_neg < -100:
        return np.min(x)
    return -(1.0 / beta) * (max_neg + np.log(np.sum(np.exp(neg_beta_x - max_neg))))

def _compute_radius(pts):
    """Compute the achievable radius for given centers."""
    if len(pts) < 2:
        return 0.0
    # Pairwise distances
    from scipy.spatial.distance import cdist
    dists = cdist(pts, pts)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Distance to boundary
    norms = np.linalg.norm(pts, axis=1)
    min_boundary = np.min(1.0 - norms)
    return 0.5 * min(min_pair, 2.0 * min_boundary)

def _gradient_step(pts, beta=30.0, lr=0.001):
    """One gradient ascent step to maximize the smooth min radius."""
    pts = pts.copy()
    n = len(pts)
    
    # Compute pairwise distances
    from scipy.spatial.distance import cdist
    dists = cdist(pts, pts)
    np.fill_diagonal(dists, np.inf)
    
    # Distance to boundary
    norms = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - norms
    
    # We want to maximize: min(min_pair_dist/2, min_boundary)
    # Use soft-min surrogate
    # The radius is min(d_ij/2 for i<j, 1-|c_i|)
    
    # Gradient of soft min of all constraints
    # For pair (i,j): contribution to gradient
    # For boundary i: contribution to gradient
    
    # Let's use a simpler approach: repulsive forces
    # Each point is pushed away from neighbors and pulled toward center
    
    forces = np.zeros_like(pts)
    
    # Pairwise repulsion: push apart if too close
    for i in range(n):
        for j in range(i + 1, n):
            d = dists[i, j]
            if d < 0.1:  # Only consider close pairs for efficiency
                diff = pts[i] - pts[j]
                if d > 1e-10:
                    force = diff / d * (1.0 - d / 0.1)
                    forces[i] += force
                    forces[j] -= force
    
    # Boundary attraction: push inward if too close to boundary
    for i in range(n):
        r = norms[i]
        if r > 0.9:  # Close to boundary
            if r > 1e-10:
                forces[i] -= pts[i] / r * (r - 0.9)
    
    # Update positions
    pts += lr * forces
    
    # Project back inside the unit circle
    new_norms = np.linalg.norm(pts, axis=1)
    mask = new_norms > 1.0
    if np.any(mask):
        pts[mask] = pts[mask] / new_norms[mask][:, np.newaxis]
    
    return pts

def _local_refine(pts, max_iter=2000, time_limit=100.0):
    """Local refinement using gradient-based approach with adaptive learning rate."""
    start_time = time.time()
    pts = pts.copy()
    best_radius = _compute_radius(pts)
    best_pts = pts.copy()
    
    beta = 20.0
    lr = 0.005
    no_improve_count = 0
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute forces based on closest neighbors
        from scipy.spatial.distance import cdist
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        
        norms = np.linalg.norm(pts, axis=1)
        
        # Find the bottleneck: which constraint is tightest
        min_pair_dist = np.min(dists)
        min_boundary = np.min(1.0 - norms)
        current_r = 0.5 * min(min_pair_dist, 2.0 * min_boundary)
        
        if current_r > best_radius + 1e-12:
            best_radius = current_r
            best_pts = pts.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count > 500:
            # Reduce learning rate and increase beta
            lr *= 0.5
            beta *= 1.5
            no_improve_count = 0
            if lr < 1e-8:
                break
        
        # Compute gradient using soft-min
        # Objective: maximize soft_min(d_ij/2, 1-|c_i|)
        # All constraints: d_ij/2 for i<j, and 1-|c_i| for each i
        
        # Build the constraint vector
        tri_idx = np.triu_indices(n, k=1)
        pair_dists = dists[tri_idx] / 2.0
        boundary = 1.0 - norms
        
        # Soft min of all constraints
        all_constraints = np.concatenate([pair_dists, boundary])
        
        # Gradient of soft min
        exp_neg = np.exp(-beta * all_constraints)
        exp_neg_max = np.max(exp_neg)
        weights = exp_neg / (np.sum(exp_neg) + 1e-30)
        
        # Gradient with respect to positions
        grad = np.zeros_like(pts)
        
        # Pairwise contributions
        for idx, (i, j) in enumerate(zip(tri_idx[0], tri_idx[1])):
            w = weights[idx]
            d = dists[i, j]
            if d > 1e-10:
                diff = pts[i] - pts[j]
                force = diff / d * w * 0.5
                grad[i] += force
                grad[j] -= force
        
        # Boundary contributions
        for i in range(n):
            w = weights[len(pair_dists) + i]
            r = norms[i]
            if r > 1e-10:
                grad[i] -= pts[i] / r * w
        
        # Update
        pts += lr * grad
        
        # Project inside unit circle
        new_norms = np.linalg.norm(pts, axis=1)
        mask = new_norms > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / new_norms[mask][:, np.newaxis]
    
    return best_pts, best_radius

def _hex_lattice_init(n_points):
    """Generate a hexagonal lattice initialization inside the unit circle."""
    # Estimate spacing
    area_per_point = np.pi / n_points
    spacing = np.sqrt(area_per_point / (np.sqrt(3.0) / 2.0))
    
    k = int(1.0 / spacing) + 3
    rows = []
    for j in range(-k, k + 1):
        y = j * spacing * np.sqrt(3.0) / 2.0
        offset = 0.5 * spacing * (j % 2)
        for i in range(-k, k + 1):
            x = i * spacing + offset
            r = np.hypot(x, y)
            if r <= 1.0 - spacing * 0.5:
                rows.append((x, y, r))
    
    rows.sort(key=lambda x: x[2])
    pts = np.array([(x, y) for x, y, r in rows[:n_points]])
    return pts

def construct():
    """Construct centers of N equal circles inside the unit circle."""
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    # Initial hexagonal lattice
    pts = _hex_lattice_init(N)
    
    if len(pts) < N:
        # Fallback: random initialization
        np.random.seed(42)
        pts = np.random.randn(N, 2)
        norms = np.linalg.norm(pts, axis=1)
        pts = pts / norms[:, np.newaxis] * 0.9
    
    # Local refinement
    pts, radius = _local_refine(pts, max_iter=3000, time_limit=time_limit - (time.time() - start_time) - 5.0)
    
    # Final validation and clipping
    norms = np.linalg.norm(pts, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        pts[mask] = pts[mask] / norms[mask][:, np.newaxis]
    
    return pts
# EVOLVE-BLOCK-END
