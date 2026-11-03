"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist, squareform


def _compute_radius(pts):
    """Compute the maximum valid radius for given points in [-1/2, 1/2]^3."""
    if len(pts) < 2:
        return 0.5
    # Min distance between any pair of centers / 2
    d = pdist(pts)
    min_pair_dist = d.min() / 2.0
    # Min distance from any center to any wall
    # For cube [-1/2, 1/2], distance to wall = 1/2 - max(|x|, |y|, |z|)
    max_coord = np.abs(pts).max(axis=1)
    wall_dist = (0.5 - max_coord).min()
    return min(min_pair_dist, wall_dist)


def _generate_bcc_points(n):
    """Generate points from BCC lattice, clipped to cube, take first n."""
    # BCC lattice: simple cubic + body center
    # In unit cube [-1/2, 1/2]^3, BCC with spacing a has points at
    # (i*a, j*a, k*a) and ((i+0.5)*a, (j+0.5)*a, (k+0.5)*a)
    # We want to fit n points. Let's try different spacings.
    
    # For n=100, a 5x5x4 grid gives 100 points in simple cubic
    # But BCC is denser. Let's use a 4x4x4 BCC = 4*4*4*2 = 128 points, take 100
    
    # Try to find good lattice parameters
    # BCC in cube: points at integer coords and half-integer coords scaled
    
    # Let's use a simpler approach: generate BCC points in a larger region,
    # then select the n that fit best in the cube
    
    # For a BCC lattice with conventional cell side a, points per cell = 2
    # In cube of side 1, we can fit roughly (1/a)^3 cells, so ~2*(1/a)^3 points
    # For 100 points: 2*(1/a)^3 ≈ 100 → (1/a)^3 ≈ 50 → 1/a ≈ 3.68 → a ≈ 0.272
    
    # Let's generate BCC points with various spacings and pick the best set
    
    best_pts = None
    best_r = 0
    
    # Try a range of lattice spacings
    for a in np.linspace(0.25, 0.35, 20):
        # Generate BCC points in region [-1/2, 1/2]
        n_cells = int(1.0 / a)
        # Points from simple cubic sublattice
        sc_pts = []
        for i in range(n_cells + 1):
            for j in range(n_cells + 1):
                for k in range(n_cells + 1):
                    x = -0.5 + i * a
                    y = -0.5 + j * a
                    z = -0.5 + k * a
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                        sc_pts.append([x, y, z])
        
        # Points from body center sublattice
        bc_pts = []
        for i in range(n_cells + 1):
            for j in range(n_cells + 1):
                for k in range(n_cells + 1):
                    x = -0.5 + (i + 0.5) * a
                    y = -0.5 + (j + 0.5) * a
                    z = -0.5 + (k + 0.5) * a
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                        bc_pts.append([x, y, z])
        
        all_pts = sc_pts + bc_pts
        if len(all_pts) < n:
            continue
        
        # Select n points that maximize minimum distance
        # Simple approach: take first n, or try to select optimally
        # For now, take a random subset and keep the best
        np.random.seed(42)
        all_pts_arr = np.array(all_pts)
        
        # Try multiple random selections
        best_subset_r = 0
        for _ in range(50):
            if len(all_pts_arr) <= n:
                subset = all_pts_arr
            else:
                idx = np.random.choice(len(all_pts_arr), n, replace=False)
                subset = all_pts_arr[idx]
            
            r = _compute_radius(subset)
            if r > best_subset_r:
                best_subset_r = r
                best_subset = subset
        
        if best_subset_r > best_r:
            best_r = best_subset_r
            best_pts = best_subset
    
    if best_pts is not None:
        return best_pts
    
    # Fallback: simple grid
    m = int(np.ceil(n ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]


def _repulsion_energy(x, n, k=10):
    """
    Compute a smooth repulsion energy that is minimized when points are well-separated.
    Uses log-sum-exp style soft-min of pairwise distances and wall distances.
    x: flat array of length 3*n
    Returns (energy, gradient)
    """
    pts = x.reshape(n, 3)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-15)
    np.fill_diagonal(dist, np.inf)
    
    # Wall distances: distance from each point to nearest wall
    wall_dists = 0.5 - np.abs(pts).max(axis=1)
    
    # Soft-min of all constraints (pair distances / 2 and wall distances)
    # We want to maximize the minimum, so minimize -soft_min
    # Use log-sum-exp for soft min: soft_min(a) = -log(sum(exp(-a_i/k))) * k ... 
    # Actually: soft_min = -logsumexp(-a/k) * k is not quite right.
    # Let's use: f = -log(sum(exp(-d_i/k))) which approximates min(d_i) for small k
    
    # Combine all distances: pair_dists/2 and wall_dists
    # For numerical stability, work with -d
    pair_half = dist / 2.0
    pair_half = pair_half[np.triu_indices(n, k=1)]
    
    all_dists = np.concatenate([pair_half, wall_dists])
    
    # Soft min via log-sum-exp
    neg_d = -all_dists
    neg_d_shifted = neg_d - neg_d.max()
    log_sum_exp = neg_d.max() + np.log(np.sum(np.exp(neg_d_shifted)))
    soft_min = -log_sum_exp  # This approximates min(all_dists)
    
    # Gradient of soft_min w.r.t. points
    # d(soft_min)/d(all_dists_i) = -exp(neg_d_i) / sum(exp(neg_d_j))
    weights = np.exp(neg_d_shifted)
    weights = weights / weights.sum()
    
    # Gradient w.r.t. pair distances
    grad_pair = -weights[:len(pair_half)]
    grad_wall = -weights[len(pair_half):]
    
    # Build gradient matrix for pair distances
    grad_dist_mat = np.zeros((n, n))
    idx_upper = np.triu_indices(n, k=1)
    grad_dist_mat[idx_upper] = grad_pair
    grad_dist_mat[(idx_upper[1], idx_upper[0])] = grad_pair
    
    # Gradient of dist[i,j] w.r.t. pts[i] = (pts[i]-pts[j]) / dist[i,j]
    # Gradient of dist[i,j] w.r.t. pts[j] = (pts[j]-pts[i]) / dist[i,j]
    grad_pts = np.zeros((n, 3))
    
    # For each pair, add contribution
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = dist[i, j]
            if d > 1e-10:
                grad_pts[i] += grad_dist_mat[i, j] * (pts[i] - pts[j]) / d
                grad_pts[j] += grad_dist_mat[i, j] * (pts[j] - pts[i]) / d
    
    # Wall gradient: d(wall_dist_i)/d(pts[i,k]) = -sign(pts[i,k]) if |pts[i,k]| is max
    # For soft handling, use: d(0.5 - max(|x|,|y|,|z|))/d(x) 
    # This is non-smooth. Use a smooth approximation.
    # For each point, the wall distance is 0.5 - max(|x|,|y|,|z|)
    # Gradient: for coordinate k, if |pts[i,k]| is the max, grad = -sign(pts[i,k])
    # Smooth version: use softmax over |coords|
    for i in range(n):
        abs_coords = np.abs(pts[i])
        # Softmax weights for which coordinate dominates
        abs_shifted = abs_coords - abs_coords.max()
        w = np.exp(abs_shifted)
        w = w / w.sum()
        # Gradient of max(|x|,|y|,|z|) w.r.t. x_k ≈ sum_j w_j * sign(x_j) * delta(j,k)
        # = w_k * sign(x_k)
        # So gradient of wall_dist = -w_k * sign(x_k)
        for k in range(3):
            if abs(pts[i, k]) > 1e-10:
                grad_pts[i, k] += grad_wall[i] * (-w[k] * np.sign(pts[i, k]))
            else:
                grad_pts[i, k] += grad_wall[i] * 0.0
    
    # Energy is -soft_min (we want to maximize soft_min, so minimize -soft_min)
    energy = -soft_min
    gradient = -grad_pts.flatten()
    
    return energy, gradient


def _local_optimize_lbfgs(pts, deadline=None, max_iter=200):
    """Local optimization using L-BFGS-B on repulsion energy."""
    from scipy.optimize import minimize
    
    n = len(pts)
    x0 = pts.flatten()
    
    best_r = _compute_radius(pts)
    best_pts = pts.copy()
    
    if deadline is not None and time.time() > deadline:
        return best_pts
    
    def obj(x):
        if deadline is not None and time.time() > deadline - 1:
            raise Exception("timeout")
        e, g = _repulsion_energy(x, n)
        return e, g
    
    try:
        result = minimize(
            obj, x0, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (3 * n),
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.x is not None:
            new_pts = np.clip(result.x.reshape(n, 3), -0.5, 0.5)
            new_r = _compute_radius(new_pts)
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts
    except Exception:
        pass
    
    return best_pts


def _polish(pts, deadline=None, max_iter=100):
    """High-precision final polish using very tight L-BFGS-B."""
    from scipy.optimize import minimize
    
    n = len(pts)
    x0 = pts.flatten()
    
    best_r = _compute_radius(pts)
    best_pts = pts.copy()
    
    if deadline is not None and time.time() > deadline:
        return best_pts
    
    def obj(x):
        if deadline is not None and time.time() > deadline - 0.5:
            raise Exception("timeout")
        e, g = _repulsion_energy(x, n, k=5)
        return e, g
    
    try:
        result = minimize(
            obj, x0, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (3 * n),
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
        )
        if result.x is not None:
            new_pts = np.clip(result.x.reshape(n, 3), -0.5, 0.5)
            new_r = _compute_radius(new_pts)
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts
    except Exception:
        pass
    
    return best_pts


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin (150s limit, leave 20s buffer)
    
    np.random.seed(42)
    
    # Generate initial BCC-based points
    pts = _generate_bcc_points(N)
    
    # L-BFGS-B local optimization from initial point
    pts = _local_optimize_lbfgs(pts, deadline=deadline, max_iter=100)
    
    # Try multiple random restarts if time permits
    best_r = _compute_radius(pts)
    best_pts = pts.copy()
    
    n_restarts = 0
    while time.time() < deadline - 30:
        n_restarts += 1
        # Generate perturbed initial points
        pts_try = _generate_bcc_points(N).copy()
        # Add random perturbation
        pts_try += np.random.randn(N, 3) * 0.02
        pts_try = np.clip(pts_try, -0.5, 0.5)
        
        # Optimize
        pts_try = _local_optimize_lbfgs(pts_try, deadline=deadline, max_iter=80)
        r_try = _compute_radius(pts_try)
        
        if r_try > best_r:
            best_r = r_try
            best_pts = pts_try.copy()
        
        # Check time for more restarts
        if n_restarts >= 5 and time.time() > deadline - 40:
            break
    
    # Final high-precision polish
    if time.time() < deadline - 10:
        best_pts = _polish(best_pts, deadline=deadline, max_iter=100)
    
    # Ensure validity
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts
# EVOLVE-BLOCK-END
