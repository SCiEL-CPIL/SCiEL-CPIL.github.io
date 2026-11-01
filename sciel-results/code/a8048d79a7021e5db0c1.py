"""Packing 200 equal circles in a unit square: Force-Directed + L-BFGS-B + Active Set Polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

N = 200
DEADLINE = 145.0  # seconds; leave margin before 150s kill

def _score(pts):
    """Return the common radius r for a valid configuration, or 0.0 if invalid."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Distance to boundary
    ax = np.abs(pts[:, 0])
    ay = np.abs(pts[:, 1])
    boundary = np.minimum(0.5 - ax, 0.5 - ay)
    min_boundary = np.min(boundary)
    if min_boundary < -1e-10:
        return 0.0
    # Pairwise distances
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    r = min(min_boundary, min_dist / 2.0)
    return r if r > 0 else 0.0

def _hex_init(jitter_scale=0.0, seed=0):
    """Generate a hexagonal lattice initialization with multiple aspect ratios."""
    best_pts = None
    best_r = 0
    rng = np.random.default_rng(seed)
    
    # Try a wider range of spacings and aspect ratios
    s_est = 1.0 / np.sqrt(N) * 1.2
    for s in np.linspace(s_est * 0.85, s_est * 1.15, 25):
        for aspect in [1.0, 0.95, 1.05, 0.9, 1.1]:
            pts_list = []
            row = 0
            while True:
                y = -0.5 + s/2 + row * s * np.sqrt(3) / 2 * aspect
                if y > 0.5 - s/2:
                    break
                offset = (row % 2) * s / 2
                x = -0.5 + s/2 + offset
                while x <= 0.5 - s/2:
                    pts_list.append((x, y))
                    x += s
                row += 1
            if len(pts_list) < N:
                continue
            pts = np.array(pts_list[:N])
            if jitter_scale > 0:
                pts += rng.normal(0, jitter_scale, size=pts.shape)
                pts = np.clip(pts, -0.49, 0.49)
            r = _score(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
    
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        pts = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = pts[:N].copy()
    return best_pts

def _random_init(seed=0, n_clusters=10):
    """Generate a random clustered initialization for diversity."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.45, 0.45, size=(N, 2))
    # Add some structure by clustering
    cluster_centers = rng.uniform(-0.4, 0.4, size=(n_clusters, 2))
    for i in range(N):
        c = i % n_clusters
        pts[i] = pts[i] * 0.5 + cluster_centers[c] * 0.5
    pts = np.clip(pts, -0.49, 0.49)
    return pts

def _force_directed(pts, time_budget):
    """
    Force-directed relaxation: repulsion between points, attraction to center/bounds.
    Uses KDTree for efficient neighbor lookup to speed up iterations.
    """
    t0 = time.time()
    pts = pts.copy()
    current_r = _score(pts)
    
    # Parameters
    learning_rate = 0.008
    min_lr = 1e-7
    repulsion_power = 1.5
    boundary_repulsion = 2.0
    neighbor_k = 15  # only consider nearest neighbors for speed
    
    while time.time() - t0 < time_budget:
        forces = np.zeros_like(pts)
        
        # Build KDTree for efficient neighbor queries
        tree = cKDTree(pts)
        dists, indices = tree.query(pts, k=neighbor_k + 1)
        
        # Pairwise repulsion from nearest neighbors only
        for i in range(N):
            for j_idx in range(1, neighbor_k + 1):
                j = int(indices[i, j_idx])
                d = dists[i, j_idx]
                if d < 1e-10:
                    continue
                # Force magnitude: stronger for closer points
                weight = 1.0 / (d ** repulsion_power)
                diff = pts[i] - pts[j]
                forces[i] += diff / d * weight
        
        # Boundary repulsion - push points away from edges
        d_x = 0.5 - np.abs(pts[:, 0])
        d_y = 0.5 - np.abs(pts[:, 1])
        boundary_dists = np.minimum(d_x, d_y)
        
        r_est = max(current_r, 0.001)
        close_to_boundary = boundary_dists < r_est * 1.5
        if np.any(close_to_boundary):
            # Push inward
            push = -pts * boundary_repulsion
            forces[close_to_boundary] += push[close_to_boundary]
        
        # Normalize forces to prevent explosion
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        max_norm = 0.5
        too_large = force_norms[:, 0] > max_norm
        if np.any(too_large):
            forces[too_large] *= (max_norm / force_norms[too_large, 0])[:, np.newaxis]
        
        # Update positions
        pts += learning_rate * forces
        pts = np.clip(pts, -0.499, 0.499)
        
        new_r = _score(pts)
        if new_r > current_r:
            current_r = new_r
            learning_rate *= 0.995
        else:
            learning_rate *= 0.97
        
        if learning_rate < min_lr:
            break
            
    return pts

def _soft_min_neg(x, alpha):
    """
    Objective function for L-BFGS-B using numerically stable log-sum-exp.
    x: flat array of size 2N.
    alpha: sharpness of the soft-min.
    Returns: -soft_min(distances)
    """
    pts = x.reshape(N, 2)
    
    # Distances to boundary
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    boundary_dists = np.minimum(d_x, d_y).flatten()
    
    # Pairwise distances
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    pairwise_dists = dist[~np.isinf(dist)]
    
    # Combine all distances
    all_dists = np.concatenate([boundary_dists, pairwise_dists])
    
    # Numerically stable log-sum-exp for soft min
    # soft_min = -1/alpha * log(sum(exp(-alpha * d)))
    # Use max trick for numerical stability
    min_d = np.min(all_dists)
    exp_terms = np.exp(-alpha * (all_dists - min_d))
    log_sum = np.log(np.sum(exp_terms)) + (-alpha * min_d)
    soft_min_val = -log_sum / alpha
    
    # We want to maximize min distance, so minimize -min distance.
    return -soft_min_val

def _optimize_lbfgs(pts, time_budget):
    """
    Run L-BFGS-B optimization with multi-temperature annealing schedule.
    Uses tighter tolerances in later phases for higher precision.
    """
    t0 = time.time()
    x0 = pts.flatten()
    
    # Multi-temperature schedule: coarse to fine, with more iterations at high alpha
    phases = [
        {"alpha": 8.0, "max_iter": 200, "gtol": 1e-3},
        {"alpha": 25.0, "max_iter": 250, "gtol": 1e-5},
        {"alpha": 80.0, "max_iter": 300, "gtol": 1e-6},
        {"alpha": 250.0, "max_iter": 300, "gtol": 1e-7},
        {"alpha": 800.0, "max_iter": 250, "gtol": 1e-8},
        {"alpha": 2000.0, "max_iter": 200, "gtol": 1e-9},
    ]
    
    for phase in phases:
        if time.time() - t0 > time_budget:
            break
            
        alpha = phase["alpha"]
        
        def obj(x):
            return _soft_min_neg(x, alpha)
        
        try:
            res = minimize(obj, x0, method='L-BFGS-B', 
                           bounds=[(-0.4999, 0.4999)] * (2 * N),
                           options={'maxiter': phase["max_iter"], 'gtol': phase["gtol"],
                                    'ftol': 1e-12, 'maxls': 50})
            x_opt = res.x
            x0 = x_opt
        except Exception:
            pass
            
    return x0.reshape(N, 2)

def _active_set_polish(pts, time_budget):
    """
    High-precision polish: only move points involved in the minimum distance constraints.
    Uses directed perturbation and exact evaluation for final convergence.
    """
    t0 = time.time()
    best_pts = pts.copy()
    best_r = _score(best_pts)
    
    # Identify active constraints (pairs and boundaries that achieve the minimum)
    def _get_active_constraints(pts, r):
        """Get indices of active constraints."""
        active_pairs = []
        active_boundaries = []
        
        dist = cdist(pts, pts)
        np.fill_diagonal(dist, np.inf)
        
        # Find all pairs within 2*r + tolerance
        tol = 1e-8
        for i in range(N):
            for j in range(i + 1, N):
                if abs(dist[i, j] - 2 * r) < tol:
                    active_pairs.append((i, j))
        
        # Find boundary constraints
        d_x = 0.5 - np.abs(pts[:, 0])
        d_y = 0.5 - np.abs(pts[:, 1])
        boundary_dists = np.minimum(d_x, d_y)
        for i in range(N):
            if abs(boundary_dists[i] - r) < tol:
                active_boundaries.append(i)
        
        return active_pairs, active_boundaries
    
    # Iterative active set refinement
    for iteration in range(50):
        if time.time() - t0 > time_budget:
            break
        
        # Get active constraints
        active_pairs, active_boundaries = _get_active_constraints(best_pts, best_r)
        
        if len(active_pairs) == 0 and len(active_boundaries) == 0:
            break
        
        # Try small perturbations of active points
        active_indices = set()
        for i, j in active_pairs:
            active_indices.add(i)
            active_indices.add(j)
        for i in active_boundaries:
            active_indices.add(i)
        
        active_indices = list(active_indices)
        
        if len(active_indices) == 0:
            break
        
        # Try moving each active point in a few directions
        improved = False
        for idx in active_indices:
            for dx in [-0.001, 0.001, -0.002, 0.002]:
                for dy in [-0.001, 0.001, -0.002, 0.002]:
                    trial = best_pts.copy()
                    trial[idx, 0] += dx
                    trial[idx, 1] += dy
                    trial = np.clip(trial, -0.4999, 0.4999)
                    
                    trial_r = _score(trial)
                    if trial_r > best_r + 1e-12:
                        best_pts = trial
                        best_r = trial_r
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try random perturbation of all active points
            rng = np.random.default_rng(iteration)
            for _ in range(100):
                if time.time() - t0 > time_budget:
                    break
                trial = best_pts.copy()
                for idx in active_indices:
                    trial[idx] += rng.normal(0, 0.001, size=2)
                trial = np.clip(trial, -0.4999, 0.4999)
                
                trial_r = _score(trial)
                if trial_r > best_r + 1e-12:
                    best_pts = trial
                    best_r = trial_r
                    improved = True
                    break
            if not improved:
                break
                
    return best_pts

def construct():
    t_start = time.time()
    
    initializers = [
        ("hex", lambda: _hex_init()),
        ("hex_jitter", lambda: _hex_init(jitter_scale=0.005, seed=0)),
        ("grid", lambda: _hex_init(jitter_scale=0.0, seed=42)),
        ("rand0", lambda: _random_init(0)),
        ("rand1", lambda: _random_init(42)),
    ]
    
    best_pts = None
    best_r = 0.0
    
    for name, init_fn in initializers:
        if time.time() > DEADLINE - 15:
            break
            
        pts = init_fn()
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
            
        # Optimize
        remaining = DEADLINE - time.time()
        if remaining < 8:
            break
            
        # Allocate time: give more to hex and grid
        if name == "hex":
            time_budget = min(remaining * 0.30, 40.0)
        elif name == "grid":
            time_budget = min(remaining * 0.20, 25.0)
        else:
            time_budget = min(remaining * 0.12, 15.0)
        
        # Phase 1: Force-directed relaxation
        fd_budget = time_budget * 0.3
        fd_pts = _force_directed(pts, fd_budget)
        
        # Phase 2: L-BFGS-B optimization
        lbfgs_budget = time_budget * 0.5
        opt_pts = _optimize_lbfgs(fd_pts, lbfgs_budget)
        opt_pts = np.clip(opt_pts, -0.4999, 0.4999)
        
        opt_r = _score(opt_pts)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()
            
    # Bundle method phase on the best solution
    remaining = DEADLINE - time.time()
    if remaining > 20 and best_pts is not None:
        # Active set polish
        polish_budget = min(remaining * 0.4, 35.0)
        polished_pts = _active_set_polish(best_pts, polish_budget)
        polished_pts = np.clip(polished_pts, -0.4999, 0.4999)
        polished_r = _score(polished_pts)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts.copy()
    
    # Final polish phases
    remaining = DEADLINE - time.time()
    if remaining > 10 and best_pts is not None:
        # Phase 1: More L-BFGS-B optimization
        time_budget = min(remaining * 0.4, 25.0)
        opt_pts = _optimize_lbfgs(best_pts, time_budget)
        opt_pts = np.clip(opt_pts, -0.4999, 0.4999)
        opt_r = _score(opt_pts)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()
        
        # Phase 2: Active set polish again with fresh start
        remaining = DEADLINE - time.time()
        if remaining > 10:
            polish_budget = min(remaining * 0.3, 20.0)
            polished_pts = _active_set_polish(best_pts, polish_budget)
            polished_pts = np.clip(polished_pts, -0.4999, 0.4999)
            polished_r = _score(polished_pts)
            if polished_r > best_r:
                best_r = polished_r
                best_pts = polished_pts.copy()
        
        # Phase 3: Force-directed final adjustment
        remaining = DEADLINE - time.time()
        if remaining > 5:
            time_budget = min(remaining * 0.7, 15.0)
            fd_pts = _force_directed(best_pts, time_budget)
            fd_pts = np.clip(fd_pts, -0.4999, 0.4999)
            fd_r = _score(fd_pts)
            if fd_r > best_r:
                best_r = fd_r
                best_pts = fd_pts.copy()
    
    if best_pts is None:
        best_pts = _hex_init()
        
    return best_pts
