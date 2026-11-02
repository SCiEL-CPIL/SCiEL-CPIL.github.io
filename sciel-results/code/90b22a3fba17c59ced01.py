import numpy as np
import time
from scipy.spatial import cKDTree

N = 50

def _fast_radius(p):
    """Fast radius computation using KD-tree."""
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(p)):
        return 0.0
    b = 0.5 - np.max(np.abs(p), axis=1)
    min_b = float(np.min(b))
    tree = cKDTree(p)
    pairs, dists = tree.query_pairs(1.0, output_type='ndarray')
    if len(pairs) == 0:
        min_d = np.inf
    else:
        min_d = float(np.min(dists)) / 2.0
    return float(min(min_b, min_d))

def _compute_radius(p):
    """Compute the valid radius for a configuration."""
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(p)):
        return 0.0
    b = 0.5 - np.max(np.abs(p), axis=1)
    min_b = float(np.min(b))
    diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_d = float(np.min(dists)) / 2.0
    return float(min(min_b, min_d))

def _seed_fcc():
    """FCC lattice seed."""
    pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                pts.append([float(i), float(j), float(k)])
                pts.append([i + 0.5, j + 0.5, float(k)])
                pts.append([i + 0.5, float(j), k + 0.5])
                pts.append([float(i), j + 0.5, k + 0.5])
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    d = np.linalg.norm(pts, axis=1)
    idx = np.argsort(d)
    return pts[idx[:N]].copy()

def _seed_grid():
    """5x5x2 grid seed."""
    xs = np.linspace(-0.35, 0.35, 5)
    ys = np.linspace(-0.35, 0.35, 5)
    zs = np.array([-0.15, 0.15])
    grid = np.array([[x, y, z] for x in xs for y in ys for z in zs])
    return grid[:N].copy()

def _seed_shell_core():
    """Shell-core seed: 8 corners, 12 edges, 6 faces, interior."""
    pts = []
    s = 0.35
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                pts.append([sx * s, sy * s, sz * s])
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            pts.append([sx * s, sy * s, 0])
            pts.append([sx * s, 0, sy * s])
            pts.append([0, sx * s, sy * s])
    for sx in [-1, 1]:
        pts.append([sx * s, 0, 0])
        pts.append([0, sx * s, 0])
        pts.append([0, 0, sx * s])
    pts.append([0, 0, 0])
    pts = np.array(pts)
    while len(pts) < N:
        pts = np.vstack([pts, np.random.uniform(-0.3, 0.3, 3)])
    return pts[:N].copy()

def _seed_random():
    """Random seed."""
    return np.random.uniform(-0.4, 0.4, (N, 3))

def _repulsion_anneal(p0, time_budget=30.0):
    """Vectorized repulsion annealing in direct space."""
    start = time.time()
    p = p0.copy()
    best_p = p.copy()
    best_r = _fast_radius(p)
    
    n_iter = 5000
    temp = 0.05
    decay = (0.001 / temp) ** (1.0 / n_iter)
    
    for it in range(n_iter):
        if time.time() - start > time_budget:
            break
        
        # Compute repulsion forces (vectorized)
        diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        np.fill_diagonal(dists, 1.0)
        
        # Soft repulsion: force = 1/d^2 in direction of separation
        force = np.zeros_like(p)
        for i in range(N):
            for j in range(i + 1, N):
                d = dists[i, j]
                if d < 0.1:
                    f = 1.0 / (d * d + 1e-8)
                    force[i] += diffs[i, j] * f
                    force[j] -= diffs[i, j] * f
        
        # Boundary repulsion
        for i in range(N):
            for k in range(3):
                if p[i, k] > 0.45:
                    force[i, k] -= (p[i, k] - 0.45) * 10.0
                elif p[i, k] < -0.45:
                    force[i, k] -= (p[i, k] + 0.45) * 10.0
        
        # Add noise and update
        noise = np.random.normal(0, temp, (N, 3))
        p_new = p + 0.01 * force + noise
        p_new = np.clip(p_new, -0.49, 0.49)
        
        r_new = _fast_radius(p_new)
        if r_new > best_r:
            best_r = r_new
            best_p = p_new.copy()
        
        # Metropolis accept
        if r_new >= best_r or np.random.rand() < np.exp((r_new - best_r) / (temp * 0.5 + 1e-8)):
            p = p_new
        
        temp *= decay
    
    return best_p, best_r

def _lloyd_step(p):
    """One Lloyd iteration: move each point to centroid of its Voronoi cell."""
    tree = cKDTree(p)
    new_p = p.copy()
    for i in range(N):
        # Find nearest neighbors to approximate Voronoi cell
        dists, indices = tree.query(p[i], k=min(8, N))
        # Weighted average of neighbors (closer = more weight)
        weights = 1.0 / (dists[1:] + 1e-6)
        neighbors = p[indices[1:]]
        centroid = np.sum(neighbors * weights[:, np.newaxis], axis=0) / np.sum(weights)
        # Blend with boundary constraint
        new_p[i] = 0.7 * centroid + 0.3 * p[i]
        # Keep in cube
        new_p[i] = np.clip(new_p[i], -0.45, 0.45)
    return new_p

def _lloyd_optimize(p0, time_budget=15.0):
    """Lloyd optimization loop."""
    start = time.time()
    p = p0.copy()
    best_p = p.copy()
    best_r = _fast_radius(p)
    
    for it in range(200):
        if time.time() - start > time_budget:
            break
        p_new = _lloyd_step(p)
        r_new = _fast_radius(p_new)
        if r_new > best_r:
            best_r = r_new
            best_p = p_new.copy()
            p = p_new
        else:
            # Try smaller step
            p_blend = 0.5 * p + 0.5 * p_new
            r_blend = _fast_radius(p_blend)
            if r_blend > best_r:
                best_r = r_blend
                best_p = p_blend.copy()
                p = p_blend
            else:
                break
    
    return best_p, best_r

def _active_set_polish(p0, time_budget=10.0):
    """Polish using only active (binding) constraints."""
    start = time.time()
    p = p0.copy()
    best_p = p.copy()
    best_r = _compute_radius(p)
    
    for outer in range(50):
        if time.time() - start > time_budget:
            break
        
        r = _compute_radius(p)
        # Find active constraints
        b = 0.5 - np.max(np.abs(p), axis=1)
        diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        active_pairs = []
        active_bounds = []
        for i in range(N):
            for j in range(i + 1, N):
                if dists[i, j] < 2.0 * (r + 1e-4):
                    active_pairs.append((i, j))
            if b[i] < r + 1e-4:
                active_bounds.append(i)
        
        if not active_pairs and not active_bounds:
            break
        
        # Compute gradient of radius w.r.t. each point
        grad = np.zeros_like(p)
        for i, j in active_pairs:
            d = p[i] - p[j]
            dn = np.linalg.norm(d)
            if dn > 1e-12:
                grad[i] += d / (2.0 * dn)
                grad[j] -= d / (2.0 * dn)
        
        for i in active_bounds:
            for k in range(3):
                if p[i, k] > 0:
                    grad[i, k] += 1.0
                else:
                    grad[i, k] -= 1.0
        
        # Gradient ascent step
        step_size = 0.001
        p_new = p + step_size * grad
        p_new = np.clip(p_new, -0.49, 0.49)
        
        r_new = _compute_radius(p_new)
        if r_new > best_r:
            best_r = r_new
            best_p = p_new.copy()
            p = p_new
        else:
            step_size *= 0.5
            if step_size < 1e-8:
                break
    
    return best_p, best_r

def construct():
    start_time = time.time()
    time_limit = 140.0  # 10s safety margin
    
    np.random.seed(42)
    
    # Generate diverse seeds
    seeds = [
        _seed_fcc(),
        _seed_grid(),
        _seed_shell_core(),
        _seed_random(),
    ]
    
    best_p = None
    best_r = 0.0
    
    # Phase 1: Repulsion annealing on each seed
    anneal_budget = 25.0
    for s in seeds:
        if time.time() - start_time > time_limit - 30.0:
            break
        remaining = min(anneal_budget, time_limit - (time.time() - start_time) - 30.0)
        if remaining < 5.0:
            break
        p_a, r_a = _repulsion_anneal(s, time_budget=remaining)
        if r_a > best_r:
            best_r = r_a
            best_p = p_a
    
    if best_p is None:
        best_p = seeds[0]
        best_r = _fast_radius(best_p)
    
    # Phase 2: Lloyd optimization
    remaining = time_limit - (time.time() - start_time)
    if remaining > 10.0:
        p_l, r_l = _lloyd_optimize(best_p, time_budget=min(remaining * 0.4, 20.0))
        if r_l > best_r:
            best_r = r_l
            best_p = p_l
    
    # Phase 3: Another anneal pass from best
    remaining = time_limit - (time.time() - start_time)
    if remaining > 15.0:
        p_a2, r_a2 = _repulsion_anneal(best_p, time_budget=min(remaining * 0.3, 15.0))
        if r_a2 > best_r:
            best_r = r_a2
            best_p = p_a2
    
    # Phase 4: Active set polish
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5.0:
        p_pol, r_pol = _active_set_polish(best_p, time_budget=min(remaining * 0.5, 15.0))
        if r_pol > best_r:
            best_r = r_pol
            best_p = p_pol
    
    # Phase 5: Final Lloyd + polish
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5.0:
        p_l2, r_l2 = _lloyd_optimize(best_p, time_budget=min(remaining * 0.3, 10.0))
        if r_l2 > best_r:
            best_r = r_l2
            best_p = p_l2
        remaining = time_limit - (time.time() - start_time)
        if remaining > 3.0:
            p_pol2, r_pol2 = _active_set_polish(best_p, time_budget=min(remaining * 0.5, 8.0))
            if r_pol2 > best_r:
                best_r = r_pol2
                best_p = p_pol2
    
    # Final validation
    final_r = _compute_radius(best_p)
    if final_r > best_r:
        best_r = final_r
    
    # Ensure valid output
    if not np.all(np.isfinite(best_p)) or best_p.shape != (N, 3):
        best_p = _seed_grid()
    
    return best_p
