"""Packing 50 equal circles in the unit square [-0.5, 0.5]^2."""
N = 50
import time
import numpy as np
from scipy.spatial import cKDTree


def _score(pts):
    """Return the achievable common radius for a centre array (N,2)."""
    d_bound = np.min(0.5 - np.abs(pts), axis=1)
    r = float(np.min(d_bound))
    if N > 1:
        tree = cKDTree(pts)
        d, _ = tree.query(pts, k=2)
        dmin = float(np.min(d[:, 1]))
        r = min(r, dmin / 2.0)
    return r


def _valid(pts):
    return (pts.shape == (N, 2)
            and np.all(np.isfinite(pts))
            and np.all(pts >= -0.5 - 1e-9)
            and np.all(pts <= 0.5 + 1e-9))


def _hex_initial():
    """Hexagonal (triangular) lattice, rows staggered, clipped to square."""
    s = 0.152
    dx = s
    dy = s * np.sqrt(3.0) / 2.0
    xs = np.arange(-0.5, 0.5 + 1e-9, dx)
    pts = []
    row = 0
    y = -0.5
    while y <= 0.5 + 1e-9:
        offset = (row % 2) * (dx / 2.0)
        for x in xs:
            xx = x + offset
            if -0.5 <= xx <= 0.5 and -0.5 <= y <= 0.5:
                pts.append([xx, y])
        y += dy
        row += 1
    pts = np.array(pts, dtype=float)
    if len(pts) < N:
        grid = _grid_initial()
        pts = np.vstack([pts, grid[len(pts):]])
    else:
        d2 = np.sum(pts * pts, axis=1)
        order = np.argsort(d2)
        pts = pts[order[:N]]
    return pts.copy()


def _grid_initial():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _force_directed(pts, time_budget, rng=None):
    """
    Force-directed layout: repulsive forces between circles (inverse power law)
    and attractive forces toward the center. Uses gradient descent on a
    smooth surrogate for the minimum distance.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    pos = pts.copy().astype(float)
    t0 = time.time()
    
    # Parameters for force-directed simulation
    repulsion_power = 6  # High power for strong short-range repulsion
    repulsion_strength = 1.0
    attraction_strength = 0.1  # Gentle pull toward center
    learning_rate = 0.001
    min_lr = 1e-6
    decay = 0.999
    lr = learning_rate
    eps = 1e-12
    
    # Precompute tree for fast distance queries
    for it in range(50000):
        if time.time() - t0 > time_budget:
            break
        
        # Compute pairwise distances using cKDTree
        tree = cKDTree(pos)
        dist, idx = tree.query(pos, k=2)
        nearest_dist = dist[:, 1]  # Distance to nearest neighbor
        
        # Repulsive forces: each pair of circles repels with force ~ 1/d^power
        # We approximate the gradient of min distance by considering
        # forces from nearest neighbors
        forces = np.zeros_like(pos)
        
        for i in range(N):
            j = idx[i, 1]
            d = nearest_dist[i]
            if d < eps:
                # Random kick if coincident
                forces[i] += rng.normal(0, 0.01, size=2)
                continue
            
            # Direction from i to j
            diff = pos[j] - pos[i]
            norm = np.linalg.norm(diff)
            if norm < eps:
                continue
            direction = diff / norm
            
            # Repulsive force magnitude: strong when close
            # Force on i is away from j
            force_mag = repulsion_strength / (d ** (repulsion_power - 1))
            forces[i] -= direction * force_mag
        
        # Attractive force toward center (gentle)
        forces -= pos * attraction_strength
        
        # Boundary repulsion: push points away from edges
        for axis in range(2):
            # Points too close to -0.5 edge
            mask_neg = pos[:, axis] < -0.45
            if np.any(mask_neg):
                forces[mask_neg, axis] += (pos[mask_neg, axis] + 0.45) * 10.0
            # Points too close to +0.5 edge  
            mask_pos = pos[:, axis] > 0.45
            if np.any(mask_pos):
                forces[mask_pos, axis] -= (pos[mask_pos, axis] - 0.45) * 10.0
        
        # Update positions with gradient ascent (maximizing min distance)
        pos += lr * forces
        
        # Clamp to square
        pos = np.clip(pos, -0.5, 0.5)
        
        # Decay learning rate
        lr = max(min_lr, lr * decay)
        
        # Periodic random perturbation to escape local minima
        if it % 1000 == 0 and it > 0:
            perturbation_scale = 0.01 * (1.0 - it / 50000.0)
            pos += rng.normal(0, perturbation_scale, size=pos.shape)
            pos = np.clip(pos, -0.5, 0.5)
    
    return pos


def _local_refine(pts, time_budget, rng):
    """Fine-tune with small random perturbations after force-directed layout."""
    best = pts.copy()
    best_s = _score(best)
    step = 0.01
    t0 = time.time()
    
    for it in range(20000):
        if time.time() - t0 > time_budget:
            break
        
        cand = best.copy()
        n_move = rng.integers(1, 4)
        idx = rng.choice(N, size=n_move, replace=False)
        cand[idx] += rng.normal(0.0, step, size=(n_move, 2))
        cand = np.clip(cand, -0.5, 0.5)
        
        s = _score(cand)
        if s > best_s + 1e-15:
            best = cand
            best_s = s
            step = max(0.005, step * 1.05)
        else:
            step *= 0.98
        step = min(step, 0.02)
    
    return best, best_s


def construct():
    start = time.time()
    deadline = start + 145.0  # safety margin before 150s kill
    
    rng = np.random.default_rng(12345)
    
    # Initial candidates
    candidates = [_hex_initial(), _grid_initial()]
    best_pts = candidates[0].copy()
    best_s = _score(best_pts)
    for c in candidates[1:]:
        s = _score(c)
        if s > best_s:
            best_s = s
            best_pts = c.copy()
    
    # Force-directed optimization with multiple restarts
    n_restarts = 5
    time_per_restart = (deadline - start) / n_restarts
    
    for r in range(n_restarts):
        elapsed = time.time() - start
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
        
        # Different initial conditions for each restart
        if r == 0:
            cur = best_pts.copy()
        elif r == 1:
            cur = _hex_initial()
            cur += rng.normal(0.0, 0.015, size=(N, 2))
            cur = np.clip(cur, -0.5, 0.5)
        elif r == 2:
            cur = _grid_initial()
            cur += rng.normal(0.0, 0.015, size=(N, 2))
            cur = np.clip(cur, -0.5, 0.5)
        elif r == 3:
            # Random scatter
            cur = rng.uniform(-0.4, 0.4, size=(N, 2))
        else:
            cur = best_pts.copy()
            cur += rng.normal(0.0, 0.02, size=(N, 2))
            cur = np.clip(cur, -0.5, 0.5)
        
        if not _valid(cur):
            cur = best_pts.copy()
        
        # Phase 1: Force-directed layout (most of the time)
        fd_budget = remaining * 0.7
        cur = _force_directed(cur, fd_budget, rng)
        cur = np.clip(cur, -0.5, 0.5)
        
        # Phase 2: Local refinement
        local_budget = remaining * 0.25
        cur, s = _local_refine(cur, local_budget, rng)
        
        if _valid(cur) and s > best_s:
            best_s = s
            best_pts = cur.copy()
    
    # Final safety
    best_pts = np.clip(best_pts, -0.5, 0.5)
    if not _valid(best_pts):
        best_pts = _grid_initial()
    
    return best_pts
