"""Improved packer: Rigid-body dynamics with Velocity Verlet + local search polish."""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 300
SEED = 12345
DEADLINE = 115.0  # safety margin before the 120 s wall-clock limit


def _score(points):
    """Return the feasible common radius (0.0 if invalid)."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    norm = np.linalg.norm(points, axis=1)
    r_bound = 1.0 - norm
    if np.min(r_bound) < 0.0:
        return 0.0
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    r_pair = 0.5 * dists[:, 1].min()
    r = min(r_bound.min(), r_pair)
    return float(r) if r > 0.0 else 0.0


def _lattice_seed(spacing):
    """Generate a hexagonal lattice and pick N points inside the disk."""
    while True:
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        pts = np.array(rows)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        keep = rad <= 1.0 - spacing / 2.0
        pts = pts[keep]
        if len(pts) >= N:
            order = np.argsort(np.hypot(pts[:, 0], pts[:, 1]), kind="stable")
            return pts[order[:N]].copy()
        spacing *= 0.97


def _ring_seed(n_rings=15):
    """Generate a ring-based initialization."""
    pts = []
    # Center
    pts.append([0.0, 0.0])
    for ring in range(1, n_rings + 1):
        r = ring * 0.06
        n = int(2 * np.pi * r / 0.06)
        if n < 1:
            n = 1
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        for a in angles:
            pts.append([r * np.cos(a), r * np.sin(a)])
    pts = np.array(pts[:N])
    # Project into unit circle
    rad = np.linalg.norm(pts, axis=1)
    mask = rad > 0.99
    if np.any(mask):
        pts[mask] = pts[mask] / rad[mask, None] * 0.99
    return pts


def _dynamics_optimize(points, t_budget, rng, target_r=None):
    """
    Rigid-body dynamics optimization using Velocity Verlet.
    
    Circles repel each other when closer than 2*target_r, and are pushed
    inward when outside the container boundary.
    """
    start = time.time()
    if target_r is None:
        target_r = _score(points)
    
    pos = points.copy().astype(np.float64)
    vel = np.zeros_like(pos)
    
    # Force parameters
    k_repel = 50.0      # repulsion stiffness
    k_boundary = 100.0  # boundary stiffness
    damping = 0.995     # velocity damping for stability
    dt = 0.005          # time step
    max_steps = 5000
    
    # Track best
    best_pos = pos.copy()
    best_r = _score(best_pos)
    
    step = 0
    while step < max_steps and (time.time() - start) < t_budget:
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Pairwise repulsion using KD-tree for efficiency
        tree = cKDTree(pos)
        # Query all pairs within a cutoff
        cutoff = 2.0 * target_r * 1.5
        pairs = tree.query_pairs(cutoff)
        
        for i, j in pairs:
            if i == j:
                continue
            d = pos[i] - pos[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                # Random nudge for coincident points
                d = rng.standard_normal(2) * 1e-6
                dist = np.linalg.norm(d)
            if dist < 2.0 * target_r:
                # Repulsive force: proportional to overlap
                overlap = 2.0 * target_r - dist
                f = k_repel * overlap * (d / dist)
                forces[i] += f
                forces[j] -= f
        
        # Boundary force: push inward if outside
        rad = np.linalg.norm(pos, axis=1)
        for i in range(N):
            if rad[i] > 1.0 - target_r:
                # Push toward center
                f = -k_boundary * (rad[i] - (1.0 - target_r)) * (pos[i] / rad[i])
                forces[i] += f
        
        # Velocity Verlet integration
        # 1. Update velocity half-step
        vel += 0.5 * dt * forces
        vel *= damping  # Apply damping
        
        # 2. Update position
        pos += dt * vel
        
        # 3. Compute new forces at new position
        forces_new = np.zeros_like(pos)
        tree_new = cKDTree(pos)
        pairs_new = tree_new.query_pairs(cutoff)
        for i, j in pairs_new:
            if i == j:
                continue
            d = pos[i] - pos[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                d = rng.standard_normal(2) * 1e-6
                dist = np.linalg.norm(d)
            if dist < 2.0 * target_r:
                overlap = 2.0 * target_r - dist
                f = k_repel * overlap * (d / dist)
                forces_new[i] += f
                forces_new[j] -= f
        
        rad_new = np.linalg.norm(pos, axis=1)
        for i in range(N):
            if rad_new[i] > 1.0 - target_r:
                f = -k_boundary * (rad_new[i] - (1.0 - target_r)) * (pos[i] / rad_new[i])
                forces_new[i] += f
        
        # 4. Update velocity second half-step
        vel += 0.5 * dt * forces_new
        vel *= damping
        
        # Check if we improved
        current_r = _score(pos)
        if current_r > best_r:
            best_r = current_r
            best_pos = pos.copy()
        
        # Occasionally try to increase target_r to push for better packing
        if step % 200 == 0 and step > 0:
            # Try slightly larger target radius
            trial_r = target_r * 1.001
            trial_pos = pos.copy()
            # Quick check if this is feasible
            trial_score = _score(trial_pos)
            if trial_score >= target_r * 0.999:
                target_r = min(trial_r, 0.06)  # Cap at reasonable value
        
        step += 1
    
    return best_pos, best_r


def _local_search(points, t_budget, rng):
    """Fast local search (hill climbing) using KD-tree for distance checks."""
    start = time.time()
    best = points.copy()
    best_r = _score(best)
    cur = points.copy()
    
    cur_min_dist, _ = _min_dist_and_pair(cur)
    cur_r_bound = 1.0 - np.linalg.norm(cur, axis=1)
    cur_r = min(cur_r_bound.min(), 0.5 * cur_min_dist)
    
    if cur_r > best_r:
        best_r = cur_r
        best = cur.copy()

    step = 0.01
    tree = cKDTree(cur)
    
    while time.time() - start < t_budget:
        idx = rng.integers(N)
        ang = rng.random() * 2.0 * np.pi
        mag = step * (0.5 + rng.random())
        dx = mag * np.cos(ang)
        dy = mag * np.sin(ang)
        
        old_pt = cur[idx].copy()
        new_pt = old_pt + np.array([dx, dy])
        
        new_norm = np.hypot(new_pt[0], new_pt[1])
        if new_norm > 1.0:
            continue
            
        dists, indices = tree.query(new_pt, k=2)
        nn_idx = indices[0]
        nn_dist = dists[0]
        
        if nn_idx == idx:
            nn_idx = indices[1]
            nn_dist = dists[1]

        new_r_pair = 0.5 * nn_dist
        new_r_bound = 1.0 - new_norm
        new_local_r = min(new_r_pair, new_r_bound)
        
        # Simple acceptance: accept if improves local constraint
        if new_local_r >= cur_r - 1e-9:
            cur[idx] = new_pt
            tree = cKDTree(cur)  # Rebuild tree (simpler, still fast for N=300)
            cur_min_dist, _ = _min_dist_and_pair(cur)
            cur_r_bound = 1.0 - np.linalg.norm(cur, axis=1)
            cur_r = min(cur_r_bound.min(), 0.5 * cur_min_dist)
            
            if cur_r > best_r:
                best_r = cur_r
                best = cur.copy()
                step *= 0.99
            else:
                cur[idx] = old_pt
                step *= 1.01
                if step > 0.1:
                    step = 0.1
                if step < 0.001:
                    step = 0.001
        else:
            step *= 1.01
            if step > 0.1:
                step = 0.1
            if step < 0.001:
                step = 0.001

    return best, best_r


def _min_dist_and_pair(points):
    """Compute minimum pairwise distance."""
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    return float(dists[:, 1].min()), (0, 0)


def construct():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # Multiple initializations
    init_strategies = [
        _lattice_seed(2.0 / (np.sqrt(N) + 1.0)),
        _ring_seed(),
    ]
    
    # Add a random initialization
    random_pts = rng.standard_normal((N, 2)) * 0.7
    rad = np.linalg.norm(random_pts, axis=1)
    mask = rad > 0.95
    if np.any(mask):
        random_pts[mask] = random_pts[mask] / rad[mask, None] * 0.95
    init_strategies.append(random_pts)
    
    best = init_strategies[0].copy()
    best_r = _score(best)
    
    # First pass: dynamics on each initialization
    for init_pts in init_strategies:
        if time.time() - t0 > DEADLINE - 10:
            break
        t_left = DEADLINE - (time.time() - t0)
        if t_left < 15:
            break
        t_alloc = min(t_left - 5, 20.0)
        
        cand, r = _dynamics_optimize(init_pts, t_alloc, rng)
        if r > best_r:
            best_r = r
            best = cand
    
    # Second pass: local search polish on best
    if time.time() - t0 < DEADLINE - 5:
        t_left = DEADLINE - (time.time() - t0)
        if t_left > 5:
            t_alloc = min(t_left - 2, 15.0)
            cand, r = _local_search(best, t_alloc, rng)
            if r > best_r:
                best_r = r
                best = cand
    
    # Final safety
    final_r = _score(best)
    if final_r <= 0.0:
        best = _lattice_seed(2.0 / (np.sqrt(N) + 1.0))

    return best
