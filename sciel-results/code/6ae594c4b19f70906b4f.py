"""
Combined packer: smooth soft-min optimization (L-BFGS-B) with KD-tree local search polish.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 300
SEED = 12345
T_START = time.time()
DEADLINE = 145.0  # safety margin before the 150s wall-clock limit


def _min_dist(points):
    """Efficiently compute the minimum pairwise distance."""
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    return float(np.min(dists[:, 1]))


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
    # minimum pairwise half-distance
    min_pair_dist = _min_dist(points)
    r_pair = 0.5 * min_pair_dist
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
            # prefer inner points (smaller radius) to maximise margin
            order = np.argsort(np.hypot(pts[:, 0], pts[:, 1]), kind="stable")
            return pts[order[:N]].copy()
        spacing *= 0.97


def _local_search(points, t_budget, rng):
    """Fast local search (hill climbing) using KD-tree for distance checks."""
    start = time.time()
    best = points.copy()
    best_r = _score(best)
    cur = points.copy()
    cur_r = best_r
    
    # Initial step size based on typical spacing
    step = 0.01
    
    while time.time() - start < t_budget:
        # Propose move
        idx = rng.integers(N)
        ang = rng.random() * 2.0 * np.pi
        mag = step * (0.5 + rng.random())
        dx = mag * np.cos(ang)
        dy = mag * np.sin(ang)
        
        old_pt = cur[idx].copy()
        new_pt = old_pt + np.array([dx, dy])
        
        # Check boundary
        if np.hypot(new_pt[0], new_pt[1]) > 1.0:
            continue
            
        cur[idx] = new_pt
        new_min_dist = _min_dist(cur)
        new_r = 0.5 * new_min_dist
        
        # Also check boundary constraint for the new point
        new_r_bound = 1.0 - np.hypot(new_pt[0], new_pt[1])
        new_r = min(new_r, new_r_bound)
        
        if new_r > cur_r:
            cur_r = new_r
            if new_r > best_r:
                best_r = new_r
                best = cur.copy()
            step *= 0.99
        else:
            cur[idx] = old_pt
            step *= 1.01
            if step > 0.1: step = 0.1
            if step < 0.001: step = 0.001

    return best, best_r


def _soft_min_optimize(points, t_budget, rng):
    """
    Smooth soft-min optimization using L-BFGS-B.
    Objective: minimize -soft_min(constraint_distances)
    """
    start = time.time()
    best = points.copy()
    best_r = _score(best)
    
    # Precompute indices for pairs
    iu = np.triu_indices(N, k=1)
    i_idx, j_idx = iu
    
    def objective_and_grad(x, beta):
        p = x.reshape(-1, 2)
        
        # Calculate distances
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        
        # Pair distances (upper triangle)
        pair_dists = dists[i_idx, j_idx] / 2.0
        
        # Boundary distances
        norms = np.linalg.norm(p, axis=1)
        bound_dists = 1.0 - norms
        
        # Soft-min calculation
        z_pairs = -beta * pair_dists
        z_bounds = -beta * bound_dists
        
        max_z = max(np.max(z_pairs), np.max(z_bounds))
        
        exp_pairs = np.exp(z_pairs - max_z)
        exp_bounds = np.exp(z_bounds - max_z)
        
        sum_exp = np.sum(exp_pairs) + np.sum(exp_bounds)
        
        # Objective value: -soft_min = (max_z + log(sum_exp)) / beta
        val = (max_z + np.log(sum_exp)) / beta
        
        # Gradient
        w_pairs = exp_pairs / sum_exp
        w_bounds = exp_bounds / sum_exp
        
        grad = np.zeros_like(p)
        
        # Pair contributions
        safe_pair_dists = np.where(pair_dists > 1e-12, pair_dists, 1.0)
        coeff = w_pairs * 0.25 / safe_pair_dists
        vecs = p[i_idx] - p[j_idx]
        
        np.add.at(grad, i_idx, coeff[:, np.newaxis] * vecs)
        np.add.at(grad, j_idx, -coeff[:, np.newaxis] * vecs)
        
        # Boundary contributions
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        bound_grad_vecs = -p / safe_norms[:, np.newaxis]
        
        grad += w_bounds[:, np.newaxis] * bound_grad_vecs
        
        # Return negative gradient (since we want to minimize -soft_min)
        return val, -grad

    # Start with a moderate beta and increase it
    beta = 50.0
    cur = points.copy().flatten()
    
    while time.time() - start < t_budget:
        t_left = t_budget - (time.time() - start)
        if t_left < 2.0:
            break
        
        # Increase beta to sharpen the soft-min
        beta *= 1.5
        if beta > 1000.0:
            beta = 1000.0
        
        # Run L-BFGS-B
        try:
            res = minimize(
                lambda x: objective_and_grad(x, beta)[0],
                cur,
                jac=lambda x: objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 500, 'gtol': 1e-6}
            )
            cur = res.x
        except Exception:
            break
        
        # Check score
        pts = cur.reshape(-1, 2)
        r = _score(pts)
        if r > best_r:
            best_r = r
            best = pts.copy()
        
        # If we're close to the deadline, break
        if time.time() - start > t_budget - 1.0:
            break
    
    return best, best_r


def construct():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # initial hex-lattice seed
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    pts = _lattice_seed(spacing)
    best = pts.copy()
    best_r = _score(best)

    # Phase 1: Soft-min optimization
    t_left = DEADLINE - (time.time() - t0)
    if t_left > 30.0:
        t_alloc = min(t_left - 10.0, 60.0)
        cand, r = _soft_min_optimize(best, t_alloc, rng)
        if r > best_r:
            best_r = r
            best = cand

    # Phase 2: Local search polish
    t_left = DEADLINE - (time.time() - t0)
    if t_left > 5.0:
        t_alloc = min(t_left - 2.0, 30.0)
        cand, r = _local_search(best, t_alloc, rng)
        if r > best_r:
            best_r = r
            best = cand

    # Phase 3: Multi-start with soft-min if time remains
    t_left = DEADLINE - (time.time() - t0)
    while t_left > 15.0:
        # Perturb best solution
        jitter = 0.01 * rng.standard_normal((N, 2))
        cand = best + jitter
        
        # Project back into unit circle if needed
        rad = np.linalg.norm(cand, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            cand[mask] = cand[mask] / rad[mask, None] * (1.0 - 1e-6)

        t_left = DEADLINE - (time.time() - t0)
        if t_left < 10.0:
            break
        t_alloc = min(t_left - 5.0, 20.0)
        
        cand, r = _soft_min_optimize(cand, t_alloc, rng)
        if r > best_r:
            best_r = r
            best = cand
        
        # Quick local search polish
        t_left = DEADLINE - (time.time() - t0)
        if t_left > 5.0:
            t_alloc = min(t_left - 2.0, 10.0)
            cand, r = _local_search(cand, t_alloc, rng)
            if r > best_r:
                best_r = r
                best = cand

    # Final safety
    final_r = _score(best)
    if final_r <= 0.0:
        best = _lattice_seed(2.0 / (np.sqrt(N) + 1.0))

    return best
