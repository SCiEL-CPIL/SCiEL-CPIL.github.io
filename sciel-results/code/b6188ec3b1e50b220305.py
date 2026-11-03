"""Packing 50 equal circles in a unit circle using simulated annealing with hexagonal init."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # safety margin before 150s kill


def _pairwise_indices(n):
    """Return array of (i, j) pairs for i < j."""
    iu = np.triu_indices(n, k=1)
    return np.column_stack((iu[0], iu[1]))


def _score_vec(points):
    """Vectorized score computation."""
    points = points.reshape(-1, 2)
    n = len(points)
    radii = 1.0 - np.linalg.norm(points, axis=1)
    r = np.min(radii)
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Only consider i < j
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu] / 2.0
    r = min(r, np.min(pair_dists))
    return r


def _hexagonal_init(n):
    """Generate a hexagonal lattice initial configuration."""
    best_points = None
    best_r = 0.0
    for spacing in np.linspace(0.05, 0.25, 50):
        s = spacing
        points = []
        max_coord = int(1.0 / s) + 2
        for j in range(-max_coord, max_coord + 1):
            for i in range(-max_coord, max_coord + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3) / 2
                if np.hypot(x, y) <= 1.0 - s / 2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= n:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:n]
            cand = points[idx]
            r = _score_vec(cand)
            if r > best_r:
                best_r = r
                best_points = cand.copy()
    if best_points is None:
        best_points = np.random.randn(n, 2) * 0.5
    return best_points


def _anneal(points, t_start, deadline, seed=None):
    """Simulated annealing to escape local minima."""
    if seed is not None:
        np.random.seed(seed)
    
    n = len(points)
    pts = points.copy()
    best_pts = pts.copy()
    best_score = _score_vec(best_pts)
    
    # Energy function: penalize overlaps and boundary violations
    # We want to maximize the minimum of pairwise half-distances and boundary distances
    # For SA, we minimize a cost that is high when constraints are violated
    
    def cost(p):
        p = p.reshape(-1, 2)
        n = len(p)
        # Boundary violation
        boundary = 1.0 - np.linalg.norm(p, axis=1)
        # Pairwise distances
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(n, 1)
        pair_dists = dists[iu]
        # We want all pair_dists >= 2*r and all boundary >= r
        # For SA, we minimize the sum of squared violations assuming a target r
        # But we don't know r yet. Instead, minimize overlap energy:
        # For each pair, if dist < 2*r_target, add penalty
        # For boundary, if dist_to_wall < r_target, add penalty
        # Use a dynamic r_target based on current best
        r_target = best_score
        overlap_energy = 0.0
        # Pairwise overlaps
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(p[i] - p[j])
                if d < 2 * r_target:
                    overlap_energy += (2 * r_target - d) ** 2
        # Boundary violations
        for i in range(n):
            d = 1.0 - np.linalg.norm(p[i])
            if d < r_target:
                overlap_energy += (r_target - d) ** 2
        return overlap_energy
    
    # SA parameters
    T = 0.05  # initial temperature
    T_min = 0.0001
    cooling = 0.9995
    step_size = 0.02
    
    current_cost = cost(pts)
    
    while time.time() < deadline:
        if T < T_min:
            break
        
        # Propose a move: random perturbation of one or two points
        n_moves = np.random.randint(1, 4)
        new_pts = pts.copy()
        for _ in range(n_moves):
            idx = np.random.randint(n)
            delta = np.random.randn(2) * step_size * (1 + T)
            new_pts[idx] += delta
        
        new_cost = cost(new_pts)
        delta_cost = new_cost - current_cost
        
        # Accept with probability exp(-delta_cost / T) if delta_cost > 0
        if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / T):
            pts = new_pts
            current_cost = new_cost
            # Update best
            s = _score_vec(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        
        T *= cooling
    
    return best_pts, best_score


def _local_optimize(c, time_limit=10.0):
    """Local optimization using L-BFGS-B with smooth-min objective."""
    pair_idx = _pairwise_indices(N)
    
    def neg_soft_min(x):
        c = x.reshape(N, 2)
        ci = c[pair_idx[:, 0]]
        cj = c[pair_idx[:, 1]]
        d = np.sqrt(np.sum((ci - cj) ** 2, axis=1)) / 2
        b = 1 - np.hypot(c[:, 0], c[:, 1])
        all_vals = np.concatenate([d, b])
        beta = 200.0
        return np.log(np.sum(np.exp(-beta * all_vals))) / beta
    
    t_start = time.time()
    try:
        res = minimize(
            neg_soft_min,
            c.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-14},
            callback=lambda x: None if time.time() - t_start < time_limit else (_ for _ in ()).throw(Exception("timeout"))
        )
        if res.success:
            return res.x.reshape(N, 2)
    except Exception:
        pass
    
    return c


def _compute_r(c):
    """Compute the max common radius for given centers."""
    return _score_vec(c)


def construct():
    t_start = time.time()
    
    # Start with hexagonal initialization
    points = _hexagonal_init(N)
    best_points = points.copy()
    best_score = _score_vec(best_points)
    
    # First pass: local optimization from hex init
    if time.time() < DEADLINE - 20:
        c_opt = _local_optimize(points, time_limit=min(15.0, DEADLINE - (time.time() - t_start)))
        r_opt = _compute_r(c_opt)
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()
    
    # Simulated annealing to escape local minima
    remaining = DEADLINE - (time.time() - t_start)
    if remaining > 30:
        sa_points, sa_score = _anneal(best_points, time.time(), time.time() + min(remaining - 10, 60), seed=42)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points.copy()
        
        # Local optimize after SA
        if time.time() < DEADLINE - 10:
            c_opt = _local_optimize(best_points, time_limit=min(15.0, DEADLINE - (time.time() - t_start)))
            r_opt = _compute_r(c_opt)
            if r_opt > best_score:
                best_score = r_opt
                best_points = c_opt.copy()
    
    # Multi-start: perturb best and try again
    n_restarts = 0
    while time.time() < DEADLINE - 10:
        perturbation_scale = 0.02 / (1.0 + 0.1 * n_restarts)
        x0 = best_points.flatten() + np.random.randn(2 * N) * perturbation_scale
        
        n_restarts += 1
        
        try:
            c_opt = _local_optimize(x0.reshape(N, 2), time_limit=min(10.0, DEADLINE - (time.time() - t_start)))
            r_opt = _compute_r(c_opt)
            if r_opt > best_score:
                best_score = r_opt
                best_points = c_opt.copy()
        except Exception:
            pass
        
        if n_restarts >= 5:
            break
    
    # Final polish
    if time.time() < DEADLINE - 3:
        c_final = _local_optimize(best_points, time_limit=min(8.0, DEADLINE - (time.time() - t_start) - 2))
        r_final = _compute_r(c_final)
        if r_final > best_score:
            best_points = c_final
            best_score = r_final
    
    # Ensure validity: clamp points that might be slightly outside
    c_final = best_points.reshape(N, 2)
    r2 = np.hypot(c_final[:, 0], c_final[:, 1])
    for i in range(N):
        if r2[i] > 1 - best_score + 1e-10:
            scale = (1 - best_score) / r2[i]
            c_final[i] *= scale
    
    return c_final
