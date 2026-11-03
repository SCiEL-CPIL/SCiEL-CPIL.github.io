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
    """Vectorized score computation: max common radius."""
    n = len(points)
    radii = 1.0 - np.linalg.norm(points, axis=1)
    r = np.min(radii)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu] / 2.0
    r = min(r, np.min(pair_dists))
    return r


def _hexagonal_init(n):
    """Generate a hexagonal lattice initial configuration."""
    best_points = None
    best_r = 0.0
    for spacing in np.linspace(0.08, 0.22, 40):
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


def _local_optimize(c, time_limit=10.0):
    """Local optimization using L-BFGS-B with soft-min objective."""
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
    c = c.reshape(N, 2)
    pair_idx = _pairwise_indices(N)
    ci = c[pair_idx[:, 0]]
    cj = c[pair_idx[:, 1]]
    d = np.sqrt(np.sum((ci - cj) ** 2, axis=1))
    min_dist = np.min(d)
    r2 = np.hypot(c[:, 0], c[:, 1])
    min_boundary = np.min(1 - r2)
    return min(min_dist / 2, min_boundary)


def _simulated_annealing(points, t_start, deadline):
    """Run simulated annealing to find a better packing."""
    best_points = points.copy()
    best_score = _score_vec(best_points)
    
    # Energy function: penalize overlaps and boundary violations
    def energy(pts):
        # Boundary penalty
        radii = np.linalg.norm(pts, axis=1)
        boundary_viol = np.maximum(0, radii - 1.0)
        boundary_pen = np.sum(boundary_viol ** 2)
        
        # Circle-circle overlap penalty
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, 1)
        pair_dists = dists[iu]
        # Target minimum distance (rough estimate)
        target = 0.25
        overlap = np.maximum(0, target - pair_dists)
        overlap_pen = np.sum(overlap ** 2)
        
        return boundary_pen + overlap_pen
    
    current = points.copy()
    current_energy = energy(current)
    
    # Annealing schedule
    t0 = 0.1
    t_end = 0.001
    n_steps = 5000
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Temperature schedule
        frac = step / n_steps
        T = t0 * (t_end / t0) ** frac
        
        # Generate a random move: perturb a random subset of points
        n_move = np.random.randint(1, 6)
        idx = np.random.choice(N, n_move, replace=False)
        move = np.random.randn(n_move, 2) * (0.05 * (1 - frac * 0.5))
        new_pts = current.copy()
        new_pts[idx] += move
        
        new_energy = energy(new_pts)
        delta_e = new_energy - current_energy
        
        # Acceptance criterion
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
            current = new_pts
            current_energy = new_energy
            
            # Check if improved
            score = _score_vec(current)
            if score > best_score:
                best_score = score
                best_points = current.copy()
    
    return best_points, best_score


def construct():
    t_start = time.time()
    
    # Start with hexagonal initialization
    points = _hexagonal_init(N)
    best_points = points.copy()
    best_score = _score_vec(best_points)
    
    # Phase 1: Local optimization from hexagonal start
    if time.time() - t_start < DEADLINE - 30:
        c_opt = _local_optimize(points, time_limit=min(15.0, DEADLINE - (time.time() - t_start)))
        r_opt = _compute_r(c_opt)
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()
    
    # Phase 2: Simulated annealing from best point
    if time.time() - t_start < DEADLINE - 20:
        sa_points, sa_score = _simulated_annealing(best_points, t_start, DEADLINE - 10)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points.copy()
    
    # Phase 3: Local optimization from annealed result
    if time.time() - t_start < DEADLINE - 10:
        c_opt = _local_optimize(best_points, time_limit=min(10.0, DEADLINE - (time.time() - t_start)))
        r_opt = _compute_r(c_opt)
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()
    
    # Phase 4: Multi-start with perturbations
    n_restarts = 0
    while time.time() - t_start < DEADLINE - 5:
        n_restarts += 1
        perturbation_scale = 0.03 / (1.0 + 0.2 * n_restarts)
        x0 = best_points.flatten() + np.random.randn(2 * N) * perturbation_scale
        c_init = x0.reshape(N, 2)
        
        # Quick local optimization
        c_opt = _local_optimize(c_init, time_limit=min(5.0, DEADLINE - (time.time() - t_start) - 2))
        r_opt = _compute_r(c_opt)
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()
        
        # Occasionally run short annealing
        if n_restarts % 3 == 0 and time.time() - t_start < DEADLINE - 15:
            sa_points, sa_score = _simulated_annealing(best_points, t_start, DEADLINE - 8)
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_points.copy()
                c_opt = _local_optimize(best_points, time_limit=min(5.0, DEADLINE - (time.time() - t_start)))
                r_opt = _compute_r(c_opt)
                if r_opt > best_score:
                    best_score = r_opt
                    best_points = c_opt.copy()
    
    # Final polish
    if time.time() - t_start < DEADLINE - 3:
        c_final = _local_optimize(best_points, time_limit=min(5.0, DEADLINE - (time.time() - t_start) - 2))
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
