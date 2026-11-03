"""Packing 50 equal circles in a unit square [-0.5, 0.5]^2.
Maximize the minimum clearance (radius). Uses force-directed relaxation +
simulated annealing + local coordinate descent, with a deadline guard."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE_FRAC = 0.92  # finish by 92% of 150s = 138s


def _radius(centers):
    """Compute the valid radius for a set of centers in [-0.5,0.5]^2."""
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min()
    # Distance to walls
    wall = np.minimum(0.5 - np.abs(centers), axis=1)
    min_wall = wall.min()
    return min(min_pair / 2.0, min_wall)


def _radius_fast(centers):
    """Faster radius using scipy's cdist for pairwise distances."""
    from scipy.spatial.distance import cdist
    D = cdist(centers, centers)
    np.fill_diagonal(D, np.inf)
    min_pair = D.min()
    wall = np.minimum(0.5 - centers[:, 0], centers[:, 0])
    wall = np.minimum(wall, 0.5 - centers[:, 1])
    wall = np.minimum(wall, centers[:, 1])
    min_wall = wall.min()
    return min(min_pair / 2.0, min_wall)


def _negative_radius(centers):
    return -_radius_fast(centers)


def _grid_init():
    """Square grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].astype(np.float64)


def _random_init():
    """Random initialization inside the square."""
    return (np.random.rand(N, 2) - 0.5) * 0.95


def _hex_init():
    """Hexagonal-ish lattice initialization for better starting spread."""
    # Use a hexagonal packing pattern
    pts = []
    # Try rows with hexagonal offset
    n_per_row = 8
    for row in range(7):
        for col in range(n_per_row):
            x = (col + 0.5) / n_per_row - 0.5 + (0.5 / n_per_row if row % 2 else 0)
            y = (row + 0.5) / 7 - 0.5
            if abs(x) < 0.5 and abs(y) < 0.5:
                pts.append([x, y])
    pts = np.array(pts[:N])
    if len(pts) < N:
        # Fill remaining randomly
        extra = (np.random.rand(N - len(pts), 2) - 0.5) * 0.9
        pts = np.vstack([pts, extra])
    return pts[:N]


def _force_relax(centers, max_iter=2000, dt=0.001, deadline=None):
    """Force-directed relaxation: push apart overlapping circles, pull toward center."""
    centers = centers.copy()
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        # Compute forces
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, 1.0)  # avoid div by zero
        
        # Repulsive force: stronger when closer
        # Use 1/d^2 for repulsion
        force = np.zeros_like(centers)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = dist[i, j]
                if d < 0.1:
                    f = 1.0 / (d * d + 1e-8)
                    force[i] += f * (centers[i] - centers[j]) / (d + 1e-8)
        
        # Wall repulsion
        for i in range(N):
            for k in range(2):
                if centers[i, k] > 0.4:
                    force[i, k] -= 5.0 * (centers[i, k] - 0.4)
                elif centers[i, k] < -0.4:
                    force[i, k] -= 5.0 * (centers[i, k] + 0.4)
        
        # Centering force (weak)
        force -= 0.01 * centers
        
        centers += dt * force
        # Clamp to square
        centers[:, 0] = np.clip(centers[:, 0], -0.49, 0.49)
        centers[:, 1] = np.clip(centers[:, 1], -0.49, 0.49)
    return centers


def _force_relax_vectorized(centers, max_iter=5000, dt=0.0005, deadline=None):
    """Vectorized force-directed relaxation."""
    centers = centers.copy()
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, 1.0)
        
        # Repulsive force: 1/d^3 * direction (so force magnitude ~ 1/d^2)
        inv_d3 = 1.0 / (dist ** 3 + 1e-10)
        np.fill_diagonal(inv_d3, 0.0)
        force = np.sum(diff * inv_d3[:, :, np.newaxis], axis=1)
        
        # Wall repulsion
        for k in range(2):
            wall_pos = centers[:, k] - 0.45
            wall_neg = centers[:, k] + 0.45
            force[:, k] -= 10.0 * np.maximum(wall_pos, 0) ** 2
            force[:, k] += 10.0 * np.maximum(wall_neg, 0) ** 2
        
        # Weak centering
        force -= 0.005 * centers
        
        centers += dt * force
        centers[:, 0] = np.clip(centers[:, 0], -0.495, 0.495)
        centers[:, 1] = np.clip(centers[:, 1], -0.495, 0.495)
    return centers


def _sa_optimize(centers, deadline, max_iter=100000, T0=0.05, T_end=0.001, alpha=0.9999):
    """Simulated annealing to maximize radius."""
    centers = centers.copy()
    best_centers = centers.copy()
    best_r = _radius_fast(centers)
    T = T0
    for it in range(max_iter):
        if time.time() > deadline:
            break
        # Random perturbation
        idx = np.random.randint(N)
        step = T * 0.5
        delta = (np.random.rand(2) - 0.5) * 2 * step
        new_centers = centers.copy()
        new_centers[idx] += delta
        new_centers[idx, 0] = np.clip(new_centers[idx, 0], -0.495, 0.495)
        new_centers[idx, 1] = np.clip(new_centers[idx, 1], -0.495, 0.495)
        
        new_r = _radius_fast(new_centers)
        delta_r = new_r - best_r
        
        if delta_r > 0 or np.random.rand() < np.exp(delta_r / max(T, 1e-10)):
            centers = new_centers
            if new_r > best_r:
                best_r = new_r
                best_centers = new_centers.copy()
        T *= alpha
    return best_centers, best_r


def _local_refine(centers, deadline):
    """Local refinement using scipy minimize on negative radius."""
    centers = centers.copy()
    best = centers.copy()
    best_r = _radius_fast(centers)
    
    # Try a few Nelder-Mead iterations with tight bounds
    bounds = [(-0.495, 0.495)] * (2 * N)
    try:
        res = minimize(_negative_radius, centers, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
        if res.x is not None and np.all(np.isfinite(res.x)):
            candidate = res.x.reshape(N, 2)
            candidate[:, 0] = np.clip(candidate[:, 0], -0.495, 0.495)
            candidate[:, 1] = np.clip(candidate[:, 1], -0.495, 0.495)
            r = _radius_fast(candidate)
            if r > best_r:
                best = candidate
                best_r = r
    except Exception:
        pass
    
    if time.time() > deadline:
        return best, best_r
    
    # Coordinate descent polish
    for coord_iter in range(200):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            for k in range(2):
                # Try small adjustments
                for step in [0.001, -0.001, 0.0005, -0.0005, 0.0002, -0.0002]:
                    new_c = best.copy()
                    new_c[i, k] += step
                    new_c[i, k] = np.clip(new_c[i, k], -0.495, 0.495)
                    r = _radius_fast(new_c)
                    if r > best_r:
                        best = new_c
                        best_r = r
                        improved = True
                        break
        if not improved:
            break
    return best, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 * DEADLINE_FRAC
    
    np.random.seed(42)
    
    # Multiple initialization strategies
    inits = [
        _grid_init(),
        _hex_init(),
        _random_init(),
        _random_init(),
    ]
    
    best_centers = None
    best_r = 0.0
    
    for init in inits:
        if time.time() > deadline:
            break
        
        # Force relaxation
        relaxed = _force_relax_vectorized(init, max_iter=3000, dt=0.0003, deadline=deadline)
        
        # Simulated annealing
        sa_centers, sa_r = _sa_optimize(relaxed, deadline, max_iter=50000, T0=0.03, T_end=0.0005, alpha=0.99995)
        
        if sa_r > best_r:
            best_r = sa_r
            best_centers = sa_centers.copy()
        
        if time.time() > deadline:
            break
    
    # Local refinement on the best
    if best_centers is not None:
        refined, final_r = _local_refine(best_centers, deadline)
        if final_r > best_r:
            best_r = final_r
            best_centers = refined
    
    # Final safety: ensure validity
    if best_centers is None:
        best_centers = _grid_init()
    
    # Ensure all values are finite and within bounds
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    best_centers = np.nan_to_num(best_centers, nan=0.0, posinf=0.0, neginf=0.0)
    
    return best_centers
