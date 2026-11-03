"""Improved packer for 50 equal circles in unit square."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
SQRT2 = np.sqrt(2.0)

def _radius_fast(centers):
    """Compute max radius. O(N^2) but vectorized, no Python loops."""
    n = len(centers)
    # Boundary distances
    dx = np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5)
    dy = np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    dist_bound = np.min(np.minimum(dx, dy))
    
    # Pairwise: use upper triangle to save half the work
    i = np.triu_indices(n, k=1)
    d = centers[i[0]] - centers[i[1]]
    dists = np.sqrt(d[:, 0]**2 + d[:, 1]**2)
    min_pair = np.min(dists)
    
    return min(min_pair / 2.0, dist_bound)

def _neg_radius(centers):
    return -_radius_fast(centers)

def _physics_repel(centers, n_steps=2000, dt=0.0005, k_repel=0.5, k_wall=2.0, temp=0.01):
    """Physics simulation with repulsive forces to spread circles."""
    n = len(centers)
    centers = centers.copy().astype(np.float64)
    vel = np.zeros_like(centers)
    
    for step in range(n_steps):
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion: F = k_repel / d^2 * direction (for d < threshold)
        # Use all pairs but only for close ones to save time
        for i in range(n):
            d = centers[i] - centers
            dist = np.sqrt(d[:, 0]**2 + d[:, 1]**2)
            # Avoid division by zero
            dist[dist < 1e-12] = 1e-12
            # Repulsion: stronger when closer
            # Use 1/d^2 falloff, but cap it
            repel = k_repel / (dist**2)
            repel[0] = 0  # no self-force
            forces[i] += np.sum(d * repel[:, np.newaxis], axis=0)
        
        # Wall repulsion
        for axis in range(2):
            # Left wall: -0.5
            dist_left = centers[:, axis] + 0.5
            wall_f = np.where(dist_left < 0.1, k_wall * (0.1 - dist_left)**2, 0.0)
            forces[:, axis] += wall_f
            # Right wall: 0.5
            dist_right = 0.5 - centers[:, axis]
            wall_f = np.where(dist_right < 0.1, k_wall * (0.1 - dist_right)**2, 0.0)
            forces[:, axis] -= wall_f
        
        # Update velocities with damping
        vel = 0.85 * vel + forces * dt
        centers += vel * dt
        
        # Hard clip to stay in bounds
        centers[:, 0] = np.clip(centers[:, 0], -0.499, 0.499)
        centers[:, 1] = np.clip(centers[:, 1], -0.499, 0.499)
    
    return centers

def _local_refine(centers, max_iter=300, tol=1e-10):
    """Refine by moving circles to increase the minimum constraint."""
    n = len(centers)
    centers = centers.copy()
    r = _radius_fast(centers)
    
    for it in range(max_iter):
        # Find the most constraining pairs
        i_idx = np.triu_indices(n, k=1)
        d = centers[i_idx[0]] - centers[i_idx[1]]
        dists = np.sqrt(d[:, 0]**2 + d[:, 1]**2)
        
        # Boundary distances
        dx = np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5)
        dy = np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
        db = np.minimum(dx, dy)
        
        # Find minimum
        min_pair_idx = np.argmin(dists)
        min_pair_dist = dists[min_pair_idx]
        min_bound_idx = np.argmin(db)
        min_bound_dist = db[min_bound_idx]
        
        new_r = min(min_pair_dist / 2.0, min_bound_dist)
        if new_r <= r + tol:
            break
        r = new_r
        
        # Try to improve: move the pair apart or move boundary circle inward
        # Strategy: move both circles of the min pair apart slightly
        p1, p2 = i_idx[0][min_pair_idx], i_idx[1][min_pair_idx]
        mid = (centers[p1] + centers[p2]) / 2
        direction = (centers[p1] - centers[p2]) / (min_pair_dist + 1e-12)
        
        # Try several step sizes
        best_step = 0
        best_r = r
        for step in [0.001, 0.0005, 0.0002, 0.0001]:
            c_test = centers.copy()
            c_test[p1] += direction * step
            c_test[p2] -= direction * step
            # Check bounds
            if (np.all(c_test[:, 0] > -0.5) and np.all(c_test[:, 0] < 0.5) and
                np.all(c_test[:, 1] > -0.5) and np.all(c_test[:, 1] < 0.5)):
                r_test = _radius_fast(c_test)
                if r_test > best_r:
                    best_r = r_test
                    best_step = step
        
        if best_step > 0:
            centers[p1] += direction * best_step
            centers[p2] -= direction * best_step
            r = best_r
        else:
            # Try moving the boundary-constrained circle inward
            c_test = centers.copy()
            ax = 0 if dx[min_bound_idx] < dy[min_bound_idx] else 1
            # Move toward center
            c_test[min_bound_idx, ax] += 0.001 if c_test[min_bound_idx, ax] < 0 else -0.001
            if (np.all(c_test[:, 0] > -0.5) and np.all(c_test[:, 0] < 0.5) and
                np.all(c_test[:, 1] > -0.5) and np.all(c_test[:, 1] < 0.5)):
                r_test = _radius_fast(c_test)
                if r_test > r:
                    centers = c_test
                    r = r_test
                else:
                    break
            else:
                break
    
    return centers, r

def _init_grid(n):
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n].astype(np.float64)

def _init_random(n, seed):
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 2))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_centers = None
    best_radius = 0.0
    
    # Multiple diverse initializations
    seeds = [42, 123, 456, 789, 101, 202, 303, 404]
    
    # Always start with grid (good baseline)
    inits = [(_init_grid(N), 'grid')]
    for s in seeds:
        inits.append((_init_random(N, s), f'rand_{s}'))
    
    for centers_init, name in inits:
        if time.time() > deadline - 15.0:
            break
        
        # Phase 1: Physics-based spreading (fast, vectorized-ish)
        t_phase1 = time.time()
        centers_phys = _physics_repel(centers_init, n_steps=1500, dt=0.0004)
        
        r_phys = _radius_fast(centers_phys)
        if r_phys > best_radius:
            best_radius = r_phys
            best_centers = centers_phys.copy()
        
        # Phase 2: Local refinement
        if time.time() < deadline - 10.0:
            centers_ref, r_ref = _local_refine(centers_phys, max_iter=200)
            if r_ref > best_radius:
                best_radius = r_ref
                best_centers = centers_ref.copy()
        
        # Phase 3: SciPy polish on the best so far
        if time.time() < deadline - 5.0:
            try:
                result = minimize(_neg_radius, best_centers.flatten(), method='Nelder-Mead',
                                 options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
                centers_nm = result.x.reshape(-1, 2)
                # Validate
                if (np.all(centers_nm[:, 0] > -0.5) and np.all(centers_nm[:, 0] < 0.5) and
                    np.all(centers_nm[:, 1] > -0.5) and np.all(centers_nm[:, 1] < 0.5) and
                    np.all(np.isfinite(centers_nm))):
                    r_nm = _radius_fast(centers_nm)
                    if r_nm > best_radius:
                        best_radius = r_nm
                        best_centers = centers_nm.copy()
            except Exception:
                pass
    
    # Final safety: ensure valid output
    if best_centers is None:
        best_centers = _init_grid(N)
    
    # Clip to strictly inside
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
