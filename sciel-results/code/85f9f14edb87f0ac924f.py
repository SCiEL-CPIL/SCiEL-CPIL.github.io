# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

N = 50

def _compute_radius(centers):
    """
    Compute the maximum valid radius for a set of centers.
    Returns r = min(min_pair_dist, min_boundary_dist) / 2.0
    """
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    abs_centers = np.abs(centers)
    dist_to_faces = 0.5 - np.max(abs_centers, axis=1)
    min_boundary_dist = np.min(dist_to_faces)
    
    return min(min_pair_dist, min_boundary_dist) / 2.0

def _relaxate(centers, n_iters=5000, cell_size=0.1):
    """
    Iterative repulsion relaxation: push overlapping spheres apart using spatial hash.
    """
    centers = centers.copy()
    
    for it in range(n_iters):
        # Build spatial hash
        grid = defaultdict(list)
        for i, c in enumerate(centers):
            key = (int(np.floor(c[0] / cell_size)),
                   int(np.floor(c[1] / cell_size)),
                   int(np.floor(c[2] / cell_size)))
            grid[key].append(i)
        
        # Find the closest pair using spatial hash
        min_dist = np.inf
        min_pair = (0, 1)
        
        for i in range(N):
            ci = centers[i]
            kx = int(np.floor(ci[0] / cell_size))
            ky = int(np.floor(ci[1] / cell_size))
            kz = int(np.floor(ci[2] / cell_size))
            
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        key = (kx + dx, ky + dy, kz + dz)
                        if key in grid:
                            for j in grid[key]:
                                if j > i:
                                    d = np.linalg.norm(centers[i] - centers[j])
                                    if d < min_dist:
                                        min_dist = d
                                        min_pair = (i, j)
        
        # Check boundary constraint
        abs_c = np.abs(centers)
        boundary_clearance = 0.5 - np.max(abs_c, axis=1)
        min_boundary = np.min(boundary_clearance)
        
        # Determine what to fix
        if min_boundary < min_dist:
            i = int(np.argmin(boundary_clearance))
            c = centers[i]
            for axis in range(3):
                if c[axis] > 0.45:
                    centers[i, axis] -= 0.005
                elif c[axis] < -0.45:
                    centers[i, axis] += 0.005
        else:
            i, j = min_pair
            diff = centers[i] - centers[j]
            d = np.linalg.norm(diff)
            if d < 1e-10:
                centers[i] += np.random.randn(3) * 0.01
                centers[j] -= np.random.randn(3) * 0.01
            else:
                unit_diff = diff / d
                push = 0.005
                centers[i] += unit_diff * push
                centers[j] -= unit_diff * push
                centers[i] = np.clip(centers[i], -0.499, 0.499)
                centers[j] = np.clip(centers[j], -0.499, 0.499)
    
    return centers

def _local_search(centers, time_limit=10.0):
    """
    Greedy local search: perturb the most-constrained spheres.
    """
    centers = centers.copy()
    current_r = _compute_radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    rng = np.random.RandomState(123)
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dists = np.min(dists, axis=1)
        
        abs_c = np.abs(centers)
        boundary_clear = 0.5 - np.max(abs_c, axis=1)
        
        clearances = np.minimum(min_pair_dists, boundary_clear)
        i = int(np.argmin(clearances))
        
        accepted = False
        for attempt in range(5):
            step = 0.01 + 0.01 * attempt
            move = rng.randn(3) * step
            new_centers = centers.copy()
            new_centers[i] += move
            new_centers[i] = np.clip(new_centers[i], -0.499, 0.499)
            
            new_r = _compute_radius(new_centers)
            if new_r > current_r + 1e-12:
                centers = new_centers
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
                accepted = True
                break
        
        if not accepted:
            i = rng.randint(N)
            move = rng.randn(3) * 0.015
            new_centers = centers.copy()
            new_centers[i] += move
            new_centers[i] = np.clip(new_centers[i], -0.499, 0.499)
            new_r = _compute_radius(new_centers)
            if new_r > current_r + 1e-12:
                centers = new_centers
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
    
    return best_centers

def _initialize_lattice(seed=0):
    """
    Initialize with a perturbed 5x5x2 lattice (50 points).
    """
    rng = np.random.RandomState(seed)
    
    nx, ny, nz = 5, 5, 2
    xs = np.linspace(-0.45, 0.45, nx)
    ys = np.linspace(-0.45, 0.45, ny)
    zs = np.linspace(-0.45, 0.45, nz)
    
    centers = np.zeros((N, 3))
    idx = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                centers[idx] = [xs[ix], ys[iy], zs[iz]]
                idx += 1
    
    centers += rng.randn(N, 3) * 0.02
    centers = np.clip(centers, -0.49, 0.49)
    
    return centers

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_centers_overall = None
    best_radius_overall = 0.0
    
    n_restarts = 5
    time_per_restart = time_limit / n_restarts
    
    for restart in range(n_restarts):
        centers = _initialize_lattice(seed=restart * 7 + 1)
        
        relaxed = _relaxate(centers, n_iters=3000, cell_size=0.1)
        r1 = _compute_radius(relaxed)
        
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed - 5
        if remaining > 5:
            searched = _local_search(relaxed, time_limit=min(15, remaining))
            r2 = _compute_radius(searched)
            final = searched if r2 >= r1 else relaxed
            r_final = max(r1, r2)
        else:
            final = relaxed
            r_final = r1
        
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed - 5
        if remaining > 8:
            relaxed2 = _relaxate(final, n_iters=2000, cell_size=0.08)
            r3 = _compute_radius(relaxed2)
            searched2 = _local_search(relaxed2, time_limit=min(10, remaining - 3))
            r4 = _compute_radius(searched2)
            final2 = searched2 if r4 >= r3 else relaxed2
            r_final2 = max(r3, r4)
            if r_final2 > r_final:
                final = final2
                r_final = r_final2
        
        if r_final > best_radius_overall:
            best_radius_overall = r_final
            best_centers_overall = final.copy()
        
        if time.time() - start_time > time_limit - 5.0:
            break
    
    return best_centers_overall
# EVOLVE-BLOCK-END
