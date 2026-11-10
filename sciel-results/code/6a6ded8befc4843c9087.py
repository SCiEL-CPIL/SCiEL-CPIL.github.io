"""Improved packing of 50 equal spheres in a unit cube using smooth energy optimization."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _exact_score(centers):
    """Exact minimum radius score."""
    if len(centers) < 2:
        return 0.5
    coords = np.abs(centers)
    wall_dists = 0.5 - coords
    min_wall = np.min(wall_dists)
    
    # Compute all pairwise distances
    from scipy.spatial.distance import cdist
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_pair / 2.0, min_wall)

def _energy_and_grad(x, k=20.0, w_wall=1.0):
    """
    Smooth energy function and its gradient.
    Energy = sum_{i<j} exp(-k * d_ij) + w_wall * sum_i sum_axis exp(-k * wall_dist_i_axis)
    Lower energy = better packing (larger minimum distances).
    """
    centers = x.reshape(N, 3)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Only consider upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Energy from pairs: sum exp(-k * d)
    exp_pairs = np.exp(-k * pair_dists)
    E_pairs = np.sum(exp_pairs)
    
    # Gradient of pair energy
    # dE/dx_i = sum_{j != i} exp(-k*d_ij) * (-k) * (x_i - x_j)/d_ij
    grad = np.zeros_like(centers)
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        d = dists[i, j]
        if d > 1e-10:
            f = -k * exp_pairs[idx] / d
            diff_ij = centers[i] - centers[j]
            grad[i] += f * diff_ij
            grad[j] -= f * diff_ij
    
    # Wall energy
    abs_c = np.abs(centers)
    wall_dists = 0.5 - abs_c  # shape (N, 3)
    exp_walls = np.exp(-k * wall_dists)
    E_walls = w_wall * np.sum(exp_walls)
    
    # Gradient of wall energy
    # dE_wall/dx_i_axis = w_wall * exp(-k * (0.5 - |x_i_axis|)) * (-k) * sign(x_i_axis)
    signs = np.sign(centers)
    grad_wall = w_wall * (-k) * exp_walls * signs
    
    grad += grad_wall
    
    E = E_pairs + E_walls
    return E, grad.flatten()

def _optimize_from_start(x0, time_limit=30.0):
    """Optimize from a starting point using L-BFGS-B."""
    start_time = time.time()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Use multiple stages with increasing k for sharper optimization
    stages = [
        (5.0, 100, 1e-8),
        (10.0, 150, 1e-10),
        (20.0, 200, 1e-12),
        (40.0, 300, 1e-14),
    ]
    
    x = x0.copy()
    for k, maxiter, ftol in stages:
        if time.time() - start_time > time_limit:
            break
        try:
            result = minimize(
                _energy_and_grad, x, args=(k, 1.0),
                method='L-BFGS-B', bounds=bounds,
                jac=True,
                options={'maxiter': maxiter, 'ftol': ftol, 'gtol': 1e-8}
            )
            x = result.x
        except Exception:
            break
    
    return np.clip(x.reshape(N, 3), -0.5, 0.5)

def _initial_grid():
    """4x4x4 grid initialization with perturbations."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    centers += np.random.uniform(-0.02, 0.02, (N, 3))
    return np.clip(centers, -0.5, 0.5)

def _initial_bcc():
    """BCC lattice initialization."""
    m = 4
    points = []
    for z in range(m):
        for y in range(m):
            for x in range(m):
                points.append(((x + 0.5) / m - 0.5, (y + 0.5) / m - 0.5, (z + 0.5) / m - 0.5))
                points.append(((x + 1.0) / m - 0.5, (y + 1.0) / m - 0.5, (z + 1.0) / m - 0.5))
    points = np.array(points)
    centers = points[:N].copy()
    centers += np.random.uniform(-0.02, 0.02, (N, 3))
    return np.clip(centers, -0.5, 0.5)

def _initial_fcc():
    """FCC-like lattice initialization."""
    m = 3
    points = []
    # FCC: corners + face centers
    for z in range(m):
        for y in range(m):
            for x in range(m):
                # Corner
                points.append(((x + 0.5) / m - 0.5, (y + 0.5) / m - 0.5, (z + 0.5) / m - 0.5))
                # Face centers
                if x < m - 1:
                    points.append(((x + 1.0) / m - 0.5, (y + 0.5) / m - 0.5, (z + 0.5) / m - 0.5))
                if y < m - 1:
                    points.append(((x + 0.5) / m - 0.5, (y + 1.0) / m - 0.5, (z + 0.5) / m - 0.5))
                if z < m - 1:
                    points.append(((x + 0.5) / m - 0.5, (y + 0.5) / m - 0.5, (z + 1.0) / m - 0.5))
    points = np.array(points)
    # Shuffle and take N
    np.random.shuffle(points)
    centers = points[:N].copy()
    centers += np.random.uniform(-0.02, 0.02, (N, 3))
    return np.clip(centers, -0.5, 0.5)

def _initial_random():
    """Random initialization."""
    centers = np.random.uniform(-0.4, 0.4, (N, 3))
    return centers

def _local_refine(centers, time_limit=10.0):
    """Local refinement using small perturbations and energy minimization."""
    current = centers.copy()
    best = current.copy()
    best_score = _exact_score(best)
    start_time = time.time()
    
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002]
    
    for step_size in step_sizes:
        if time.time() - start_time > time_limit:
            break
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for _ in range(50):
                if time.time() - start_time > time_limit:
                    break
                # Try moving a random sphere
                idx = np.random.randint(N)
                old_pos = current[idx].copy()
                current[idx] += np.random.uniform(-step_size, step_size, 3)
                current[idx] = np.clip(current[idx], -0.5, 0.5)
                
                score = _exact_score(current)
                if score > best_score:
                    best = current.copy()
                    best_score = score
                    improved = True
                else:
                    current[idx] = old_pos
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    best_centers = None
    best_score = 0.0
    
    # Try multiple initializations
    initializations = [
        _initial_grid(),
        _initial_bcc(),
        _initial_fcc(),
        _initial_random(),
    ]
    
    # Add more random initializations if time permits
    for _ in range(3):
        initializations.append(_initial_random())
    
    for init in initializations:
        if time.time() > deadline - 25.0:
            break
        
        # Optimize from this starting point
        optimized = _optimize_from_start(init, time_limit=20.0)
        score = _exact_score(optimized)
        
        # Local refinement
        if time.time() < deadline - 10.0:
            refined, ref_score = _local_refine(optimized, time_limit=8.0)
            if ref_score > score:
                score = ref_score
                optimized = refined
        
        if best_centers is None or score > best_score:
            best_score = score
            best_centers = optimized
    
    if best_centers is None:
        best_centers = _initial_grid()
    
    return best_centers
# EVOLVE-BLOCK-END
