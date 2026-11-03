"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools

# Deadline guard: keep a safety margin before 120s
_DEADLINE = 115.0  # seconds
_start_time = None

def _elapsed():
    return time.time() - _start_time if _start_time is not None else 0.0

def _is_valid(arr, tol=1e-6):
    """Check if the array of circles is valid: disjoint interiors, in rectangle with perimeter <= 4."""
    if arr is None or len(arr) != N:
        return False
    centers = arr[:, :2]
    radii = arr[:, 2]
    if np.any(radii < -tol):
        return False
    # Check non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < radii[i] + radii[j] - tol:
                return False
    # Check bounding rectangle perimeter
    min_x = np.min(centers[:, 0] - radii)
    max_x = np.max(centers[:, 0] + radii)
    min_y = np.min(centers[:, 1] - radii)
    max_y = np.max(centers[:, 1] + radii)
    w = max_x - min_x
    h = max_y - min_y
    if w < -tol or h < -tol:
        return False
    perim = 2 * (w + h)
    if perim > 4 + 1e-4:
        return False
    return True

def _sum_radii(arr):
    return float(np.sum(arr[:, 2]))

def _bounding_rect_perim(arr):
    centers = arr[:, :2]
    radii = arr[:, 2]
    min_x = np.min(centers[:, 0] - radii)
    max_x = np.max(centers[:, 0] + radii)
    min_y = np.min(centers[:, 1] - radii)
    max_y = np.max(centers[:, 1] + radii)
    return 2 * ((max_x - min_x) + (max_y - min_y))

def _hex_init(n_circles):
    """Initialize circles on a hexagonal lattice pattern."""
    # For 21 circles, use a hexagonal arrangement
    # Rows: 5,4,5,4,3 or similar patterns
    # Let's try a compact hexagonal packing
    # 21 = 1 + 6 + 12 (hexagonal numbers: H_1=1, H_2=7, H_3=19) - not quite
    # Let's use rows: 4,5,4,4,4 or 5,4,5,4,3
    # Actually, let's just place them in a hexagonal grid
    positions = []
    r = 0.1  # initial radius guess
    dx = 2 * r
    dy = 2 * r * np.sqrt(3) / 2
    
    # Generate hexagonal lattice points
    for row in range(8):
        for col in range(8):
            x = col * dx + (row % 2) * dx / 2
            y = row * dy
            positions.append([x, y])
    
    # Take first n_circles
    positions = positions[:n_circles]
    positions = np.array(positions, dtype=float)
    
    # Center them
    positions[:, 0] -= np.mean(positions[:, 0])
    positions[:, 1] -= np.mean(positions[:, 1])
    
    return positions, r

def _random_init(n_circles, scale=1.0):
    """Random initialization of circle centers and radii."""
    centers = np.random.uniform(-scale, scale, size=(n_circles, 2))
    radii = np.full(n_circles, 0.1)
    return np.column_stack([centers, radii])

def _force_directed_refine(arr, steps=200, dt=0.01, repulse_strength=1.0, attract_strength=0.01):
    """Physics-based force-directed refinement to escape local minima."""
    arr = arr.copy()
    centers = arr[:, :2].copy()
    radii = arr[:, 2].copy()
    
    for step in range(steps):
        if _elapsed() > _DEADLINE - 5:
            break
        forces = np.zeros_like(centers)
        
        # Repulsive forces between overlapping circles
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                min_dist = radii[i] + radii[j]
                if dist < 1e-8:
                    dist = 1e-8
                    diff = np.array([1e-8, 0.0])
                if dist < min_dist:
                    # Strong repulsion when overlapping
                    force = repulse_strength * (min_dist - dist) / dist
                    forces[i] += force * diff
                    forces[j] -= force * diff
        
        # Mild attraction to center to keep things compact
        for i in range(N):
            forces[i] += -attract_strength * centers[i]
        
        # Update positions
        centers += dt * forces
        arr[:, :2] = centers
    
    return arr

def _optimize_packing(arr_init, max_iter=500):
    """Optimize the packing using scipy minimize."""
    def objective_and_grad(x):
        centers = x[:2*N].reshape(N, 2)
        radii = x[2*N:]
        
        # Penalty for overlapping circles
        overlap_penalty = 0.0
        grad_centers = np.zeros_like(centers)
        grad_radii = np.zeros_like(radii)
        
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                min_dist = radii[i] + radii[j]
                if dist < 1e-8:
                    dist = 1e-8
                    diff = np.array([1e-8, 0.0])
                if dist < min_dist:
                    overlap = min_dist - dist
                    overlap_penalty += overlap ** 2
                    force = 2 * overlap / dist
                    grad_centers[i] += force * diff
                    grad_centers[j] -= force * diff
                    grad_radii[i] += 2 * overlap
                    grad_radii[j] += 2 * overlap
        
        # Penalty for exceeding perimeter
        min_x = np.min(centers[:, 0] - radii)
        max_x = np.max(centers[:, 0] + radii)
        min_y = np.min(centers[:, 1] - radii)
        max_y = np.max(centers[:, 1] + radii)
        w = max_x - min_x
        h = max_y - min_y
        perim = 2 * (w + h)
        
        perim_penalty = 0.0
        if perim > 4:
            perim_penalty = (perim - 4) ** 2 * 100
        
        # Objective: maximize sum of radii, penalize violations
        obj = -np.sum(radii) + overlap_penalty + perim_penalty
        
        # Gradient
        grad = np.zeros_like(x)
        grad[:2*N] = grad_centers.reshape(-1)
        grad[2*N:] = -np.ones(N) + grad_radii
        
        return obj, grad
    
    x0 = arr_init.flatten()
    
    result = minimize(
        objective_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': max_iter, 'gtol': 1e-6}
    )
    
    x_opt = result.x
    arr_opt = x_opt.reshape(-1, 3)
    arr_opt[:, :2] = x_opt[:2*N].reshape(N, 2)
    arr_opt[:, 2] = x_opt[2*N:]
    
    return arr_opt

def construct():
    global _start_time
    _start_time = time.time()
    
    best_arr = None
    best_score = -np.inf
    
    # Try multiple initialization strategies
    init_strategies = []
    
    # Strategy 1: Hexagonal lattice
    hex_pos, hex_r = _hex_init(N)
    hex_arr = np.column_stack([hex_pos, np.full(N, hex_r)])
    init_strategies.append(hex_arr)
    
    # Strategy 2: Random initializations
    for _ in range(3):
        random_arr = _random_init(N, scale=0.8)
        init_strategies.append(random_arr)
    
    # Strategy 3: Grid-like initialization (similar to seed but better)
    side = int(np.ceil(np.sqrt(N)))
    grid_cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                  for i in range(side) for j in range(side)]
    grid_arr = np.array(grid_cells[:N])
    init_strategies.append(grid_arr)
    
    for init_arr in init_strategies:
        if _elapsed() > _DEADLINE - 10:
            break
        
        # Force-directed refinement first
        refined = _force_directed_refine(init_arr, steps=100, dt=0.005)
        
        # Check if valid
        if _is_valid(refined):
            score = _sum_radii(refined)
            if score > best_score:
                best_score = score
                best_arr = refined.copy()
        
        # Optimize
        if _elapsed() > _DEADLINE - 5:
            break
        optimized = _optimize_packing(refined, max_iter=300)
        
        # Check if valid
        if _is_valid(optimized):
            score = _sum_radii(optimized)
            if score > best_score:
                best_score = score
                best_arr = optimized.copy()
        
        # Another round of force-directed + optimization
        if _elapsed() > _DEADLINE - 5:
            break
        refined2 = _force_directed_refine(optimized, steps=100, dt=0.003)
        optimized2 = _optimize_packing(refined2, max_iter=200)
        
        if _is_valid(optimized2):
            score = _sum_radii(optimized2)
            if score > best_score:
                best_score = score
                best_arr = optimized2.copy()
    
    # If no valid solution found, fall back to seed
    if best_arr is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_arr = np.array(cells[:N])
    
    return best_arr
# EVOLVE-BLOCK-END
