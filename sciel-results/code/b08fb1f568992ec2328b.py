"""Circle packing for 200 equal circles in a unit square using physics-based optimization."""
N = 200
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

def _radius_of_placement(centers):
    """Compute the largest common radius for a given set of centers."""
    dist_to_wall = np.min([
        0.5 - centers[:, 0],
        centers[:, 0] + 0.5,
        0.5 - centers[:, 1],
        centers[:, 1] + 0.5
    ], axis=0)
    min_wall = np.min(dist_to_wall)
    
    if len(centers) <= 1:
        min_pair = np.inf
    else:
        tree = cKDTree(centers)
        dists, _ = tree.query(centers, k=2)
        min_pair = np.min(dists[:, 1]) / 2.0
    
    return min(min_wall, min_pair)


def _hex_lattice_init(rows=None, cols=None):
    """Create hexagonal lattice initialization."""
    if rows is None:
        rows = 14
    if cols is None:
        cols = 15
    
    dx = 1.0 / (cols + 0.5)
    dy = dx * np.sqrt(3) / 2
    
    centers = []
    for row in range(rows):
        for col in range(cols):
            if len(centers) >= N:
                break
            x = (col + 0.5) * dx - 0.5 + (0.5 * dx if row % 2 == 1 else 0)
            y = (row + 0.5) * dy - 0.5
            centers.append([x, y])
    
    if len(centers) < N:
        # Fallback: add more points
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        centers = list(grid[:N])
    
    centers = np.array(centers[:N])
    centers = np.clip(centers, -0.5, 0.5)
    return centers


def _energy(centers_flat):
    """
    Physics-based energy function: minimize total potential energy.
    Uses soft repulsive potential between circles and wall penalties.
    """
    centers = centers_flat.reshape(-1, 2)
    n = len(centers)
    
    # Wall penalty: push points away from boundaries
    wall_penalty = 0.0
    for i in range(n):
        for coord in range(2):
            val = centers[i, coord]
            if val < -0.45:
                wall_penalty += (val + 0.45) ** 2
            elif val > 0.45:
                wall_penalty += (val - 0.45) ** 2
    
    # Pairwise repulsive potential (soft, to avoid singularities)
    # Use a potential that is large when points are close
    total_repulsion = 0.0
    for i in range(n):
        for j in range(i+1, n):
            diff = centers[i] - centers[j]
            dist_sq = np.dot(diff, diff)
            # Soft repulsion: 1/(dist^2 + eps)
            total_repulsion += 1.0 / (dist_sq + 1e-6)
    
    return wall_penalty + 0.01 * total_repulsion


def _energy_grad(centers_flat):
    """Gradient of the energy function."""
    centers = centers_flat.reshape(-1, 2)
    n = len(centers)
    grad = np.zeros_like(centers)
    
    # Wall gradient
    for i in range(n):
        for coord in range(2):
            val = centers[i, coord]
            if val < -0.45:
                grad[i, coord] += 2 * (val + 0.45)
            elif val > 0.45:
                grad[i, coord] += 2 * (val - 0.45)
    
    # Pairwise repulsion gradient
    for i in range(n):
        for j in range(i+1, n):
            diff = centers[i] - centers[j]
            dist_sq = np.dot(diff, diff)
            # d/dx_i of 1/(dist^2 + eps) = -2*diff / (dist^2 + eps)^2
            factor = -2.0 / (dist_sq + 1e-6) ** 2
            grad[i] += factor * diff
            grad[j] -= factor * diff
    
    grad *= 0.01
    return grad.flatten()


def _physics_optimize(centers, max_time=30):
    """
    Optimize using physics-based L-BFGS-B with soft potentials.
    This explores the energy landscape more thoroughly.
    """
    start_time = time.time()
    centers_flat = centers.flatten()
    
    # Use L-BFGS-B with bounds to keep points in the square
    bounds = [(-0.5, 0.5)] * (N * 2)
    
    result = minimize(
        _energy,
        centers_flat,
        jac=_energy_grad,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 2000, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    optimized = result.x.reshape(-1, 2)
    optimized = np.clip(optimized, -0.5, 0.5)
    
    elapsed = time.time() - start_time
    return optimized, elapsed


def _local_refinement(centers, max_iters=2000, time_limit=15):
    """
    Fine-tuning using KDTree to identify and fix close pairs.
    """
    current = centers.copy()
    current_r = _radius_of_placement(current)
    best = current.copy()
    best_r = current_r
    
    start_time = time.time()
    step_size = 0.001
    no_improve = 0
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Find the closest pair
        tree = cKDTree(current)
        dists, neighbors = tree.query(current, k=2)
        min_idx = np.argmin(dists[:, 1])
        i = min_idx
        j = neighbors[min_idx, 1]
        min_dist = dists[min_idx, 1]
        
        if min_dist > 2 * current_r * 1.05:
            # Good separation, try small random moves
            idx = np.random.randint(len(current))
            perturbation = np.random.randn(2) * step_size
            candidate = current.copy()
            candidate[idx] += perturbation
            candidate[idx] = np.clip(candidate[idx], -0.5, 0.5)
            candidate_r = _radius_of_placement(candidate)
            if candidate_r > current_r:
                current = candidate
                current_r = candidate_r
                if current_r > best_r:
                    best_r = current_r
                    best = current.copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve % 50 == 0:
                    step_size *= 0.9
                    if step_size < 1e-7:
                        break
            continue
        
        # Fix the closest pair
        diff = current[i] - current[j]
        dist = np.linalg.norm(diff)
        if dist < 1e-10:
            direction = np.random.randn(2)
            direction /= np.linalg.norm(direction)
        else:
            direction = diff / dist
        
        move = step_size * 0.5
        candidate = current.copy()
        candidate[i] += direction * move
        candidate[j] -= direction * move
        candidate = np.clip(candidate, -0.5, 0.5)
        
        candidate_r = _radius_of_placement(candidate)
        
        if candidate_r > current_r:
            current = candidate
            current_r = candidate_r
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve % 30 == 0:
                step_size *= 0.9
                if step_size < 1e-7:
                    break
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 135  # Leave 15s safety margin
    
    # Start with hexagonal lattice
    centers = _hex_lattice_init()
    best_centers = centers.copy()
    best_r = _radius_of_placement(centers)
    
    # Phase 1: Physics-based optimization
    if time.time() < deadline - 60:
        optimized, elapsed = _physics_optimize(centers, max_time=30)
        r = _radius_of_placement(optimized)
        if r > best_r:
            best_r = r
            best_centers = optimized.copy()
    
    # Phase 2: Local refinement
    if time.time() < deadline - 25:
        refined = _local_refinement(best_centers, max_iters=2000, time_limit=20)
        r = _radius_of_placement(refined)
        if r > best_r:
            best_r = r
            best_centers = refined.copy()
    
    # Phase 3: Multiple restarts with physics optimization
    remaining_time = deadline - time.time()
    if remaining_time > 15:
        n_restarts = min(int(remaining_time / 15), 5)
        for restart in range(n_restarts):
            if time.time() >= deadline - 10:
                break
            
            # New hex lattice with different parameters
            rows = 12 + restart
            cols = 16 - restart
            centers = _hex_lattice_init(rows=rows, cols=cols)
            # Add small random perturbation
            centers += np.random.randn(N, 2) * 0.003
            centers = np.clip(centers, -0.5, 0.5)
            
            # Quick physics optimization
            optimized, _ = _physics_optimize(centers, max_time=10)
            r = _radius_of_placement(optimized)
            
            if r > best_r:
                best_r = r
                best_centers = optimized.copy()
                
                # Quick local refinement
                if time.time() < deadline - 5:
                    refined = _local_refinement(best_centers, max_iters=500, time_limit=5)
                    r2 = _radius_of_placement(refined)
                    if r2 > best_r:
                        best_r = r2
                        best_centers = refined.copy()
    
    # Final refinement
    if time.time() < deadline - 3:
        final = _local_refinement(best_centers, max_iters=1000, time_limit=3)
        final_r = _radius_of_placement(final)
        if final_r > best_r:
            best_r = final_r
            best_centers = final
    
    return best_centers
