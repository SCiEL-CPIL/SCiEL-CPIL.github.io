"""
Circle packing in unit square: maximize sum of radii for 26 disjoint circles.
Uses Delaunay contact graph optimization with L-BFGS-B and Parallel Tempering Monte Carlo.
"""
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import sys

N = 26
DEADLINE = 115.0  # Safety margin before 120s limit

def _remaining_time():
    return DEADLINE - time.time()

def _softplus(x, k=100):
    """Numerically stable softplus approximation for penalty."""
    return np.where(x > 0, x, np.log1p(np.exp(-x)) / k)

def _validate(circles):
    """Check if circles are valid (disjoint, inside unit square)."""
    n = len(circles)
    if n == 0:
        return False
    # Boundary check
    if np.any(circles[:, 0] - circles[:, 2] < -1e-7) or np.any(circles[:, 0] + circles[:, 2] > 1 + 1e-7):
        return False
    if np.any(circles[:, 1] - circles[:, 2] < -1e-7) or np.any(circles[:, 1] + circles[:, 2] > 1 + 1e-7):
        return False
    
    # Overlap check
    # Vectorized distance calculation
    dx = circles[:, 0][:, None] - circles[:, 0][None, :]
    dy = circles[:, 1][:, None] - circles[:, 1][None, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    min_dists = circles[:, 2][:, None] + circles[:, 2][None, :]
    
    # Mask out self-distances
    np.fill_diagonal(dist, np.inf)
    np.fill_diagonal(min_dists, 0)
    
    # Check if any distance is less than sum of radii
    if np.any(dist < min_dists - 1e-7):
        return False
    return True

def _get_contact_edges(circles):
    """Get contact edges using Delaunay triangulation."""
    points = circles[:, :2]
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                a = simplex[i]
                b = simplex[(i + 1) % 3]
                if a > b:
                    a, b = b, a
                edges.add((a, b))
        return list(edges)
    except Exception:
        # Fallback to all pairs if Delaunay fails (e.g., duplicate points)
        return [(i, j) for i in range(len(circles)) for j in range(i+1, len(circles))]

def _objective(params, penalty_weight=1000.0, contact_edges=None):
    """
    Objective: minimize -sum(radii) + penalty.
    params: flat array [x1, y1, r1, x2, y2, r2, ...]
    """
    n = len(params) // 3
    circles = params.reshape(n, 3)
    
    # Ensure radii are positive
    # We use a soft constraint for radii > 0 in the bounds, but clip here for safety in distance calc
    radii = np.maximum(circles[:, 2], 1e-6)
    circles[:, 2] = radii
    
    # Boundary penalties
    # x - r >= 0  => r - x <= 0
    # x + r <= 1  => 1 - x - r <= 0
    # y - r >= 0  => r - y <= 0
    # y + r <= 1  => 1 - y - r <= 0
    
    bound_pen = 0.0
    # Vectorized boundary penalty
    # We want to penalize violations of [r, 1-r] for x and y
    x_min_viol = np.maximum(0, radii - circles[:, 0])
    x_max_viol = np.maximum(0, circles[:, 0] + radii - 1)
    y_min_viol = np.maximum(0, radii - circles[:, 1])
    y_max_viol = np.maximum(0, circles[:, 1] + radii - 1)
    
    bound_pen = np.sum(x_min_viol**2 + x_max_viol**2 + y_min_viol**2 + y_max_viol**2)
    
    # Overlap penalties
    overlap_pen = 0.0
    if contact_edges is None:
        contact_edges = _get_contact_edges(circles)
        
    for i, j in contact_edges:
        dx = circles[i, 0] - circles[j, 0]
        dy = circles[i, 1] - circles[j, 1]
        dist = np.sqrt(dx*dx + dy*dy)
        min_dist = radii[i] + radii[j]
        if dist < min_dist:
            overlap_pen += (min_dist - dist)**2
            
    return -np.sum(radii) + penalty_weight * (overlap_pen + bound_pen)

def _local_optimize(circles, start_time, max_iter=500):
    """Local optimization using L-BFGS-B."""
    n = len(circles)
    params = circles.flatten().copy()
    
    # Bounds: x in [0,1], y in [0,1], r in [0.01, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0.01, 0.5))
        
    # We need to update contact edges during optimization
    # L-BFGS-B doesn't easily support dynamic objective functions with external state updates in every call
    # Strategy: Use a wrapper that updates contact edges periodically or use a fixed set of "potential" contacts
    # For N=26, calculating all pairs is 325 pairs. Delaunay is ~100 edges.
    # Using all pairs is safer for gradient correctness but slower.
    # Let's use a hybrid: start with Delaunay, but if we detect significant movement, recompute.
    # However, scipy.optimize.minimize calls the objective many times.
    # To keep it simple and robust, we will use a "static" contact graph for the duration of one L-BFGS run,
    # but we will run L-BFGS multiple times, updating the graph between runs.
    
    edges = _get_contact_edges(circles)
    
    def obj_wrapper(p):
        return _objective(p, penalty_weight=1000.0, contact_edges=edges)
        
    # Use L-BFGS-B
    res = minimize(obj_wrapper, params, method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
    
    if res.success:
        return res.x.reshape(n, 3)
    else:
        # Return the best found even if not fully converged
        return res.x.reshape(n, 3)

def _refine_with_recalc(circles, start_time, num_passes=5):
    """
    Iteratively optimize, recalculating contact graph between passes.
    """
    current = circles.copy()
    for _ in range(num_passes):
        if time.time() > start_time + _remaining_time() - 5:
            break
        current = _local_optimize(current, start_time, max_iter=100)
        # Recalculate edges for next pass
        # No need to do it explicitly here as _local_optimize does it at start
    return current

def _pt_mc_search(circles, start_time, deadline, n_replicas=4, max_steps=2000):
    """
    Parallel Tempering Monte Carlo search.
    """
    n = len(circles)
    temps = [0.01, 0.05, 0.1, 0.5]
    replicas = [circles.copy() for _ in range(n_replicas)]
    
    best_circles = circles.copy()
    best_sum = np.sum(circles[:, 2])
    
    step_size = 0.02
    
    # Pre-compute edges for initial state? 
    # Edges change as we move. We'll compute on the fly for energy calc.
    # To speed up, we can use a fixed set of "potential" edges from the start, 
    # or just use all pairs for N=26 (325 pairs is cheap).
    # Let's use all pairs for simplicity and robustness in MC, as it's fast enough.
    # 325 pairs * 4 replicas * 2000 steps = 2.6M distance calcs. Very fast.
    
    def compute_energy(circs):
        return _objective(circs.flatten(), penalty_weight=100.0, contact_edges=None) # None means all pairs
        
    for step in range(max_steps):
        if time.time() > deadline - 5:
            break
        
        for rep in range(n_replicas):
            # Propose move
            idx = np.random.randint(n)
            move_type = np.random.choice(3)
            
            new_circles = replicas[rep].copy()
            if move_type == 0:
                new_circles[idx, 0] += np.random.normal(0, step_size)
                new_circles[idx, 1] += np.random.normal(0, step_size)
            elif move_type == 1:
                new_circles[idx, 2] *= np.exp(np.random.normal(0, step_size * 0.5))
                new_circles[idx, 2] = np.clip(new_circles[idx, 2], 0.01, 0.5)
            else:
                new_circles[idx, 0] += np.random.normal(0, step_size)
                new_circles[idx, 1] += np.random.normal(0, step_size)
                new_circles[idx, 2] *= np.exp(np.random.normal(0, step_size * 0.3))
                new_circles[idx, 2] = np.clip(new_circles[idx, 2], 0.01, 0.5)
            
            # Clip positions
            new_circles[idx, 0] = np.clip(new_circles[idx, 0], 0, 1)
            new_circles[idx, 1] = np.clip(new_circles[idx, 1], 0, 1)
            
            # Energy change
            old_e = compute_energy(replicas[rep])
            new_e = compute_energy(new_circles)
            delta_e = new_e - old_e
            
            # Metropolis acceptance
            if delta_e < 0 or np.random.random() < np.exp(-delta_e / temps[rep]):
                replicas[rep] = new_circles
            
            # Track best valid
            if _validate(replicas[rep]):
                s = np.sum(replicas[rep][:, 2])
                if s > best_sum:
                    best_sum = s
                    best_circles = replicas[rep].copy()
        
        # Swap replicas
        for rep in range(n_replicas - 1):
            if np.random.random() < 0.3:
                e1 = compute_energy(replicas[rep])
                e2 = compute_energy(replicas[rep + 1])
                delta_swap = (temps[rep] - temps[rep + 1]) * (e1 - e2)
                if delta_swap < 0 or np.random.random() < np.exp(-delta_swap):
                    replicas[rep], replicas[rep + 1] = replicas[rep + 1], replicas[rep]
        
        # Adaptive step size
        if step % 100 == 0:
            step_size *= 0.95
            
    return best_circles

def _initial_hex(n_circles):
    """Hexagonal packing initialization."""
    r0 = 0.15
    dx = 2 * r0
    dy = 2 * r0 * np.sqrt(3) / 2
    circles = []
    for row in range(8):
        for col in range(8):
            x = col * dx + (row % 2) * dx / 2 + 0.1
            y = row * dy + 0.1
            if 0 <= x <= 1 and 0 <= y <= 1:
                circles.append([x, y, r0])
    if len(circles) > n_circles:
        circles = circles[:n_circles]
    while len(circles) < n_circles:
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        circles.append([x, y, r0])
    return np.array(circles)

def _initial_grid(n_circles):
    """Grid initialization."""
    side = int(np.ceil(np.sqrt(n_circles)))
    circles = []
    for i in range(side):
        for j in range(side):
            if len(circles) < n_circles:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.005
                circles.append([x, y, r])
    return np.array(circles)

def _initial_corner_weighted(n_circles):
    """Corner weighted initialization."""
    circles = []
    corner_positions = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9),
                        (0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5)]
    for x, y in corner_positions:
        if len(circles) < n_circles:
            circles.append([x + np.random.uniform(-0.05, 0.05),
                            y + np.random.uniform(-0.05, 0.05),
                            0.12])
    while len(circles) < n_circles:
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        circles.append([x, y, 0.1])
    return np.array(circles)

def _initial_random(n_circles):
    """Random initialization."""
    circles = []
    for _ in range(n_circles):
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        r = np.random.uniform(0.05, 0.15)
        circles.append([x, y, r])
    return np.array(circles)

def construct():
    start_time = time.time()
    
    initializations = [
        _initial_hex(N),
        _initial_grid(N),
        _initial_corner_weighted(N),
        _initial_random(N)
    ]
    
    best_circles = None
    best_sum = 0.0
    
    # Phase 1: Local Optimization from multiple starts
    for init in initializations:
        if time.time() > start_time + _remaining_time() - 30:
            break
        
        # Refine with recalculating contact graph
        optimized = _refine_with_recalc(init, start_time, num_passes=3)
        
        if _validate(optimized):
            s = np.sum(optimized[:, 2])
            if s > best_sum:
                best_sum = s
                best_circles = optimized.copy()
        
        # Update best even if invalid? No, we need valid to score.
        # But we can use it as a starting point for PTMC if it's "close" to valid.
        # For now, stick to valid bests for PTMC start.
        
    if best_circles is None:
        # Fallback if all local optimizations failed to produce valid results
        best_circles = _initial_grid(N)
        # Force validity by shrinking
        for i in range(len(best_circles)):
            best_circles[i, 2] *= 0.5
            best_circles[i, 0] = np.clip(best_circles[i, 0], best_circles[i, 2], 1 - best_circles[i, 2])
            best_circles[i, 1] = np.clip(best_circles[i, 1], best_circles[i, 2], 1 - best_circles[i, 2])
        best_sum = np.sum(best_circles[:, 2])

    # Phase 2: Parallel Tempering MC
    if time.time() < start_time + _remaining_time() - 20:
        refined = _pt_mc_search(best_circles, start_time, DEADLINE)
        if _validate(refined):
            s = np.sum(refined[:, 2])
            if s > best_sum:
                best_sum = s
                best_circles = refined.copy()

    # Phase 3: Final Polish
    if time.time() < start_time + _remaining_time() - 5:
        polished = _local_optimize(best_circles, start_time, max_iter=200)
        if _validate(polished):
            s = np.sum(polished[:, 2])
            if s > best_sum:
                best_circles = polished.copy()
                best_sum = s

    # Final sanity check and fix
    if not _validate(best_circles):
        # Shrink radii slightly and re-center if needed
        for i in range(len(best_circles)):
            best_circles[i, 2] *= 0.9
            best_circles[i, 0] = np.clip(best_circles[i, 0], best_circles[i, 2], 1 - best_circles[i, 2])
            best_circles[i, 1] = np.clip(best_circles[i, 1], best_circles[i, 2], 1 - best_circles[i, 2])
            
    return best_circles

if __name__ == "__main__":
    res = construct()
    print(f"Sum of radii: {np.sum(res[:, 2])}")
    print(f"Valid: {_validate(res)}")
