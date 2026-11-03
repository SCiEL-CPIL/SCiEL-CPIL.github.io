"""Pack 50 equal circles in a unit square using Discrete MIS + Continuous Refinement."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
import sys

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers (exact, for scoring)."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall radius
    wall_r = np.min(0.5 - np.abs(centers))
    if wall_r < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _soft_min_radius(centers, beta=100.0):
    """
    Compute a smooth approximation of the radius using log-sum-exp.
    """
    # Wall constraints
    c1 = 0.5 - centers[:, 0]
    c2 = 0.5 + centers[:, 0]
    c3 = 0.5 - centers[:, 1]
    c4 = 0.5 + centers[:, 1]
    
    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    # Take upper triangle
    iu = np.triu_indices(N, 1)
    pair_dists = dist[iu] / 2.0
    
    # Combine all constraints
    all_constraints = np.concatenate([c1, c2, c3, c4, pair_dists])
    
    y = -beta * all_constraints
    max_y = np.max(y)
    # Handle case where max_y is very large (constraints very small/negative)
    # exp(y - max_y) will be <= 1
    sum_exp = np.sum(np.exp(y - max_y))
    log_sum_exp = max_y + np.log(sum_exp)
    
    return -log_sum_exp / beta

def _objective_smooth(x, beta=100.0):
    """Objective to minimize: -soft_min_radius."""
    centers = x.reshape(N, 2)
    r = _soft_min_radius(centers, beta)
    return -r

def _local_refine_smooth(centers, time_limit=5.0, beta=100.0):
    """
    Use L-BFGS-B to refine the configuration using the smooth objective.
    """
    x0 = centers.ravel()
    
    start_time = time.time()
    
    def callback(xk):
        if time.time() - start_time > time_limit:
            raise RuntimeError("Time limit exceeded")
        return False

    try:
        res = minimize(
            _objective_smooth, 
            x0, 
            args=(beta,),
            method='L-BFGS-B',
            options={
                'maxiter': 2000,
                'maxfun': 10000,
                'ftol': 1e-12,
                'gtol': 1e-10,
                'disp': False
            },
            callback=callback
        )
        if res.x is not None:
            return res.x.reshape(N, 2)
    except RuntimeError:
        pass
    except Exception:
        pass
    
    return centers

def _discrete_mis_search(grid_size, target_r, time_limit=10.0):
    """
    Find a set of N non-overlapping points on a grid with separation >= 2*target_r.
    Uses a greedy randomized approach with backtracking-like restarts if stuck,
    or simply multiple random greedy passes.
    
    Returns: (success, centers)
    """
    start_time = time.time()
    
    # Grid coordinates
    # We want points in [-0.5, 0.5]
    # If grid_size is e.g. 20, step is 1/19? Or 1/20?
    # Let's use linspace for better coverage of boundaries
    coords = np.linspace(-0.5, 0.5, grid_size)
    
    # Precompute all valid pairs (distance >= 2*target_r)
    # Grid points: (i, j) -> (coords[i], coords[j])
    # Total points: grid_size^2
    # For 30x30 = 900 points.
    # Pairs: ~400k.
    # We can build an adjacency matrix where 1 means "compatible" (distance >= 2r)
    # And find a clique of size N? No, MIS on the conflict graph (edges where dist < 2r).
    # Or Clique on the compatibility graph.
    
    # Let's use a simpler heuristic: Random Greedy with multiple restarts.
    # 1. Shuffle all grid points.
    # 2. Iterate, adding a point if it is compatible with all already selected points.
    # 3. If we reach N points, we have a solution.
    
    n_points = grid_size * grid_size
    points = np.array([(x, y) for y in coords for x in coords])
    
    # Precompute distances? 900x900 is 810k floats. ~6.4MB. Okay.
    # But checking compatibility is O(N_selected) per candidate.
    # Total O(N * n_points * N_selected) ~ 50 * 900 * 50 = 2.25M ops per restart.
    # Very fast.
    
    # To make it more robust, we can bias the random selection towards "central" points
    # or just uniform random.
    
    best_centers = None
    
    for attempt in range(200):
        if time.time() - start_time > time_limit:
            break
            
        # Shuffle indices
        indices = np.random.permutation(n_points)
        
        selected = []
        selected_indices = []
        
        for idx in indices:
            if len(selected) == N:
                break
                
            p = points[idx]
            is_ok = True
            
            for s in selected:
                if np.linalg.norm(p - s) < 2 * target_r - 1e-9:
                    is_ok = False
                    break
            
            if is_ok:
                selected.append(p)
                selected_indices.append(idx)
                
        if len(selected) == N:
            return True, np.array(selected)
            
    return False, None

def _grid_to_continuous(centers, target_r, time_limit=5.0):
    """
    Take discrete centers and refine them continuously to maximize radius.
    """
    # The discrete centers are already separated by at least 2*target_r.
    # So the current radius is at least target_r.
    # We can use L-BFGS-B to push them apart further.
    
    return _local_refine_smooth(centers, time_limit=time_limit, beta=100.0)

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    # Strategy:
    # 1. Try to find a discrete packing with radius r_target.
    #    Start with a high r_target and decrease if failed.
    # 2. Once a discrete packing is found, refine it continuously.
    # 3. Also try a few random continuous starts as fallback.
    
    # Known optimum is ~0.07138.
    # Let's try targets from 0.070 down to 0.060.
    
    # Grid size:
    # If r = 0.07, diameter = 0.14.
    # Grid step should be smaller than diameter to allow flexibility.
    # 0.05 step -> 20x20 grid.
    # 0.025 step -> 40x40 grid.
    
    grid_sizes = [25, 35, 45]
    
    # Phase 1: Discrete Search
    for grid_size in grid_sizes:
        if time.time() - start_time > DEADLINE - 15:
            break
            
        # Try a few target radii
        for r_target in [0.071, 0.070, 0.069, 0.068]:
            if time.time() - start_time > DEADLINE - 15:
                break
                
            time_left = DEADLINE - 15 - (time.time() - start_time)
            alloc_search = min(5.0, time_left * 0.2)
            
            if alloc_search < 0.5:
                continue
                
            success, disc_centers = _discrete_mis_search(grid_size, r_target, time_limit=alloc_search)
            
            if success:
                # Refine continuously
                time_left = DEADLINE - 10 - (time.time() - start_time)
                if time_left > 1.0:
                    refined = _grid_to_continuous(disc_centers, r_target, time_limit=min(10.0, time_left))
                    r_ref = _radius(refined)
                    
                    if r_ref > best_r:
                        best_r = r_ref
                        best_centers = refined.copy()
                        
                    # If we found a good one, we can break or try to improve
                    if best_r > 0.071:
                        # Good enough? Maybe try one more refinement with higher beta
                        pass
                        
    # Phase 2: Continuous Random Starts (Fallback / Improvement)
    # If discrete didn't find anything, or to try to beat it.
    np.random.seed(42)
    
    for i in range(5):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        
        time_left = DEADLINE - 5 - (time.time() - start_time)
        if time_left < 1.0:
            break
            
        # Quick refinement
        polished = _local_refine_smooth(c_rand, time_limit=min(5.0, time_left), beta=100.0)
        r_pol = _radius(polished)
        
        if r_pol > best_r:
            best_r = r_pol
            best_centers = polished.copy()

    # Final Polish on Best
    if best_centers is not None and time.time() - start_time < DEADLINE - 2:
        time_left = DEADLINE - 2 - (time.time() - start_time)
        if time_left > 1.0:
            # Use a higher beta for final precision?
            # L-BFGS-B with higher beta might get stuck if not careful, but we are close.
            final_polished = _local_refine_smooth(best_centers, time_limit=time_left, beta=200.0)
            r_final = _radius(final_polished)
            
            if r_final > best_r:
                best_r = r_final
                best_centers = final_polished.copy()

    if best_centers is None:
        # Fallback: Grid
        ticks_x = (np.arange(7) + 0.5) / 7 - 0.5
        ticks_y = (np.arange(8) + 0.5) / 8 - 0.5
        best_centers = np.array([(x, y) for y in ticks_y for x in ticks_x])[:N]

    return best_centers
