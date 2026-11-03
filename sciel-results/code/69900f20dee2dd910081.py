"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return np.sum(r), True


def _de_objective(params, n):
    """
    Objective function for Differential Evolution.
    Returns a cost value to be minimized.
    We want to maximize sum(r) and minimize violations.
    Cost = -sum(r) + large_penalty * violations
    """
    xy = params[:2*n].reshape(n, 2)
    r = params[2*n:]
    
    # Clip radii to be positive
    r = np.maximum(r, 0.001)
    
    # Wall violations
    wall_violation = 0.0
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        wall_violation += max(0.0, ri - x)**2
        wall_violation += max(0.0, ri - (1.0 - x))**2
        wall_violation += max(0.0, ri - y)**2
        wall_violation += max(0.0, ri - (1.0 - y))**2
    
    # Overlap violations
    overlap_violation = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                overlap_violation += (min_dist - dist)**2
    
    penalty = 1000.0 * (wall_violation + overlap_violation)
    total_radius = np.sum(r)
    
    return -total_radius + penalty


def _local_refine(xy, r, deadline):
    """
    Refine a valid packing using L-BFGS with penalty, starting from a good DE result.
    """
    n = len(xy)
    params = np.zeros(3 * n)
    params[0::3] = xy[:, 0]
    params[1::3] = xy[:, 1]
    params[2::3] = r
    
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.4))
    
    best_total, best_valid = _eval_packing(xy, r)
    best_params = params.copy()
    
    if not best_valid:
        return xy, r, best_total, best_valid

    def _penalized_objective(p, n, penalty_weight=500.0):
        xy_p = p[0::3].reshape(-1, 2)
        r_p = np.maximum(p[1::3], 0.001)
        
        wall_violation = 0.0
        for i in range(n):
            x, y, ri = xy_p[i, 0], xy_p[i, 1], r_p[i]
            wall_violation += max(0.0, ri - x)**2
            wall_violation += max(0.0, ri - (1.0 - x))**2
            wall_violation += max(0.0, ri - y)**2
            wall_violation += max(0.0, ri - (1.0 - y))**2
        
        overlap_violation = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy_p[i, 0] - xy_p[j, 0]
                dy = xy_p[i, 1] - xy_p[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r_p[i] + r_p[j]
                if dist < min_dist:
                    overlap_violation += (min_dist - dist)**2
        
        penalty = penalty_weight * (wall_violation + overlap_violation)
        total_radius = np.sum(r_p)
        return -total_radius + penalty

    try:
        result = minimize(
            _penalized_objective,
            params,
            args=(n, 500.0),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        opt_params = result.x
        opt_xy = opt_params[0::3].reshape(-1, 2)
        opt_r = np.maximum(opt_params[1::3], 0.001)
        
        total, valid = _eval_packing(opt_xy, opt_r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_params = opt_params.copy()
            
    except Exception:
        pass
    
    final_xy = best_params[0::3].reshape(-1, 2)
    final_r = np.maximum(best_params[1::3], 0.001)
    return final_xy, final_r, best_total, best_valid


def _greedy_delaunay_init():
    """
    Initialize using a greedy algorithm that places circles one by one
    at the position that maximizes the minimum distance to existing circles/walls.
    This creates a Voronoi-like packing which is a good starting point.
    """
    xy = np.zeros((N, 2))
    r = np.zeros(N)
    
    # First circle in center
    xy[0] = [0.5, 0.5]
    r[0] = 0.5
    
    for i in range(1, N):
        # Grid search for best position for new circle
        # We want to maximize the radius it can have, which is min(dist_to_others - r_others, dist_to_walls)
        best_r = 0.0
        best_pos = [0.5, 0.5]
        
        # Coarse grid first
        for side in [20, 40]:
            gx = np.linspace(0.01, 0.99, side)
            gy = np.linspace(0.01, 0.99, side)
            for x in gx:
                for y in gy:
                    # Max possible radius is limited by walls
                    r_max_wall = min(x, 1-x, y, 1-y)
                    if r_max_wall <= best_r:
                        continue
                    
                    # Check against existing circles
                    r_max_circle = r_max_wall
                    for j in range(i):
                        dx = x - xy[j, 0]
                        dy = y - xy[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        r_allowed = dist - r[j]
                        if r_allowed < r_max_circle:
                            r_max_circle = r_allowed
                            if r_max_circle <= best_r:
                                break
                    if r_max_circle > best_r:
                        best_r = r_max_circle
                        best_pos = [x, y]
        
        xy[i] = best_pos
        r[i] = best_r
    
    return xy, r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False
    
    # 1. Greedy Delaunay-like initialization
    xy_init, r_init = _greedy_delaunay_init()
    
    # Refine this initial packing locally
    if time.time() < deadline - 30.0:
        xy_ref, r_ref, total_ref, valid_ref = _local_refine(xy_init, r_init, deadline - 30.0)
        if valid_ref and total_ref > best_total:
            best_total = total_ref
            best_valid = True
            best_xy = xy_ref
            best_r = r_ref
    
    # 2. Differential Evolution Global Search
    # DE explores the whole space, potentially finding better topologies than the greedy init
    bounds_de = []
    for i in range(N):
        bounds_de.append((0.01, 0.99))  # x
        bounds_de.append((0.01, 0.99))  # y
        bounds_de.append((0.01, 0.3))   # r
    
    # Use a subset of the time for DE
    de_deadline = deadline - 20.0
    if time.time() < de_deadline:
        try:
            # We use a wrapper to check time
            class TimeExceededException(Exception):
                pass
            
            def de_obj_with_time(params):
                if time.time() > de_deadline:
                    raise TimeExceededException()
                return _de_objective(params, N)
            
            # Run DE with limited iterations
            # popsize=10 means 10*N = 260 candidates
            # maxiter=50 means 50 generations
            result_de = differential_evolution(
                de_obj_with_time,
                bounds_de,
                seed=42,
                maxiter=30,
                popsize=15,
                tol=1e-6,
                init='sobol',
                updating='immediate'
            )
            
            if not result_de.success and 'TimeExceeded' not in str(result_de.message):
                 # It might have just finished or been cut off, check if we have a result
                 pass
                 
            opt_params_de = result_de.x
            xy_de = opt_params_de[:2*N].reshape(N, 2)
            r_de = np.maximum(opt_params_de[2*N:], 0.001)
            
            total_de, valid_de = _eval_packing(xy_de, r_de)
            if valid_de and total_de > best_total:
                best_total = total_de
                best_valid = True
                best_xy = xy_de
                best_r = r_de
                
        except TimeExceededException:
            pass
        except Exception:
            pass
    
    # 3. Final Polish on the best result found
    if best_valid and time.time() < deadline - 5.0:
        p_xy, p_r, p_total, p_valid = _local_refine(best_xy, best_r, deadline - 2.0)
        if p_valid and p_total > best_total:
            best_total = p_total
            best_xy = p_xy
            best_r = p_r

    # Fallback if no valid result
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
        return result

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
    
    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
            
    return result
# EVOLVE-BLOCK-END
