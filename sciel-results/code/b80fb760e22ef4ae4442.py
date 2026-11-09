# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

_N = 10
_TOL = 1e-7
_DEADLINE = 590.0  # wall-clock safety margin (evaluator kills at 630s)

def _evaluate(centers, sides):
    """
    Evaluate the solution.
    Returns (objective, max_violation).
    objective: sum of sides
    max_violation: max(0, boundary_violation, overlap_violation)
    """
    cx = centers[:, 0]
    cy = centers[:, 1]
    obj = float(np.sum(sides))
    
    # Boundary violations
    lo_x = cx - sides * 0.5
    hi_x = cx + sides * 0.5
    lo_y = cy - sides * 0.5
    hi_y = cy + sides * 0.5
    
    bv = np.maximum(0.0, np.maximum(np.maximum(-lo_x, hi_x - 1.0),
                                    np.maximum(-lo_y, hi_y - 1.0)))
    max_bv = float(np.max(bv))
    
    # Pairwise overlap violations
    # |cx_i - cx_j| + tol < (s_i + s_j)/2  AND  |cy_i - cy_j| + tol < (s_i + s_j)/2
    # Violation = max(0, (s_i + s_j)/2 - tol - |cx_i - cx_j|) * max(0, (s_i + s_j)/2 - tol - |cy_i - cy_j|)
    # Or simpler: if they overlap, the violation is the amount of overlap.
    # The task says: overlap if |cx_i - cx_j| < t and |cy_i - cy_j| < t, where t = (s_i + s_j)/2 - tol.
    # So violation is 0 if either distance >= t.
    # If both < t, violation is positive.
    # Let's use the product of "excess" closeness as a smooth penalty.
    
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    t = 0.5 * (sides[:, None] + sides[None, :]) - _TOL
    
    # excess_x = max(0, t - |dx|)
    # excess_y = max(0, t - |dy|)
    # overlap = excess_x * excess_y
    # This is 0 if they don't overlap.
    
    ex_x = np.maximum(0.0, t - np.abs(dx))
    ex_y = np.maximum(0.0, t - np.abs(dy))
    ov = ex_x * ex_y
    ov = np.triu(ov, 1)
    max_ov = float(np.max(ov))
    
    return obj, max(max_bv, max_ov)

def _unpack_solution(x):
    """Unpack a flat vector of length 30 into centers and sides."""
    sides = x[:_N]
    centers = x[_N:].reshape(_N, 2)
    return centers, sides

def _pack_solution(centers, sides):
    """Pack centers and sides into a flat vector of length 30."""
    return np.concatenate([sides, centers.flatten()])

def _penalized_objective(x):
    """
    Objective for minimization (DE minimizes).
    We want to maximize sum(sides) and minimize violations.
    So minimize: -sum(sides) + penalty * violations
    """
    centers, sides = _unpack_solution(x)
    # Ensure sides are positive (DE bounds should handle this, but just in case)
    if np.any(sides <= 0):
        return 1e10
    
    obj, viol = _evaluate(centers, sides)
    # Large penalty for violations
    penalty = 1e6
    return -obj + penalty * viol

def _local_refine(centers, sides, start_time, deadline, rng):
    """
    Local search using Nelder-Mead on the penalized objective.
    """
    x0 = _pack_solution(centers, sides)
    
    # Bounds for Nelder-Mead are not directly supported in the same way, 
    # but we can clip after each step or rely on the penalty.
    # Nelder-Mead can be unstable. Let's use a simple adaptive random step or COBYLA?
    # COBYLA is good for inequality constraints.
    # Let's try COBYLA.
    
    # Define constraints for COBYLA
    # 1. sides > 0
    # 2. Boundary constraints
    # 3. Non-overlap constraints
    
    # This might be too complex to set up dynamically for every restart.
    # Let's stick to a robust random walk with annealing, but smarter.
    # Or just use differential_evolution with more iterations if time permits?
    # DE is already global. A local polish is good.
    
    # Let's implement a simple gradient-free local search with adaptive steps.
    best_obj, best_viol = _evaluate(centers, sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    
    step = 0.05
    current_obj, current_viol = best_obj, best_viol
    
    # Use a penalty function for the local search
    def local_score(c, s):
        o, v = _evaluate(c, s)
        return -o + 1e5 * v
        
    current_score = local_score(centers, sides)
    
    while time.time() < deadline - 5.0:
        # Random perturbation
        idx = rng.integers(0, 30)
        new_x = _pack_solution(centers, sides).copy()
        new_x[idx] += rng.normal(0, step)
        
        # Keep sides positive
        if idx < _N:
            new_x[idx] = max(1e-4, new_x[idx])
            
        nc, ns = _unpack_solution(new_x)
        nscore = local_score(nc, ns)
        
        if nscore < current_score: # Minimizing score
            centers = nc
            sides = ns
            current_score = nscore
            step *= 1.1
            if step > 0.5: step = 0.5
            
            o, v = _evaluate(centers, sides)
            if v < 1e-9 and o > best_obj:
                best_obj = o
                best_centers = centers.copy()
                best_sides = sides.copy()
        else:
            step *= 0.9
            if step < 1e-5: step = 1e-5
            
    return best_centers, best_sides, best_obj

def squares_in_square10() -> tuple:
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Baseline
    base_centers = np.array([
        (0.225, 0.775), (0.775, 0.775),
        (0.125, 0.125), (0.375, 0.125), (0.625, 0.125), (0.875, 0.125),
        (0.125, 0.375), (0.375, 0.375), (0.625, 0.375), (0.875, 0.375),
    ], dtype=np.float64)
    base_sides = np.array([0.45, 0.45, 0.25, 0.25, 0.25, 0.25,
                          0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    
    best_centers = base_centers.copy()
    best_sides = base_sides.copy()
    best_obj, best_viol = _evaluate(best_centers, best_sides)
    
    # Bounds for DE
    # Sides: [0.01, 0.5]
    # Centers: [0.0, 1.0]
    bounds = []
    for i in range(_N):
        bounds.append((0.01, 0.5)) # sides
    for i in range(_N):
        bounds.append((0.0, 1.0)) # cx
        bounds.append((0.0, 1.0)) # cy
        
    # DE settings
    # maxiter: 100
    # popsize: 15
    # tol: 1e-4
    # seed: varies
    
    restart = 0
    while time.time() < deadline - 20.0:
        seed = 1000 + restart
        rng = np.random.default_rng(seed)
        
        try:
            # Run DE
            res = differential_evolution(
                _penalized_objective, 
                bounds=bounds, 
                maxiter=50, 
                popsize=10, 
                tol=1e-4,
                seed=seed,
                polish=False # We will do our own polish
            )
        except Exception as e:
            # If DE fails, skip
            restart += 1
            continue
            
        if res.success or res.fun < 1e5: # If it didn't explode
            c, s = _unpack_solution(res.x)
            
            # Ensure feasibility check
            obj, viol = _evaluate(c, s)
            
            # Local refinement
            c, s, obj = _local_refine(c, s, time.time(), deadline, rng)
            
            # Final validation
            final_obj, final_viol = _evaluate(c, s)
            
            if final_viol < 1e-7:
                if final_obj > best_obj:
                    best_obj = final_obj
                    best_centers = c.copy()
                    best_sides = s.copy()
            else:
                # If local search made it infeasible, try to repair or discard
                # For now, discard if infeasible, rely on DE's best valid?
                # DE doesn't track valid best explicitly.
                # We should track best valid during DE? No, DE is black box.
                # We can check the result of DE before local search.
                # Let's re-evaluate: if DE result is valid and better, save it.
                pass
            
        restart += 1
        
    # Final clamp and validation
    sides = np.clip(best_sides, 1e-5, None)
    cx = np.clip(best_centers[:, 0], sides * 0.5, 1.0 - sides * 0.5)
    cy = np.clip(best_centers[:, 1], sides * 0.5, 1.0 - sides * 0.5)
    centers = np.stack([cx, cy], axis=1)
    
    obj, viol = _evaluate(centers, sides)
    if viol > 1e-7:
        # Fallback to baseline
        centers = base_centers
        sides = base_sides
        
    return centers, sides

# EVOLVE-BLOCK-END
