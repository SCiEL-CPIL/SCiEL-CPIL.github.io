# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
from scipy.optimize import minimize

TOL = 1e-7
N = 10
DEADLINE_FRACTION = 0.95


def _check_validity(centers, sides):
    """Return (is_valid, score) for a candidate packing."""
    if centers.shape != (N, 2) or sides.shape != (N,):
        return False, 0.0
    if not (np.all(np.isfinite(centers)) and np.all(np.isfinite(sides))):
        return False, 0.0
    if np.any(sides <= 0):
        return False, 0.0
    # Boundary check
    for i in range(N):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s / 2 < -TOL or cx + s / 2 > 1 + TOL:
            return False, 0.0
        if cy - s / 2 < -TOL or cy + s / 2 > 1 + TOL:
            return False, 0.0
    # Overlap check
    for i in range(N):
        for j in range(i + 1, N):
            t = (sides[i] + sides[j]) / 2 - TOL
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False, 0.0
    score = float(np.sum(sides))
    return True, score


def _grid_packing(n, k, side):
    """Baseline grid packing."""
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    return centers, sides


def _separation_graph(centers, sides, tol=TOL):
    """
    Determine which pairs of squares are separated in x, y, or both.
    Returns a set of (i, j) that are 'separated' (non-overlapping in at least one axis).
    Pairs NOT in this set are considered 'close' and need explicit separation constraints.
    """
    sep_pairs = set()
    for i in range(N):
        for j in range(i + 1, N):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2
            # If separated in x or y (with a small margin), they are safe
            if dx >= t + 1e-9 or dy >= t + 1e-9:
                sep_pairs.add((i, j))
    return sep_pairs


def _solve_fixed_graph(sep_pairs, centers_init, sides_init, time_limit=30.0):
    """
    Solve the optimization problem with a fixed set of separation constraints.
    sep_pairs: set of (i,j) that are allowed to be 'separated' (i.e., we don't need to enforce
              non-overlap explicitly because they are already well-separated in the init).
              For pairs NOT in sep_pairs, we enforce non-overlap via disjunctive constraints.
    """
    start_time = time.time()

    # Variables: x_0..x_9, y_0..y_9, s_0..s_9 (30 vars)
    def unpack(v):
        cx = v[:N]
        cy = v[N:2*N]
        s = v[2*N:]
        return cx, cy, s

    # Objective: maximize sum(s) -> minimize -sum(s)
    def objective(v):
        _, _, s = unpack(v)
        return -np.sum(s)

    # Inequality constraints: g(v) >= 0
    # 1. Boundary: cx - s/2 >= 0, cx + s/2 <= 1, etc.
    # 2. Non-overlap for 'close' pairs (not in sep_pairs)
    #    For each close pair (i,j), we need:
    #    (cx_i - s_i/2 >= cx_j + s_j/2) OR (cx_j - s_j/2 >= cx_i + s_i/2) OR
    #    (cy_i - s_i/2 >= cy_j + s_j/2) OR (cy_j - s_j/2 >= cy_i + s_i/2)
    #    This is a disjunction. SLSQP handles single inequalities, not disjunctions.
    #    Strategy: Use the initial configuration to determine the 'active' separation direction
    #    for each close pair, and fix it. If the solution violates it, we'd need to re-assign.
    #    For now, we enforce the specific direction that holds in the initial guess.

    constraints = []

    # Boundary constraints
    for i in range(N):
        # x lower: cx_i - s_i/2 >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]/2})
        # x upper: 1 - (cx_i + s_i/2) >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[i] - v[2*N+i]/2})
        # y lower
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]/2})
        # y upper
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[N+i] - v[2*N+i]/2})

    # Determine active separation directions for close pairs
    cx_init, cy_init, s_init = unpack(np.concatenate([centers_init, sides_init]))
    # Wait, unpack expects full vector. Let's just use centers_init and sides_init directly.
    
    close_pairs = []
    for i in range(N):
        for j in range(i+1, N):
            if (i, j) not in sep_pairs:
                close_pairs.append((i, j))
                
    # For each close pair, determine the separation axis and direction from init
    # If init is feasible, at least one direction must hold. We pick the one with the largest margin.
    for (i, j) in close_pairs:
        dx = centers_init[i,0] - centers_init[j,0]
        dy = centers_init[i,1] - centers_init[j,1]
        # Possible constraints:
        # 1. i left of j: cx_i + s_i/2 <= cx_j - s_j/2  => cx_j - cx_i - (s_i+s_j)/2 >= 0
        # 2. i right of j: cx_i - s_i/2 >= cx_j + s_j/2  => cx_i - cx_j - (s_i+s_j)/2 >= 0
        # 3. i below j: cy_i + s_i/2 <= cy_j - s_j/2  => cy_j - cy_i - (s_i+s_j)/2 >= 0
        # 4. i above j: cy_i - s_i/2 >= cy_j + s_j/2  => cy_i - cy_j - (s_i+s_j)/2 >= 0
        
        # Calculate margins in init
        m1 = (centers_init[j,0] - centers_init[i,0]) - (sides_init[i] + sides_init[j])/2
        m2 = (centers_init[i,0] - centers_init[j,0]) - (sides_init[i] + sides_init[j])/2
        m3 = (centers_init[j,1] - centers_init[i,1]) - (sides_init[i] + sides_init[j])/2
        m4 = (centers_init[i,1] - centers_init[j,1]) - (sides_init[i] + sides_init[j])/2
        
        # Pick the direction with the largest margin (most slack)
        # If all are negative, the init is infeasible for this graph, but we proceed anyway.
        margins = [m1, m2, m3, m4]
        max_idx = np.argmax(margins)
        
        if max_idx == 0:
            # i left of j
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i, j=j: v[j] - v[i] - (v[2*N+i] + v[2*N+j])/2})
        elif max_idx == 1:
            # i right of j
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i, j=j: v[i] - v[j] - (v[2*N+i] + v[2*N+j])/2})
        elif max_idx == 2:
            # i below j
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i, j=j: v[N+j] - v[N+i] - (v[2*N+i] + v[2*N+j])/2})
        else:
            # i above j
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i, j=j: v[N+i] - v[N+j] - (v[2*N+i] + v[2*N+j])/2})

    # Bounds
    # cx in [0, 1], cy in [0, 1], s in [0.01, 1.0]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # cx
    for i in range(N):
        bounds.append((0.0, 1.0)) # cy
    for i in range(N):
        bounds.append((0.01, 1.0)) # s

    x0 = np.concatenate([centers_init[:,0], centers_init[:,1], sides_init])

    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': 200, 'ftol': 1e-9, 'disp': False})
        
        if res.success:
            cx, cy, s = unpack(res.x)
            centers = np.stack([cx, cy], axis=1)
            valid, score = _check_validity(centers, s)
            if valid:
                return centers, s, score
    except Exception:
        pass
    
    return None, None, 0.0


def squares_in_square10() -> tuple:
    start_time = time.time()
    deadline = start_time + 630 * DEADLINE_FRACTION

    # 1. Get a good initial packing (grid)
    k = int(np.ceil(np.sqrt(N)))
    side = 1.0 / k
    centers, sides = _grid_packing(N, k, side)
    
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_score = float(np.sum(sides))
    valid, _ = _check_validity(best_centers, best_sides)
    if not valid:
        # Fallback to a very safe small packing
        best_sides = np.full(N, 0.1)
        best_centers = np.array([[0.1, 0.1], [0.3, 0.1], [0.5, 0.1], [0.7, 0.1], [0.9, 0.1],
                                 [0.1, 0.3], [0.3, 0.3], [0.5, 0.3], [0.7, 0.3], [0.9, 0.3]])
        best_score = float(np.sum(best_sides))

    # 2. Iterative improvement loop
    # We will try to improve the packing by:
    # a) Determining the current separation graph
    # b) Solving the optimization with that graph fixed
    # c) If the solution is valid and better, update best
    # d) Repeat with the new configuration to potentially discover new separation patterns

    for iteration in range(50):
        if time.time() > deadline - 60: # Leave 60s buffer
            break

        # Determine separation graph from current best
        sep_pairs = _separation_graph(best_centers, best_sides)
        
        # Solve with this graph
        new_centers, new_sides, new_score = _solve_fixed_graph(
            sep_pairs, best_centers, best_sides, 
            time_limit=30.0
        )
        
        if new_score is not None and new_score > best_score:
            best_centers = new_centers
            best_sides = new_sides
            best_score = new_score
        else:
            # If no improvement, try perturbing the initial guess slightly to escape local optima
            # or break if we've been stuck for a while
            pass

    # Final polish: try a few random restarts if time permits
    if time.time() < deadline - 30:
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            # Random perturbation of the best solution
            perturbed_centers = best_centers + np.random.randn(N, 2) * 0.01
            perturbed_sides = best_sides + np.random.randn(N) * 0.01
            perturbed_sides = np.clip(perturbed_sides, 0.01, 1.0)
            
            sep_pairs = _separation_graph(perturbed_centers, perturbed_sides)
            new_centers, new_sides, new_score = _solve_fixed_graph(
                sep_pairs, perturbed_centers, perturbed_sides,
                time_limit=20.0
            )
            if new_score is not None and new_score > best_score:
                best_centers = new_centers
                best_sides = new_sides
                best_score = new_score

    # Ensure output is valid
    valid, score = _check_validity(best_centers, best_sides)
    if not valid:
        # Emergency fallback
        best_centers, best_sides = _grid_packing(N, k, side)
        best_score = float(np.sum(best_sides))

    return best_centers, best_sides


# EVOLVE-BLOCK-END
