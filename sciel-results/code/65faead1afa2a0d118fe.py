"""Circle packing in unit square - 26 circles maximizing sum of radii.
Strategy: Combinatorial Selection via Maximum Weight Independent Set (MWIS) on a Conflict Graph.
1. Generate a diverse library of candidate circles (grid-based, random, corner-biased).
2. Build a conflict graph where nodes are candidates and edges represent overlapping circles.
3. Use a randomized greedy algorithm to find a large independent set of exactly 26 circles.
4. Polish the selected set using SLSQP to refine positions and radii to the boundary of the feasible region.
This approach avoids getting stuck in continuous local optima by treating the problem as a discrete selection task.
"""

import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 26
# EVOLVE-BLOCK-START

def _is_valid(circles, tol=1e-7):
    n = len(circles)
    if n != N:
        return False
    for i in range(n):
        x, y, r = circles[i]
        if r < -tol:
            return False
        if x - r < -tol or x + r > 1.0 + tol:
            return False
        if y - r < -tol or y + r > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - tol:
                return False
    return True


def _post_process(circles, tol=1e-9):
    out = []
    for x, y, r in circles:
        r = max(0.0, r - tol)
        x = min(1.0 - r, max(r, x))
        y = min(1.0 - r, max(r, y))
        out.append((x, y, r))
    return out


def _sum_radii(circles):
    return sum(c[2] for c in circles)


def _generate_candidates():
    """Generate a diverse set of candidate circles."""
    candidates = []
    
    # 1. Grid-based candidates with varying radii
    # Fine grid to allow many positions
    grid_res = 20
    step = 1.0 / (grid_res - 1)
    for i in range(grid_res):
        for j in range(grid_res):
            x = i * step
            y = j * step
            # Max radius at this position is distance to nearest boundary
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if max_r < 0.01:
                continue
            # Try several radii as fractions of max_r
            for frac in [1.0, 0.8, 0.6, 0.4, 0.2]:
                r = max_r * frac
                if r > 0.005:
                    candidates.append((x, y, r))
                    
    # 2. Random candidates
    random.seed(42)
    for _ in range(2000):
        x = random.uniform(0.0, 1.0)
        y = random.uniform(0.0, 1.0)
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if max_r < 0.01:
            continue
        r = max_r * random.uniform(0.5, 1.0)
        candidates.append((x, y, r))
        
    # 3. Corner-focused candidates (often good for packing)
    corners = [(0.05, 0.05), (0.95, 0.05), (0.05, 0.95), (0.95, 0.95),
               (0.5, 0.05), (0.5, 0.95), (0.05, 0.5), (0.95, 0.5)]
    for cx, cy in corners:
        max_r = min(cx, 1.0 - cx, cy, 1.0 - cy)
        for frac in [1.0, 0.9, 0.8, 0.7]:
            r = max_r * frac
            if r > 0.005:
                candidates.append((cx, cy, r))

    # Deduplicate and store as array
    # We keep them as list of tuples for now, convert to array for vectorized ops
    return np.array(candidates)

def _build_conflict_graph(candidates):
    """
    Build a boolean matrix M where M[i,j] is True if candidate i and j overlap.
    Also handles boundary violations (though candidates are generated to be inside).
    """
    n_cand = len(candidates)
    # Vectorized distance calculation
    # To save memory, we might chunk this if n_cand is huge, but 2000-5000 is okay.
    # Let's check size. Grid 20x20=400 * 5 fracs = 2000. Random 2000. Total ~4000.
    # 4000x4000 matrix is 16M booleans ~ 16MB. Acceptable.
    
    coords = candidates[:, :2]
    radii = candidates[:, 2]
    
    # Compute pairwise distances
    # cdist is efficient
    dists = cdist(coords, coords)
    
    # Overlap condition: dist < r_i + r_j
    # r_i + r_j can be computed as radii[:, None] + radii[None, :]
    sum_radii = radii[:, None] + radii[None, :]
    
    conflict = dists < sum_radii
    
    # Remove self-conflicts (diagonal)
    np.fill_diagonal(conflict, False)
    
    return conflict

def _greedy_mwis(conflict, weights, n_select, n_restarts=50, time_limit=10.0):
    """
    Find a set of n_select indices that are mutually non-conflicting, maximizing sum of weights.
    Uses randomized greedy with restarts.
    """
    n_cand = len(weights)
    best_val = -1
    best_indices = None
    start_time = time.time()
    
    for restart in range(n_restarts):
        if time.time() > start_time + time_limit:
            break
            
        # Randomly shuffle candidate order, biased by weight? 
        # Or just random permutation.
        order = np.arange(n_cand)
        np.random.shuffle(order)
        
        # To improve quality, we can sort by weight descending with some noise
        # Let's try: sort by (weight + noise)
        noise = np.random.normal(0, 0.1, n_cand)
        scores = weights + noise
        order = np.argsort(scores)[::-1]
        
        selected = []
        selected_set = set()
        current_val = 0
        
        for idx in order:
            if len(selected) == n_select:
                break
            if idx in selected_set:
                continue
            # Check if idx conflicts with any selected
            # conflict[idx, selected] should be all False
            if selected:
                if np.any(conflict[idx, selected]):
                    continue
            
            selected.append(idx)
            selected_set.add(idx)
            current_val += weights[idx]
            
        if len(selected) == n_select:
            if current_val > best_val:
                best_val = current_val
                best_indices = selected
                
    return best_indices, best_val

def _slsqp_polish(circles):
    """SLSQP polish with hard inequality constraints."""
    params = np.array([val for c in circles for val in c])
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.001, 0.5)] * N
    
    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[3*i] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[3*i+1] - p[3*i+2]})
    
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (p[3*i]-p[3*j])**2 + (p[3*i+1]-p[3*j+1])**2 - (p[3*i+2]+p[3*j+2])**2
            })
    
    def objective(p):
        return -sum(p[3*i+2] for i in range(N))
    
    try:
        result = minimize(
            objective, params,
            method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        if result.success:
            return [(result.x[3*i], result.x[3*i+1], result.x[3*i+2]) for i in range(N)]
    except Exception:
        pass
    return circles

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Generate Candidates
    candidates = _generate_candidates()
    n_cand = len(candidates)
    
    # 2. Build Conflict Graph
    conflict = _build_conflict_graph(candidates)
    weights = candidates[:, 2]  # Radius is the weight
    
    # 3. Solve MWIS
    best_indices, best_val = _greedy_mwis(conflict, weights, N, n_restarts=100, time_limit=30.0)
    
    if best_indices is None:
        # Fallback: Grid init
        side = int(np.ceil(np.sqrt(N)))
        best_circles = []
        for k in range(N):
            i = k % side
            j = k // side
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.4 / side
            best_circles.append((x, y, r))
    else:
        best_circles = [tuple(candidates[i]) for i in best_indices]
        
    # 4. Polish the selected set
    if time.time() < deadline - 20.0:
        polished = _slsqp_polish(best_circles)
        if _is_valid(polished, tol=1e-7):
            if _sum_radii(polished) > _sum_radii(best_circles):
                best_circles = polished
                
    # 5. Try a second round with a different MWIS strategy or more restarts if time permits
    if time.time() < deadline - 40.0:
        # Try again with more restarts
        best_indices2, best_val2 = _greedy_mwis(conflict, weights, N, n_restarts=100, time_limit=30.0)
        if best_indices2 is not None:
            circles2 = [tuple(candidates[i]) for i in best_indices2]
            if _sum_radii(circles2) > _sum_radii(best_circles):
                best_circles = circles2
                if time.time() < deadline - 20.0:
                    polished2 = _slsqp_polish(best_circles)
                    if _is_valid(polished2, tol=1e-7):
                        if _sum_radii(polished2) > _sum_radii(best_circles):
                            best_circles = polished2

    # 6. Local Search / Refinement
    # Simple hill climbing: try to swap one circle with a better non-conflicting candidate
    if time.time() < deadline - 10.0:
        current_val = _sum_radii(best_circles)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # For each circle in the current set, try to replace it with a better one
            # This is expensive O(N * N_cand * N). 
            # Let's do a simpler version: just re-run SLSQP with slight perturbations
            # Or just trust SLSQP.
            # Let's try a few random perturbations of the current best
            for _ in range(5):
                if time.time() > deadline - 5.0:
                    break
                perturbed = []
                for x, y, r in best_circles:
                    x += random.uniform(-0.01, 0.01)
                    y += random.uniform(-0.01, 0.01)
                    r += random.uniform(-0.005, 0.005)
                    perturbed.append((x, y, r))
                polished_p = _slsqp_polish(perturbed)
                if _is_valid(polished_p, tol=1e-7):
                    p_val = _sum_radii(polished_p)
                    if p_val > current_val:
                        current_val = p_val
                        best_circles = polished_p
                        improved = True

    best_circles = _post_process(best_circles)
    
    if not _is_valid(best_circles):
        # Emergency fallback
        side = int(np.ceil(np.sqrt(N)))
        best_circles = []
        for k in range(N):
            i = k % side
            j = k // side
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.4 / side
            best_circles.append((x, y, r))
        best_circles = _post_process(best_circles)

    return np.array(best_circles)

# EVOLVE-BLOCK-END
