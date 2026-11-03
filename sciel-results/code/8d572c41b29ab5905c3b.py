import math
import time
import numpy as np
from scipy.optimize import linprog
from itertools import combinations

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 135.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_enclosing_hex_side_lp(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with flat-top orientation offset by outer_angle_deg)
    that contains all points in verts. Uses LP to find optimal center and inradius.
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    A_rows = []
    b_vals = []
    
    for i in range(M):
        vi = verts[i]
        for k in range(6):
            nk = normals[k]
            A_rows.append([-nk[0], -nk[1], -1.0])
            b_vals.append(-nk[0]*vi[0] - nk[1]*vi[1])
            A_rows.append([nk[0], nk[1], -1.0])
            b_vals.append(nk[0]*vi[0] + nk[1]*vi[1])
    
    A = np.array(A_rows)
    b = np.array(b_vals)
    c_obj = np.array([0.0, 0.0, 1.0])
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.status == 0:
            cx, cy, r = res.x
            side = r * 2.0 / _SQRT3
            return cx, cy, side
    except Exception:
        pass
    
    # Fallback: centroid-based
    cx, cy = np.mean(verts, axis=0)
    rel = verts - np.array([cx, cy])
    projections = rel @ normals.T
    max_proj = np.max(projections, axis=0)
    min_proj = np.min(projections, axis=0)
    r = np.max(np.maximum(max_proj, -min_proj))
    if r < 0: r = 0
    side = r * 2.0 / _SQRT3
    return cx, cy, side

def _build_lattice_points():
    """
    Generate a set of candidate lattice points for hexagon centers.
    Uses a hexagonal lattice with spacing sqrt(3) (touching unit hexagons).
    We generate points in a 3x3 grid of hexagonal cells around the origin.
    """
    sqrt3 = _SQRT3
    # Scale slightly > 1 to ensure strict disjointness (no overlap)
    scale = 1.001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    points = []
    # Generate a 5x5 grid of lattice points to have enough candidates
    # Indices i, j from -2 to 2
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            points.append((x, y))
    
    return points

def _evaluate_subset(indices, points, outer_angles):
    """
    Evaluate a subset of indices from points.
    Returns the minimum enclosing hexagon side length.
    """
    selected_points = [points[i] for i in indices]
    # All inner hexagons have angle 0 (or 30, doesn't matter much for lattice aligned, 
    # but let's stick to 0 for consistency with lattice geometry if we want).
    # Actually, the current program uses 30.0. Let's use 30.0 to match previous bests.
    inner_angle = 30.0
    all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in selected_points])
    
    best_side = float('inf')
    for outer_angle_deg in outer_angles:
        res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
    
    return best_side

def _refine_positions(positions, outer_angle_deg):
    """
    Refine positions slightly? 
    For lattice points, the positions are fixed relative to each other.
    However, we can optimize the outer hexagon center and angle more precisely.
    This function just runs the LP with a finer angle search around the given angle.
    """
    inner_angle = 30.0
    all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in positions])
    
    best_side = float('inf')
    best_center = (0,0)
    
    # Fine search around the provided angle
    for ang in np.arange(outer_angle_deg - 0.5, outer_angle_deg + 0.5, 0.01):
        res = _min_enclosing_hex_side_lp(all_vertices, ang)
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
    
    return best_side, best_center, outer_angle_deg

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # 1. Generate candidate lattice points
    points = _build_lattice_points()
    num_points = len(points)
    
    # 2. Initial coarse search over subsets
    # We have 25 points, need to choose 11. C(25, 11) is too large.
    # We restrict to a smaller set or use heuristics.
    # Let's restrict to a 5x5 grid (25 points) but maybe prune points far from center?
    # Or just iterate through a smaller candidate set.
    # Let's use a 4x4 grid (16 points) for the search space? C(16, 11) = C(16, 5) = 4368. This is very fast.
    # Let's use a 5x5 grid (25 points) but limit to those within a certain radius?
    # Or just use the 16 points from a 4x4 grid (-2 to 1? or -1 to 2?).
    
    # Let's generate a specific set of 16-20 points that are likely to be part of the optimal packing.
    # The optimal packing of 11 hexagons is likely a hexagonal cluster.
    # A hexagon of side 2 has 1+6+12 = 19 hexagons.
    # A hexagon of side 1 has 7 hexagons.
    # So 11 is between.
    
    # Let's define the candidate points as a 4x4 block of the lattice.
    # Indices i, j in [-2, 1] gives 16 points.
    sqrt3 = _SQRT3
    scale = 1.001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    candidate_points = []
    for i in range(-2, 2):
        for j in range(-2, 2):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            candidate_points.append((x, y))
            
    num_candidates = len(candidate_points) # 16
    
    # Coarse outer angles to test
    coarse_angles = np.arange(0, 30.0, 1.0)
    
    best_side = float('inf')
    best_indices = None
    best_center = (0,0)
    best_outer_angle = 0.0
    
    # Iterate over all combinations of 11 indices from the 16 candidates
    # C(16, 11) = 4368 combinations.
    # For each, we test ~30 angles. 4368 * 30 = 131k LP solves.
    # Each LP solve is small (66 constraints, 3 vars). This should be fast enough.
    
    import itertools
    
    start_search_time = time.time()
    
    for indices in itertools.combinations(range(num_candidates), N):
        if time.time() > _DEADLINE - 10.0:
            break
        
        side = _evaluate_subset(indices, candidate_points, coarse_angles)
        
        if side < best_side:
            best_side = side
            best_indices = indices
            # We need to recover the center and angle that gave this side.
            # _evaluate_subset doesn't return them. Let's re-run the best angle for this subset to get details.
            # Or modify _evaluate_subset to return details.
            
            # For now, let's just store the indices. We'll refine later.
            # Actually, we need the center and angle for the final output.
            # Let's find the best angle for this specific subset now.
            selected_points = [candidate_points[i] for i in indices]
            inner_angle = 30.0
            all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in selected_points])
            
            local_best_side = float('inf')
            local_best_center = (0,0)
            local_best_angle = 0.0
            
            for ang in coarse_angles:
                res = _min_enclosing_hex_side_lp(all_vertices, ang)
                if res:
                    cx, cy, s = res
                    if s < local_best_side:
                        local_best_side = s
                        local_best_center = (cx, cy)
                        local_best_angle = ang
            
            best_center = local_best_center
            best_outer_angle = local_best_angle
            best_side = local_best_side

    # 3. Refine the best subset
    if best_indices is not None:
        best_positions = [candidate_points[i] for i in best_indices]
        
        # Refine outer angle with finer steps
        if time.time() < _DEADLINE - 15.0:
            refined_side, refined_center, refined_angle = _refine_positions(best_positions, best_outer_angle)
            if refined_side < best_side:
                best_side = refined_side
                best_center = refined_center
                best_outer_angle = refined_angle
        
        # Also try a few other subsets near the best one?
        # Maybe swap one point?
        # For now, let's stick with the best found.
        
        # Construct the inner list
        inner = [[p[0], p[1], 30.0] for p in best_positions]
        
        return inner, best_center, best_side, best_outer_angle
    
    # Fallback if no valid subset found (shouldn't happen)
    inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
    return inner, (0.0, 0.0), 40.0, 0.0
