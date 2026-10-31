"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math
from itertools import combinations, product

def _hex_vertices_array(cx, cy, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon as (6,2) array."""
    a = np.radians(angle_deg)
    angles = a + np.pi / 6 + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _all_vertices(positions, angles, side=1.0):
    """Get all vertices of all hexagons as (N*6, 2) array."""
    n = len(positions)
    verts = np.zeros((n * 6, 2))
    for i in range(n):
        verts[i*6:(i+1)*6] = _hex_vertices_array(positions[i][0], positions[i][1], side, angles[i])
    return verts

def _min_enclosing_hex_side(positions, angles, outer_center=(0.0, 0.0), outer_rotation_deg=0.0):
    """Compute minimum side length of outer hexagon centered at outer_center with given rotation
    to contain all inner hexagons."""
    verts = _all_vertices(positions, angles, 1.0)
    verts = verts - np.array(outer_center)
    phi = np.radians(outer_rotation_deg)
    thetas = phi + np.array([(2*k+1)*np.pi/6 for k in range(6)])
    max_proj = 0.0
    for theta in thetas:
        proj = verts[:, 0] * np.cos(theta) + verts[:, 1] * np.sin(theta)
        mp = np.max(proj)
        if mp > max_proj:
            max_proj = mp
    return max_proj / (np.sqrt(3) / 2)

def _sat_gap(p1, a1, p2, a2, side=1.0):
    """Return the minimum gap between two hexagons using SAT.
    Positive gap means separated, negative means overlapping.
    Returns a large positive number if they are far apart (optimization shortcut)."""
    v1 = _hex_vertices_array(p1[0], p1[1], side, a1)
    v2 = _hex_vertices_array(p2[0], p2[1], side, a2)
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    min_gap = float('inf')
    for e in list(edges1) + list(edges2):
        n_vec = np.array([-e[1], e[0]])
        norm = np.linalg.norm(n_vec)
        if norm < 1e-12:
            continue
        n_vec = n_vec / norm
        proj1 = np.dot(v1, n_vec)
        proj2 = np.dot(v2, n_vec)
        gap = min(np.max(proj1) - np.min(proj2), np.max(proj2) - np.min(proj1))
        if gap < min_gap:
            min_gap = gap
    return min_gap

def _check_all_disjoint(positions, angles, tol=1e-9):
    """Check all pairs are disjoint (non-overlapping)."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            gap = _sat_gap(positions[i], angles[i], positions[j], angles[j])
            if gap < -tol:  # negative gap means overlap
                return False
    return True

def _objective_and_grad(x, angles, outer_rotation_deg):
    """Objective: outer hex side length. x is flattened positions (2N,)."""
    positions = [(x[2*i], x[2*i+1]) for i in range(N)]
    side = _min_enclosing_hex_side(positions, angles, (0.0, 0.0), outer_rotation_deg)
    # Numerical gradient is expensive; use finite differences via scipy
    return side

def _optimize_arrangement(positions, angles, outer_rotation_deg, deadline, max_iter=200):
    """Run local optimization to reduce outer hex side length, maintaining disjointness.
    Returns (optimized_positions, optimized_side) or (positions, original_side) if no improvement."""
    original_side = _min_enclosing_hex_side(positions, angles, (0.0, 0.0), outer_rotation_deg)
    x0 = np.array([p[0] for p in positions] + [p[1] for p in positions])
    # Actually interleave: x = [x0, y0, x1, y1, ...]
    x0 = np.zeros(2 * N)
    for i in range(N):
        x0[2*i] = positions[i][0]
        x0[2*i+1] = positions[i][1]
    
    def obj(x):
        if time.time() > deadline:
            return float('inf')
        positions = [(x[2*i], x[2*i+1]) for i in range(N)]
        # Hard constraint: if any overlap, return large penalty
        for i in range(N):
            for j in range(i + 1, N):
                gap = _sat_gap(positions[i], angles[i], positions[j], angles[j])
                if gap < -1e-8:
                    return 1e6 + original_side
        return _min_enclosing_hex_side(positions, angles, (0.0, 0.0), outer_rotation_deg)
    
    # Try SLSQP with a few iterations
    try:
        result = minimize(obj, x0, method='SLSQP', options={'maxiter': max_iter, 'ftol': 1e-10})
        if result.success or (result.fun < original_side and result.fun < 1e5):
            opt_positions = [(result.x[2*i], result.x[2*i+1]) for i in range(N)]
            # Verify disjointness strictly
            if _check_all_disjoint(opt_positions, angles):
                opt_side = _min_enclosing_hex_side(opt_positions, angles, (0.0, 0.0), outer_rotation_deg)
                if opt_side < original_side:
                    return opt_positions, opt_side
    except Exception:
        pass
    return positions, original_side

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    sqrt3 = math.sqrt(3)
    
    def axial_to_xy(q, r):
        return (1.5 * q, sqrt3 * (r + q / 2.0))
    
    arrangements = []
    
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    ring2 = [(2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2)]
    
    # Option 1: 4 consecutive from ring2
    for start in range(12):
        subset = [ring2[(start + k) % 12] for k in range(4)]
        axial_coords = [(0,0)] + ring1 + subset
        positions = [axial_to_xy(q, r) for q, r in axial_coords]
        arrangements.append(positions)
    
    # Option 2: Symmetric selections (opposite pairs)
    for i in range(6):
        for j in range(i+1, 6):
            subset = [ring2[i], ring2[i+6], ring2[j], ring2[j+6]]
            axial_coords = [(0,0)] + ring1 + subset
            positions = [axial_to_xy(q, r) for q, r in axial_coords]
            arrangements.append(positions)
    
    # Option 3: All combinations of 4 from ring2 (for more diversity)
    for combo in combinations(range(12), 4):
        subset = [ring2[i] for i in combo]
        axial_coords = [(0,0)] + ring1 + subset
        positions = [axial_to_xy(q, r) for q, r in axial_coords]
        arrangements.append(positions)
    
    # Deduplicate
    unique_arrangements = []
    seen = set()
    for arr in arrangements:
        key = tuple((round(p[0], 6), round(p[1], 6)) for p in arr)
        if key not in seen:
            seen.add(key)
            unique_arrangements.append(arr)
    arrangements = unique_arrangements
    
    # Sort arrangements by their initial outer hex side (for both rotations) to prioritize good seeds
    def arr_score(arr):
        angles = [0.0] * N
        if not _check_all_disjoint(arr, angles):
            return float('inf')
        s0 = _min_enclosing_hex_side(arr, angles, (0.0, 0.0), 0.0)
        s30 = _min_enclosing_hex_side(arr, angles, (0.0, 0.0), 30.0)
        return min(s0, s30)
    
    scored = []
    for arr in arrangements:
        if time.time() > deadline - 20.0:
            break
        s = arr_score(arr)
        if s < float('inf'):
            scored.append((s, arr))
    scored.sort(key=lambda x: x[0])
    
    # Keep top K seeds for optimization
    K = 15
    top_seeds = [arr for _, arr in scored[:K]]
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_outer_rotation = 0.0
    
    # Phase 1: Evaluate all valid arrangements with outer rotation 0 and 30 (no optimization)
    for arr in arrangements:
        if time.time() > deadline - 30.0:
            break
        angles = [0.0] * N
        if not _check_all_disjoint(arr, angles):
            continue
        for outer_rot in [0.0, 30.0]:
            side = _min_enclosing_hex_side(arr, angles, (0.0, 0.0), outer_rot)
            if side < best_side:
                best_side = side
                best_positions = arr[:]
                best_angles = angles[:]
                best_outer_rotation = outer_rot
    
    # Phase 2: Local optimization on top seeds with both outer rotations
    for arr in top_seeds:
        if time.time() > deadline - 10.0:
            break
        angles = [0.0] * N
        if not _check_all_disjoint(arr, angles):
            continue
        for outer_rot in [0.0, 30.0]:
            opt_positions, opt_side = _optimize_arrangement(arr, angles, outer_rot, deadline, max_iter=100)
            if opt_side < best_side:
                best_side = opt_side
                best_positions = opt_positions[:]
                best_angles = angles[:]
                best_outer_rotation = outer_rot
    
    # Phase 3: Also try fine-tuning outer rotation (every 5 degrees) on the best arrangement found
    if best_positions is not None:
        for outer_rot in np.arange(0, 360, 5.0):
            if time.time() > deadline - 5.0:
                break
            side = _min_enclosing_hex_side(best_positions, best_angles, (0.0, 0.0), outer_rot)
            if side < best_side:
                best_side = side
                best_outer_rotation = outer_rot
    
    if best_positions is None:
        # Fallback: linear arrangement
        best_positions = [(4.0 * k, 0.0) for k in range(-(N//2), N - N//2)]
        best_angles = [0.0] * N
        best_side = _min_enclosing_hex_side(best_positions, best_angles, (0.0, 0.0), 0.0)
        best_outer_rotation = 0.0
    
    # Add tiny epsilon for strict containment
    best_side *= (1.0 + 1e-9)
    
    inner = [[best_positions[i][0], best_positions[i][1], best_angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), best_side, best_outer_rotation
# EVOLVE-BLOCK-END
