"""Combined polyhex search + continuous refinement for packing 11 unit hexagons
in a minimal outer regular hexagon."""
import numpy as np
import math
import time
from itertools import combinations

N = 11

# ---- Geometry helpers ------------------------------------------------------

def _hex_vertices(cx, cy, s, ang_deg):
    """Return 6 vertices of a regular hexagon (flat-top at ang=0)."""
    a = math.radians(ang_deg)
    # Flat-top: vertices at angles 0, 60, 120, 180, 240, 300 degrees
    angs = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + s * np.cos(angs), cy + s * np.sin(angs)])

def _all_vertices(positions, angles, s=1.0):
    """Return all 6*n vertices."""
    n = len(positions)
    v = np.empty((n * 6, 2))
    for i in range(n):
        v[i*6:(i+1)*6] = _hex_vertices(positions[i][0], positions[i][1], s, angles[i])
    return v

def _min_enclosing_hex_side(positions, angles, phi_rad, center=(0.0, 0.0)):
    """
    Compute the exact side length of the smallest regular hexagon with rotation phi_rad
    and center `center` that encloses all given unit hexagons.
    
    For a regular hexagon with side length S and center at origin:
    - The apothem (distance from center to each face) is S * sqrt(3)/2
    - The 6 face normals are at angles phi_rad + (2k+1)*pi/6 for k=0..5
    - A point (x,y) is inside if for all k: (x*cos(theta_k) + y*sin(theta_k)) <= S*sqrt(3)/2
    """
    verts = _all_vertices(positions, angles, 1.0) - np.array(center)
    thetas = phi_rad + np.array([(2*k + 1) * np.pi / 6 for k in range(6)])
    max_proj = 0.0
    for t in thetas:
        proj = verts[:, 0] * math.cos(t) + verts[:, 1] * math.sin(t)
        mp = np.max(proj)
        if mp > max_proj:
            max_proj = mp
    return max_proj / (math.sqrt(3) / 2)

# ---- SAT overlap check -----------------------------------------------------

def _sat_gap(p1, a1, p2, a2, s=1.0):
    """Compute minimum gap between two hexagons using SAT. Negative means overlap."""
    v1 = _hex_vertices(p1[0], p1[1], s, a1)
    v2 = _hex_vertices(p2[0], p2[1], s, a2)
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    min_gap = float('inf')
    for e in list(edges1) + list(edges2):
        nv = np.array([-e[1], e[0]])
        n = np.linalg.norm(nv)
        if n < 1e-12:
            continue
        nv /= n
        proj1 = v1 @ nv
        proj2 = v2 @ nv
        gap = min(np.max(proj1) - np.min(proj2), np.max(proj2) - np.min(proj1))
        if gap < min_gap:
            min_gap = gap
    return min_gap

def _check_disjoint(positions, angles, tol=1e-9):
    """Check that all hexagons are pairwise disjoint (non-overlapping)."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if _sat_gap(positions[i], angles[i], positions[j], angles[j]) < -tol:
                return False
    return True

# ---- Axial coordinate helpers ----------------------------------------------

def _axial_to_xy(q, r):
    """Convert axial coordinates to Cartesian for flat-top hexagons, side 1."""
    s3 = math.sqrt(3)
    return (1.5 * q, s3 * (r + q / 2.0))

# ---- Polyhex generation via BFS on axial grid ------------------------------

def _generate_polyhexes(n_cells=11):
    """
    Generate all free polyhexes with n_cells cells using BFS.
    Returns a list of sets of axial coordinates (normalized to center at origin).
    """
    # Axial directions for flat-top hexagons
    directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    
    def normalize(cells):
        """Normalize a set of axial coords so that min q and min r are at 0."""
        cells = list(cells)
        min_q = min(c[0] for c in cells)
        min_r = min(c[1] for c in cells)
        translated = [(c[0] - min_q, c[1] - min_r) for c in cells]
        return frozenset(translated)
    
    def canonical(cells):
        """Get canonical form by trying all 6 rotations and 6 reflections (12 total)."""
        cells_list = list(cells)
        forms = set()
        for rot in range(6):
            # Rotate by rot * 60 degrees
            # In axial coords, rotation by 60 degrees: (q,r) -> (-r, q+r)
            c = list(cells_list)
            for _ in range(rot):
                c = [(-r, q + r) for q, r in c]
            forms.add(normalize(c))
            # Reflection: (q,r) -> (-q-r, r)
            c_ref = [(-q - r, r) for q, r in c]
            forms.add(normalize(c_ref))
        # Return the lexicographically smallest
        return min(forms)
    
    # BFS: start with single cell, add cells
    seen_canonical = set()
    polyhexes = []
    
    # Start with 1-cell polyhex
    initial = frozenset([(0, 0)])
    
    def bfs_expand(current_set, depth, target):
        """Recursively expand a polyhex by adding one cell at a time."""
        if depth == target:
            canon = canonical(current_set)
            if canon not in seen_canonical:
                seen_canonical.add(canon)
                polyhexes.append(set(canon))
            return
        
        # Find all boundary positions (adjacent to existing cells but not occupied)
        boundary = set()
        for q, r in current_set:
            for dq, dr in directions:
                pos = (q + dq, r + dr)
                if pos not in current_set:
                    boundary.add(pos)
        
        for pos in boundary:
            new_set = frozenset(current_set | {pos})
            bfs_expand(new_set, depth + 1, target)
    
    bfs_expand(initial, 1, n_cells)
    
    return polyhexes

def _polyhex_to_positions(cells):
    """Convert a set of axial coordinates to Cartesian positions (flat-top, side 1)."""
    return [_axial_to_xy(q, r) for q, r in cells]

# ---- Nelder-Mead refinement -------------------------------------------------

def _refine_positions(positions, angles, outer_rotation_deg, max_iter=200, tol=1e-8):
    """
    Use Nelder-Mead to refine positions and outer rotation to minimize enclosing hex side.
    """
    from scipy.optimize import minimize
    
    n = len(positions)
    
    def objective(x):
        # x contains 2n position coordinates + 1 outer rotation
        pos = []
        for i in range(n):
            pos.append((x[2*i], x[2*i+1]))
        phi = math.radians(x[2*n])
        
        # Check disjointness - if invalid, return large penalty
        if not _check_disjoint(pos, angles, tol=1e-6):
            return 1e6
        
        return _min_enclosing_hex_side(pos, angles, phi)
    
    x0 = []
    for i in range(n):
        x0.append(positions[i][0])
        x0.append(positions[i][1])
    x0.append(outer_rotation_deg)
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol})
        if result.fun < 1e5:  # Valid solution
            new_pos = []
            for i in range(n):
                new_pos.append((result.x[2*i], result.x[2*i+1]))
            new_rot = result.x[2*n]
            return new_pos, new_rot, result.fun
        else:
            return positions, outer_rotation_deg, _min_enclosing_hex_side(positions, angles, math.radians(outer_rotation_deg))
    except Exception:
        return positions, outer_rotation_deg, _min_enclosing_hex_side(positions, angles, math.radians(outer_rotation_deg))

# ---- Main construct ----------------------------------------------------------

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    sqrt3 = math.sqrt(3)
    
    # Generate polyhexes (this can be slow, so limit it)
    polyhexes = None
    if time.time() < deadline - 60.0:
        try:
            polyhexes = _generate_polyhexes(N)
        except Exception:
            polyhexes = None
    
    # Also prepare hex grid arrangements as fallback
    arrangements = []
    
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    ring2 = [(2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2)]
    
    # Option 1: 4 consecutive from ring2
    for start in range(12):
        subset = [ring2[(start + k) % 12] for k in range(4)]
        axial_coords = [(0,0)] + ring1 + subset
        positions = [_axial_to_xy(q, r) for q, r in axial_coords]
        arrangements.append(positions)
    
    # Option 2: Symmetric selections (opposite pairs)
    for i in range(6):
        for j in range(i+1, 6):
            subset = [ring2[i], ring2[i+6], ring2[j], ring2[j+6]]
            axial_coords = [(0,0)] + ring1 + subset
            positions = [_axial_to_xy(q, r) for q, r in axial_coords]
            arrangements.append(positions)
    
    # Add polyhex-based arrangements
    if polyhexes:
        for cells in polyhexes:
            if time.time() > deadline - 30.0:
                break
            positions = _polyhex_to_positions(cells)
            arrangements.append(positions)
    
    # Evaluate all arrangements and find best few
    candidates = []
    for arr in arrangements:
        if time.time() > deadline - 20.0:
            break
        angles = [0.0] * N
        if not _check_disjoint(arr, angles):
            continue
        
        # Try multiple outer rotations
        best_side_for_arr = float('inf')
        best_rot_for_arr = 0.0
        for rot_deg in [0.0, 30.0, 15.0, 45.0]:
            side = _min_enclosing_hex_side(arr, angles, math.radians(rot_deg))
            if side < best_side_for_arr:
                best_side_for_arr = side
                best_rot_for_arr = rot_deg
        
        candidates.append((best_side_for_arr, arr, best_rot_for_arr))
    
    # Sort by side length
    candidates.sort(key=lambda x: x[0])
    
    # Refine top candidates with Nelder-Mead
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_outer_rot = 0.0
    
    for side_est, arr, rot_est in candidates[:10]:  # Top 10
        if time.time() > deadline - 15.0:
            break
        
        refined_pos, refined_rot, refined_side = _refine_positions(
            arr, [0.0] * N, rot_est, max_iter=100
        )
        
        if refined_side < best_side:
            best_side = refined_side
            best_positions = refined_pos
            best_angles = [0.0] * N
            best_outer_rot = refined_rot
    
    # Fallback if no valid candidates
    if best_positions is None:
        # Linear arrangement
        best_positions = [(4.0 * k, 0.0) for k in range(-(N//2), N - N//2)]
        best_angles = [0.0] * N
        best_outer_rot = 0.0
        best_side = _min_enclosing_hex_side(best_positions, best_angles, 0.0)
    
    # Add tiny epsilon for strict containment
    best_side *= (1.0 + 1e-9)
    
    inner = [[best_positions[i][0], best_positions[i][1], best_angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), best_side, best_outer_rot
