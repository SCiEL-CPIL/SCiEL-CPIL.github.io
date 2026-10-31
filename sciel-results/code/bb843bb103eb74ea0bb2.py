"""
Combined approach: Generate all 11-polyhexes, evaluate minimal enclosing hexagon 
by optimizing outer rotation, and refine top candidates with local continuous 
optimization allowing slight position adjustments.
"""
import numpy as np
import math
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11

# --- Geometry Helpers ---

def _hex_vertices(cx, cy, s, ang_deg):
    """Return 6 vertices of a regular hexagon (flat-top at ang=0)."""
    a = math.radians(ang_deg)
    # Flat-top hexagon vertices are at 0, 60, 120, 180, 240, 300 degrees
    angs = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + s * np.cos(angs), cy + s * np.sin(angs)])

def _all_vertices(positions, angles, s=1.0):
    """Return all 6*n vertices as (6n, 2) array."""
    n = len(positions)
    v = np.empty((n * 6, 2))
    for i in range(n):
        v[i*6:(i+1)*6] = _hex_vertices(positions[i][0], positions[i][1], s, angles[i])
    return v

def _min_enclosing_hex_side(positions, angles, phi_rad, center=(0.0, 0.0)):
    """
    Compute the exact side length of the smallest regular hexagon with rotation phi_rad
    and center `center` that encloses all given unit hexagons.
    
    The outer hexagon has 6 face normals at angles phi_rad + (2k+1)*pi/6 for k=0..5.
    Side length S is determined by: max_proj / (sqrt(3)/2)
    where max_proj is the maximum projection of any vertex onto any face normal.
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

# --- Polyhex Generation ---

def _axial_to_xy(q, r):
    """Convert axial coordinates to Cartesian for flat-top hexagons, side 1."""
    s3 = math.sqrt(3)
    return (1.5 * q, s3 * (r + q / 2.0))

def _generate_polyhexes(n_cells=11):
    """
    Generate all free polyhexes with n_cells cells using BFS.
    Returns a list of sets of axial coordinates (normalized).
    """
    directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    
    def normalize(cells):
        cells = list(cells)
        min_q = min(c[0] for c in cells)
        min_r = min(c[1] for c in cells)
        translated = [(c[0] - min_q, c[1] - min_r) for c in cells]
        return frozenset(translated)
    
    def canonical(cells):
        """Get canonical form by trying all 6 rotations and 6 reflections."""
        cells_list = list(cells)
        forms = set()
        for rot in range(6):
            c = list(cells_list)
            for _ in range(rot):
                c = [(-r, q + r) for q, r in c]
            forms.add(normalize(c))
            c_ref = [(-q - r, r) for q, r in c]
            forms.add(normalize(c_ref))
        return min(forms)
    
    seen_canonical = set()
    polyhexes = []
    initial = frozenset([(0, 0)])
    
    def bfs_expand(current_set, depth, target):
        if depth == target:
            canon = canonical(current_set)
            if canon not in seen_canonical:
                seen_canonical.add(canon)
                polyhexes.append(set(canon))
            return
        
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

def _evaluate_polyhex(positions, angles, deadline):
    """
    Evaluate a polyhex by finding the optimal outer rotation.
    Returns (best_side, best_phi_rad).
    """
    # Coarse search for optimal rotation
    best_side = float('inf')
    best_phi = 0.0
    # The side length function is periodic with 60 degrees.
    # We search in [0, pi/3).
    # Use a few points to find the minimum, then refine.
    phis = np.linspace(0, math.pi/3, 12)
    for phi in phis:
        if time.time() > deadline:
            break
        s = _min_enclosing_hex_side(positions, angles, phi)
        if s < best_side:
            best_side = s
            best_phi = phi
    
    # Fine-tune rotation with golden section search
    a, b = 0, math.pi/3
    if best_phi > 0 and best_phi < math.pi/3:
        # Refine around best_phi
        a = max(0, best_phi - math.pi/36)
        b = min(math.pi/3, best_phi + math.pi/36)
    
    # Simple Nelder-Mead on phi
    def obj_phi(phi):
        return _min_enclosing_hex_side(positions, angles, phi[0])
    
    x0 = np.array([best_phi])
    try:
        res = minimize(obj_phi, x0, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 50})
        if res.fun < best_side:
            best_side = res.fun
            best_phi = res.x[0]
    except Exception:
        pass
        
    return best_side, best_phi

def _refine_arrangement(positions, angles, phi_init, deadline):
    """
    Refine positions and angles of inner hexagons and outer rotation.
    Variables: [x1, y1, a1, ..., xN, yN, aN, phi_outer]
    Objective: Outer hex side length + Penalty for overlaps.
    """
    n = len(positions)
    # Flatten: 2*N positions + N angles + 1 outer rotation
    x0 = np.zeros(2*n + n + 1)
    for i in range(n):
        x0[2*i] = positions[i][0]
        x0[2*i+1] = positions[i][1]
        x0[2*n + i] = angles[i]
    x0[3*n] = phi_init
    
    def objective(x):
        if time.time() > deadline:
            return float('inf')
        
        pos = [(x[2*i], x[2*i+1]) for i in range(n)]
        ang = [x[2*n + i] for i in range(n)]
        phi = x[3*n]
        
        # Check overlaps
        penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                gap = _sat_gap(pos[i], ang[i], pos[j], ang[j])
                if gap < 0:
                    # Quadratic penalty for overlap
                    penalty += 1000.0 * gap * gap
        
        side = _min_enclosing_hex_side(pos, ang, phi)
        return side + penalty
    
    try:
        # Use Nelder-Mead as it doesn't require gradients and handles non-smooth penalties well
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 200, 'adaptive': True})
        
        # Extract results
        if res.fun < float('inf') and res.fun < 100: # Sanity check
            pos_opt = [(res.x[2*i], res.x[2*i+1]) for i in range(n)]
            ang_opt = [res.x[2*n + i] for i in range(n)]
            phi_opt = res.x[3*n]
            
            # Verify disjointness strictly
            if _check_disjoint(pos_opt, ang_opt):
                side_opt = _min_enclosing_hex_side(pos_opt, ang_opt, phi_opt)
                return pos_opt, ang_opt, phi_opt, side_opt
    except Exception:
        pass
        
    return positions, angles, phi_init, float('inf')

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Generate all 11-polyhexes
    polyhexes = _generate_polyhexes(N)
    
    # 2. Evaluate each polyhex to find top candidates
    candidates = []
    for i, cells in enumerate(polyhexes):
        if time.time() > deadline - 20.0:
            break
        positions = _polyhex_to_positions(cells)
        angles = [0.0] * N
        
        # Quick check: are they disjoint? (They should be, by construction, but check just in case)
        if not _check_disjoint(positions, angles):
            continue
            
        # Evaluate optimal outer rotation
        best_side, best_phi = _evaluate_polyhex(positions, angles, deadline)
        candidates.append((best_side, best_phi, positions, angles))
        
    # Sort by side length
    candidates.sort(key=lambda x: x[0])
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    
    # 3. Refine top candidates with continuous optimization
    # Take top 10 candidates for refinement
    top_k = 10
    for side, phi, pos, ang in candidates[:top_k]:
        if time.time() > deadline - 10.0:
            break
            
        # If this candidate is already worse than our best, skip? 
        # No, refinement might improve it. But if side is much larger, skip to save time.
        if side > best_side * 1.05 and best_side < float('inf'):
            continue
            
        opt_pos, opt_ang, opt_phi, opt_side = _refine_arrangement(pos, ang, phi, deadline)
        
        if opt_side < best_side:
            best_side = opt_side
            best_positions = opt_pos
            best_angles = opt_ang
            best_phi = opt_phi
    
    # 4. If no candidate was refined or found, use the best from step 2
    if best_positions is None:
        if candidates:
            best_side, best_phi, best_positions, best_angles = candidates[0]
        else:
            # Fallback: linear arrangement
            best_positions = [(4.0 * k, 0.0) for k in range(-(N//2), N - N//2)]
            best_angles = [0.0] * N
            best_side = _min_enclosing_hex_side(best_positions, best_angles, 0.0)
            best_phi = 0.0

    # Add tiny epsilon for strict containment
    best_side *= (1.0 + 1e-9)
    
    inner = [[best_positions[i][0], best_positions[i][1], best_angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), best_side, math.degrees(best_phi)

# EVOLVE-BLOCK-END
