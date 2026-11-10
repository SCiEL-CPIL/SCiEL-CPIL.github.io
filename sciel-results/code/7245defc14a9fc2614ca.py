# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

def hex_vertices(cx, cy, s, a_deg):
    """Compute vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _check_validity(inner, outer, side):
    """Check if the packing is valid."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False
    if side < np.sqrt(11):
        return False
    
    ocx, ocy, oang = outer[0], outer[1], outer[2]
    outer_verts = hex_vertices(ocx, ocy, side, oang)
    
    # Shift inner hexagons so outer center is at origin
    shifted_inner = inner.copy()
    shifted_inner[:, 0] -= ocx
    shifted_inner[:, 1] -= ocy
    
    # Check edge lengths of shifted inner hexagons
    for i in range(11):
        cx, cy, ang = shifted_inner[i]
        verts = hex_vertices(cx, cy, 1.0, ang)
        for j in range(6):
            v1 = verts[j]
            v2 = verts[(j + 1) % 6]
            edge_len = np.linalg.norm(v2 - v1)
            if abs(edge_len - 1.0) > 1e-8:
                return False
    
    # Check separation between all pairs of inner hexagons
    for i in range(11):
        for j in range(i + 1, 11):
            v1 = hex_vertices(shifted_inner[i, 0], shifted_inner[i, 1], 1.0, shifted_inner[i, 2])
            v2 = hex_vertices(shifted_inner[j, 0], shifted_inner[j, 1], 1.0, shifted_inner[j, 2])
            
            # Get all 12 edge normals (6 from each hexagon)
            normals = []
            for k in range(6):
                e1 = v1[(k + 1) % 6] - v1[k]
                n1 = np.array([-e1[1], e1[0]])
                n1 /= (np.linalg.norm(n1) + 1e-15)
                normals.append(n1)
            
            for k in range(6):
                e2 = v2[(k + 1) % 6] - v2[k]
                n2 = np.array([-e2[1], e2[0]])
                n2 /= (np.linalg.norm(n2) + 1e-15)
                normals.append(n2)
            
            # Check if any normal separates the two hexagons
            separated = False
            for n in normals:
                proj1 = v1 @ n
                proj2 = v2 @ n
                max1 = np.max(proj1)
                min1 = np.min(proj1)
                max2 = np.max(proj2)
                min2 = np.min(proj2)
                
                # Intervals are apart if max1 < min2 or max2 < min1
                if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                    separated = True
                    break
            
            if not separated:
                return False
    
    # Check containment: all inner vertices must be inside outer hexagon
    for i in range(11):
        verts = hex_vertices(shifted_inner[i, 0], shifted_inner[i, 1], 1.0, shifted_inner[i, 2])
        for p in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < -1e-6:
                    return False
    
    return True

def _compute_objective(params, s):
    """
    Compute objective: maximize the minimum separation constraint violation.
    We want to find positions that are valid. 
    If params represents inner hexagons (x,y,angle), we check constraints.
    Objective: minimize the maximum violation (we want all violations <= 0).
    Actually, for optimization, we want to MINIMIZE the side length S required.
    But S is fixed in the search? No, S is a variable.
    Let's use a different approach:
    Minimize S. Subject to:
    1. Inner hexes don't overlap.
    2. Inner hexes are inside outer hex of side S.
    
    We can define a penalty function that is 0 if valid, and positive if invalid.
    But to find the minimum S, we need to search S as well?
    Or we can fix S and check if valid, then binary search S?
    
    Better: Optimize positions (x,y,angle) for a given S to minimize a "tightness" metric?
    No, we want the smallest S.
    
    Strategy:
    1. Define a function that takes positions and S, returns a penalty (0 if valid, >0 if invalid).
    2. We want to find positions and S such that penalty is 0, minimizing S.
    
    This is a non-linear optimization.
    Let's use a penalty method.
    Objective: S + lambda * (sum of constraint violations)^2
    
    Constraint 1: Separation. For each pair (i,j), they must be separated.
    Separation means: max over all 12 normals n of (min(proj_j) - max(proj_i)) > 1e-6 OR (min(proj_i) - max(proj_j)) > 1e-6.
    Actually, the condition is: "on at least one of their 12 edge normals the projected intervals are apart by more than 1e-6".
    So for a pair to be INVALID, for ALL 12 normals, the intervals overlap or touch within 1e-6.
    Let d_ij be the minimum gap over all 12 normals.
    If d_ij <= 1e-6, it's a violation.
    Violation amount: max(0, 1e-6 - d_ij) is not quite right if they overlap.
    If they overlap, the "gap" is negative.
    Let's define gap_ij = max_{n in normals} (min(proj_j) - max(proj_i), min(proj_i) - max(proj_j)).
    If gap_ij > 1e-6, valid.
    If gap_ij <= 1e-6, violation.
    Penalty term: max(0, 1e-6 - gap_ij).
    
    Constraint 2: Containment. All inner vertices must be inside outer hex.
    For each inner vertex p, and each outer edge (p1, p2), cross(p2-p1, p-p1) >= -1e-6.
    Violation: max(0, -1e-6 - cross).
    
    Objective: S + K * (Sum(pair penalties) + Sum(vertex penalties))
    
    We need to optimize over x,y,angle for 11 hexes (33 vars) AND S (1 var).
    Total 34 vars.
    """
    inner = params[:33].reshape(11, 3)
    S = params[33]
    
    # Calculate all inner vertices
    all_verts = []
    for i in range(11):
        cx, cy, ang = inner[i]
        verts = hex_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    
    # Separation penalties
    pair_pen = 0.0
    for i in range(11):
        for j in range(i + 1, 11):
            v1 = all_verts[i]
            v2 = all_verts[j]
            
            normals = []
            for k in range(6):
                e1 = v1[(k + 1) % 6] - v1[k]
                n1 = np.array([-e1[1], e1[0]])
                n1 /= (np.linalg.norm(n1) + 1e-15)
                normals.append(n1)
            
            for k in range(6):
                e2 = v2[(k + 1) % 6] - v2[k]
                n2 = np.array([-e2[1], e2[0]])
                n2 /= (np.linalg.norm(n2) + 1e-15)
                normals.append(n2)
            
            min_gap = np.inf
            for n in normals:
                proj1 = v1 @ n
                proj2 = v2 @ n
                max1 = np.max(proj1)
                min1 = np.min(proj1)
                max2 = np.max(proj2)
                min2 = np.min(proj2)
                
                gap1 = min2 - max1
                gap2 = min1 - max2
                gap = max(gap1, gap2)
                if gap < min_gap:
                    min_gap = gap
            
            if min_gap < 1e-6:
                pair_pen += (1e-6 - min_gap)
    
    # Containment penalties
    outer_verts = hex_vertices(0, 0, S, 0)  # Center at 0,0 for optimization simplicity
    vert_pen = 0.0
    for i in range(11):
        verts = all_verts[i]
        for p in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < -1e-6:
                    vert_pen += (-1e-6 - cross)
    
    total_pen = pair_pen + vert_pen
    K = 100.0
    return S + K * total_pen

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit
    
    # Initial honeycomb lattice arrangement
    # Center hexagon at origin, 6 around it, 4 more
    hex_spacings = np.sqrt(3)  # Distance between centers of touching hexagons
    
    initial_positions = np.array([
        [0, 0, 0],  # center
        [hex_spacings, 0, 0],  # right
        [hex_spacings/2, hex_spacings*np.sqrt(3)/2, 0],  # top-right
        [-hex_spacings/2, hex_spacings*np.sqrt(3)/2, 0],  # top-left
        [-hex_spacings, 0, 0],  # left
        [-hex_spacings/2, -hex_spacings*np.sqrt(3)/2, 0],  # bottom-left
        [hex_spacings/2, -hex_spacings*np.sqrt(3)/2, 0],  # bottom-right
        [3*hex_spacings/2, hex_spacings*np.sqrt(3)/2, 0],  # far top-right
        [3*hex_spacings/2, -hex_spacings*np.sqrt(3)/2, 0],  # far bottom-right
        [-3*hex_spacings/2, hex_spacings*np.sqrt(3)/2, 0],  # far top-left
        [-3*hex_spacings/2, -hex_spacings*np.sqrt(3)/2, 0],  # far bottom-left
    ])
    
    # Estimate initial S
    all_verts = []
    for i in range(11):
        cx, cy, ang = initial_positions[i]
        verts = hex_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    center = np.mean(all_verts, axis=0)
    distances = np.linalg.norm(all_verts - center, axis=1)
    max_dist = np.max(distances)
    initial_S = 2 * max_dist / np.sqrt(3) * 1.01
    
    # Shift initial positions to center
    initial_positions[:, 0] -= center[0]
    initial_positions[:, 1] -= center[1]
    
    x0 = np.hstack([initial_positions.flatten(), [initial_S]])
    
    best_params = x0.copy()
    best_obj = _compute_objective(best_params, None)
    
    # Nelder-Mead optimization
    if time.time() < deadline - 100:
        try:
            result = minimize(
                _compute_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False}
            )
            if result.fun < best_obj:
                best_params = result.x
                best_obj = result.fun
        except:
            pass
    
    # Extract best result
    inner = best_params[:33].reshape(11, 3)
    S = best_params[33]
    
    # Ensure S is at least sqrt(11)
    if S < np.sqrt(11):
        S = np.sqrt(11)
    
    # Add a small margin to S to ensure validity due to floating point
    S *= 1.001
    
    # Shift inner hexagons so outer center is at origin (already done in optimization)
    inner_shifted = inner.copy()
    
    # Verify validity
    if not _check_validity(inner_shifted, np.array([0, 0, 0]), S):
        # Fallback: increase S until valid
        S *= 1.1
        if not _check_validity(inner_shifted, np.array([0, 0, 0]), S):
            # Emergency: use the initial expanded arrangement
            inner_shifted = initial_positions.copy()
            all_verts = []
            for i in range(11):
                cx, cy, ang = inner_shifted[i]
                verts = hex_vertices(cx, cy, 1.0, ang)
                all_verts.append(verts)
            all_verts = np.vstack(all_verts)
            center = np.mean(all_verts, axis=0)
            distances = np.linalg.norm(all_verts - center, axis=1)
            max_dist = np.max(distances)
            S = 2 * max_dist / np.sqrt(3) * 1.01
            inner_shifted[:, 0] -= center[0]
            inner_shifted[:, 1] -= center[1]
    
    return inner_shifted, np.array([0, 0, 0]), S


# EVOLVE-BLOCK-END
