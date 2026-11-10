# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _min_s_for_hex(verts, angle_deg):
    """
    Given vertices of a polygon and the orientation angle of the outer hexagon,
    compute the minimum side length s of a regular hexagon with that orientation
    (centered at origin) that contains all vertices.
    
    A regular hexagon with side s, angle 0, centered at origin has vertices at
    s * (cos(60i), sin(60i)) for i=0..5.
    
    The hexagon can be described as the intersection of 6 half-planes.
    For a hexagon with orientation angle a (degrees), the outward normals of the
    6 edges are at angles a + 60i + 30 degrees (for i=0..5) if we use the convention
    where vertices are at a + 60i.
    
    Actually, let's think more carefully. If vertices are at angles a + 60i, then
    the edges are between consecutive vertices. The normal to the edge between
    vertex i and vertex i+1 points outward.
    
    For a regular hexagon centered at origin with vertices at angle a + 60i * side s:
    The edge between vertex i (at angle a+60i) and vertex i+1 (at angle a+60(i+1))
    has its midpoint at angle a + 60i + 30, at distance s * cos(30°) = s * sqrt(3)/2.
    The outward normal to this edge is in the direction of the midpoint, i.e., 
    at angle a + 60i + 30.
    
    The half-plane constraint is: dot(p, n_i) <= s * sqrt(3)/2 for all vertices p,
    where n_i = (cos(a + 60i + 30), sin(a + 60i + 30)).
    
    So the minimum s is: s = max_i [ dot(p, n_i) / (sqrt(3)/2) ] over all vertices p and all i.
    But we need s >= 0, so s = max(0, max over all p, i of dot(p, n_i) / (sqrt(3)/2)).
    
    Wait, but the hexagon is centered at origin. If all vertices are in one half-plane,
    this might not be tight. Actually for a convex polygon containing the origin,
    the tightest constraint gives the minimum s.
    
    More precisely: s_min = (2/sqrt(3)) * max_{p in verts, i in 0..5} dot(p, n_i)
    where n_i are the 6 outward unit normals.
    
    But we also need to handle the case where the polygon doesn't contain the origin.
    In that case, the hexagon might need to be shifted, but we're fixing the center at origin.
    Actually, the problem says the outer hexagon has a center, so we can choose it.
    
    Hmm, let me reconsider. The outer hexagon has center (cx, cy) and angle a and side s.
    The constraint is that all inner vertices must be inside the outer hexagon.
    
    For a fixed outer hexagon orientation a, the minimum s such that all points p are inside
    the outer hexagon (centered at some (cx,cy)) is more complex.
    
    Actually, let's simplify: for a given set of points and a fixed hexagon orientation,
    the minimum enclosing regular hexagon (with that orientation) can be found by
    considering the 6 support functions.
    
    Let me use a different approach. For a regular hexagon with orientation a, centered at c,
    side s, the condition that point p is inside is:
    for each edge i (i=0..5): dot(p - c, n_i) <= s * sqrt(3)/2
    where n_i = (cos(a + 60i + 30), sin(a + 60i + 30)).
    
    So: dot(p, n_i) - dot(c, n_i) <= s * sqrt(3)/2 for all p, i.
    
    This means: s >= [dot(p, n_i) - dot(c, n_i)] / (sqrt(3)/2) for all p, i.
    
    And also we need the "other side": since the hexagon is bounded, we also need
    dot(p, n_i) - dot(c, n_i) >= -s * sqrt(3)/2, which gives:
    s >= [dot(c, n_i) - dot(p, n_i)] / (sqrt(3)/2) for all p, i.
    
    So: s >= max_{p, i} |dot(p, n_i) - dot(c, n_i)| / (sqrt(3)/2)
    
    To minimize s over c, we want to minimize max_{p, i} |dot(p, n_i) - dot(c, n_i)|.
    
    This is equivalent to: for each i, let a_i = dot(c, n_i). We want to choose c (2D)
    to minimize max_{p, i} |dot(p, n_i) - a_i|.
    
    Since c is 2D and n_i are 6 directions, a_i = dot(c, n_i) is constrained:
    a_i = c_x * n_i[0] + c_y * n_i[1].
    
    This is a convex optimization problem in c. But it's getting complex.
    
    Let me just use a simpler approach: for a given orientation, compute the bounding
    box in the 6 normal directions and find the optimal center.
    
    Actually, the simplest correct approach: the minimum enclosing regular hexagon
    with fixed orientation can be found by noting that it's determined by the 6
    support lines. The optimal center is such that the max extent in each pair of
    opposite directions is balanced.
    
    For direction i and i+3 (opposite), we have:
    max_p dot(p, n_i) - min_p dot(p, n_i) = 2 * s * sqrt(3)/2 = s * sqrt(3)
    (when the hexagon is centered optimally)
    
    Wait no. If the hexagon is centered at c, then:
    max_p [dot(p, n_i) - dot(c, n_i)] <= s * sqrt(3)/2
    min_p [dot(p, n_i) - dot(c, n_i)] >= -s * sqrt(3)/2
    
    So: max_p dot(p, n_i) - dot(c, n_i) <= s * sqrt(3)/2
    dot(c, n_i) - min_p dot(p, n_i) <= s * sqrt(3)/2
    
    Adding: max_p dot(p, n_i) - min_p dot(p, n_i) <= s * sqrt(3)
    
    So s >= (max_p dot(p, n_i) - min_p dot(p, n_i)) / sqrt(3) for each i.
    
    And the optimal c_i (projection of c on n_i) is the midpoint:
    dot(c, n_i) = (max_p dot(p, n_i) + min_p dot(p, n_i)) / 2
    
    But c must be consistent across all 6 directions. The 6 constraints on c are:
    dot(c, n_i) = m_i for i=0..5, where m_i are the midpoints.
    
    Since n_0..n_5 span R^2 (they're 6 directions), the system is overdetermined.
    We need to find c that best satisfies all 6 constraints in a least-squares sense,
    or more precisely, we need to find c that minimizes the maximum violation.
    
    This is getting too complex. Let me just use a numerical approach:
    For a given orientation, use scipy to minimize s over c and s.
    """
    a = np.radians(angle_deg)
    normals = np.column_stack([
        np.cos(a + np.radians(60 * np.arange(6) + 30)),
        np.sin(a + np.radians(60 * np.arange(6) + 30))
    ])
    
    # For each normal direction, compute max and min dot product with vertices
    dots = verts @ normals.T  # shape (n_verts, 6)
    max_dots = np.max(dots, axis=0)  # shape (6,)
    min_dots = np.min(dots, axis=0)  # shape (6,)
    
    # The width in each direction
    widths = max_dots - min_dots  # shape (6,)
    
    # s must satisfy: s >= widths[i] / sqrt(3) for all i
    # So s_min_lower_bound = max(widths) / sqrt(3)
    s_lower = np.max(widths) / np.sqrt(3)
    
    # The optimal center in each direction is the midpoint
    midpoints = (max_dots + min_dots) / 2.0  # shape (6,)
    
    # We need to find c such that dot(c, normals[i]) ≈ midpoints[i] for all i
    # This is a least squares problem: normals.T @ c ≈ midpoints
    # normals is 6x2, so normals.T is 2x6
    # We want c (2,) such that normals @ c = midpoints (6,)
    # Least squares: c = pinv(normals) @ midpoints
    c = np.linalg.lstsq(normals, midpoints, rcond=None)[0]
    
    # Now compute the actual s needed with this center
    # s = max_{p, i} |dot(p, n_i) - dot(c, n_i)| / (sqrt(3)/2)
    c_dots = normals @ c  # shape (6,)
    deviations = np.abs(dots - c_dots[np.newaxis, :])  # shape (n_verts, 6)
    s = np.max(deviations) / (np.sqrt(3) / 2)
    
    return s, c

def _is_valid(inner, outer, s, tol=1e-6):
    """Check if the packing is valid."""
    # Check finite values
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    
    # Check s >= sqrt(12)
    if s < np.sqrt(12) - 1e-9:
        return False
    
    # Compute outer vertices
    outer_verts = _hex_vertices(outer[0], outer[1], s, outer[2])
    
    # Check outer vertices are distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    
    # Shift all centers by minus outer center
    shift = np.array([outer[0], outer[1]])
    
    # Check each inner hexagon's edges have length 1
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1) % 6]
            edge_len = np.linalg.norm(v2 - v1)
            if abs(edge_len - 1.0) > 1e-8:
                return False
    
    # Check separation between inner hexagons using SAT
    inner_verts = []
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        inner_verts.append(verts)
    
    for i in range(12):
        for j in range(i+1, 12):
            v1 = inner_verts[i]
            v2 = inner_verts[j]
            
            # Get edge normals for both
            edges1 = np.roll(v1, -1, axis=0) - v1
            normals1 = np.column_stack([edges1[:, 1], -edges1[:, 0]])
            normals1 = normals1 / np.linalg.norm(normals1, axis=1, keepdims=True)
            
            edges2 = np.roll(v2, -1, axis=0) - v2
            normals2 = np.column_stack([edges2[:, 1], -edges2[:, 0]])
            normals2 = normals2 / np.linalg.norm(normals2, axis=1, keepdims=True)
            
            # Check all 12 normals
            all_normals = np.vstack([normals1, normals2])
            separated = False
            for n in all_normals:
                proj1 = v1 @ n
                proj2 = v2 @ n
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                if max1 < min2 - tol or max2 < min1 - tol:
                    separated = True
                    break
            
            if not separated:
                return False
    
    # Check all inner vertices are inside the outer hexagon
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for v in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1) % 6]
                edge = p2 - p1
                vec = v - p1
                cross = edge[0] * vec[1] - edge[1] * vec[0]
                if cross < -1e-6:
                    return False
    
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600  # Safety margin
    
    # Start with a hexagonal lattice arrangement
    # 12 hexagons in a honeycomb pattern
    # Center at origin, neighbors at distance sqrt(3)
    
    sqrt3 = np.sqrt(3)
    
    # Define 12 positions in a hexagonal lattice
    # Ring 0: center (1 hex)
    # Ring 1: 6 hexes at distance sqrt(3)
    # Ring 2: 5 more hexes to make 12 total
    
    # Let's use a 3-4-3 or similar arrangement
    # Or just use the hexagonal lattice positions
    
    positions = []
    angles_list = []
    
    # Center
    positions.append([0, 0])
    angles_list.append(0.0)
    
    # Ring 1: 6 hexes
    for i in range(6):
        ang = np.radians(60 * i)
        x = sqrt3 * np.cos(ang)
        y = sqrt3 * np.sin(ang)
        positions.append([x, y])
        angles_list.append(0.0)
    
    # Ring 2: 5 more hexes (to total 12)
    # Pick 5 of the 12 positions in ring 2
    for i in range(5):
        ang = np.radians(60 * i + 30)  # Offset by 30 degrees
        x = 2 * sqrt3 * np.cos(ang)
        y = 2 * sqrt3 * np.sin(ang)
        positions.append([x, y])
        angles_list.append(0.0)
    
    positions = np.array(positions)
    angles_arr = np.array(angles_list)
    
    # Initial inner data
    inner = np.column_stack([positions, angles_arr])
    
    # Compute initial outer hexagon
    all_verts = []
    for i in range(12):
        verts = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    s, center = _min_s_for_hex(all_verts, 0.0)
    outer = np.array([center[0], center[1], 0.0])
    
    # Verify initial solution
    if _is_valid(inner, outer, s):
        best_inner = inner.copy()
        best_outer = outer.copy()
        best_s = s
        best_score = 1.0 / s
    else:
        best_inner = None
        best_score = -1e10
    
    # Try different orientations of the outer hexagon
    # and optimize inner positions
    
    best_inner = inner.copy()
    best_outer = outer.copy()
    best_s = s
    best_score = 1.0 / s
    
    # Try multiple outer hexagon orientations
    for outer_angle in [0, 30, 60, 90, 120, 150]:
        if time.time() > deadline:
            break
        
        # For this outer orientation, try to optimize inner positions
        # using a simple local search
        
        current_inner = best_inner.copy()
        
        # Try to shrink by moving hexagons closer
        for iteration in range(100):
            if time.time() > deadline:
                break
            
            # Compute current outer
            all_verts = []
            for i in range(12):
                verts = _hex_vertices(current_inner[i, 0], current_inner[i, 1], 1.0, current_inner[i, 2])
                all_verts.append(verts)
            all_verts = np.vstack(all_verts)
            
            s_curr, center_curr = _min_s_for_hex(all_verts, outer_angle)
            outer_curr = np.array([center_curr[0], center_curr[1], float(outer_angle)])
            
            if not _is_valid(current_inner, outer_curr, s_curr):
                break
            
            score_curr = 1.0 / s_curr
            if score_curr > best_score:
                best_score = score_curr
                best_inner = current_inner.copy()
                best_outer = outer_curr.copy()
                best_s = s_curr
            
            # Try small random perturbations
            improved = False
            for _ in range(20):
                if time.time() > deadline:
                    break
                
                trial = current_inner.copy()
                idx = np.random.randint(0, 12)
                dx = np.random.uniform(-0.1, 0.1)
                dy = np.random.uniform(-0.1, 0.1)
                da = np.random.uniform(-10, 10)
                
                trial[idx, 0] += dx
                trial[idx, 1] += dy
                trial[idx, 2] += da
                
                # Check validity
                all_verts_t = []
                for i in range(12):
                    verts = _hex_vertices(trial[i, 0], trial[i, 1], 1.0, trial[i, 2])
                    all_verts_t.append(verts)
                all_verts_t = np.vstack(all_verts_t)
                
                s_t, center_t = _min_s_for_hex(all_verts_t, outer_angle)
                outer_t = np.array([center_t[0], center_t[1], float(outer_angle)])
                
                if _is_valid(trial, outer_t, s_t):
                    score_t = 1.0 / s_t
                    if score_t > score_curr:
                        current_inner = trial
                        improved = True
                        break
            
            if not improved:
                break
    
    # Final verification
    if best_inner is not None:
        if _is_valid(best_inner, best_outer, best_s):
            return best_inner, best_outer, best_s
    
    # Fallback: return the initial solution if valid, else a known valid one
    if _is_valid(inner, outer, s):
        return inner, outer, s
    
    # Absolute fallback
    inner_fb = np.array(
        [
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
            [0, -4, 0],
        ]
    )
    outer_fb = np.array([0, 0, 0])
    s_fb = 8.0
    return inner_fb, outer_fb, s_fb

# EVOLVE-BLOCK-END
