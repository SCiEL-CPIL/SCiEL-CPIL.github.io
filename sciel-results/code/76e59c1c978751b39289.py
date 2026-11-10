# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _hex_edge_normals(vertices):
    """Return outward normals (unit) for each edge of a hexagon given its 6 vertices in CCW order."""
    edges = np.roll(vertices, -1, axis=0) - vertices
    # For CCW polygon, outward normal is (dy, -dx) normalized... wait.
    # For CCW, the interior is to the left of each edge. So outward is to the right.
    # Edge vector e = (ex, ey). Right normal is (ey, -ex).
    normals = np.column_stack([edges[:, 1], -edges[:, 0]])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals = normals / norms
    return normals

def _sat_overlap(v1, n1, v2, n2):
    """
    Check if two convex polygons (given by vertices and outward normals) overlap using SAT.
    Returns True if they overlap (i.e., there is NO separating axis), False if separated.
    We want to detect overlap to penalize it.
    """
    # Project vertices of poly1 onto each normal of poly2, and vice versa
    # Actually, SAT says: if there exists any axis where projections are disjoint, they don't overlap.
    # We need to check all 12 axes (6 from each).
    
    # Project v1 onto n1 axes
    proj1_on_n1 = v1 @ n1.T  # shape (6, 6) -> row i is dot of v1[i] with n1[j]
    # For each normal n1[j], the projection interval of poly1 is [min(proj1_on_n1[:,j]), max(proj1_on_n1[:,j])]
    # Similarly for poly2 on n1[j]
    proj2_on_n1 = v2 @ n1.T
    
    for j in range(6):
        min1 = np.min(proj1_on_n1[:, j])
        max1 = np.max(proj1_on_n1[:, j])
        min2 = np.min(proj2_on_n1[:, j])
        max2 = np.max(proj2_on_n1[:, j])
        if max1 < min2 or max2 < min1:
            return False  # Separated along this axis
    
    # Check normals of poly2
    proj1_on_n2 = v1 @ n2.T
    proj2_on_n2 = v2 @ n2.T
    
    for j in range(6):
        min1 = np.min(proj1_on_n2[:, j])
        max1 = np.max(proj1_on_n2[:, j])
        min2 = np.min(proj2_on_n2[:, j])
        max2 = np.max(proj2_on_n2[:, j])
        if max1 < min2 or max2 < min1:
            return False  # Separated along this axis
    
    return True  # No separating axis found -> overlap

def _point_in_hexagon(v, outer_vertices):
    """
    Check if point v is inside or on the boundary of the convex hexagon given by outer_vertices (CCW).
    For each edge (p1, p2), cross(p2-p1, v-p1) should be >= 0 for CCW polygon (interior on left).
    Wait, the task says: cross(p2 - p1, v - p1) must be >= -1e-6.
    So we just check that all cross products are >= -1e-6.
    """
    for i in range(6):
        p1 = outer_vertices[i]
        p2 = outer_vertices[(i + 1) % 6]
        edge = p2 - p1
        vec = v - p1
        cross = edge[0] * vec[1] - edge[1] * vec[0]
        if cross < -1e-6:
            return False
    return True

def _compute_penalty(x, penalty_weight):
    """
    x: array of shape (49,)
    [0:24] = 12 centers (x,y)
    [24:36] = 12 angles (radians)
    [36] = outer center x
    [37] = outer center y
    [38] = outer angle (radians)
    [39] = outer side length s
    """
    centers = x[0:24].reshape(12, 2)
    angles = x[24:36]
    outer_cx, outer_cy = x[36], x[37]
    outer_angle = x[38]
    s = x[39]
    
    if s <= 0:
        return penalty_weight * 1e10
    
    # Compute inner hexagon vertices and normals
    inner_verts = []
    inner_normals = []
    for i in range(12):
        v = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, np.degrees(angles[i]))
        n = _hex_edge_normals(v)
        inner_verts.append(v)
        inner_normals.append(n)
    
    # Compute outer hexagon vertices
    outer_verts = _hex_vertices(outer_cx, outer_cy, s, np.degrees(outer_angle))
    outer_normals = _hex_edge_normals(outer_verts)
    
    penalty = 0.0
    
    # 1. SAT overlap penalty between inner hexagons
    for i in range(12):
        for j in range(i + 1, 12):
            if _sat_overlap(inner_verts[i], inner_normals[i], inner_verts[j], inner_normals[j]):
                # Overlap detected. Add penalty.
                # To make it differentiable-ish, we can just add a fixed penalty or estimate penetration.
                # For simplicity, add a large constant penalty.
                penalty += penalty_weight * 100.0
    
    # 2. Inner vertices outside outer hexagon penalty
    for i in range(12):
        for k in range(6):
            v = inner_verts[i][k]
            if not _point_in_hexagon(v, outer_verts):
                # Estimate violation depth
                # Find the most violated edge
                min_cross = np.inf
                for e in range(6):
                    p1 = outer_verts[e]
                    p2 = outer_verts[(e + 1) % 6]
                    edge = p2 - p1
                    vec = v - p1
                    cross = edge[0] * vec[1] - edge[1] * vec[0]
                    if cross < min_cross:
                        min_cross = cross
                # min_cross is negative. Violation depth is roughly -min_cross / |edge|
                # |edge| = s
                if s > 0:
                    violation = -min_cross / s
                    penalty += penalty_weight * violation * violation
    
    return penalty

def _objective(x, penalty_weight):
    """
    Objective: minimize s + penalty
    """
    s = x[39]
    if s <= 0:
        return 1e10
    pen = _compute_penalty(x, penalty_weight)
    return s + pen

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 620  # 10s safety margin
    
    # Initial guess: a reasonable packing
    # Let's start with a hexagonal lattice arrangement
    # Side 1 hexagons.
    # Center at (0,0)
    # Neighbors at distance 2*sqrt(3) ~ 3.464? No, center-to-center for touching hexagons of side 1 is 2*sqrt(3)/2 * 2?
    # Distance between centers of touching hexagons of side d is d*sqrt(3).
    # So d=1, distance = sqrt(3) ~ 1.732.
    # Wait, width of hexagon is 2*d. Height is sqrt(3)*d.
    # Flat-top hexagon (angle 0): width 2, height sqrt(3).
    # Neighbors in x direction: distance 2? No.
    # Let's just use a known good starting config or random.
    
    # Let's try a 3-4-3 arrangement or similar.
    # Or just random initialization with a large s.
    
    np.random.seed(42)
    
    best_x = None
    best_score = -1e10
    
    # Multiple restarts
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Random initial positions in a circle of radius 5
        centers = np.random.uniform(-5, 5, (12, 2))
        angles = np.random.uniform(0, 2 * np.pi, 12)
        
        # Random outer parameters
        outer_cx, outer_cy = np.random.uniform(-1, 1, 2)
        outer_angle = np.random.uniform(0, 2 * np.pi)
        s = 10.0  # Start large
        
        x0 = np.concatenate([centers.flatten(), angles, [outer_cx, outer_cy, outer_angle, s]])
        
        # Optimization
        penalty_weight = 10.0
        success = False
        res = None
        
        try:
            res = minimize(_objective, x0, args=(penalty_weight,), method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-8})
            success = res.success
        except Exception:
            pass
        
        if res is not None:
            x_opt = res.x
            # Check validity and compute score
            # We need to ensure constraints are actually met.
            # If penalty is non-zero, it might be invalid.
            # Let's check the penalty without weight to see if it's ~0
            pen_check = _compute_penalty(x_opt, 1.0)
            
            if pen_check < 1e-6:
                s_val = x_opt[39]
                score = 1.0 / s_val
                if score > best_score:
                    best_score = score
                    best_x = x_opt.copy()
        
        # Increase penalty weight and re-optimize from previous best?
        # Or just do multiple random restarts.
        
        # Adaptive penalty: if we found a good one, try to refine with higher penalty
        if best_x is not None:
            # Refine the best solution
            penalty_weight = 1000.0
            try:
                res2 = minimize(_objective, best_x, args=(penalty_weight,), method='L-BFGS-B',
                                options={'maxiter': 100, 'ftol': 1e-10})
                x_opt2 = res2.x
                pen_check2 = _compute_penalty(x_opt2, 1.0)
                if pen_check2 < 1e-6:
                    s_val2 = x_opt2[39]
                    score2 = 1.0 / s_val2
                    if score2 > best_score:
                        best_score = score2
                        best_x = x_opt2.copy()
            except Exception:
                pass
    
    if best_x is None:
        # Fallback to seed solution
        inner_hex_data = np.array(
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
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    # Extract results
    centers = best_x[0:24].reshape(12, 2)
    angles_rad = best_x[24:36]
    angles_deg = np.degrees(angles_rad)
    
    inner_hex_data = np.column_stack([centers, angles_deg])
    
    outer_cx, outer_cy = best_x[36], best_x[37]
    outer_angle_deg = np.degrees(best_x[38])
    s = best_x[39]
    
    outer_hex_data = np.array([outer_cx, outer_cy, outer_angle_deg])
    
    return inner_hex_data, outer_hex_data, s

# EVOLVE-BLOCK-END
