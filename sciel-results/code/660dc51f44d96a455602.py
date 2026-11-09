"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from itertools import product

N = 11

def hexagon_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side s, rotation angle_deg."""
    theta = np.radians(angle_deg)
    angles = theta + np.pi * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def hexagon_support(cx, cy, s, angle_deg, direction):
    """Support function of a regular hexagon in a given direction (unit vector)."""
    # A regular hexagon with side s has circumradius s, inradius s*sqrt(3)/2
    # The support function in direction d is max over vertices of dot(v, d)
    # For a regular hexagon centered at origin with side s and rotation theta:
    # vertices at angles theta + k*pi/3, radius s
    # support = s * max_k cos(d_angle - theta - k*pi/3)
    # This is s * max over 6 directions of cos(delta)
    # The max of cos over 6 equally spaced angles is cos(0) if aligned, but generally:
    # For a regular hexagon, the support function in direction phi is:
    # s * max_{k=0..5} cos(phi - theta - k*pi/3)
    # This equals s * (cos(min_k |phi - theta - k*pi/3| mod 2pi adjusted))
    # More simply: the support function of a regular hexagon (flat-top or pointy-top) 
    # can be computed as the maximum dot product with vertices.
    
    angles = angle_deg * np.pi / 180.0 + np.pi * np.arange(6)
    verts = s * np.column_stack([np.cos(angles), np.sin(angles)])
    # direction is a unit vector
    dots = verts @ direction
    return cx * direction[0] + cy * direction[1] + np.max(dots)

def hexagon_distance(v1, v2):
    """
    Compute the minimum distance between two regular hexagons.
    v1, v2 are dicts with keys cx, cy, s, angle_deg.
    If distance <= 0, they overlap.
    """
    # For two convex polygons, the minimum distance is the minimum over all edge pairs
    # of the distance between the edges, or 0 if they intersect.
    # A simpler approach: use the support function approach.
    # The distance between two convex sets A and B is min_{a in A, b in B} |a - b|
    # This can be found via: min_d (h_A(d) + h_B(-d) - d·(c_B - c_A)) ... no that's not right.
    
    # Let's just compute vertex-vertex, vertex-edge distances.
    verts1 = hexagon_vertices(v1['cx'], v1['cy'], v1['s'], v1['angle_deg'])
    verts2 = hexagon_vertices(v2['cx'], v2['cy'], v2['s'], v2['angle_deg'])
    
    min_dist = np.inf
    for i in range(6):
        p1 = verts1[i]
        for j in range(6):
            p2 = verts2[j]
            d = np.linalg.norm(p1 - p2)
            if d < min_dist:
                min_dist = d
    
    # Also check edge-edge distances
    for i in range(6):
        a = verts1[i]
        b = verts1[(i + 1) % 6]
        for j in range(6):
            c = verts2[j]
            d = verts2[(j + 1) % 6]
            # Distance between line segment ab and cd
            # Parameterize: find min |a + t*(b-a) - (c + u*(d-c))| for t,u in [0,1]
            # This is a small quadratic minimization
            r = b - a
            s_vec = d - c
            p = a - c
            a_coeff = r @ r
            e = r @ s_vec
            f = s_vec @ s_vec
            c_coeff = p @ s_vec
            b_coeff = r @ p
            
            if a_coeff > 1e-12 and f > 1e-12:
                denom = a_coeff * f - e * e
                if abs(denom) > 1e-12:
                    s_t = (e * c_coeff - b_coeff * f) / denom
                    t_t = (a_coeff * c_coeff - e * b_coeff) / denom
                    if s_t < 0: s_t = 0
                    elif s_t > 1: s_t = 1
                    if t_t < 0: t_t = 0
                    elif t_t > 1: t_t = 1
                    pt = a + s_t * r
                    qt = c + t_t * s_vec
                    dist = np.linalg.norm(pt - qt)
                    if dist < min_dist:
                        min_dist = dist
            else:
                # Degenerate case
                pass
    
    return min_dist

def check_outer_boundary(inner_hexes, outer_cx, outer_cy, outer_s, outer_angle):
    """
    Check if all inner hexagons are inside the outer hexagon.
    Returns True if valid, False otherwise.
    """
    outer_verts = hexagon_vertices(outer_cx, outer_cy, outer_s, outer_angle)
    
    for ih in inner_hexes:
        inner_verts = hexagon_vertices(ih['cx'], ih['cy'], ih['s'], ih['angle_deg'])
        # Check if all inner vertices are inside outer hexagon
        for iv in inner_verts:
            # Point in polygon test
            inside = False
            n = len(outer_verts)
            for k in range(n):
                v1 = outer_verts[k]
                v2 = outer_verts[(k + 1) % n]
                # Cross product to check if point is to the left of edge
                cross = (v2[0] - v1[0]) * (iv[1] - v1[1]) - (v2[1] - v1[1]) * (iv[0] - v1[0])
                if cross < -1e-9:
                    inside = False
                    break
            if inside:
                continue
            else:
                # Check if on edge or inside
                # More robust: use barycentric or winding number
                # For a convex polygon, if all cross products are >= 0 (or all <= 0), point is inside
                all_pos = True
                all_neg = True
                for k in range(n):
                    v1 = outer_verts[k]
                    v2 = outer_verts[(k + 1) % n]
                    cross = (v2[0] - v1[0]) * (iv[1] - v1[1]) - (v2[1] - v1[1]) * (iv[0] - v1[0])
                    if cross < -1e-9:
                        all_pos = False
                    if cross > 1e-9:
                        all_neg = False
                if not all_pos and not all_neg:
                    return False
                # If all_pos or all_neg, point is inside or on boundary
    return True

def objective_and_constraints(x, N_inner):
    """
    x: [cx1, cy1, a1, cx2, cy2, a2, ..., cxN, cyN, aN, outer_cx, outer_cy, outer_s, outer_angle]
    Returns (outer_s, violations)
    """
    n_inner = N_inner
    stride = 3
    inner_params = []
    for i in range(n_inner):
        cx = x[i * stride]
        cy = x[i * stride + 1]
        a = x[i * stride + 2]
        inner_params.append({'cx': cx, 'cy': cy, 's': 1.0, 'angle_deg': a})
    
    base = n_inner * stride
    outer_cx = x[base]
    outer_cy = x[base + 1]
    outer_s = x[base + 2]
    outer_angle = x[base + 3]
    
    # Check inner-inner distances
    violations = 0.0
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            d = hexagon_distance(inner_params[i], inner_params[j])
            if d < 1e-6:
                violations += (1e-6 - d) ** 2
    
    # Check inner-outer containment
    if not check_outer_boundary(inner_params, outer_cx, outer_cy, outer_s, outer_angle):
        # Estimate violation magnitude
        outer_verts = hexagon_vertices(outer_cx, outer_cy, outer_s, outer_angle)
        for ih in inner_params:
            inner_verts = hexagon_vertices(ih['cx'], ih['cy'], ih['s'], ih['angle_deg'])
            for iv in inner_verts:
                # Find distance to outer boundary
                min_dist_to_boundary = np.inf
                for k in range(6):
                    v1 = outer_verts[k]
                    v2 = outer_verts[(k + 1) % 6]
                    # Project iv onto edge v1-v2
                    edge = v2 - v1
                    t = np.dot(iv - v1, edge) / (np.dot(edge, edge) + 1e-12)
                    t = np.clip(t, 0, 1)
                    proj = v1 + t * edge
                    dist = np.linalg.norm(iv - proj)
                    if dist < min_dist_to_boundary:
                        min_dist_to_boundary = dist
                # If iv is outside, min_dist_to_boundary is the distance to the nearest edge
                # We need to check if it's actually outside
                all_pos = True
                all_neg = True
                for k in range(6):
                    v1 = outer_verts[k]
                    v2 = outer_verts[(k + 1) % 6]
                    cross = (v2[0] - v1[0]) * (iv[1] - v1[1]) - (v2[1] - v1[1]) * (iv[0] - v1[0])
                    if cross < -1e-9:
                        all_pos = False
                    if cross > 1e-9:
                        all_neg = False
                if not all_pos and not all_neg:
                    violations += min_dist_to_boundary ** 2
    
    return outer_s, violations

def construct():
    N_inner = N
    
    # Initial guess: place hexagons in a grid-like pattern
    # For N=11, use a hexagonal packing
    # Center one at origin, 6 around it, 4 more
    positions = []
    # Hexagonal lattice with spacing 2 (since side=1, distance between centers of touching hexes is 2)
    # Actually for regular hexagons of side 1, the distance between centers of adjacent hexes in a hexagonal packing is 2.
    
    # Let's start with a simple arrangement and let the optimizer work
    # Place them in a row first, then optimize
    x0 = np.zeros(N_inner * 3 + 4)
    
    # Initial positions: spread them out
    for i in range(N_inner):
        x0[i * 3] = 2.0 * (i - N_inner // 2)  # cx
        x0[i * 3 + 1] = 0.0  # cy
        x0[i * 3 + 2] = 0.0  # angle
    
    # Outer hexagon initial guess
    base = N_inner * 3
    x0[base] = 0.0  # outer_cx
    x0[base + 1] = 0.0  # outer_cy
    x0[base + 2] = 15.0  # outer_s (start large)
    x0[base + 3] = 0.0  # outer_angle
    
    # Use scipy minimize with penalty method
    def penalized_objective(x):
        outer_s, violations = objective_and_constraints(x, N_inner)
        penalty = 100.0 * violations
        return outer_s + penalty
    
    # Bounds
    bounds = []
    for i in range(N_inner):
        bounds.append((-20, 20))  # cx
        bounds.append((-20, 20))  # cy
        bounds.append((0, 180))   # angle
    bounds.append((-20, 20))  # outer_cx
    bounds.append((-20, 20))  # outer_cy
    bounds.append((5, 30))    # outer_s
    bounds.append((0, 180))   # outer_angle
    
    # Try a few random starts
    best_s = np.inf
    best_x = x0.copy()
    
    np.random.seed(42)
    for trial in range(3):
        if trial > 0:
            x_trial = x0.copy()
            # Perturb inner positions
            for i in range(N_inner):
                x_trial[i * 3] += np.random.uniform(-2, 2)
                x_trial[i * 3 + 1] += np.random.uniform(-2, 2)
                x_trial[i * 3 + 2] = np.random.uniform(0, 180)
            x_trial[base + 2] = np.random.uniform(8, 20)
        else:
            x_trial = x0.copy()
        
        try:
            result = minimize(penalized_objective, x_trial, method='L-BFGS-B', 
                             bounds=bounds, options={'maxiter': 500, 'ftol': 1e-6})
            if result.success or result.fun < best_s:
                outer_s, violations = objective_and_constraints(result.x, N_inner)
                if violations < 1e-4:
                    if outer_s < best_s:
                        best_s = outer_s
                        best_x = result.x.copy()
        except Exception:
            pass
    
    # Extract results
    inner = []
    for i in range(N_inner):
        cx = best_x[i * 3]
        cy = best_x[i * 3 + 1]
        a = best_x[i * 3 + 2]
        inner.append([cx, cy, a])
    
    outer_cx = best_x[base]
    outer_cy = best_x[base + 1]
    outer_s = best_x[base + 2]
    outer_angle = best_x[base + 3]
    
    return inner, (outer_cx, outer_cy), outer_s, outer_angle
