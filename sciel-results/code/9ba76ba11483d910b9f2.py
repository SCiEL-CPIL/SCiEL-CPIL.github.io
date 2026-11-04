"""Hexagon packing: 11 unit hexagons in minimal outer hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = 140.0  # seconds, safety margin before 150s kill

def hex_vertices(side, cx, cy, angle_deg):
    """Return 6 vertices of a regular hexagon with given side, center, rotation."""
    angle = np.radians(angle_deg)
    ca, sa = np.cos(angle), np.sin(angle)
    base = np.array([np.cos(np.pi/3 * k) for k in range(6)])
    base_y = np.array([np.sin(np.pi/3 * k) for k in range(6)])
    verts = np.column_stack([base * side, base_y * side])
    # Rotate
    rot = np.array([[ca, -sa], [sa, ca]])
    verts = verts @ rot.T
    return verts + np.array([cx, cy])

def hex_support_function(verts, angle_rad):
    """Max projection of hexagon vertices onto direction at angle_rad."""
    d = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    return np.max(verts @ d)

def outer_hex_side_from_points(points, outer_center, outer_angle_deg):
    """Compute minimum side length of outer hexagon containing all points."""
    outer_verts = hex_vertices(1.0, 0, 0, outer_angle_deg)
    # For a regular hexagon, the support function in direction u is side * (cos of angle between u and nearest vertex normal)
    # Actually for a regular hexagon with side s centered at origin:
    # The 6 face normals are at angles 0, 60, 120, 180, 240, 300 degrees
    # The support in direction u is s * max over the 6 normal directions of (dot(u, n_i) + 0) ... 
    # Wait, let me think more carefully.
    # A regular hexagon with side s has inradius r = s * sqrt(3)/2
    # The distance from center to each face is r = s * sqrt(3)/2
    # The 6 outward face normals are at angles: 0, 60, 120, 180, 240, 300 (for a "flat-top" hex)
    # Actually for a pointy-top hexagon (vertices at 0, 60, ...), the face normals are at 30, 90, 150, 210, 270, 330
    # Let me just use: for a hexagon with face normals at angles theta_i, 
    # the support function h(u) = r * max_i cos(u_angle - theta_i) where r is inradius
    # But that's not quite right either. Let me use the vertex-based approach.
    
    # For a regular hexagon with side s, centered at origin, the 6 vertices are at:
    # distance s from center, at angles 0, 60, 120, 180, 240, 300 (if pointy-top)
    # The 6 face normals are perpendicular to the edges, at angles 30, 90, 150, 210, 270, 330
    # The distance from center to each face (inradius) = s * sqrt(3)/2
    
    # For the outer hexagon to contain all points, for each of the 6 face normals n_i:
    # max_j (p_j - center) . n_i <= inradius = s * sqrt(3)/2
    # So s >= max_j (p_j - center) . n_i / (sqrt(3)/2) for each i
    # s >= max_i [max_j (p_j - center) . n_i] / (sqrt(3)/2)
    
    n_face = np.radians(outer_angle_deg + 30)  # face normal angles
    normals = np.column_stack([np.cos(n_face + np.arange(6) * np.pi/3),
                                np.sin(n_face + np.arange(6) * np.pi/3)])
    
    rel = points - np.array(outer_center)
    # rel shape (M, 2), normals shape (6, 2)
    projections = rel @ normals.T  # (M, 6)
    max_proj = np.max(projections, axis=0)  # (6,)
    required_side = np.max(max_proj) / (np.sqrt(3) / 2)
    return required_side

def pairwise_hex_collision(p1, a1, p2, a2, side=1.0):
    """Check if two unit hexagons overlap. Returns violation amount (negative = separated)."""
    v1 = hex_vertices(side, p1[0], p1[1], a1)
    v2 = hex_vertices(side, p2[0], p2[1], a2)
    # Use separating axis theorem: check all 12 edge normals
    all_norms = []
    for v in [v1, v2]:
        for i in range(6):
            edge = v[(i+1)%6] - v[i]
            norm = np.array([-edge[1], edge[0]])
            norm = norm / (np.linalg.norm(norm) + 1e-12)
            all_norms.append(norm)
    
    min_sep = np.inf
    for n in all_norms:
        proj1 = v1 @ n
        proj2 = v2 @ n
        sep = max(proj2.min() - proj1.max(), proj1.min() - proj2.max())
        if sep < 0:
            return sep  # Overlapping
        min_sep = min(min_sep, sep)
    return -min_sep  # Negative means separated

def smooth_collision(v1, v2, beta=5.0):
    """Smooth approximation of overlap amount between two hexagons."""
    all_norms = []
    for v in [v1, v2]:
        for i in range(6):
            edge = v[(i+1)%6] - v[i]
            norm = np.array([-edge[1], edge[0]])
            norm = norm / (np.linalg.norm(norm) + 1e-12)
            all_norms.append(norm)
    
    overlaps = []
    for n in all_norms:
        proj1 = v1 @ n
        proj2 = v2 @ n
        # Overlap in this direction: max(0, max1 - min2, max2 - min1)
        ov = max(proj1.max() - proj2.min(), proj2.max() - proj1.min(), 0)
        overlaps.append(ov)
    overlaps = np.array(overlaps)
    # Smooth max via log-sum-exp
    m = np.max(overlaps)
    if m < 1e-8:
        return 0.0
    return m + np.log(np.sum(np.exp(beta * (overlaps - m)))) / beta

def objective(x):
    """x = [p1x, p1y, a1, p2x, p2y, a2, ..., p11x, p11y, a11, ocx, ocy, oa, s]
    Minimize s + penalty * sum(collision violations)"""
    positions = x[0:3*N:3]
    ys = x[1:3*N:3]
    angles = x[2:3*N:3]
    ocx, ocy = x[3*N], x[3*N+1]
    oa = x[3*N+2]
    s = x[3*N+3]
    
    points = np.column_stack([positions, ys])
    
    # Outer hexagon containment: all hex vertices must be inside
    # Required side for containment
    required_s = outer_hex_side_from_points(points, (ocx, ocy), oa)
    
    # Also check all vertices of inner hexagons are inside outer hex
    all_verts = []
    for i in range(N):
        v = hex_vertices(1.0, positions[i], ys[i], angles[i])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    req_s2 = outer_hex_side_from_points(all_verts, (ocx, ocy), oa)
    required_s = max(required_s, req_s2)
    
    # Penalty for s being too small
    s_violation = max(0, required_s - s)
    
    # Pairwise collisions
    collision_penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            v1 = hex_vertices(1.0, positions[i], ys[i], angles[i])
            v2 = hex_vertices(1.0, positions[j], ys[j], angles[j])
            coll = smooth_collision(v1, v2, beta=10.0)
            collision_penalty += coll
    
    # Also penalize if s is negative or too small
    s_penalty = max(0, 2.0 - s) ** 2
    
    return s + 10.0 * s_violation**2 + 5.0 * collision_penalty + 0.1 * s_penalty

def construct():
    start_time = time.time()
    
    # Initial guess: hexagonal lattice arrangement
    # 11 hexagons: 1 center + 6 around + 4 more in second ring
    # Standard hex lattice positions (side=1, so center-to-center = 2*inradius = sqrt(3) for edge-adjacent)
    # Actually for two unit hexagons touching edge-to-edge, center distance = 2 * (sqrt(3)/2) = sqrt(3)
    # But if rotated, they can be closer. Let's start with a compact arrangement.
    
    # Hexagonal lattice: centers at distance sqrt(3) apart in 6 directions
    d = np.sqrt(3)  # center-to-center for edge-adjacent hexagons
    init_positions = []
    # Center
    init_positions.append((0.0, 0.0))
    # 6 neighbors
    for k in range(6):
        ang = np.pi/3 * k
        init_positions.append((d * np.cos(ang), d * np.sin(ang)))
    # 4 more: extend some of the 6 directions
    # Add at 2*d in directions 0, 1, 2, 3 (or some selection)
    for k in [0, 1, 2, 3]:
        ang = np.pi/3 * k
        init_positions.append((2*d * np.cos(ang), 2*d * np.sin(ang)))
    
    init_angles = [0.0] * N
    
    # Initial outer hexagon: centered at origin, angle 0, side estimated
    # The bounding hexagon of the initial arrangement
    init_pts = np.array(init_positions)
    init_oc = (0.0, 0.0)
    init_oa = 0.0
    # Estimate required side
    required = outer_hex_side_from_points(init_pts, init_oc, init_oa)
    init_s = required + 0.5  # Add margin
    
    x0 = []
    for i in range(N):
        x0.extend([init_positions[i][0], init_positions[i][1], init_angles[i]])
    x0.extend([init_oc[0], init_oc[1], init_oa, init_s])
    x0 = np.array(x0)
    
    best_x = x0.copy()
    best_val = objective(x0)
    
    # Try multiple optimization methods and restarts
    methods = ['L-BFGS-B', 'Nelder-Mead']
    restarts = 3
    
    for method in methods:
        for r in range(restarts):
            if time.time() - start_time > DEADLINE:
                break
            
            # Perturb initial guess for restarts
            if r > 0:
                np.random.seed(r * 1000 + hash(method) % 10000)
                x0_perturbed = x0 + np.random.randn(len(x0)) * 0.3
                x0_perturbed[-1] = init_s  # Keep s reasonable
            else:
                x0_perturbed = x0.copy()
            
            try:
                if method == 'L-BFGS-B':
                    res = minimize(objective, x0_perturbed, method='L-BFGS-B',
                                   options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8})
                else:
                    res = minimize(objective, x0_perturbed, method='Nelder-Mead',
                                   options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
                
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
            
            if time.time() - start_time > DEADLINE:
                break
    
    # Extract best solution
    positions = best_x[0:3*N:3]
    ys = best_x[1:3*N:3]
    angles = best_x[2:3*N:3]
    ocx, ocy = best_x[3*N], best_x[3*N+1]
    oa = best_x[3*N+2]
    s = best_x[3*N+3]
    
    # Final polish: compute exact required side
    points = np.column_stack([positions, ys])
    all_verts = []
    for i in range(N):
        v = hex_vertices(1.0, positions[i], ys[i], angles[i])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    exact_s = max(outer_hex_side_from_points(points, (ocx, ocy), oa),
                  outer_hex_side_from_points(all_verts, (ocx, ocy), oa))
    s = max(s, exact_s)
    
    # Build inner list
    inner = []
    for i in range(N):
        inner.append([float(positions[i]), float(ys[i]), float(angles[i])])
    
    # If we ran out of time or got a bad result, fall back to seed
    if s > 10.0 or len(inner) != N:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return inner, (float(ocx), float(ocy)), float(s), float(oa)
