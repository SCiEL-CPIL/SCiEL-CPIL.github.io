import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = time.time() + 140.0

def hex_vertices(cx, cy, angle_deg, side=1.0):
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def stp_overlap_penalty(verts_i, verts_j, eps=1e-4):
    """Separating Axis Theorem overlap check for two convex polygons.
    Returns penetration depth (positive if overlapping)."""
    # For each edge of both polygons, check if all points of other polygon are on one side
    # If for all edges, there's a separating axis, no overlap.
    # Otherwise, compute minimum penetration.
    
    min_pen = np.inf
    all_verts = [verts_i, verts_j]
    
    for poly in all_verts:
        for k in range(6):
            # Edge from vertex k to vertex k+1
            p1 = poly[k]
            p2 = poly[(k+1) % 6]
            # Edge vector
            edge = p2 - p1
            # Normal (perpendicular, pointing outward)
            normal = np.array([edge[1], -edge[0]])
            # Ensure normal points outward (away from center of this polygon)
            center = np.mean(poly, axis=0)
            if np.dot(normal, center - p1) < 0:
                normal = -normal
            normal = normal / (np.linalg.norm(normal) + 1e-12)
            
            # Project both polygons onto this normal
            proj_i = np.dot(verts_i - p1, normal)
            proj_j = np.dot(verts_j - p1, normal)
            
            min_i, max_i = np.min(proj_i), np.max(proj_i)
            min_j, max_j = np.min(proj_j), np.max(proj_j)
            
            # Check for separation
            if max_i < min_j - eps or max_j < min_i - eps:
                return 0.0  # Separated
            
            # Penetration depth along this axis
            pen = min(max_i, max_j) - max(min_i, min_j)
            min_pen = min(min_pen, pen)
    
    return min_pen

def overlap_penalty(positions, angles, eps=1e-4):
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            # Quick reject
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            if dx*dx + dy*dy > 4.5:  # (2.25)^2, generous bound
                continue
            vi = hex_vertices(*positions[i], angles[i])
            vj = hex_vertices(*positions[j], angles[j])
            pen = stp_overlap_penalty(vi, vj, eps)
            penalty += max(0, pen)
    return penalty

def container_side(positions, angles, container_angle_deg):
    """Min side of regular hexagon (given angle) centered at origin containing all inner hexagons."""
    a = np.radians(container_angle_deg)
    all_verts = []
    for (cx, cy), ang in zip(positions, angles):
        verts = hex_vertices(cx, cy, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Rotate to make container axis-aligned
    cos_a, sin_a = np.cos(-a), np.sin(-a)
    rx = all_verts[:, 0] * cos_a - all_verts[:, 1] * sin_a
    ry = all_verts[:, 0] * sin_a + all_verts[:, 1] * cos_a
    
    # For axis-aligned hexagon with vertices at angles 0,60,...,300 (radius=s):
    # Point (x,y) inside iff |x*cos(k*60) + y*sin(k*60)| <= s for all k
    dirs = np.column_stack([np.cos(np.arange(6)*np.pi/3), np.sin(np.arange(6)*np.pi/3)])
    dots = np.abs(rx * dirs[:, 0] + ry * dirs[:, 1])
    s = np.max(dots)
    return s

def objective(params):
    if time.time() > DEADLINE:
        return 1e10
    positions = [(params[3*i], params[3*i+1]) for i in range(N)]
    angles = [params[3*i+2] for i in range(N)]
    container_angle = params[3*N]
    
    s = container_side(positions, angles, container_angle)
    pen = overlap_penalty(positions, angles)
    return s + 50.0 * pen

def construct():
    start_time = time.time()
    sqrt3 = np.sqrt(3)
    
    # Hexagonal lattice initial positions
    init_positions = [(0.0, 0.0)]
    init_angles = [0.0]
    for k in range(6):
        ang = k * np.pi / 3
        init_positions.append((2.0 * np.cos(ang), 2.0 * np.sin(ang)))
        init_angles.append(0.0)
    for k in [0, 1, 3, 4]:
        ang = k * np.pi / 3
        init_positions.append((4.0 * np.cos(ang), 4.0 * np.sin(ang)))
        init_angles.append(0.0)
    
    params0 = []
    for (px, py), ang in zip(init_positions, init_angles):
        params0.extend([px, py, ang])
    params0.append(0.0)
    params0 = np.array(params0)
    
    best_params = params0.copy()
    best_val = objective(params0)
    
    rng = np.random.default_rng(123)
    n_restarts = 8
    
    for r in range(n_restarts):
        if time.time() > DEADLINE - 40:
            break
        
        if r == 0:
            p = params0.copy()
        elif r == 1:
            p = params0 + rng.normal(0, 0.3, size=params0.shape)
        else:
            p = params0 + rng.normal(0, 0.8, size=params0.shape)
        
        try:
            res = minimize(objective, p, method='L-BFGS-B', 
                          options={'maxiter': 300, 'ftol': 1e-7, 'gtol': 1e-7})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    positions = [(best_params[3*i], best_params[3*i+1]) for i in range(N)]
    angles = [best_params[3*i+2] for i in range(N)]
    container_angle = best_params[3*N]
    
    s = container_side(positions, angles, container_angle)
    pen = overlap_penalty(positions, angles)
    
    if pen > 1e-3:
        # Fallback to simple line
        positions = [(4.0 * k, 0.0) for k in range(-(N//2), N - N//2)]
        angles = [0.0] * N
        container_angle = 0.0
        s = container_side(positions, angles, container_angle)
    
    inner = [[px, py, ang] for (px, py), ang in zip(positions, angles)]
    return inner, (0.0, 0.0), s, container_angle

