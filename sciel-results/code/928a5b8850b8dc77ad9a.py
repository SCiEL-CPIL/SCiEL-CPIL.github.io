"""
Packing 12 unit regular hexagons into a regular hexagon.
Strategy:
1. SDF-based smooth distance for hexagon-hexagon separation.
2. L-BFGS-B optimization with smooth penalty for overlaps.
3. Nelder-Mead polish for tight packing.
4. Grid search over outer hexagon rotation and center.
5. Deadline guard for 150s limit.
"""
import time
import math
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # Safety margin before the 150s kill

# EVOLVE-BLOCK-START

def _hex_vertices(theta, center=(0.0, 0.0), side=1.0):
    """
    Returns vertices of a regular hexagon with given side length,
    centered at 'center', rotated by 'theta'.
    For a regular hexagon of side s, circumradius R = s.
    """
    cx, cy = center
    verts = np.array([
        [side * math.cos(theta + k * math.pi / 3.0),
         side * math.sin(theta + k * math.pi / 3.0)]
        for k in range(6)
    ])
    return verts + np.array([cx, cy])

def _hexagon_sdf_point(px, py, center, theta, side=1.0):
    """
    Signed distance from point (px, py) to a regular hexagon.
    Negative inside, positive outside.
    """
    cx, cy = center
    # Transform point to hexagon's local frame
    dx = px - cx
    dy = py - cy
    # Rotate by -theta
    cos_t = math.cos(-theta)
    sin_t = math.sin(-theta)
    lx = dx * cos_t - dy * sin_t
    ly = dx * sin_t + dy * cos_t
    
    # In local frame, hexagon has vertices at angles k*pi/3
    # The hexagon is symmetric; we can check against the 6 sides
    # For a unit hexagon, the apothem is sqrt(3)/2
    apothem = side * math.sqrt(3.0) / 2.0
    
    # The hexagon can be described by 6 half-planes
    # For each edge, check if point is outside
    min_dist = float('inf')
    for k in range(6):
        # Edge normal direction (pointing outward)
        angle = theta + math.pi / 6.0 + k * math.pi / 3.0
        nx = math.cos(angle)
        ny = math.sin(angle)
        # Distance from center to this edge along normal is apothem
        # Project point onto this normal
        proj = dx * nx + dy * ny
        # If proj > apothem, point is outside this half-plane
        if proj > apothem:
            dist = proj - apothem
            min_dist = min(min_dist, dist)
    
    # Also need to check corner regions
    # For a regular hexagon, the SDF can be computed more carefully
    # But for optimization purposes, the half-plane approximation is smooth enough
    # Actually, let's use a proper SDF
    
    # Proper SDF for regular hexagon:
    # In local frame, the hexagon has vertices at (cos(k*pi/3), sin(k*pi/3)) for k=0..5
    # The SDF is the minimum distance to any edge, considering the 6 edges
    
    # Let's use the approach: for each edge, compute distance to the line segment
    # But that's not smooth. Instead, use the half-plane approach with a correction.
    
    # For optimization, we'll use the maximum of the 6 half-plane violations
    # This gives a smooth-ish penalty
    violations = []
    for k in range(6):
        angle = math.pi / 6.0 + k * math.pi / 3.0
        nx = math.cos(angle)
        ny = math.sin(angle)
        proj = lx * nx + ly * ny
        violations.append(proj - apothem)
    
    # The SDF is approximately the max of these violations (clipped to 0)
    # But this overestimates in corner regions. For penalty purposes, it's fine.
    sdf = max(violations)
    return sdf

def _hexagon_hexagon_distance(A_center, A_theta, B_center, B_theta, side=1.0):
    """
    Approximate signed distance between two hexagons.
    Negative if overlapping, positive if separated.
    Uses SDF of one hexagon evaluated at vertices of the other.
    """
    # Get vertices of both hexagons
    verts_A = _hex_vertices(A_theta, A_center, side)
    verts_B = _hex_vertices(B_theta, B_center, side)
    
    # Compute min SDF of B's vertices w.r.t. A, and vice versa
    min_sdf = float('inf')
    for vx, vy in verts_B:
        sdf = _hexagon_sdf_point(vx, vy, A_center, A_theta, side)
        min_sdf = min(min_sdf, sdf)
    for vx, vy in verts_A:
        sdf = _hexagon_sdf_point(vx, vy, B_center, B_theta, side)
        min_sdf = min(min_sdf, sdf)
    
    return min_sdf

def _sat_overlap_depth(A, B):
    """
    Checks overlap between two convex polygons A and B (arrays of (x,y)).
    Returns (overlap_depth, axis) if overlapping, else (0.0, None).
    """
    min_pen = np.inf
    min_axis = None
    
    for poly in [A, B]:
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            nl = math.hypot(normal[0], normal[1])
            if nl < 1e-9:
                continue
            normal = normal / nl
            
            projA = A @ normal
            projB = B @ normal
            
            maxA = projA.max()
            minA = projA.min()
            maxB = projB.max()
            minB = projB.min()
            
            if maxA < minB or maxB < minA:
                return 0.0, None
            
            depth = min(maxA, maxB) - max(minA, minB)
            if depth < min_pen:
                min_pen = depth
                cA = A.mean(axis=0) @ normal
                cB = B.mean(axis=0) @ normal
                if cB > cA:
                    min_axis = normal
                else:
                    min_axis = -normal
    
    if min_pen == np.inf:
        return 0.0, None
    return min_pen, min_axis

def _outer_hex_side(centers, angles, outer_theta=0.0):
    """
    Calculates the minimum side length of a regular hexagon (orientation outer_theta)
    that contains all inner hexagons.
    """
    norms = np.array([
        [math.cos(outer_theta + math.pi / 6.0 + k * math.pi / 3.0),
         math.sin(outer_theta + math.pi / 6.0 + k * math.pi / 3.0)]
        for k in range(6)
    ])
    
    min_proj = np.inf * np.ones(6)
    max_proj = -np.inf * np.ones(6)
    
    for i in range(N):
        verts = _hex_vertices(angles[i], centers[i])
        projs = verts @ norms.T
        min_proj = np.minimum(min_proj, projs.min(axis=0))
        max_proj = np.maximum(max_proj, projs.max(axis=0))
    
    widths = max_proj - min_proj
    return float(np.max(widths) / math.sqrt(3.0))

def _is_valid(centers, angles, tol=1e-9):
    """
    Checks if the configuration has no overlaps.
    """
    verts = [_hex_vertices(angles[i], centers[i]) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            depth, _ = _sat_overlap_depth(verts[i], verts[j])
            if depth > tol:
                return False
    return True

def _smooth_penalty(centers, angles, penalty_weight=10.0, smooth_eps=0.01):
    """
    Smooth penalty for overlaps using SDF-based distances.
    """
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dist = _hexagon_hexagon_distance(
                centers[i], angles[i], centers[j], angles[j]
            )
            if dist < 0:
                # Overlap: penalty proportional to overlap squared
                penalty += penalty_weight * dist * dist
            else:
                # Small smooth transition
                if dist < smooth_eps:
                    penalty += penalty_weight * 0.1 * (smooth_eps - dist) ** 2
    return penalty

def _cost_function(params, outer_theta=0.0, penalty_weight=10.0):
    """
    Cost function for optimization.
    params: array of shape (3*N,) -> [x0, y0, a0, x1, y1, a1, ...]
    """
    centers = params[0::3].reshape(-1, 2)
    angles = params[1::3]
    
    side = _outer_hex_side(centers, angles, outer_theta)
    penalty = _smooth_penalty(centers, angles, penalty_weight)
    
    return side + penalty

def _nm_cost(params, outer_theta=0.0):
    """
    Nelder-Mead cost: hard penalty for overlaps.
    """
    centers = params[0::3].reshape(-1, 2)
    angles = params[1::3]
    
    side = _outer_hex_side(centers, angles, outer_theta)
    
    verts = [_hex_vertices(angles[i], centers[i]) for i in range(N)]
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth, _ = _sat_overlap_depth(verts[i], verts[j])
            if depth > 0:
                overlap_penalty += depth ** 2
    
    return side + 100.0 * overlap_penalty

def construct():
    start_time = time.time()
    random.seed(42)
    
    best_side = float('inf')
    best_params = None
    best_outer_center = (0.0, 0.0)
    best_outer_theta = 0.0

    # Initializer: Hexagonal lattice
    def init_lattice():
        d = 2.0 * math.sqrt(3.0)
        a1 = np.array([d, 0.0])
        a2 = np.array([d / 2.0, d * math.sqrt(3.0) / 2.0])
        pts = []
        for i in range(-1, 3):
            for j in range(-1, 2):
                p = i * a1 + j * a2
                pts.append(p)
        pts = np.array(pts)
        c = pts.mean(axis=0)
        pts = pts - c
        centers = pts.tolist()
        angles = [0.0] * N
        return centers, angles

    def init_random():
        centers = []
        angles = []
        for _ in range(N):
            centers.append([random.uniform(-3, 3), random.uniform(-3, 3)])
            angles.append(random.uniform(0, math.pi / 3))
        return centers, angles

    def init_ring():
        # Place hexagons in a ring pattern
        centers = []
        angles = []
        # Center hexagon
        centers.append([0.0, 0.0])
        angles.append(0.0)
        # 6 hexagons around
        d = 2.0 * math.sqrt(3.0)
        for k in range(6):
            angle = k * math.pi / 3.0
            centers.append([d * math.cos(angle), d * math.sin(angle)])
            angles.append(0.0)
        # 5 more in outer ring
        d2 = d * math.sqrt(3.0)
        for k in range(5):
            angle = k * 2 * math.pi / 5.0 + math.pi / 5.0
            centers.append([d2 * math.cos(angle), d2 * math.sin(angle)])
            angles.append(0.0)
        # Center all
        c = np.array(centers).mean(axis=0)
        centers = [list(np.array(cen) - c) for cen in centers]
        return centers, angles

    # Physics Relaxation
    def physics_relax(centers, angles, steps=200, dt=0.01):
        centers = np.array(centers, dtype=float)
        angles = np.array(angles, dtype=float)
        vel = np.zeros((N, 2))
        
        for s in range(steps):
            forces = np.zeros((N, 2))
            
            for i in range(N):
                forces[i] -= 0.05 * centers[i]
            
            verts = [_hex_vertices(angles[i], centers[i]) for i in range(N)]
            for i in range(N):
                for j in range(i + 1, N):
                    depth, axis = _sat_overlap_depth(verts[i], verts[j])
                    if depth > 0 and axis is not None:
                        f = 5.0 * depth
                        forces[i] -= f * axis
                        forces[j] += f * axis
            
            vel += forces * dt
            vel *= 0.9
            centers += vel * dt
            
        return centers, angles

    # Optimization Loop
    initializers = [init_lattice, init_ring, init_random, init_random]
    
    for init_idx, init_func in enumerate(initializers):
        if time.time() - start_time > DEADLINE - 25:
            break
            
        centers, angles = init_func()
        
        # 1. Physics Relaxation
        c_rel, a_rel = physics_relax(centers, angles, steps=150)
        
        # Flatten parameters
        params = np.zeros(3 * N)
        for i in range(N):
            params[3 * i] = c_rel[i][0]
            params[3 * i + 1] = c_rel[i][1]
            params[3 * i + 2] = a_rel[i]
            
        # Try a few outer thetas
        outer_thetas = [0.0, math.pi / 6, math.pi / 3, math.pi / 12]
        
        for outer_theta in outer_thetas:
            if time.time() - start_time > DEADLINE - 10:
                break
                
            # L-BFGS-B optimization
            try:
                res_lbfgs = minimize(
                    _cost_function,
                    params,
                    method='L-BFGS-B',
                    args=(outer_theta, 10.0),
                    options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                params = res_lbfgs.x
            except Exception:
                pass
            
            # Nelder-Mead polish
            try:
                res_nm = minimize(
                    _nm_cost,
                    params,
                    method='Nelder-Mead',
                    args=(outer_theta,),
                    options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-7}
                )
                params = res_nm.x
            except Exception:
                pass
            
            # Extract results
            c_opt = params[0::3].reshape(-1, 2)
            a_opt = params[1::3]
            
            # Check validity
            if _is_valid(c_opt, a_opt):
                side = _outer_hex_side(c_opt, a_opt, outer_theta)
                if side < best_side:
                    best_side = side
                    best_params = params.copy()
                    best_outer_theta = outer_theta
                    
                    # Calculate optimal center
                    norms = np.array([
                        [math.cos(outer_theta + math.pi / 6.0 + k * math.pi / 3.0),
                         math.sin(outer_theta + math.pi / 6.0 + k * math.pi / 3.0)]
                        for k in range(6)
                    ])
                    min_proj = np.inf * np.ones(6)
                    max_proj = -np.inf * np.ones(6)
                    for i in range(N):
                        verts = _hex_vertices(a_opt[i], c_opt[i])
                        projs = verts @ norms.T
                        min_proj = np.minimum(min_proj, projs.min(axis=0))
                        max_proj = np.maximum(max_proj, projs.max(axis=0))
                    
                    # Center is midpoint of projections
                    center_proj = (min_proj + max_proj) / 2.0
                    # Invert: center in xy space
                    # For a regular hexagon, the center can be estimated as the average of all vertices
                    all_verts = []
                    for i in range(N):
                        all_verts.append(_hex_vertices(a_opt[i], c_opt[i]))
                    all_verts = np.vstack(all_verts)
                    best_outer_center = tuple(float(x) for x in all_verts.mean(axis=0))
            
            # Grid search over outer center offset
            if best_params is not None and time.time() - start_time < DEADLINE - 5:
                for dx in np.linspace(-0.3, 0.3, 7):
                    for dy in np.linspace(-0.3, 0.3, 7):
                        if time.time() - start_time > DEADLINE - 5:
                            break
                        # Shift all centers
                        shifted = params.copy()
                        shifted[0::3] += dx
                        shifted[1::3] += dy
                        c_s = shifted[0::3].reshape(-1, 2)
                        a_s = shifted[1::3]
                        if _is_valid(c_s, a_s):
                            s_s = _outer_hex_side(c_s, a_s, outer_theta)
                            if s_s < best_side:
                                best_side = s_s
                                best_params = shifted
                                best_outer_center = (float(dx), float(dy))

    if best_params is None:
        # Fallback
        centers, angles = init_lattice()
        best_params = np.zeros(3 * N)
        for i in range(N):
            best_params[3 * i] = centers[i][0]
            best_params[3 * i + 1] = centers[i][1]
            best_params[3 * i + 2] = angles[i]
        best_side = _outer_hex_side(np.array(centers), np.array(angles), 0.0)
        best_outer_theta = 0.0
        best_outer_center = (0.0, 0.0)

    # Finalize
    inner = []
    for i in range(N):
        x = float(best_params[3 * i])
        y = float(best_params[3 * i + 1])
        a_deg = float(math.degrees(best_params[3 * i + 2]))
        inner.append([x, y, a_deg])
        
    return inner, best_outer_center, best_side, float(math.degrees(best_outer_theta))

# EVOLVE-BLOCK-END
