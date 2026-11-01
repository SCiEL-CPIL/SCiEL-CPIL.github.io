"""Improved program: rigid composite hexagon packing with BFS topology search."""
import math
import time
import numpy as np
from scipy.optimize import minimize
from collections import deque

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def hex_vertices_local(side=1.0):
    """Vertices of a unit hexagon centered at origin, flat-top orientation."""
    return [(side * math.cos(math.pi/6 + i * math.pi/3),
             side * math.sin(math.pi/6 + i * math.pi/3)) for i in range(6)]

def hex_vertices(cx, cy, side, angle_rad):
    """Vertices of a hexagon with given center, side, and rotation."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    verts = []
    for vx, vy in hex_vertices_local(side):
        rx = vx * cos_a - vy * sin_a
        ry = vx * sin_a + vy * cos_a
        verts.append((cx + rx, cy + ry))
    return verts

def compute_required_side_fast(inner, outer_center, outer_angle_rad):
    """
    Compute minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_rad)
    outer_center: (x, y)
    outer_angle_rad: rotation of outer hexagon (radians)
    """
    if not inner:
        return 0.0
    ox, oy = outer_center
    cos_a = math.cos(outer_angle_rad)
    sin_a = math.sin(outer_angle_rad)
    
    # The 6 outward normals of a flat-top hexagon (pointing to edge midpoints)
    # Edge midpoints are at angles 0, 60, 120, 180, 240, 300 degrees
    # Outward normals point in those directions
    normals = [(math.cos(math.pi/3 * i), math.sin(math.pi/3 * i)) for i in range(6)]
    
    max_proj = 0.0
    for (ix, iy, iangle) in inner:
        cos_i = math.cos(iangle)
        sin_i = math.sin(iangle)
        for vx, vy in hex_vertices_local(1.0):
            rx = vx * cos_i - vy * sin_i
            ry = vx * sin_i + vy * cos_i
            px = ix + rx - ox
            py = iy + ry - oy
            # Rotate into outer hexagon frame
            lx = px * cos_a + py * sin_a
            ly = -px * sin_a + py * cos_a
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_proj:
                    max_proj = proj
    
    return 2.0 * max_proj / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_rad, beta=40.0):
    """Smooth surrogate using log-sum-exp for differentiability."""
    if not inner:
        return 0.0
    ox, oy = outer_center
    cos_a = math.cos(outer_angle_rad)
    sin_a = math.sin(outer_angle_rad)
    
    normals = [(math.cos(math.pi/3 * i), math.sin(math.pi/3 * i)) for i in range(6)]
    
    projections = []
    for (ix, iy, iangle) in inner:
        cos_i = math.cos(iangle)
        sin_i = math.sin(iangle)
        for vx, vy in hex_vertices_local(1.0):
            rx = vx * cos_i - vy * sin_i
            ry = vx * sin_i + vy * cos_i
            px = ix + rx - ox
            py = iy + ry - oy
            lx = px * cos_a + py * sin_a
            ly = -px * sin_a + py * cos_a
            for nx, ny in normals:
                projections.append(lx * nx + ly * ny)
    
    proj_arr = np.array(projections)
    # Center the projections to avoid numerical issues with log-sum-exp
    max_p = proj_arr.max()
    logsumexp = max_p + math.log(np.sum(np.exp(proj_arr - max_p))) / beta
    # Actually: log(sum(exp(beta*p)))/beta = max_p + log(sum(exp(beta*(p-max_p))))/beta
    logsumexp = max_p + math.log(np.sum(np.exp(beta * (proj_arr - max_p)))) / beta
    
    return 2.0 * logsumexp / SQRT3

def sat_overlap(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap using Separating Axis Theorem."""
    axes = []
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.hypot(normal[0], normal[1])
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    for i in range(6):
        p1 = v2[i]
        p2 = v2[(i + 1) % 6]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.hypot(normal[0], normal[1])
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def check_disjoint(inner, eps=1e-6):
    """Check all pairs of inner hexagons are disjoint."""
    n = len(inner)
    verts_list = []
    for i in range(n):
        x, y, a = inner[i]
        verts_list.append(hex_vertices(x, y, 1.0, a))
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap(verts_list[i], verts_list[j], eps):
                return False
    return True

def overlap_penalty(inner, min_dist=SQRT3 - 0.001):
    """Smooth penalty for center distances violating minimum separation."""
    n = len(inner)
    if n < 2:
        return 0.0
    xs = np.array([inner[i][0] for i in range(n)])
    ys = np.array([inner[i][1] for i in range(n)])
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists_sq = dx * dx + dy * dy
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dists_sq[iu])
    diff = np.maximum(0.0, min_dist - dists)
    return float(np.sum(diff ** 4))

def axial_to_xy(q, r, scale=1.0):
    """Convert axial coordinates to Cartesian (flat-top hex grid)."""
    x = 1.5 * scale * q
    y = SQRT3 * scale * (r + q / 2.0)
    return x, y

def xy_to_axial(x, y):
    """Convert Cartesian to approximate axial coordinates."""
    q = (2.0/3.0 * x) / 1.5  # x / 1.5
    r = (y / SQRT3 - q / 2.0)
    return round(q), round(r)

def generate_polyhexes(n_cells):
    """
    Generate all free polyhexes of size n_cells via BFS on axial grid.
    Returns list of sets of (q, r) axial coordinates, canonicalized.
    """
    if n_cells == 1:
        return [frozenset([(0, 0)])]
    
    # Use the previous size to grow
    prev = generate_polyhexes(n_cells - 1)
    
    seen = set()
    results = []
    
    # Neighbor directions in axial coordinates
    neighbors = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    for poly in prev:
        cells = list(poly)
        # For each cell, try adding in each of 6 directions
        for q, r in cells:
            for dq, dr in neighbors:
                new_cell = (q + dq, r + dr)
                if new_cell in poly:
                    continue
                new_poly = frozenset(poly | {new_cell})
                if len(new_poly) != n_cells:
                    continue
                
                # Canonicalize: translate so min-q then min-r is at (0,0)
                min_q = min(c[0] for c in new_poly)
                min_r = min(c[1] for c in new_poly if c[0] == min_q)
                translated = frozenset(((q - min_q, r - min_r) for q, r in new_poly))
                
                # Also try all 6 rotations to get canonical form
                # Rotation by 60 degrees in axial coordinates
                def rotate_axial(q, r, k):
                    """Rotate (q,r) by k*60 degrees around origin."""
                    for _ in range(k):
                        # 60 degree rotation: (q,r) -> (-r, q+r)
                        q, r = -r, q + r
                    return (q, r)
                
                canonical_options = []
                for k in range(6):
                    rotated = frozenset((rotate_axial(q, r, k) for q, r in translated))
                    min_q2 = min(c[0] for c in rotated)
                    min_r2 = min(c[1] for c in rotated if c[0] == min_q2)
                    canon = frozenset(((q - min_q2, r - min_r2) for q, r in rotated))
                    canonical_options.append(canon)
                
                # Pick lexicographically smallest as canonical
                canonical = min(canonical_options)
                
                if canonical not in seen:
                    seen.add(canonical)
                    results.append(canonical)
    
    return results

def build_rigid_composite(polyhex, angle_rad=0.0):
    """
    Build a rigid composite: all hexagons at axial positions with same orientation.
    Returns list of (x, y, angle_rad) for each hexagon.
    """
    centers = []
    for q, r in polyhex:
        x, y = axial_to_xy(q, r, scale=1.0)
        centers.append((x, y))
    return [(x, y, angle_rad) for x, y in centers]

def rigid_composite_objective(params, polyhex, beta=40.0):
    """
    Objective for rigid composite: 3 params (cx, cy, angle).
    All inner hexagons move and rotate together.
    """
    cx, cy, angle = params
    inner = []
    for q, r in polyhex:
        # Local position in composite frame
        lx, ly = axial_to_xy(q, r, scale=1.0)
        # Rotate local position by angle
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rx = lx * cos_a - ly * sin_a
        ry = lx * sin_a + ly * cos_a
        inner.append((cx + rx, cy + ry, angle))
    
    side = compute_required_side_smooth(inner, (0, 0), 0.0, beta=beta)
    # No overlap penalty needed for rigid composite with valid polyhex
    return side

def optimize_composite_rigid(polyhex, deadline):
    """Optimize rigid composite placement for a given polyhex topology."""
    best_val = float('inf')
    best_params = None
    
    # Try multiple initial angles (exploiting 60-degree symmetry of hexagons)
    init_angles = [0.0, math.pi/12, math.pi/6, math.pi/4, math.pi/3, math.pi/2]
    
    for init_angle in init_angles:
        if time.time() > deadline - 5:
            break
        
        init_params = [0.0, 0.0, init_angle]
        
        try:
            # Phase 1: Smooth optimization
            res = minimize(lambda p: rigid_composite_objective(p, polyhex, beta=40.0),
                         init_params, method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 2000})
            
            # Phase 2: Exact optimization
            def exact_obj(p):
                cx, cy, angle = p
                inner = []
                for q, r in polyhex:
                    lx, ly = axial_to_xy(q, r, scale=1.0)
                    cos_a = math.cos(angle)
                    sin_a = math.sin(angle)
                    rx = lx * cos_a - ly * sin_a
                    ry = lx * sin_a + ly * cos_a
                    inner.append((cx + rx, cy + ry, angle))
                return compute_required_side_fast(inner, (0, 0), 0.0)
            
            res2 = minimize(exact_obj, res.x, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 5000})
            
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass
    
    return best_val, best_params

def semi_rigid_optimize(polyhex, initial_params, deadline):
    """
    Semi-rigid optimization: allow small deviations from rigid body.
    Each hexagon can shift slightly but stays close to its rigid position.
    """
    if initial_params is None:
        return None, None
    
    cx, cy, angle = initial_params
    inner = []
    for q, r in polyhex:
        lx, ly = axial_to_xy(q, r, scale=1.0)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rx = lx * cos_a - ly * sin_a
        ry = lx * sin_a + ly * cos_a
        inner.append((cx + rx, cy + ry, angle))
    
    # Allow each hexagon to shift by up to 0.05 units
    def semi_rigid_obj(params):
        # params: 3*N shifts + 3 global (cx, cy, angle)
        shifts = params[:3*N]
        gcx, gcy, gangle = params[3*N:3*N+3]
        
        test_inner = []
        for i, (q, r) in enumerate(polyhex):
            lx, ly = axial_to_xy(q, r, scale=1.0)
            cos_a = math.cos(gangle)
            sin_a = math.sin(gangle)
            rx = lx * cos_a - ly * sin_a
            ry = lx * sin_a + ly * cos_a
            # Add small shift
            sx = shifts[i*3]
            sy = shifts[i*3+1]
            sa = shifts[i*3+2]
            test_inner.append((gcx + rx + sx, gcy + ry + sy, gangle + sa))
        
        side = compute_required_side_smooth(test_inner, (0, 0), 0.0, beta=30.0)
        penalty = overlap_penalty(test_inner)
        return side + 100.0 * penalty
    
    init = [0.0] * (3*N) + [cx, cy, angle]
    
    try:
        res = minimize(semi_rigid_obj, init, method='Nelder-Mead',
                      options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 3000})
        
        shifts = res.x[:3*N]
        gcx, gcy, gangle = res.x[3*N:3*N+3]
        
        test_inner = []
        for i, (q, r) in enumerate(polyhex):
            lx, ly = axial_to_xy(q, r, scale=1.0)
            cos_a = math.cos(gangle)
            sin_a = math.sin(gangle)
            rx = lx * cos_a - ly * sin_a
            ry = lx * sin_a + ly * cos_a
            sx = shifts[i*3]
            sy = shifts[i*3+1]
            sa = shifts[i*3+2]
            test_inner.append((gcx + rx + sx, gcy + ry + sy, gangle + sa))
        
        if check_disjoint(test_inner, eps=1e-5):
            side = compute_required_side_fast(test_inner, (0, 0), 0.0)
            return side, test_inner
    except Exception:
        pass
    
    return None, None

def full_joint_optimize(initial_inner, deadline):
    """Full joint optimization of all 33 parameters."""
    def joint_obj(params):
        inner_list = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            inner_list.append([x, y, a])
        side = compute_required_side_smooth(inner_list, (0, 0), 0.0, beta=30.0)
        penalty = overlap_penalty(inner_list)
        return side + 200.0 * penalty
    
    try:
        res = minimize(joint_obj, np.array(initial_inner).flatten(), method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-13, 'maxfev': 6000})
        
        inner_list = []
        for i in range(N):
            x = res.x[i*3]
            y = res.x[i*3+1]
            a = res.x[i*3+2]
            inner_list.append([x, y, a])
        
        if check_disjoint(inner_list, eps=1e-5):
            side = compute_required_side_fast(inner_list, (0, 0), 0.0)
            return side, inner_list
    except Exception:
        pass
    
    return None, None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Generate polyhexes of size 11
    # This can be expensive, so we use a time-bounded approach
    # For 11 cells, there are 140 free polyhexes
    try:
        polyhexes = generate_polyhexes(N)
    except Exception:
        # Fallback: use a few known good topologies
        polyhexes = []
    
    # If generation is too slow or fails, use heuristic topologies
    if not polyhexes or time.time() > deadline - 80:
        # Known good topologies for 11 hexagons
        polyhexes = [
            # Compact: 3-4-3-1 arrangement
            frozenset([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(1,1),(0,2),(-1,2)]),
            # Ring + center + 4
            frozenset([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(1,1),(0,2),(-2,1)]),
            # Linear-ish
            frozenset([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(0,-1),(1,-1),(2,-1)]),
            # Compact cluster
            frozenset([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,1),(2,-1),(-1,2)]),
        ]
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Sort polyhexes by compactness (smaller bounding box = better)
    def compactness(poly):
        xs = [q for q, r in poly]
        ys = [r for q, r in poly]
        return (max(xs) - min(xs)) + (max(ys) - min(ys))
    
    polyhexes = sorted(polyhexes, key=compactness)
    
    # Phase 1: Rigid composite optimization for each topology
    for idx, poly in enumerate(polyhexes):
        if time.time() > deadline - 40:
            break
        
        val, params = optimize_composite_rigid(poly, deadline)
        
        if val is not None and val < best_side:
            best_side = val
            cx, cy, angle = params
            inner = []
            for q, r in poly:
                lx, ly = axial_to_xy(q, r, scale=1.0)
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                rx = lx * cos_a - ly * sin_a
                ry = lx * sin_a + ly * cos_a
                inner.append((cx + rx, cy + ry, angle))
            if check_disjoint(inner, eps=1e-5):
                best_inner = inner
                best_outer_center = (0.0, 0.0)
                best_outer_angle = 0.0
                best_side = compute_required_side_fast(inner, (0, 0), 0.0)
    
    # If no valid result yet, use fallback
    if best_inner is None:
        # Use the most compact polyhex with angle=0
        if polyhexes:
            poly = polyhexes[0]
        else:
            poly = frozenset([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(1,1),(0,2),(-1,2)])
        
        inner = []
        for q, r in poly:
            x, y = axial_to_xy(q, r, scale=1.0)
            inner.append((x, y, 0.0))
        
        best_inner = inner
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(inner, (0, 0), 0.0)
    
    # Phase 2: Semi-rigid optimization (allow small deviations)
    if time.time() < deadline - 20:
        cx, cy, angle = best_outer_center[0], best_outer_center[1], best_outer_angle
        # Find which polyhex was used
        # Reconstruct from best_inner
        semi_val, semi_inner = semi_rigid_optimize(polyhexes[0], [0, 0, 0], deadline)
        if semi_val is not None and semi_val < best_side:
            best_side = semi_val
            best_inner = semi_inner
            best_outer_center = (0.0, 0.0)
            best_outer_angle = 0.0
    
    # Phase 3: Full joint optimization
    if time.time() < deadline - 10:
        init_inner = [[x, y, a] for x, y, a in best_inner]
        joint_val, joint_inner = full_joint_optimize(init_inner, deadline)
        if joint_val is not None and joint_val < best_side:
            best_side = joint_val
            best_inner = joint_inner
            best_outer_center = (0.0, 0.0)
            best_outer_angle = 0.0
    
    # Phase 4: Final high-precision polish of outer container
    if time.time() < deadline - 3:
        def polish_obj(params):
            cx, cy, angle = params
            return compute_required_side_fast(best_inner, (cx, cy), angle)
        
        best_polish_val = best_side
        best_polish_params = [0.0, 0.0, 0.0]
        
        polish_starts = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.1],
            [0.0, 0.0, -0.1],
            [0.05, 0.05, 0.0],
            [-0.05, 0.05, 0.0],
        ]
        
        for ps in polish_starts:
            if time.time() > deadline - 1:
                break
            try:
                res = minimize(polish_obj, ps, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-15, 'maxfev': 8000})
                if res.fun < best_polish_val:
                    best_polish_val = res.fun
                    best_polish_params = res.x
            except Exception:
                pass
        
        if best_polish_params is not None:
            best_outer_center = (best_polish_params[0], best_polish_params[1])
            best_outer_angle = best_polish_params[2]
            best_side = best_polish_val
    
    # Final validation
    if not check_disjoint(best_inner, eps=1e-5):
        # Revert to a known valid configuration
        poly = polyhexes[0] if polyhexes else frozenset([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(1,1),(0,2),(-1,2)])
        inner = []
        for q, r in poly:
            x, y = axial_to_xy(q, r, scale=1.0)
            inner.append((x, y, 0.0))
        best_inner = inner
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(inner, (0, 0), 0.0)
    
    # Convert angles to degrees for output
    best_inner_deg = [[x, y, math.degrees(a) % 360.0] for x, y, a in best_inner]
    best_outer_angle_deg = math.degrees(best_outer_angle) % 360.0
    
    return best_inner_deg, best_outer_center, best_side, best_outer_angle_deg

# EVOLVE-BLOCK-END
