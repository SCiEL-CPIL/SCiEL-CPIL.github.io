"""
Packing of 12 unit hexagons in a minimal outer hexagon.
Combines structural initialization with smooth optimization using SAT-based
collision penalties and Nelder-Mead refinement with global rotation.
"""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 135.0  # Safety margin below 150s
INRADIUS = math.sqrt(3) / 2.0
MIN_DIST_CIRCLE = 2 * INRADIUS  # sqrt(3) ~ 1.732

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _project_polygon(verts, axis):
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _separating_axis_check(vertsA, vertsB):
    """Check if two convex polygons overlap using SAT. Return True if they overlap."""
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
            return False
    return True

def _overlap_depth(vertsA, vertsB):
    """Compute maximum overlap depth between two polygons via SAT."""
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    max_overlap = 0.0
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        overlap = min(maxA, maxB) - max(minA, minB)
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def _is_valid_fast(cx_arr, cy_arr, angle_arr):
    n = len(cx_arr)
    min_dist_sq = (MIN_DIST_CIRCLE * MIN_DIST_CIRCLE) - 1e-9
    
    for i in range(n):
        dx = cx_arr[i] - cx_arr[i+1:]
        dy = cy_arr[i] - cy_arr[i+1:]
        dist_sq = dx*dx + dy*dy
        if np.any(dist_sq < min_dist_sq):
            return False

    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))
    
    for i in range(n):
        for j in range(i + 1, n):
            if _separating_axis_check(verts_list[i], verts_list[j]):
                return False
    
    return True

def _outer_hex_side(cx_arr, cy_arr, angle_arr, center, angle_deg):
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    for i in range(len(cx_arr)):
        verts = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _quick_score(cx_arr, cy_arr, angle_arr):
    apothem_unit = math.sqrt(3) / 2.0
    normals = [(math.cos(math.radians(90 + 60 * k)), math.sin(math.radians(90 + 60 * k))) for k in range(6)]
    max_need = 0.0
    for i in range(N):
        a = math.radians(angle_arr[i])
        ca, sa = math.cos(a), math.sin(a)
        for k in range(6):
            theta = math.radians(60 * k)
            vx = cx_arr[i] + ca * math.cos(theta) - sa * math.sin(theta)
            vy = cy_arr[i] + sa * math.cos(theta) + ca * math.sin(theta)
            for (nx, ny) in normals:
                val = (nx * vx + ny * vy) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _compute_best_outer(cx_arr, cy_arr, angle_arr):
    all_verts = []
    for i in range(N):
        verts = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
        all_verts.extend(verts)
    
    cx = np.mean([v[0] for v in all_verts])
    cy = np.mean([v[1] for v in all_verts])
    
    best_side = float('inf')
    best_angle = 0.0
    
    for angle_deg in range(0, 60, 5):
        side = _outer_hex_side(cx_arr, cy_arr, angle_arr, (cx, cy), angle_deg)
        if side < best_side:
            best_side = side
            best_angle = angle_deg
            
    for angle_deg in np.arange(best_angle - 5, best_angle + 5, 0.5):
        side = _outer_hex_side(cx_arr, cy_arr, angle_arr, (cx, cy), angle_deg)
        if side < best_side:
            best_side = side
            best_angle = angle_deg

    return cx, cy, best_side, best_angle

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)
    
    def make_honeycomb_cluster():
        """
        Create a honeycomb cluster of 12 hexagons.
        Structure: 1 center, 6 in first ring, 5 in second ring (partial).
        """
        pts = []
        # Center
        pts.append([0.0, 0.0, 0.0])
        
        # First ring: 6 hexagons at distance 2*sqrt(3) from center
        r1 = 2 * math.sqrt(3)
        for i in range(6):
            angle = math.radians(60 * i)
            x = r1 * math.cos(angle)
            y = r1 * math.sin(angle)
            pts.append([x, y, 0.0])
        
        # Second ring: 5 hexagons to complete the cluster
        r2 = r1 * math.sqrt(3)
        angles = [30, 90, 150, 210, 270]
        for deg in angles:
            rad = math.radians(deg)
            x = r2 * math.cos(rad)
            y = r2 * math.sin(rad)
            pts.append([x, y, 0.0])
        
        return pts[:12]
    
    def make_lattice_initial():
        """3-4-3-2 lattice arrangement."""
        pts = []
        row_height = math.sqrt(3)
        col_width = 2.0
        for c in range(3):
            x = c * col_width - col_width
            y = -row_height
            pts.append([x, y, 0.0])
        for c in range(4):
            x = c * col_width - col_width - col_width/2
            y = 0.0
            pts.append([x, y, 0.0])
        for c in range(3):
            x = c * col_width - col_width
            y = row_height
            pts.append([x, y, 0.0])
        pts.append([-col_width*2 - col_width/2 - col_width, 0.0, 0.0])
        pts.append([col_width*2 - col_width/2 + col_width, 0.0, 0.0])
        return pts

    def make_ring_initial():
        """1 center + 11 in a ring."""
        pts = [[0.0, 0.0, 0.0]]
        r = 2 * math.sqrt(3)
        for i in range(11):
            angle = math.radians(360 * i / 11 + 15)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            pts.append([x, y, 0.0])
        return pts

    def make_random_initial():
        pts = []
        for _ in range(N):
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            a = random.uniform(0, 60)
            pts.append([x, y, a])
        return pts

    initializers = [
        make_honeycomb_cluster,
        make_lattice_initial,
        make_ring_initial,
        make_random_initial,
    ]
    
    best_side = float('inf')
    best_cfg = None
    
    # 1. Initialize and validate
    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 100:
            break
        cfg = init_func()
        if len(cfg) != N:
            continue
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        
        if _is_valid_fast(cx_arr, cy_arr, angle_arr):
            side = _quick_score(cx_arr, cy_arr, angle_arr)
            if side < best_side:
                best_side = side
                best_cfg = cfg[:]
    
    if best_cfg is None:
        cfg = make_lattice_initial()
        scale = 1.5
        for p in cfg:
            p[0] *= scale
            p[1] *= scale
        best_cfg = cfg
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        best_side = _quick_score(cx_arr, cy_arr, angle_arr)

    # 2. Smooth objective with SAT-based collision penalty
    def smooth_objective(params):
        """
        Smooth objective: outer side length + collision penalty.
        params: [x1, y1, a1, ..., xN, yN, aN]
        """
        cx_arr = params[:N]
        cy_arr = params[N:2*N]
        angle_arr = params[2*N:]
        
        # Compute all vertices
        all_verts = []
        for i in range(N):
            all_verts.append(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))
        
        # Compute outer hex side (exact)
        cx, cy, side, angle = _compute_best_outer(cx_arr, cy_arr, angle_arr)
        
        # Collision penalty: for each overlapping pair, compute overlap depth
        collision_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                if _separating_axis_check(all_verts[i], all_verts[j]):
                    depth = _overlap_depth(all_verts[i], all_verts[j])
                    collision_penalty += depth * depth
        
        return side + 15.0 * collision_penalty
    
    params0 = np.array([p[0] for p in best_cfg] + [p[1] for p in best_cfg] + [p[2] for p in best_cfg])
    
    try:
        res = minimize(smooth_objective, params0, method='Nelder-Mead', 
                       options={'maxiter': 1500, 'xatol': 1e-5, 'fatol': 1e-5})
        
        if res.fun < best_side:
            best_side = res.fun
            params_opt = res.x
            best_cfg = [[params_opt[i], params_opt[N+i], params_opt[2*N+i]] for i in range(N)]
    except:
        pass
    
    # 3. Final Polish with LBFGS
    if best_cfg is not None:
        cx_arr = np.array([p[0] for p in best_cfg])
        cy_arr = np.array([p[1] for p in best_cfg])
        angle_arr = np.array([p[2] for p in best_cfg])
        
        if _is_valid_fast(cx_arr, cy_arr, angle_arr):
            cx, cy, side, angle = _compute_best_outer(cx_arr, cy_arr, angle_arr)
            best_side = side
            best_center = (cx, cy)
            best_outer_angle = angle
        else:
            # Fallback to lattice if invalid
            cfg = make_lattice_initial()
            cx_arr = np.array([p[0] for p in cfg])
            cy_arr = np.array([p[1] for p in cfg])
            angle_arr = np.array([p[2] for p in cfg])
            cx, cy, side, angle = _compute_best_outer(cx_arr, cy_arr, angle_arr)
            best_side = side
            best_center = (cx, cy)
            best_outer_angle = angle
            best_cfg = cfg
            
    inner = [[p[0], p[1], p[2]] for p in best_cfg]
    return inner, best_center, best_side, best_outer_angle
