import math
import time
import random
import numpy as np

N = 12
TIME_LIMIT = 145.0  # Safety margin below 150s

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with rotation angle_deg."""
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
    """Project polygon vertices onto axis, return (min, max)."""
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
    """Get edge normals (perpendicular to edges) for SAT."""
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
    """Check if two convex polygons overlap using SAT. Return True if they overlap (not disjoint)."""
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
            return False  # Found separating axis -> no overlap
    return True  # No separating axis -> overlap

def _is_valid_fast(cx_arr, cy_arr, angle_arr):
    """
    Fast validity check using distance-based pre-filter + SAT fallback.
    cx_arr, cy_arr, angle_arr: numpy arrays of shape (N,)
    Returns True if all hexagons are pairwise disjoint.
    """
    n = len(cx_arr)
    # Precompute vertices for all hexagons
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))
    
    # Precompute axes for all hexagons
    axes_list = []
    for i in range(n):
        axes_list.append(_axes_from_polygon(verts_list[i]))
    
    for i in range(n):
        xi, yi = cx_arr[i], cy_arr[i]
        vi = verts_list[i]
        ai = axes_list[i]
        for j in range(i + 1, n):
            xj, yj = cx_arr[j], cy_arr[j]
            vj = verts_list[j]
            aj = axes_list[j]
                
            # Quick distance check
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx * dx + dy * dy
                
            # If distance > 2*sqrt(3), definitely disjoint
            if dist_sq > (2 * math.sqrt(3)) ** 2:
                continue
                
            # If distance < 2.0, definitely overlapping (min separation is 2.0)
            if dist_sq < 4.0 - 1e-9:
                return False
                
            # Intermediate case: do SAT check with early exit
            found_sep = False
            for axis in ai:
                ax, ay = axis
                minA = min(vx * ax + vy * ay for vx, vy in vi)
                maxA = max(vx * ax + vy * ay for vx, vy in vi)
                minB = min(vx * ax + vy * ay for vx, vy in vj)
                maxB = max(vx * ax + vy * ay for vx, vy in vj)
                if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                    found_sep = True
                    break
            if not found_sep:
                for axis in aj:
                    ax, ay = axis
                    minA = min(vx * ax + vy * ay for vx, vy in vi)
                    maxA = max(vx * ax + vy * ay for vx, vy in vi)
                    minB = min(vx * ax + vy * ay for vx, vy in vj)
                    maxB = max(vx * ax + vy * ay for vx, vy in vj)
                    if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                        found_sep = True
                        break
            if not found_sep:
                return False
    
    return True

def _outer_hex_side(cx_arr, cy_arr, angle_arr, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    # 6 outward normals of the outer hexagon
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

def construct():
    t0 = time.time()
    random.seed(123)
    
    # Strategy: Use a hexagonal lattice grid search with a single global rotation
    # The inner hexagons are constrained to positions on a triangular lattice,
    # and we search over the global rotation angle and lattice offset.
    
    # First, generate a good initial configuration using simulated annealing
    # on a continuous space, then refine with lattice-based search.
    
    def make_flower_initial():
        """
        Create a 'flower' like arrangement.
        1 center, 6 around in ring 1, 5 in ring 2 valleys.
        For aligned hexagons, touching distance is 2*sqrt(3).
        """
        pts = []
        # Center
        pts.append([0.0, 0.0, 0.0])
        # Ring 1: 6 hexagons touching the center
        r1 = 2 * math.sqrt(3)
        for i in range(6):
            angle = math.radians(60 * i + 30)  # offset by 30° to sit in valleys
            x = r1 * math.cos(angle)
            y = r1 * math.sin(angle)
            pts.append([x, y, 0.0])
        
        # We have 7. Need 5 more.
        # Ring 2: place 5 hexagons at distance ~3.464 from center, in gaps
        r2 = 3.46410161514  # 2*sqrt(3)
        angles2 = [0, 72, 144, 216, 288]  # deg, pentagonal spacing
        for deg in angles2:
            rad = math.radians(deg)
            x = r2 * math.cos(rad)
            y = r2 * math.sin(rad)
            pts.append([x, y, 0.0])
            
        return pts[:12]
    
    def make_lattice_initial():
        """
        Hexagonal lattice arrangement.
        """
        pts = []
        row_height = math.sqrt(3)
        col_width = 2.0
        # Rows: 3, 4, 3 = 10 hexes
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
        
        # Add 2 more
        pts.append([-col_width*2 - col_width/2 - col_width, 0.0, 0.0])
        pts.append([col_width*2 - col_width/2 + col_width, 0.0, 0.0])
        
        return pts
    
    def make_random_initial():
        """Random scatter."""
        pts = []
        for _ in range(N):
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            a = random.uniform(0, 60)
            pts.append([x, y, a])
        return pts
    
    # Generate multiple initial configurations
    initializers = [
        make_flower_initial,
        make_lattice_initial,
        make_random_initial,
        make_random_initial,
        make_random_initial,
    ]
    
    best_side = float('inf')
    best_cfg = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Try each initializer and find the best valid starting point
    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 60:
            break
            
        cfg = init_func()
        if len(cfg) != N:
            continue
        
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        
        # Check validity
        if _is_valid_fast(cx_arr, cy_arr, angle_arr):
            side = _outer_hex_side(cx_arr, cy_arr, angle_arr, (0.0, 0.0), 0.0)
            if side < best_side:
                best_side = side
                best_cfg = cfg[:]
                best_center = (0.0, 0.0)
                best_angle = 0.0
    
    # If no valid configuration found, use a loose one
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
        best_side = _outer_hex_side(cx_arr, cy_arr, angle_arr, (0.0, 0.0), 0.0)
        best_center = (0.0, 0.0)
        best_angle = 0.0

    # Phase 1: Simulated Annealing to find a good continuous configuration
    cfg = [row[:] for row in best_cfg]
    cx_arr = np.array([p[0] for p in cfg])
    cy_arr = np.array([p[1] for p in cfg])
    angle_arr = np.array([p[2] for p in cfg])
    
    cur_score = best_side
    T = 0.5
    T_end = 0.002
    alpha = 0.997
    
    # Precompute normals for outer hexagon (angle=0, center=(0,0))
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    def _quick_score(cx_arr, cy_arr, angle_arr):
        """Fast score computation assuming center (0,0) and angle 0."""
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

    # Main SA loop with adaptive step size
    step_scale = 1.0
    while time.time() - t0 < TIME_LIMIT - 30:
        r = random.random()
        idx = random.randrange(N)
        
        # Save old values
        old_x, old_y, old_a = cx_arr[idx], cy_arr[idx], angle_arr[idx]
        
        if r < 0.65:
            # Move one hexagon
            cx_arr[idx] += random.gauss(0, 0.15 * step_scale)
            cy_arr[idx] += random.gauss(0, 0.15 * step_scale)
            angle_arr[idx] += random.gauss(0, 3.0 * step_scale)
        elif r < 0.80:
            # Rotate one hexagon
            angle_arr[idx] += random.gauss(0, 8.0 * step_scale)
        else:
            # Move two hexagons
            idx2 = random.randrange(N)
            if idx2 == idx:
                idx2 = (idx + 1) % N
            old_x2, old_y2, old_a2 = cx_arr[idx2], cy_arr[idx2], angle_arr[idx2]
            cx_arr[idx] += random.gauss(0, 0.10 * step_scale)
            cy_arr[idx] += random.gauss(0, 0.10 * step_scale)
            cx_arr[idx2] += random.gauss(0, 0.10 * step_scale)
            cy_arr[idx2] += random.gauss(0, 0.10 * step_scale)
            angle_arr[idx] += random.gauss(0, 2.0 * step_scale)
            angle_arr[idx2] += random.gauss(0, 2.0 * step_scale)
        
        # Check validity
        valid = _is_valid_fast(cx_arr, cy_arr, angle_arr)
        
        if valid:
            new_score = _quick_score(cx_arr, cy_arr, angle_arr)
            
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                # Decrease step size slightly on improvement
                step_scale = max(0.3, step_scale * 0.98)
                if new_score < best_side:
                    best_side = new_score
                    best_cfg = [[cx_arr[i], cy_arr[i], angle_arr[i]] for i in range(N)]
                    best_center = (0.0, 0.0)
                    best_angle = 0.0
            else:
                # Revert
                cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
                if r >= 0.80:
                    cx_arr[idx2], cy_arr[idx2], angle_arr[idx2] = old_x2, old_y2, old_a2
        else:
            # Revert
            cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
            if r >= 0.80:
                cx_arr[idx2], cy_arr[idx2], angle_arr[idx2] = old_x2, old_y2, old_a2
        
        T = max(T_end, T * alpha)
    
    # Phase 2: Refinement with outer hexagon optimization
    cx_best = np.array([p[0] for p in best_cfg])
    cy_best = np.array([p[1] for p in best_cfg])
    angle_best = np.array([p[2] for p in best_cfg])
    
    # Try different outer hexagon rotations and centers
    best_final_side = best_side
    best_final_cfg = best_cfg[:]
    best_final_center = best_center
    best_final_angle = best_angle
    
    # Try a range of outer hexagon angles
    for outer_angle in range(0, 60, 3):
        if time.time() - t0 > TIME_LIMIT - 10:
            break
        
        # Compute side length for current config with this outer angle
        side = _outer_hex_side(cx_best, cy_best, angle_best, (0.0, 0.0), float(outer_angle))
        
        if side < best_final_side:
            best_final_side = side
            best_final_cfg = best_cfg[:]
            best_final_center = (0.0, 0.0)
            best_final_angle = float(outer_angle)
    
    # Also try shifting the center to reduce asymmetry
    for outer_angle in range(0, 60, 5):
        if time.time() - t0 > TIME_LIMIT - 5:
            break
        
        # Try small center offsets
        for dx in [-0.3, -0.1, 0.0, 0.1, 0.3]:
            for dy in [-0.3, -0.1, 0.0, 0.1, 0.3]:
                side = _outer_hex_side(cx_best, cy_best, angle_best, (dx, dy), float(outer_angle))
                if side < best_final_side:
                    best_final_side = side
                    best_final_cfg = best_cfg[:]
                    best_final_center = (dx, dy)
                    best_final_angle = float(outer_angle)
    
    # Final verification
    cx_final = np.array([p[0] for p in best_final_cfg])
    cy_final = np.array([p[1] for p in best_final_cfg])
    angle_final = np.array([p[2] for p in best_final_cfg])
    
    if not _is_valid_fast(cx_final, cy_final, angle_final):
        # Fallback: use a known valid loose configuration
        cfg = make_lattice_initial()
        scale = 1.3
        for p in cfg:
            p[0] *= scale
            p[1] *= scale
        best_final_cfg = cfg
        cx_final = np.array([p[0] for p in cfg])
        cy_final = np.array([p[1] for p in cfg])
        angle_final = np.array([p[2] for p in cfg])
        best_final_side = _outer_hex_side(cx_final, cy_final, angle_final, (0.0, 0.0), 0.0)
    else:
        best_final_side = _outer_hex_side(cx_final, cy_final, angle_final, best_final_center, best_final_angle)
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_final_cfg]
    return inner, best_final_center, best_final_side, best_final_angle

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
