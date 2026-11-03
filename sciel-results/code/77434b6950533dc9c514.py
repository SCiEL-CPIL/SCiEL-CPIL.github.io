<<<<<<< SEARCH
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 115.0  # Safety margin

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
        # Edge direction
        dx, dy = x2 - x1, y2 - y1
        # Normal
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

def _outer_hex_side(centers_angles, center, angle_deg):
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
    for (px, py, pangle) in centers_angles:
        verts = _hex_vertices(px, py, pangle)
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _is_valid(centers_angles):
    """Check all pairs of hexagons are disjoint."""
    m = len(centers_angles)
    polys = [_hex_vertices(p[0], p[1], p[2]) for p in centers_angles]
    for i in range(m):
        for j in range(i + 1, m):
            if _separating_axis_check(polys[i], polys[j]):
                return False
    return True

def _objective(params):
    """
    Objective function for scipy optimizer.
    Returns the outer side length if valid, otherwise a large penalty.
    params: [x0, y0, a0, x1, y1, a1, ...]
    """
    cfg = []
    for i in range(N):
        x = params[i * 3]
        y = params[i * 3 + 1]
        a = params[i * 3 + 2]
        cfg.append((x, y, a))
    
    # Center is assumed to be (0,0) for simplicity in local opt, 
    # but we can optimize center too if needed. 
    # For now, fix center at (0,0) and optimize inner positions.
    # Actually, shifting the whole cluster doesn't change the side length, 
    # so fixing center at (0,0) is WLOG for the relative positions, 
    # but the outer hexagon center is also variable in the return interface.
    # However, minimizing side length is invariant to global translation.
    # So we can fix outer center at (0,0) and just find positions relative to it.
    
    if not _is_valid(cfg):
        return 1000.0  # Penalty for overlap
    
    return _outer_hex_side(cfg, (0.0, 0.0), 0.0)

def construct():
    t0 = time.time()
    random.seed(42)
    
    def make_flower_initial():
        """
        Create a 'flower' like arrangement.
        1 center, 6 around, 5 in gaps.
        """
        pts = []
        # Center
        pts.append([0.0, 0.0, 0.0])
        # Ring 1: 6 hexagons touching the center
        r1 = math.sqrt(3)
        for i in range(6):
            angle = math.radians(60 * i)
            x = r1 * math.cos(angle)
            y = r1 * math.sin(angle)
            # Rotate to fit? For regular hexagons, rotation doesn't change shape, 
            # but for packing, alignment matters. Standard flower has aligned hexes.
            pts.append([x, y, 0.0])
        
        # We have 7. Need 5 more.
        # Place them in the 'valleys' of the first ring, further out.
        # The distance from center to the "corner" of the ring 1 is 2*apothem? No.
        # The outer boundary of ring 1 is roughly radius 2 (vertex to vertex).
        # Let's place 5 more around the perimeter.
        r2 = r1 + math.sqrt(3) # approximate distance to next layer
        # Angles for the 5 extra: try to distribute them
        angles = [30, 90, 150, 210, 270] # deg
        for deg in angles:
            rad = math.radians(deg)
            x = r2 * math.cos(rad)
            y = r2 * math.sin(rad)
            pts.append([x, y, 0.0])
            
        # Ensure we have exactly 12
        return pts[:12]

    def make_lattice_initial():
        """
        Hexagonal lattice 3-4-3 arrangement (10 hexes) + 2 extra.
        """
        pts = []
        row_height = math.sqrt(3)
        col_width = 2.0
        # Rows: 3, 4, 3
        # Row 0 (3 hexes)
        for c in range(3):
            x = c * col_width - col_width # center around 0
            y = -row_height
            pts.append([x, y, 0.0])
        # Row 1 (4 hexes)
        for c in range(4):
            x = c * col_width - col_width - col_width/2
            y = 0.0
            pts.append([x, y, 0.0])
        # Row 2 (3 hexes)
        for c in range(3):
            x = c * col_width - col_width
            y = row_height
            pts.append([x, y, 0.0])
        
        # Add 2 more at the ends of row 1
        x_end1 = -col_width*2 - col_width/2 - col_width
        x_end2 = col_width*2 - col_width/2 + col_width
        pts.append([x_end1, 0.0, 0.0])
        pts.append([x_end2, 0.0, 0.0])
        
        return pts

    def make_random_initial():
        """Random scatter."""
        pts = []
        for _ in range(N):
            x = random.uniform(-2, 2)
            y = random.uniform(-2, 2)
            a = random.uniform(0, 60)
            pts.append([x, y, a])
        return pts

    def to_params(cfg):
        params = []
        for x, y, a in cfg:
            params.extend([x, y, a])
        return params

    def from_params(params):
        cfg = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            cfg.append([x, y, a])
        return cfg

    best_side = float('inf')
    best_cfg = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Strategies
    initializers = [
        make_flower_initial,
        make_lattice_initial,
        make_random_initial,
        make_random_initial,
        make_random_initial
    ]
    
    # Run local optimization from each initial guess
    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 10:
            break
            
        cfg = init_func()
        if len(cfg) != N:
            continue
            
        params = to_params(cfg)
        
        # Try a few random perturbations of the initial guess
        for attempt in range(3):
            if time.time() - t0 > TIME_LIMIT - 10:
                break
                
            p_current = params[:]
            if attempt > 0:
                # Add noise
                for i in range(len(p_current)):
                    p_current[i] += random.gauss(0, 0.2)
            
            try:
                # Use Nelder-Mead for low-dimensional non-smooth (SAT boundary)
                # maxiter depends on time, but let's give it a reasonable number
                res = minimize(_objective, p_current, method='Nelder-Mead', 
                               options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
                
                if res.success or res.fun < best_side:
                    final_params = res.x
                    cfg_res = from_params(final_params)
                    # Verify validity and get exact side
                    if _is_valid(cfg_res):
                        side = _outer_hex_side(cfg_res, (0.0, 0.0), 0.0)
                        if side < best_side:
                            best_side = side
                            best_cfg = cfg_res
                            best_center = (0.0, 0.0)
                            best_angle = 0.0
            except Exception:
                pass

    # If we still have time, do Simulated Annealing to refine the best found or find better
    if best_cfg is None:
        # Fallback
        cfg = make_lattice_initial()
        best_cfg = cfg
        best_side = _outer_hex_side(cfg, (0.0, 0.0), 0.0)
        best_center = (0.0, 0.0)
        best_angle = 0.0

    # Simulated Annealing Refinement
    if time.time() - t0 < TIME_LIMIT - 5:
        cfg = [row[:] for row in best_cfg]
        center = [0.0, 0.0] # Keep center fixed at 0 for simplicity, 
                           # as global translation doesn't affect side length
        angle = 0.0
        
        cur_score = best_side
        T = 0.1
        T_end = 0.001
        alpha = 0.99
        
        # Pre-compute vertices for speed? No, they change.
        while time.time() - t0 < TIME_LIMIT - 5:
            r = random.random()
            if r < 0.8:
                idx = random.randrange(N)
                old_x, old_y, old_a = cfg[idx]
                cfg[idx][0] += random.gauss(0, 0.1)
                cfg[idx][1] += random.gauss(0, 0.1)
                cfg[idx][2] += random.gauss(0, 2.0)
            else:
                # Small global rotation of one hex
                idx = random.randrange(N)
                old_a = cfg[idx][2]
                cfg[idx][2] += random.gauss(0, 5.0)
                
            if not _is_valid(cfg):
                # Revert
                if r < 0.8:
                    cfg[idx] = [old_x, old_y, old_a]
                else:
                    cfg[idx][2] = old_a
                continue
            
            new_score = _outer_hex_side(cfg, tuple(center), angle)
            
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                if new_score < best_side:
                    best_side = new_score
                    best_cfg = [row[:] for row in cfg]
                    best_center = tuple(center)
                    best_angle = angle
            else:
                if r < 0.8:
                    cfg[idx] = [old_x, old_y, old_a]
                else:
                    cfg[idx][2] = old_a
            
            T = max(T_end, T * alpha)

    # Final verification and return
    # Ensure the returned center and angle are consistent with the side length calculation
    # The side length calculated assumes center (0,0) and angle 0.
    # If we moved the center in SA (we didn't in this version), we would need to return it.
    # Here, we fixed center at (0,0).
    
    # One last check: Is the configuration valid?
    if not _is_valid(best_cfg):
        # If something went wrong, fallback to a known valid loose configuration
        cfg = make_lattice_initial()
        best_cfg = cfg
        best_side = _outer_hex_side(cfg, (0.0, 0.0), 0.0)

    inner = [[p[0], p[1], p[2]] for p in best_cfg]
    return inner, best_center, best_side, best_angle

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
=======
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 115.0  # Safety margin

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
        # Edge direction
        dx, dy = x2 - x1, y2 - y1
        # Normal
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

def _outer_hex_side(centers_angles, center, angle_deg):
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
    for (px, py, pangle) in centers_angles:
        verts = _hex_vertices(px, py, pangle)
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _is_valid(centers_angles):
    """Check all pairs of hexagons are disjoint."""
    m = len(centers_angles)
    polys = [_hex_vertices(p[0], p[1], p[2]) for p in centers_angles]
    for i in range(m):
        for j in range(i + 1, m):
            if _separating_axis_check(polys[i], polys[j]):
                return False
    return True

def _separation_direction(vertsA, vertsB):
    """
    Find the direction to separate two overlapping polygons.
    Returns (dx, dy) unit vector to move A away from B, or None if not overlapping.
    """
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    best_overlap = float('inf')
    best_axis = None
    for axis in axes:
        ax, ay = axis
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        overlap = min(maxA, maxB) - max(minA, minB)
        if overlap <= 1e-9:
            return None  # Already separated
        if overlap < best_overlap:
            best_overlap = overlap
            best_axis = axis
    
    if best_axis is None:
        return None
    
    # Determine direction: move A in direction that increases separation
    ax, ay = best_axis
    minA, maxA = _project_polygon(vertsA, (ax, ay))
    minB, maxB = _project_polygon(vertsB, (ax, ay))
    
    # If A is to the left of B, move A left (negative direction)
    # If A is to the right of B, move A right (positive direction)
    centerA = (minA + maxA) / 2
    centerB = (minB + maxB) / 2
    
    if centerA < centerB:
        return (-ax, -ay)
    else:
        return (ax, ay)

def _resolve_collisions(cfg, max_iters=100):
    """
    Iteratively resolve collisions by moving overlapping hexagons apart.
    Returns modified cfg if all collisions resolved, else None.
    """
    cfg = [list(row) for row in cfg]
    
    for _ in range(max_iters):
        changed = False
        polys = [_hex_vertices(p[0], p[1], p[2]) for p in cfg]
        
        for i in range(N):
            for j in range(i + 1, N):
                if _separating_axis_check(polys[i], polys[j]):
                    # Overlapping - find separation direction
                    sep_dir = _separation_direction(polys[i], polys[j])
                    if sep_dir is not None:
                        dx, dy = sep_dir
                        # Move both hexagons apart by half the overlap
                        cfg[i][0] += dx * 0.05
                        cfg[i][1] += dy * 0.05
                        cfg[j][0] -= dx * 0.05
                        cfg[j][1] -= dy * 0.05
                        changed = True
        
        if not changed:
            break
    
    if _is_valid(cfg):
        return cfg
    return None

def _optimize_outer_angle(cfg):
    """Optimize the outer hexagon angle to minimize side length."""
    def angle_objective(theta_rad):
        angle_deg = math.degrees(theta_rad)
        return _outer_hex_side(cfg, (0.0, 0.0), angle_deg)
    
    best_angle = 0.0
    best_side = float('inf')
    
    # Try multiple starting angles
    for start_angle in range(0, 60, 5):
        res = minimize(angle_objective, [math.radians(start_angle)], 
                      method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-6})
        if res.fun < best_side:
            best_side = res.fun
            best_angle = math.degrees(res.x[0])
    
    return best_angle, best_side

def construct():
    t0 = time.time()
    random.seed(42)
    
    def make_dense_lattice():
        """
        Create a dense hexagonal lattice arrangement.
        For 12 hexagons, try various row configurations.
        """
        configs = []
        
        # Configuration 1: 3-4-3-2 (12 hexes)
        row_height = math.sqrt(3)
        col_width = 2.0
        
        # Row 0: 3 hexes
        for c in range(3):
            x = c * col_width - col_width
            y = -1.5 * row_height
            configs.append([x, y, 0.0])
        
        # Row 1: 4 hexes (offset)
        for c in range(4):
            x = c * col_width - 1.5 * col_width
            y = -0.5 * row_height
            configs.append([x, y, 0.0])
        
        # Row 2: 3 hexes
        for c in range(3):
            x = c * col_width - col_width
            y = 0.5 * row_height
            configs.append([x, y, 0.0])
        
        # Row 3: 2 hexes (offset)
        for c in range(2):
            x = c * col_width - 0.5 * col_width
            y = 1.5 * row_height
            configs.append([x, y, 0.0])
        
        return configs[:12]
    
    def make_fan_arrangement():
        """
        Fan arrangement: hexagons radiating from center.
        """
        pts = []
        # Center hex
        pts.append([0.0, 0.0, 0.0])
        
        # 5 hexes in a fan around center
        r1 = math.sqrt(3)
        for i in range(5):
            angle = math.radians(-60 + 30 * i)  # -60 to 60 degrees
            x = r1 * math.cos(angle)
            y = r1 * math.sin(angle)
            pts.append([x, y, 0.0])
        
        # 6 more hexes in a second fan
        r2 = 2 * math.sqrt(3)
        for i in range(6):
            angle = math.radians(-75 + 30 * i)  # -75 to 75 degrees
            x = r2 * math.cos(angle)
            y = r2 * math.sin(angle)
            pts.append([x, y, 0.0])
        
        return pts[:12]
    
    def make_circular_arrangement():
        """
        Circular arrangement: hexagons on a circle.
        """
        pts = []
        # Center
        pts.append([0.0, 0.0, 0.0])
        
        # 5 on inner ring
        r1 = math.sqrt(3)
        for i in range(5):
            angle = math.radians(72 * i)
            x = r1 * math.cos(angle)
            y = r1 * math.sin(angle)
            pts.append([x, y, 0.0])
        
        # 6 on outer ring
        r2 = 2 * math.sqrt(3)
        for i in range(6):
            angle = math.radians(60 * i + 30)
            x = r2 * math.cos(angle)
            y = r2 * math.sin(angle)
            pts.append([x, y, 0.0])
        
        return pts[:12]
    
    def make_random_dense():
        """Random dense arrangement."""
        pts = []
        # Place hexagons in a small region
        for _ in range(N):
            x = random.uniform(-2, 2)
            y = random.uniform(-2, 2)
            a = random.uniform(0, 60)
            pts.append([x, y, a])
        return pts
    
    initializers = [
        make_dense_lattice,
        make_fan_arrangement,
        make_circular_arrangement,
        make_random_dense,
        make_random_dense,
        make_random_dense
    ]
    
    best_side = float('inf')
    best_cfg = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Strategy 1: Try each initializer, resolve collisions, optimize
    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 15:
            break
        
        cfg = init_func()
        if len(cfg) != N:
            continue
        
        # Try to resolve collisions
        resolved = _resolve_collisions(cfg, max_iters=200)
        
        if resolved is not None:
            # Valid configuration found - optimize outer angle
            angle, side = _optimize_outer_angle(resolved)
            if side < best_side:
                best_side = side
                best_cfg = resolved
                best_center = (0.0, 0.0)
                best_angle = angle
        
        # Also try with random perturbations
        for attempt in range(3):
            if time.time() - t0 > TIME_LIMIT - 15:
                break
            
            cfg_perturbed = [row[:] for row in cfg]
            for i in range(N):
                cfg_perturbed[i][0] += random.gauss(0, 0.3)
                cfg_perturbed[i][1] += random.gauss(0, 0.3)
                cfg_perturbed[i][2] += random.gauss(0, 5.0)
            
            resolved = _resolve_collisions(cfg_perturbed, max_iters=200)
            
            if resolved is not None:
                angle, side = _optimize_outer_angle(resolved)
                if side < best_side:
                    best_side = side
                    best_cfg = resolved
                    best_center = (0.0, 0.0)
                    best_angle = angle
    
    # Strategy 2: If no valid config found, use a loose fallback
    if best_cfg is None:
        # Use a very loose arrangement that's guaranteed valid
        cfg = []
        for i in range(N):
            angle = math.radians(60 * i)
            r = 3.0
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            cfg.append([x, y, 0.0])
        
        best_cfg = cfg
        best_center = (0.0, 0.0)
        best_angle = 0.0
        best_side = _outer_hex_side(best_cfg, best_center, best_angle)
    
    # Strategy 3: Local refinement with small moves
    if best_cfg is not None and time.time() - t0 < TIME_LIMIT - 10:
        cfg = [row[:] for row in best_cfg]
        cur_score = best_side
        T = 0.05
        T_end = 0.001
        alpha = 0.995
        
        while time.time() - t0 < TIME_LIMIT - 10:
            idx = random.randrange(N)
            old_x, old_y, old_a = cfg[idx][0], cfg[idx][1], cfg[idx][2]
            
            # Small move
            cfg[idx][0] += random.gauss(0, 0.05)
            cfg[idx][1] += random.gauss(0, 0.05)
            cfg[idx][2] += random.gauss(0, 2.0)
            
            if not _is_valid(cfg):
                cfg[idx] = [old_x, old_y, old_a]
                continue
            
            # Try to optimize angle quickly
            angle, new_score = _optimize_outer_angle(cfg)
            
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                if new_score < best_side:
                    best_side = new_score
                    best_cfg = [row[:] for row in cfg]
                    best_center = (0.0, 0.0)
                    best_angle = angle
            else:
                cfg[idx] = [old_x, old_y, old_a]
            
            T = max(T_end, T * alpha)
    
    # Final verification
    if not _is_valid(best_cfg):
        # Fallback to guaranteed valid loose config
        cfg = []
        for i in range(N):
            angle = math.radians(60 * i)
            r = 3.0
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            cfg.append([x, y, 0.0])
        best_cfg = cfg
        best_center = (0.0, 0.0)
        best_angle = 0.0
        best_side = _outer_hex_side(best_cfg, best_center, best_angle)
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, best_center, float(best_side), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
>>>>>>> REPLACE
