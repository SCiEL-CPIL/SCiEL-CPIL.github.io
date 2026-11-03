"""Packing 12 unit hexagons in minimal regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 145.0  # safety margin before 150s kill

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of regular hexagon with given center, side, rotation."""
    angle = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        a = angle + i * np.pi / 3
        x = cx + side * np.cos(a)
        y = cy + side * np.sin(a)
        vertices.append((x, y))
    return np.array(vertices)

def point_in_polygon(pt, poly):
    """Check if point is inside polygon using ray casting."""
    x, y = pt
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def polygons_overlap_sat(p1, p2, tol=1e-9):
    """Check if two convex polygons overlap using SAT. Returns True if they overlap (interior intersection)."""
    polys = [p1, p2]
    for poly in polys:
        n = len(poly)
        for i in range(n):
            ax, ay = poly[i]
            bx, by = poly[(i + 1) % n]
            ex, ey = bx - ax, by - ay
            nx, ny = -ey, ex
            norm = np.sqrt(nx*nx + ny*ny)
            if norm < 1e-12:
                continue
            nx, ny = nx/norm, ny/norm
            proj1 = [np.dot(p, [nx, ny]) for p in p1]
            proj2 = [np.dot(p, [nx, ny]) for p in p2]
            min1, max1 = min(proj1), max(proj1)
            min2, max2 = min(proj2), max(proj2)
            if max1 + tol < min2 or max2 + tol < min1:
                return False
    return True

def hexagon_in_hexagon(inner_hex, outer_hex):
    """Check if inner hexagon is entirely inside outer hexagon."""
    return all(point_in_polygon(v, outer_hex) for v in inner_hex)

def check_disjoint(positions, angles, side, outer_center, outer_angle):
    """Check all inner hexagons are disjoint and inside outer hexagon."""
    outer_hex = hex_vertices(outer_center[0], outer_center[1], side, outer_angle)
    
    for pos, ang in zip(positions, angles):
        inner_hex = hex_vertices(pos[0], pos[1], 1.0, ang)
        if not hexagon_in_hexagon(inner_hex, outer_hex):
            return False
    
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            h1 = hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
            h2 = hex_vertices(positions[j][0], positions[j][1], 1.0, angles[j])
            if polygons_overlap_sat(h1, h2):
                return False
    return True

def compute_min_outer_side(inner_positions, inner_angles, outer_center, outer_angle, lo=2.0, hi=20.0, iters=30):
    """Binary search for minimum side length."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if check_disjoint(inner_positions, inner_angles, mid, outer_center, outer_angle):
            hi = mid
        else:
            lo = mid
    return hi

def construct():
    start_time = time.time()
    
    # Strategy: Discrete topology search + SLSQP with hard constraints
    # Step 1: Generate candidate topologies (hex lattice variants, grids, etc.)
    # Step 2: For each, use SLSQP to minimize outer side with non-overlap constraints
    
    def make_hex_lattice(scale, inner_rot, outer_rot):
        """Hexagonal lattice: center + ring of 6 + partial ring of 5."""
        d = scale * np.sqrt(3)
        positions = [(0.0, 0.0)]
        for i in range(6):
            a = np.radians(60 * i)
            positions.append((d * np.cos(a), d * np.sin(a)))
        for i in range(6):
            if i == 2:
                continue
            a = np.radians(60 * i + 30)
            positions.append((2*d * np.cos(a), 2*d * np.sin(a)))
        angles = [inner_rot] * len(positions)
        return positions, angles, outer_rot
    
    def make_grid(scale, inner_rot, outer_rot):
        """3x4 grid arrangement."""
        d = scale * np.sqrt(3)
        positions = []
        for row in range(3):
            y_off = row * d * np.sqrt(3) / 2
            x_start = -1.5 * d
            if row % 2 == 1:
                x_start += d / 2
            for col in range(4):
                positions.append((x_start + col * d, y_off))
        angles = [inner_rot] * len(positions)
        return positions, angles, outer_rot
    
    # Topology generators
    topologies = [
        ("hex", make_hex_lattice),
        ("grid", make_grid),
    ]
    
    best_side = 100.0
    best_config = None
    
    for top_name, top_func in topologies:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Try several initial parameters
        inits = [
            (1.0, 0.0, 0.0),
            (0.95, 0.0, 0.0),
            (1.0, 30.0, 0.0),
            (1.0, 0.0, 30.0),
            (0.9, 0.0, 0.0),
        ]
        
        for init in inits:
            if time.time() - start_time > DEADLINE - 5:
                break
            
            scale0, rot0, orot0 = init
            positions0, angles0, orot0 = top_func(scale0, rot0, orot0)
            
            # Objective: minimize outer side length
            # Parameters: scale, inner_rot, outer_rot
            def objective(params, positions0, angles0, orot0, top_func):
                scale, inner_rot, outer_rot = params
                positions, angles, orot = top_func(scale, inner_rot, outer_rot)
                
                # Estimate required side
                all_x = [p[0] for p in positions]
                all_y = [p[1] for p in positions]
                extent = max(max(all_x) - min(all_x), max(all_y) - min(all_y))
                est = extent / np.sqrt(3) + 2.0
                
                # Binary search for exact side
                side = compute_min_outer_side(positions, angles, (0.0, 0.0), orot, lo=2.5, hi=12.0, iters=15)
                return side
            
            try:
                result = minimize(objective, init, args=(positions0, angles0, orot0, top_func),
                                 method='Nelder-Mead',
                                 options={'maxiter': 80, 'xatol': 1e-5, 'fatol': 1e-5})
                
                if result.fun < best_side:
                    best_side = result.fun
                    best_config = (top_name, top_func, result.x.copy())
            except Exception:
                continue
    
    # Final evaluation with best config
    if best_config is not None:
        top_name, top_func, params = best_config
        scale, inner_rot, outer_rot = params
        positions, angles, orot = top_func(scale, inner_rot, outer_rot)
        
        # Precise binary search
        side = compute_min_outer_side(positions, angles, (0.0, 0.0), orot, lo=2.5, hi=12.0, iters=35)
        
        inner = [[p[0], p[1], a] for p, a in zip(positions, angles)]
        return inner, (0.0, 0.0), side, orot
    else:
        # Fallback
        positions = [[4.0 * k, 0.0, 0.0] for k in range(-5, 7)]
        return positions, (0.0, 0.0), 40.0, 0.0
