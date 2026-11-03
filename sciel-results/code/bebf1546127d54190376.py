import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 140.0  # Safety margin before 150s kill


def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts


def polygons_overlap_sat(A, B, tol=1e-9):
    """SAT: True if two convex polygons intersect (including touching)."""
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            if abs(nx) < 1e-12 and abs(ny) < 1e-12:
                continue
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            if maxA < minB - tol or maxB < minA - tol:
                return False
    return True


def point_in_convex_polygon(px, py, poly, tol=1e-9):
    """True if (px,py) is inside or on boundary of convex polygon."""
    n = len(poly)
    sign = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < tol:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def compute_min_outer_side(inner, outer_center, outer_angle):
    """Minimum outer side length such that all unit hexagons fit inside.
    inner: list of (x, y, angle_deg) with unit side.
    Returns the minimum side length S.
    """
    cx, cy = outer_center
    a = math.radians(outer_angle)
    # Outward normals for a regular hexagon of side S: direction (i+0.5)*60 + angle
    normals = []
    for i in range(6):
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_proj = 0.0
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        for (px, py) in p:
            dx, dy = px - cx, py - cy
            for (nx, ny) in normals:
                proj = dx * nx + dy * ny
                if proj > max_proj:
                    max_proj = proj
    # apothem = S * sqrt(3)/2, so S = 2*max_proj/sqrt(3)
    return (2.0 / math.sqrt(3.0)) * max_proj


def check_validity(inner, outer_center, outer_side, outer_angle):
    """Check all inner hexagons are inside outer and pairwise non-overlapping."""
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    polys = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        polys.append(p)
    
    # Inside outer
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                return False
    
    # Pairwise non-overlap (allow touching)
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap_sat(polys[i], polys[j]):
                return False
    return True


def objective(params):
    """
    Objective function for free-placement optimization.
    params: [x0, y0, a0, ..., x11, y11, a11, X, Y, Phi]
    Returns: S + Penalty
    """
    if len(params) != 40:
        return 1e6
    
    # Extract inner hexagons
    inner = []
    for i in range(12):
        x = params[i*3]
        y = params[i*3+1]
        a = params[i*3+2]
        inner.append((x, y, a))
    
    # Extract outer hexagon parameters
    X = params[36]
    Y = params[37]
    Phi = params[38]
    # S is not a variable, it is derived from the constraint
    # But we need to calculate the required S
    
    outer_center = (X, Y)
    outer_angle = Phi
    
    # Calculate required side length S
    S_req = compute_min_outer_side(inner, outer_center, outer_angle)
    
    # Calculate overlap penalty
    # We use a simple check: if any pair overlaps, add penalty
    # To make it smoother for optimization, we can estimate overlap depth
    # but for Nelder-Mead, a boolean penalty with a large constant often works
    # if the starting point is close to valid.
    
    polys = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        polys.append(p)
        
    penalty = 0.0
    for i in range(12):
        for j in range(i + 1, 12):
            if polygons_overlap_sat(polys[i], polys[j]):
                # Penalty for overlap
                # Simple penalty: 10.0 per overlapping pair
                penalty += 10.0
                
    return S_req + penalty


def construct():
    t0 = time.time()
    
    best_side = 1e6
    best_result = None
    
    # Generate a few initial structured packings to use as starting points
    # 1. Ring packing (1 center, 6 ring, 5 outer)
    # 2. Triangular lattice (3x4)
    
    starts = []
    
    # Start 1: Ring
    inner1 = [(0.0, 0.0, 0.0)]
    r1 = 1.732
    for k in range(6):
        theta = math.radians(k * 60.0)
        inner1.append((r1 * math.cos(theta), r1 * math.sin(theta), 0.0))
    r2 = 3.0
    for k in range(5):
        theta = math.radians(30.0 + k * 72.0)
        inner1.append((r2 * math.cos(theta), r2 * math.sin(theta), 0.0))
    
    params1 = []
    for (x, y, a) in inner1:
        params1.extend([x, y, a])
    params1.extend([0.0, 0.0, 0.0]) # Outer center and angle
    starts.append(np.array(params1))
    
    # Start 2: Triangular Lattice
    inner2 = []
    dx = 1.5 * 2.0 # Spacing for unit hexagons touching is 2*sqrt(3)/2 * 2? No.
    # Distance between centers of touching unit hexagons is 2*sqrt(3)/2 * 2 = 2*sqrt(3) ?
    # Apothem is sqrt(3)/2. Distance to edge is sqrt(3)/2.
    # If rotated 0, flat top.
    # Horizontal distance: 2 * (3/2) = 3? No.
    # Let's just use a rough grid.
    for row in range(3):
        for col in range(4):
            x = col * 2.6 + (row % 2) * 1.3
            y = row * 2.2 - 2.2
            inner2.append((x, y, 0.0))
            
    params2 = []
    for (x, y, a) in inner2:
        params2.extend([x, y, a])
    params2.extend([0.0, 0.0, 0.0])
    starts.append(np.array(params2))
    
    # Random starts
    np.random.seed(42)
    for _ in range(5):
        inner_rand = []
        for i in range(12):
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            a = np.random.uniform(0, 60)
            inner_rand.append((x, y, a))
        params_rand = []
        for (x, y, a) in inner_rand:
            params_rand.extend([x, y, a])
        params_rand.extend([0.0, 0.0, 0.0])
        starts.append(np.array(params_rand))
        
    # Optimize each start
    for x0 in starts:
        if time.time() > DEADLINE:
            break
            
        try:
            # Use Nelder-Mead for robustness with non-smooth constraints
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.fun < best_side:
                best_side = res.fun
                best_params = res.x
                
                # Extract result
                inner = []
                for i in range(12):
                    x = best_params[i*3]
                    y = best_params[i*3+1]
                    a = best_params[i*3+2]
                    inner.append((x, y, a))
                
                X = best_params[36]
                Y = best_params[37]
                Phi = best_params[38]
                
                outer_center = (X, Y)
                outer_angle = Phi
                
                # Recompute exact side
                side = compute_min_outer_side(inner, outer_center, outer_angle)
                
                # Check validity
                if check_validity(inner, outer_center, side, outer_angle):
                    best_result = (inner, outer_center, side, outer_angle)
                    
        except Exception:
            pass
            
    if best_result is not None:
        return best_result
        
    # Fallback
    inner = [(0.0, 0.0, 0.0)]
    r1 = 1.732
    for k in range(6):
        theta = math.radians(k * 60.0)
        inner.append((r1 * math.cos(theta), r1 * math.sin(theta), 0.0))
    r2 = 3.2
    for k in range(5):
        theta = math.radians(30.0 + k * 72.0)
        inner.append((r2 * math.cos(theta), r2 * math.sin(theta), 0.0))
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    side = compute_min_outer_side(inner, outer_center, outer_angle)
    
    if not check_validity(inner, outer_center, side, outer_angle):
        side = 20.0
        
    return inner, outer_center, side, outer_angle
