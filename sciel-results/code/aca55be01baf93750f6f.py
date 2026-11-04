"""Hexagon packing: 11 unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    # Standard hexagon vertices at 0, 60, 120, 180, 240, 300 degrees
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _point_in_polygon(pt, poly):
    """Check if point is inside convex polygon using cross products."""
    n = len(poly)
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i+1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cross > 1e-12:
            if sign < 0:
                return False
            sign = 1
        elif cross < -1e-12:
            if sign > 0:
                return False
            sign = -1
    return True

def _seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between two line segments p1-p2 and p3-p4."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(p1 - p3)
    
    if a < 1e-12:
        t = f / e
        t = max(0, min(1, t))
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)
    
    c = np.dot(d1, r)
    if e < 1e-12:
        s = c / a
        s = max(0, min(1, s))
        d = (p1 + s * d1) - p3
        return np.linalg.norm(d)
    
    b = np.dot(d1, d2)
    denom = a * e - b * b
    s = 0
    t = 0
    
    if denom > 1e-12:
        s = (b * f - c * e) / denom
        s = max(0, min(1, s))
        t = (b * s + f) / e
        if t < 0:
            t = 0
            s = max(0, min(1, c / a))
        elif t > 1:
            t = 1
            s = max(0, min(1, (b + f) / a))
        d = (p1 + s * d1) - (p3 + t * d2)
        return np.linalg.norm(d)
    else:
        # Parallel
        t = f / e
        t = max(0, min(1, t))
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)

def _poly_separation(p1, p2):
    """
    Compute the minimum distance between two convex polygons.
    Returns 0 if they overlap (interiors intersect or share boundary),
    positive if separated.
    """
    # Check if any vertex of p1 is inside p2
    for v in p1:
        if _point_in_polygon(v, p2):
            return 0.0
    # Check if any vertex of p2 is inside p1
    for v in p2:
        if _point_in_polygon(v, p1):
            return 0.0
    
    # Check edge-edge distances
    min_dist = float('inf')
    n1 = len(p1)
    n2 = len(p2)
    for i in range(n1):
        a1 = p1[i]
        a2 = p1[(i+1) % n1]
        for j in range(n2):
            b1 = p2[j]
            b2 = p2[(j+1) % n2]
            d = _seg_seg_dist(a1, a2, b1, b2)
            if d < min_dist:
                min_dist = d
    return min_dist

def _outer_hex_vertices(cx, cy, side, angle_deg):
    """Vertices of outer regular hexagon."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([side * np.cos(angles), side * np.sin(angles)])
    return vs + np.array([cx, cy])

def _compute_side(inner_params, outer_angle_deg):
    """
    Compute the minimum outer hexagon side length such that all inner hexagons
    are contained within the outer hexagon.
    inner_params: array of shape (n, 3) with [x, y, angle_deg]
    """
    n = len(inner_params)
    
    # Binary search on side length
    lo = 2.0
    hi = 15.0
    
    for _ in range(40):
        mid = (lo + hi) / 2
        outer_v = _outer_hex_vertices(0.0, 0.0, mid, outer_angle_deg)
        
        valid = True
        for i in range(n):
            x = inner_params[i, 0]
            y = inner_params[i, 1]
            a = inner_params[i, 2]
            hex_v = _hex_vertices(x, y, a)
            for v in hex_v:
                if not _point_in_polygon(v, outer_v):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            hi = mid
        else:
            lo = mid
            
    return hi

def _overlap_penalty(inner_params, penalty_weight=100.0):
    """
    Compute penalty for overlapping inner hexagons.
    Returns 0 if all hexagons are disjoint, positive otherwise.
    """
    n = len(inner_params)
    penalty = 0.0
    for i in range(n):
        x1 = inner_params[i, 0]
        y1 = inner_params[i, 1]
        a1 = inner_params[i, 2]
        v1 = _hex_vertices(x1, y1, a1)
        
        for j in range(i+1, n):
            x2 = inner_params[j, 0]
            y2 = inner_params[j, 1]
            a2 = inner_params[j, 2]
            v2 = _hex_vertices(x2, y2, a2)
            
            sep = _poly_separation(v1, v2)
            if sep < 0.01:
                # Penalty for being too close (penalize if distance < 0.01)
                penalty += penalty_weight * (0.01 - sep) ** 2
                
    return penalty

def _objective(params, outer_angle_deg, n, deadline):
    """
    Objective function: outer side length + overlap penalty.
    params: flat array of length 3*n: [x1, y1, a1, x2, y2, a2, ...]
    """
    if time.time() > deadline:
        return 1e10
    inner_params = params.reshape(n, 3)
    side = _compute_side(inner_params, outer_angle_deg)
    overlap_pen = _overlap_penalty(inner_params)
    return side + overlap_pen

def _is_valid(inner_params, side, outer_angle_deg, min_sep=0.0):
    """Check if the configuration is valid."""
    n = len(inner_params)
    outer_v = _outer_hex_vertices(0.0, 0.0, side, outer_angle_deg)
    
    # Check containment
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        hex_v = _hex_vertices(x, y, a)
        for v in hex_v:
            if not _point_in_polygon(v, outer_v):
                return False
    
    # Check non-overlap
    for i in range(n):
        x1 = inner_params[i, 0]
        y1 = inner_params[i, 1]
        a1 = inner_params[i, 2]
        v1 = _hex_vertices(x1, y1, a1)
        
        for j in range(i+1, n):
            x2 = inner_params[j, 0]
            y2 = inner_params[j, 1]
            a2 = inner_params[j, 2]
            v2 = _hex_vertices(x2, y2, a2)
            
            sep = _poly_separation(v1, v2)
            if sep < min_sep:
                return False
                
    return True

def _seed_solution():
    """
    Return the concentric ring seed solution.
    1 central, 6 surrounding, 4 in second ring.
    """
    eps = 1e-9
    centers = []
    angles = []
    
    # Central hexagon
    centers.append((0.0, 0.0))
    angles.append(30.0)
    
    # 6 surrounding hexagons
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (np.sqrt(3) + eps) * np.cos(angle)
        y = (np.sqrt(3) + eps) * np.sin(angle)
        centers.append((x, y))
        angles.append(30.0)
        
    # 4 more hexagons in the second ring
    # Placed at 30, 90, 150, 210 degrees relative to the center
    # Distance approx 3.0
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (3.0 + 2*eps) * np.cos(angle)
        y = (3.0 + 2*eps) * np.sin(angle)
        centers.append((x, y))
        angles.append(30.0)
        
    inner = []
    for i in range(N):
        inner.append([centers[i][0], centers[i][1], angles[i]])
    
    # Compute side length for this seed
    inner_params = np.array(inner)
    side = _compute_side(inner_params, 30.0)
    side += 1e-9 # Small margin
    
    return inner, (0.0, 0.0), side, 30.0

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    n = 11
    outer_angle = 30.0
    
    # Get seed solution
    seed_inner, seed_center, seed_side, seed_angle = _seed_solution()
    best_inner = seed_inner
    best_side = seed_side
    best_valid = _is_valid(np.array(best_inner), best_side, seed_angle)
    
    # Convert seed to flat params
    init_params = np.zeros(3 * n)
    for i, (x, y, a) in enumerate(seed_inner):
        init_params[i*3] = x
        init_params[i*3+1] = y
        init_params[i*3+2] = a
        
    # List of starting points for local optimization
    starts = [init_params]
    
    # Add some perturbed starts
    np.random.seed(42)
    for _ in range(5):
        perturbed = init_params.copy()
        # Perturb positions slightly
        perturbed[:2*n:3] += np.random.uniform(-0.5, 0.5, n)
        perturbed[1:2*n:3] += np.random.uniform(-0.5, 0.5, n)
        # Perturb angles slightly
        perturbed[2*n:] += np.random.uniform(-10, 10, n)
        starts.append(perturbed)
        
    # Run Nelder-Mead on each start
    for start in starts:
        if time.time() > deadline - 10:
            break
            
        def nm_obj(params):
            return _objective(params, outer_angle, n, deadline)
            
        try:
            res = minimize(nm_obj, start, method='Nelder-Mead',
                           options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 500, 'maxfev': 1000})
            
            if res.fun < best_side:
                params = res.x
                inner_params = params.reshape(n, 3)
                side = _compute_side(inner_params, outer_angle)
                pen = _overlap_penalty(inner_params)
                
                # Check validity
                if pen < 1e-4 and side < best_side:
                    if _is_valid(inner_params, side, outer_angle):
                        best_side = side
                        best_inner = inner_params.tolist()
        except Exception:
            pass
            
    # If time permits, try Differential Evolution for a global search
    if time.time() < deadline - 30:
        def de_obj(params):
            return _objective(params, outer_angle, n, deadline)
            
        bounds = []
        for i in range(n):
            bounds.append((-5.0, 5.0))  # x
            bounds.append((-5.0, 5.0))  # y
            bounds.append((0.0, 180.0)) # angle
            
        try:
            res = differential_evolution(
                de_obj, bounds,
                seed=42,
                maxiter=20,
                popsize=10,
                tol=1e-4,
                mutation=(0.5, 1.5),
                recombination=0.8,
                polish=False,
                init='latinhypercube'
            )
            
            if res.fun < best_side:
                params = res.x
                inner_params = params.reshape(n, 3)
                side = _compute_side(inner_params, outer_angle)
                pen = _overlap_penalty(inner_params)
                
                if pen < 1e-4 and side < best_side:
                    if _is_valid(inner_params, side, outer_angle):
                        best_side = side
                        best_inner = inner_params.tolist()
        except Exception:
            pass

    # Final verification
    if not _is_valid(np.array(best_inner), best_side, outer_angle):
        # If invalid, recompute side with margin
        side = _compute_side(np.array(best_inner), outer_angle)
        best_side = side * 1.02

    # Format output
    inner = []
    for i in range(N):
        inner.append([float(best_inner[i][0]), float(best_inner[i][1]), float(best_inner[i][2])])
    
    return inner, (0.0, 0.0), float(best_side), float(outer_angle)

# EVOLVE-BLOCK-END
