"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
_DEADLINE = 145.0

def _hex_inradius(s):
    return s * math.cos(math.radians(30))

def _hex_verts(cx, cy, s, rot):
    """Get the 6 vertices of a hexagon centered at (cx,cy) with side s and rotation rot (radians)."""
    vs = np.empty((6, 2))
    for k in range(6):
        a = math.radians(60.0 * k) + rot
        vs[k, 0] = cx + s * math.cos(a)
        vs[k, 1] = cy + s * math.sin(a)
    return vs

def _point_in_hex(px, py, cx, cy, s, rot):
    """Check if point (px, py) is inside a hexagon centered at (cx, cy) with side s and rotation rot (radians)."""
    dx = px - cx
    dy = py - cy
    ca, sa = math.cos(-rot), math.sin(-rot)
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    r_in = _hex_inradius(s)
    c30 = math.cos(math.radians(30))
    d1 = abs(c30 * lx + 0.5 * ly)
    d2 = abs(ly)
    d3 = abs(-c30 * lx + 0.5 * ly)
    return max(d1, d2, d3) <= r_in + 1e-9

def _support(A, d):
    """Find vertex of polygon A (array of shape (6,2)) that maximizes dot product with direction d."""
    dots = A @ d
    return A[np.argmax(dots)]

def _gjk_distance(A, B, max_iter=50):
    """
    Compute minimum distance between two convex polygons A and B (each shape (6,2)) using GJK.
    Returns the distance (0 if overlapping).
    """
    # Start with a direction
    d = np.array([1.0, 0.0])
    
    simplex = []
    
    for _ in range(max_iter):
        # Support points
        a = _support(A, d)
        b = _support(B, -d)
        
        # New point in Minkowski difference space (A - B)
        p = a - b
        
        # If p is in direction d, we've made progress
        if np.dot(p, d) < 0:
            # No progress, we're at minimum distance
            break
        
        # Add to simplex
        simplex.append(p)
        
        # Check if origin is in simplex
        if len(simplex) == 1:
            # Single point
            d = -simplex[0]
            if np.dot(d, d) < 1e-12:
                return 0.0  # Origin is at this point
        elif len(simplex) == 2:
            # Two points: line segment
            p0 = simplex[0]
            p1 = simplex[1]
            v = p1 - p0
            w = -p0
            # Check if origin is on the segment
            t = np.dot(w, v) / np.dot(v, v) if np.dot(v, v) > 1e-12 else 0.0
            if t >= 0 and t <= 1:
                # Origin is on the segment
                closest = p0 + t * v
                if np.dot(closest, closest) < 1e-12:
                    return 0.0
                # Direction perpendicular to segment
                d = np.array([-v[1], v[0]])
                # Choose direction away from origin
                if np.dot(d, closest) > 0:
                    d = -d
            else:
                # Direction toward the closest endpoint
                if t < 0:
                    d = -p0
                    simplex = [p0]
                else:
                    d = -p1
                    simplex = [p1]
        else:
            # Three points: triangle
            p0 = simplex[0]
            p1 = simplex[1]
            p2 = simplex[2]
            
            # Check if origin is in triangle
            v0 = p2 - p0
            v1 = p1 - p0
            w = -p0
            
            # Barycentric coordinates
            d00 = np.dot(v0, v0)
            d01 = np.dot(v0, v1)
            d11 = np.dot(v1, v1)
            d20 = np.dot(w, v0)
            d21 = np.dot(w, v1)
            
            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-12:
                # Degenerate triangle, fall back to edge check
                d = -p0
                simplex = [p0]
                continue
            
            u = (d11 * d20 - d01 * d21) / denom
            v = (d00 * d21 - d01 * d20) / denom
            
            if u >= 0 and v >= 0 and u + v <= 1:
                # Origin is in triangle
                return 0.0
            elif u >= 0 and v < 0:
                # Edge p0-p1
                d = -p0
                simplex = [p0]
            elif u < 0 and v >= 0:
                # Edge p0-p2
                d = -p0
                simplex = [p0]
            elif u >= 0 and v >= 0 and u + v > 1:
                # Edge p1-p2
                d = -(p1 + p2) / 2
                simplex = [p1, p2]
            elif u < 0 and v < 0:
                # Vertex p0
                d = -p0
                simplex = [p0]
            else:
                # Should not happen, but fallback
                d = -p0
                simplex = [p0]
    
    # Compute distance from origin to closest point in final simplex
    if len(simplex) == 1:
        return np.linalg.norm(simplex[0])
    elif len(simplex) == 2:
        p0 = simplex[0]
        p1 = simplex[1]
        v = p1 - p0
        t = np.dot(-p0, v) / np.dot(v, v) if np.dot(v, v) > 1e-12 else 0.0
        t = max(0, min(1, t))
        closest = p0 + t * v
        return np.linalg.norm(closest)
    else:
        # For triangle, just return min distance to vertices as approximation
        return min(np.linalg.norm(p) for p in simplex)

def _is_valid_config(inner, center, side, angle):
    """
    Strictly validate the configuration.
    inner: list of (x, y, angle_deg)
    center: (cx, cy)
    side: outer side length
    angle: outer rotation (radians)
    """
    cx, cy = center
    s_out = side
    r_out = _hex_inradius(s_out)
    
    # Check boundary
    for i in range(len(inner)):
        xi, yi, ai_deg = inner[i]
        ai = math.radians(ai_deg)
        verts = _hex_verts(xi, yi, 1.0, ai)
        for v in verts:
            if not _point_in_hex(v[0], v[1], cx, cy, s_out, angle):
                return False

    # Check overlap using GJK
    for i in range(len(inner)):
        xi, yi, ai_deg = inner[i]
        ai = math.radians(ai_deg)
        vi = _hex_verts(xi, yi, 1.0, ai)
        for j in range(i + 1, len(inner)):
            xj, yj, aj_deg = inner[j]
            aj = math.radians(aj_deg)
            vj = _hex_verts(xj, yj, 1.0, aj)
            
            # Quick reject
            dc = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dc > 2.0:
                continue
            
            # Use GJK to check for overlap
            dist = _gjk_distance(vi, vj)
            if dist < 1e-9:
                return False
                
    return True

def _honeycomb_positions():
    """Generate honeycomb packing positions for 11 hexagons."""
    positions = []
    # Center
    positions.append((0.0, 0.0, 0.0))
    # Ring 1: 6 hexagons at distance 2
    for k in range(6):
        a = math.radians(60.0 * k + 30.0)
        positions.append((2.0*math.cos(a), 2.0*math.sin(a), 0.0))
    # Ring 2: 4 hexagons at distance 4, alternating angles
    for k in range(4):
        a = math.radians(60.0 * k + 30.0)
        positions.append((4.0*math.cos(a), 4.0*math.sin(a), 0.0))
    return positions[:N]

def construct():
    t_start = time.time()
    n = N

    def _remaining():
        return _DEADLINE - (time.time() - t_start)

    # Generate initial valid configuration
    init_pos = _honeycomb_positions()
    
    # Calculate required outer side for initial config
    max_r = 0.0
    for (xi, yi, ai) in init_pos:
        verts = _hex_verts(xi, yi, 1.0, ai)
        for v in verts:
            r = math.sqrt(v[0]**2 + v[1]**2)
            if r > max_r:
                max_r = r
    
    s_init = max_r / math.cos(math.radians(30))
    s_init *= 1.01
    
    # Prepare initial parameters for optimization
    p0 = np.zeros(3*n + 1)
    for i in range(n):
        p0[3*i] = init_pos[i][0]
        p0[3*i+1] = init_pos[i][1]
        p0[3*i+2] = 0.0
    p0[3*n] = s_init
    
    def objective(p):
        x = p[0:3*n:3]
        y = p[1:3*n:3]
        angles = p[2:3*n:3]
        S = p[3*n]
        
        if S < 1.0:
            return 1e6
            
        pen = 0.0
        r_out = _hex_inradius(S)
        
        # Boundary penalty
        for i in range(n):
            verts = _hex_verts(x[i], y[i], 1.0, angles[i])
            for v in verts:
                dx = v[0]
                dy = v[1]
                c30 = math.cos(math.radians(30))
                d1 = abs(c30 * dx + 0.5 * dy)
                d2 = abs(dy)
                d3 = abs(-c30 * dx + 0.5 * dy)
                d_max = max(d1, d2, d3)
                if d_max > r_out:
                    pen += (d_max - r_out)**2
                    
        # Overlap penalty using GJK
        for i in range(n):
            for j in range(i+1, n):
                dc = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dc > 2.0:
                    continue
                
                vi = _hex_verts(x[i], y[i], 1.0, angles[i])
                vj = _hex_verts(x[j], y[j], 1.0, angles[j])
                
                dist = _gjk_distance(vi, vj)
                if dist < 0.01:
                    # Penalty grows quadratically as hexagons get closer
                    pen += max(0, 0.01 - dist) ** 2 * 1000.0
        
        return 0.5 * S * S + pen

    best_val = float('inf')
    best_p = p0.copy()
    
    # Try optimization
    rng = np.random.RandomState(42)
    
    for trial in range(8):
        if _remaining() < 25.0:
            break
            
        p_init = p0.copy()
        if trial > 0:
            noise = 0.05 + 0.02 * trial
            p_init[0:3*n] += rng.randn(3*n) * noise
            p_init[3*n] = s_init * (1.0 + rng.rand()*0.05)
            
        try:
            res = minimize(objective, p_init, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_val:
                best_val = res.fun
                best_p = res.x.copy()
        except Exception:
            pass

    # Extract best solution
    x = best_p[0:3*n:3]
    y = best_p[1:3*n:3]
    angles = best_p[2:3*n:3]
    S = best_p[3*n]
    
    inner = [[float(x[i]), float(y[i]), float(math.degrees(angles[i]))] for i in range(n)]
    center = (0.0, 0.0)
    angle_out = 0.0
    
    # Validate
    if _is_valid_config(inner, center, S, angle_out):
        return inner, center, float(S), angle_out
    else:
        # Fallback to initial valid configuration
        inner_init = [[float(p[0]), float(p[1]), float(math.degrees(p[2]))] for p in init_pos]
        return inner_init, center, float(s_init), angle_out
