import math
import random
import numpy as np

N = 12

def _hex_verts(angle_deg=0.0, cx=0.0, cy=0.0, side=1.0):
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        th = math.radians(60 * k)
        x = side * math.cos(th)
        y = side * math.sin(th)
        verts.append((cx + x * cos_a - y * sin_a,
                      cy + x * sin_a + y * cos_a))
    return verts

def _convex_hull(points):
    """Andrew's monotone chain; returns list of (x,y) in CCW order."""
    pts = sorted(set((round(p[0], 12), round(p[1], 12)) for p in points))
    if len(pts) <= 2:
        return pts
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def _minkowski_sum(poly1, poly2):
    """Minkowski sum of two convex polygons (CCW)."""
    if not poly1 or not poly2:
        return []
    # Sum all vertex pairs
    pts = [(a[0] + b[0], a[1] + b[1]) for a in poly1 for b in poly2]
    return _convex_hull(pts)

def _signed_dist_point_convex(poly, p):
    """
    Returns signed distance from point p to convex polygon poly.
    Positive if outside, negative if inside.
    Assumes poly is CCW.
    """
    if len(poly) < 3:
        return 0.0
    
    # Check if inside
    # Use cross products for edges
    inside = True
    min_dist_sq = float('inf')
    n = len(poly)
    
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        
        # Vector edge
        ex = x2 - x1
        ey = y2 - y1
        # Vector point to edge start
        px = p[0] - x1
        py = p[1] - y1
        
        # Cross product to check side
        cross = ex * py - ey * px
        if cross < 0:
            inside = False
            break
        
        # Distance to segment
        # Project p onto line
        len_sq = ex*ex + ey*ey
        if len_sq > 0:
            t = (px*ex + py*ey) / len_sq
            if t < 0:
                t = 0
            elif t > 1:
                t = 1
            qx = x1 + t * ex
            qy = y1 + t * ey
            dx = p[0] - qx
            dy = p[1] - qy
            dist_sq = dx*dx + dy*dy
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
    
    dist = math.sqrt(min_dist_sq)
    if inside:
        return -dist
    else:
        return dist

def _energy(configs, outer_side, outer_center=(0.0, 0.0), outer_angle=0.0):
    """
    Calculate energy (penalty) for a configuration.
    configs: list of (cx, cy, angle_deg)
    """
    penalty = 0.0
    n = len(configs)
    
    # Precompute vertices for inner hexagons
    inner_verts = []
    for cx, cy, ang in configs:
        inner_verts.append(_hex_verts(ang, cx, cy, 1.0))
        
    # Precompute outer hexagon vertices
    outer_verts = _hex_verts(outer_angle, outer_center[0], outer_center[1], outer_side)
    
    # Containment penalty
    # For each inner hexagon, check if it is inside outer.
    # This is equivalent to checking if the center is inside the Minkowski difference
    # Outer - Inner. But we can just check if all vertices of inner are inside outer.
    # Since inner is convex and outer is convex, if all vertices of inner are inside outer, inner is inside outer.
    for i in range(n):
        # Check vertices of inner[i] against outer
        # If any vertex is outside, penalty
        # We use signed distance from vertex to outer polygon?
        # No, we want the inner hexagon to be inside.
        # If a vertex of inner is outside outer, it's a violation.
        # The "depth" of violation is the distance of the vertex to the outer boundary.
        for v in inner_verts[i]:
            # Distance from v to outer_verts
            # If v is outside, dist > 0. We want dist <= 0.
            # So penalty is max(0, dist)^2
            d = _signed_dist_point_convex(outer_verts, v)
            if d > 0:
                penalty += d * d
                
    # Collision penalty
    # For each pair, check if they overlap.
    # Overlap if 0 is inside Minkowski Sum of (Inner_i - Inner_j)?
    # Let's use the center difference method.
    # K_ij = Hex_i (at origin) + (-Hex_j (at origin))
    # Vector d = c_i - c_j
    # If d is inside K_ij, they overlap.
    # Penalty = max(0, -dist(d, K_ij))^2 ?
    # If d is inside, dist is negative. We want it to be outside (dist > 0).
    # So if dist < 0, penalty is (-dist)^2.
    
    for i in range(n):
        for j in range(i + 1, n):
            # Get hexagons at origin
            vi = _hex_verts(configs[i][2], 0.0, 0.0, 1.0)
            vj = _hex_verts(configs[j][2], 0.0, 0.0, 1.0)
            # Negate vj
            vj_neg = [(-x, -y) for x, y in vj]
            
            # Minkowski Sum
            K = _minkowski_sum(vi, vj_neg)
            
            # Center difference
            dx = configs[i][0] - configs[j][0]
            dy = configs[i][1] - configs[j][1]
            
            # Signed distance from (dx, dy) to K
            d = _signed_dist_point_convex(K, (dx, dy))
            
            if d < 0:
                penalty += (-d) * (-d)
                
    return penalty

def _anneal(outer_side, outer_center=(0.0, 0.0), outer_angle=0.0, max_iters=2000, T_start=10.0, T_end=0.01):
    """
    Simulated Annealing to minimize energy.
    """
    random.seed(42)
    
    # Initial configuration: spread out
    configs = []
    # Place in a circle
    for k in range(N):
        th = 2 * math.pi * k / N
        r = 3.0
        cx = r * math.cos(th)
        cy = r * math.sin(th)
        ang = 0.0
        configs.append([cx, cy, ang])
        
    current_energy = _energy(configs, outer_side, outer_center, outer_angle)
    best_configs = configs[:]
    best_energy = current_energy
    
    T = T_start
    cooling = (T_end / T_start) ** (1.0 / max_iters)
    
    for it in range(max_iters):
        T *= cooling
        
        # Pick a random hexagon to move
        i = random.randint(0, N - 1)
        
        # Perturb
        new_configs = [list(c) for c in configs]
        new_configs[i][0] += random.uniform(-0.5, 0.5)
        new_configs[i][1] += random.uniform(-0.5, 0.5)
        new_configs[i][2] += random.uniform(-15, 15)
        
        new_energy = _energy(new_configs, outer_side, outer_center, outer_angle)
        
        delta = new_energy - current_energy
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            configs = new_configs
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_configs = configs[:]
                
    return best_configs, best_energy

def _minimize_side(outer_center=(0.0, 0.0), outer_angle=0.0, lo=3.0, hi=10.0, tol=0.01):
    """
    Binary search for minimum side length.
    """
    # Check if hi is valid
    configs, energy = _anneal(hi, outer_center, outer_angle)
    if energy > 1e-6:
        # Try larger
        hi *= 1.2
        while hi < 20.0:
            configs, energy = _anneal(hi, outer_center, outer_angle)
            if energy < 1e-6:
                break
            hi *= 1.2
        if energy > 1e-6:
            return None, hi

    # Binary search
    for _ in range(20):
        if hi - lo < tol:
            break
        mid = (lo + hi) / 2
        configs, energy = _anneal(mid, outer_center, outer_angle)
        if energy < 1e-6:
            hi = mid
        else:
            lo = mid
            
    # Final check
    configs, energy = _anneal(hi, outer_center, outer_angle)
    if energy > 1e-6:
        return None, hi
        
    return configs, hi

def construct():
    """
    Construct 12 unit hexagons inside a regular hexagon, minimizing outer side.
    """
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    configs, side = _minimize_side(outer_center, outer_angle, lo=3.5, hi=8.0, tol=0.01)
    
    if configs is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    inner = [[c[0], c[1], c[2]] for c in configs]
    return inner, outer_center, side, outer_angle
