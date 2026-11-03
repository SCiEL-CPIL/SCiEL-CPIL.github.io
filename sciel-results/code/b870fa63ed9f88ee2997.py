"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def hex_vertices(cx, cy, side, angle_deg):
        """Vertices of regular hexagon (pointy-top at angle=0)."""
        a = math.radians(angle_deg)
        # Pointy-top: vertices at 30, 90, 150, 210, 270, 330 degrees
        verts = []
        for i in range(6):
            theta = a + math.radians(60 * i + 30)
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return np.array(verts)

    def point_in_convex_polygon(px, py, verts):
        """Check if point is inside convex polygon using cross products."""
        n = len(verts)
        sign = 0
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
            if cross > 1e-10:
                if sign < 0:
                    return False
                sign = 1
            elif cross < -1e-10:
                if sign > 0:
                    return False
                sign = -1
        return True

    def hex_contains_point(hx, hy, hangle, px, py):
        """Check if point (px,py) is inside hexagon at (hx,hy) with rotation hangle, side=1."""
        # Transform point to hexagon's local frame
        dx = px - hx
        dy = py - hy
        a = math.radians(hangle)
        # Rotate point by -angle to get local coords
        lx = dx * math.cos(a) + dy * math.sin(a)
        ly = -dx * math.sin(a) + dy * math.cos(a)
        # For pointy-top hex with side 1, check if in hexagon
        # Hexagon vertices at angles 30,90,...,330 with radius 1
        # In local frame, hexagon has vertices at (cos(30°), sin(30°)), etc.
        # Use the standard hexagon containment: max of |x|/1, |x/2 + y*sqrt(3)/2|/1, |-x/2 + y*sqrt(3)/2|/1 <= 1
        # Actually for pointy-top: the hexagon is defined by 6 half-planes
        # Let me just check against the 6 vertices
        verts = hex_vertices(0, 0, 1.0, 0.0)
        return point_in_convex_polygon(lx, ly, verts)

    def min_distance_to_hexagon(hx, hy, hangle, px, py):
        """Minimum distance from point to hexagon boundary (negative if inside)."""
        # Transform to local frame
        dx = px - hx
        dy = py - hy
        a = math.radians(hangle)
        lx = dx * math.cos(a) + dy * math.sin(a)
        ly = -dx * math.sin(a) + dy * math.cos(a)
        
        # For a regular hexagon with side 1 (pointy-top), the inradius is sqrt(3)/2
        # The hexagon can be described as: for each of 6 directions, the projection <= inradius
        # Directions (outward normals) for pointy-top hex: at angles 0, 60, 120, 180, 240, 300
        # Inradius = sqrt(3)/2 ≈ 0.866
        inradius = math.sqrt(3) / 2
        max_proj = 0
        for i in range(6):
            theta = math.radians(60 * i)
            proj = lx * math.cos(theta) + ly * math.sin(theta)
            max_proj = max(max_proj, proj)
        return max_proj - inradius  # negative if inside

    def hexagon_distance(h1, h2):
        """Minimum distance between two unit hexagons (negative if overlapping)."""
        # Use center-to-center distance and orientation difference
        # For two regular hexagons of side 1, if they don't overlap,
        # the minimum distance is at least related to center distance
        # Simple approach: check if any vertex of one is inside the other,
        # and compute min distance between edges
        # Fast approximation: if center distance > 2*side, no overlap
        dx = h2[0] - h1[0]
        dy = h2[1] - h1[1]
        d = math.sqrt(dx*dx + dy*dy)
        if d > 2.0:
            return d - 2.0  # approximate: hexagon diameter is 2
        # More precise: check containment
        v1 = hex_vertices(h1[0], h1[1], 1.0, h1[2])
        v2 = hex_vertices(h2[0], h2[1], 1.0, h2[2])
        min_dist = float('inf')
        # Check all vertices of hex1 against hex2 boundary and vice versa
        for px, py in v1:
            dist = min_distance_to_hexagon(h2[0], h2[1], h2[2], px, py)
            min_dist = min(min_dist, dist)
        for px, py in v2:
            dist = min_distance_to_hexagon(h1[0], h1[1], h1[2], px, py)
            min_dist = min(min_dist, dist)
        return min_dist

    def outer_hex_contains_point(ocx, ocy, oside, oangle, px, py):
        """Check if point is inside outer hexagon."""
        verts = hex_vertices(ocx, ocy, oside, oangle)
        return point_in_convex_polygon(px, py, verts)

    def min_distance_to_outer_hex(ocx, ocy, oside, oangle, px, py):
        """Minimum distance from point to outer hexagon boundary (negative if inside)."""
        verts = hex_vertices(ocx, ocy, oside, oangle)
        # Transform to local frame
        dx = px - ocx
        dy = py - ocy
        a = math.radians(oangle)
        lx = dx * math.cos(a) + dy * math.sin(a)
        ly = -dx * math.sin(a) + dy * math.cos(a)
        
        # For pointy-top hex with side s, inradius = s*sqrt(3)/2
        inradius = oside * math.sqrt(3) / 2
        max_proj = 0
        for i in range(6):
            theta = math.radians(60 * i)
            proj = lx * math.cos(theta) + ly * math.sin(theta)
            max_proj = max(max_proj, proj)
        return max_proj - inradius

    def evaluate_config(params):
        """Evaluate a configuration: return (outer_side, penalty).
        params: [x1,y1,a1, x2,y2,a2, ..., x12,y2,a12, ocx, ocy, oside, oangle]
        """
        inner = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner.append((x, y, a))
        ocx = params[3*N]
        ocy = params[3*N+1]
        oside = params[3*N+2]
        oangle = params[3*N+3]
        
        # Check all inner hexagons are inside outer hexagon
        penalty = 0.0
        for hx, hy, ha in inner:
            verts = hex_vertices(hx, hy, 1.0, ha)
            for px, py in verts:
                d = min_distance_to_outer_hex(ocx, ocy, oside, oangle, px, py)
                if d > 0:
                    penalty += d * d
        
        # Check no overlaps between inner hexagons
        for i in range(N):
            for j in range(i+1, N):
                d = hexagon_distance(inner[i], inner[j])
                if d < 0:
                    penalty += d * d
        
        return oside, penalty

    def objective(params):
        """Objective: minimize outer side + large penalty for violations."""
        if time.time() > deadline:
            return 1e6
        oside, penalty = evaluate_config(params)
        return oside + 100.0 * penalty

    # Initial guess: hexagonal lattice arrangement
    def make_initial():
        # Place hexagons in a hexagonal pattern
        # For 12 hexagons: 1 center + 6 around + 5 more
        # Try a compact arrangement
        positions = []
        # Center
        positions.append((0, 0, 0))
        # 6 around center
        for i in range(6):
            angle = math.radians(60 * i)
            positions.append((2.0*math.cos(angle), 2.0*math.sin(angle), 0))
        # 5 more: extend the pattern
        # Place at distance ~4 in various directions
        for i in range(5):
            angle = math.radians(60 * i + 30)
            positions.append((4.0*math.cos(angle), 4.0*math.sin(angle), 0))
        
        params = []
        for x, y, a in positions:
            params.extend([x, y, a])
        # Outer hexagon: large enough to contain all
        params.extend([0.0, 0.0, 6.0, 0.0])
        return np.array(params)

    best_params = None
    best_score = float('inf')

    # Try multiple initial configurations
    initial_configs = []
    
    # Config 1: hexagonal lattice
    initial_configs.append(make_initial())
    
    # Config 2: linear arrangement with some rotation
    params2 = []
    for i in range(N):
        x = 2.0 * (i - N//2)
        y = 0.0
        a = 0.0
        params2.extend([x, y, a])
    params2.extend([0.0, 0.0, 8.0, 0.0])
    initial_configs.append(np.array(params2))
    
    # Config 3: two rows
    params3 = []
    for i in range(6):
        x = 2.0 * (i - 3)
        y = 0.0
        params3.extend([x, y, 0.0])
    for i in range(6):
        x = 2.0 * (i - 3) + 1.0
        y = math.sqrt(3)
        params3.extend([x, y, 0.0])
    params3.extend([0.0, math.sqrt(3)/2, 5.0, 0.0])
    initial_configs.append(np.array(params3))

    for idx, init in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        # Evaluate initial
        oside, penalty = evaluate_config(init)
        score = oside + 100.0 * penalty
        if score < best_score:
            best_score = score
            best_params = init.copy()
        
        # Local optimization
        try:
            result = minimize(objective, init, method='L-BFGS-B', 
                             options={'maxiter': 200, 'ftol': 1e-8})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 20:
            break

    # Try differential evolution if time remains
    if time.time() < deadline - 40:
        try:
            bounds = []
            for i in range(N):
                bounds.append((-10, 10))
                bounds.append((-10, 10))
                bounds.append((-180, 180))
            bounds.append((-5, 5))   # ocx
            bounds.append((-5, 5))   # ocy
            bounds.append((2.0, 15.0))  # oside
            bounds.append((-180, 180))  # oangle
            
            de_result = differential_evolution(objective, bounds, maxiter=20, 
                                               popsize=15, tol=1e-6, seed=42,
                                               polish=False)
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_params = de_result.x.copy()
        except Exception:
            pass

    # Final local polish
    if best_params is not None and time.time() < deadline - 10:
        try:
            result = minimize(objective, best_params, method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-10})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    # Extract final configuration
    if best_params is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        inner.append([x, y, a])
    
    ocx = best_params[3*N]
    ocy = best_params[3*N+1]
    oside = best_params[3*N+2]
    oangle = best_params[3*N+3]
    
    return inner, (ocx, ocy), oside, oangle
# EVOLVE-BLOCK-END
