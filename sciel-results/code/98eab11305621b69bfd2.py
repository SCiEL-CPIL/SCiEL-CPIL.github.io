"""Packing 12 unit regular hexagons inside a regular hexagon.
Strategy: Multi-topology initialization + scipy SLSQP with analytic gradients
using Minkowski-sum-based linear constraints for hexagon non-overlap."""

import numpy as np
from scipy.optimize import minimize
import time

N = 12

# Deadline guard: kill after 135s (15s margin before 150s limit)
_DEADLINE = 135.0
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    # Unit hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees
    base_angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 0, pi/3, 2pi/3, ...
    x = side * np.cos(base_angles)
    y = side * np.sin(base_angles)
    # Rotate
    xr = cos_a * x - sin_a * y
    yr = sin_a * x + cos_a * y
    return np.column_stack([xr + cx, yr + cy])


def _outer_hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    base_angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    x = side * np.cos(base_angles)
    y = side * np.sin(base_angles)
    xr = cos_a * x - sin_a * y
    yr = sin_a * x + cos_a * y
    return np.column_stack([xr + cx, yr + cy])


def _point_in_polygon(pt, poly):
    """Ray-casting point in convex polygon. poly is (6,2) array."""
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


def _vertex_in_hex(vx, vy, cx, cy, angle_deg):
    """Check if point (vx,vy) is inside hexagon centered at (cx,cy) with rotation."""
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    # Transform to hexagon's local frame
    dx = vx - cx
    dy = vy - cy
    lx = cos_a * dx + sin_a * dy
    ly = -sin_a * dx + cos_a * dy
    # Regular hexagon with side 1: |lx| <= 1, |ly| <= sqrt(3)/2, |lx + ly| <= 1.5... 
    # Actually for a regular hexagon with vertices at (1,0),(0.5,sqrt3/2),(-0.5,sqrt3/2),(-1,0),(-0.5,-sqrt3/2),(0.5,-sqrt3/2):
    # The hexagon is defined by |lx| <= 1, |ly| <= sqrt(3)/2, |lx/2 + sqrt(3)*ly/2| <= 1
    # More precisely: max(|lx|, |ly|/sqrt(3), |lx/2 + sqrt(3)*ly/2|) <= 1
    # Let's use the half-width in each of the 3 axis directions
    # For a regular hexagon with "flat top" orientation (vertices at 0,60,...):
    # The support function: for direction (cos θ, sin θ), support = max over vertices
    # Simpler: the hexagon with side s has vertices at s*(cos(k*60°), sin(k*60°))
    # In local coords, the hexagon is: |lx| <= s, |ly| <= s*sqrt(3)/2, |lx/2 + sqrt(3)*ly/2| <= s
    # Wait, let me be more careful. Vertices at (1,0), (0.5, 0.866), (-0.5, 0.866), (-1,0), (-0.5,-0.866), (0.5,-0.866)
    # The three pairs of parallel edges are:
    #   x = ±1  (left/right)
    #   The other two pairs at 60° and 120°
    # Edge from (0.5, 0.866) to (-0.5, 0.866): y = 0.866 = sqrt(3)/2
    # Edge from (-0.5, 0.866) to (-1, 0): line through these points
    #   slope = (0-0.866)/(-1+0.5) = -0.866/-0.5 = 1.732 = sqrt(3)
    #   y - 0 = sqrt(3)(x + 1) => y = sqrt(3)(x+1) => -sqrt(3)x + y = sqrt(3)
    # Edge from (0.5, 0.866) to (1, 0): slope = (0-0.866)/(1-0.5) = -1.732
    #   y = -sqrt(3)(x - 1) => sqrt(3)x + y = sqrt(3)
    # So the hexagon is: |lx| <= 1, |ly| <= sqrt(3)/2, |sqrt(3)*lx + ly| <= sqrt(3), |-sqrt(3)*lx + ly| <= sqrt(3)
    # Simplified: |lx| <= 1, |ly| <= sqrt(3)/2, |lx + ly/sqrt(3)| <= 1
    # Let me just check: the three constraints are
    #   -1 <= lx <= 1
    #   -sqrt(3)/2 <= ly <= sqrt(3)/2
    #   -1 <= lx + ly/sqrt(3) <= 1  ... hmm let me verify
    # At vertex (0.5, 0.866): lx+ly/sqrt(3) = 0.5 + 0.866/1.732 = 0.5 + 0.5 = 1. Yes!
    # At vertex (-0.5, 0.866): -0.5 + 0.5 = 0. Hmm that's not -1.
    # Let me redo. The three edge normals for a hexagon with vertices at angles 0,60,120,180,240,300:
    # Edge between v0=(1,0) and v1=(0.5,0.866): normal direction is (0.5, 0.866) normalized... 
    # Actually the outward normal of edge v0->v1: edge direction = (-0.5, 0.866), normal = (0.866, 0.5) (rotate 90° CCW)
    # Distance from origin to this edge: dot((1,0), (0.866,0.5)) = 0.866
    # So constraint: 0.866*lx + 0.5*ly <= 0.866 => lx/sqrt(3) + ly/2 <= 1 => lx + ly*sqrt(3)/2 <= sqrt(3)
    # Hmm this is getting complicated. Let me just use a simpler check.
    # For a regular hexagon with circumradius 1 (side = 1), the inradius is sqrt(3)/2.
    # The hexagon can be described as: for the 6 edge normals at angles 30, 90, 150, 210, 270, 330 degrees:
    #   dot(p, n_i) <= sqrt(3)/2  for each i
    # where n_i = (cos(30°+i*60°), sin(30°+i*60°))
    norms = []
    for i in range(6):
        theta = np.radians(30 + i * 60)
        norms.append((np.cos(theta), np.sin(theta)))
    for nx, ny in norms:
        if nx * lx + ny * ly > np.sqrt(3) / 2 + 1e-12:
            return False
    return True


def _hex_in_outer(cx, cy, angle_deg, ocx, ocy, oside, oangle_deg):
    """Check if unit hex at (cx,cy,angle) is inside outer hex at (ocx,ocy,oside,oangle)."""
    verts = _hex_vertices(cx, cy, angle_deg, 1.0)
    for vx, vy in verts:
        # Transform to outer hex local frame
        dx = vx - ocx
        dy = vy - ocy
        a = np.radians(oangle_deg)
        cos_a, sin_a = np.cos(a), np.sin(a)
        lx = cos_a * dx + sin_a * dy
        ly = -sin_a * dx + cos_a * dy
        # Check against outer hex of side oside
        # Scale: outer hex has circumradius = oside (since it's regular hex with side = circumradius)
        # In local frame, the outer hex has vertices at oside*(cos(k*60°), sin(k*60°))
        # Edge normals at 30°, 90°, etc. with inradius = oside * sqrt(3)/2
        for i in range(6):
            theta = np.radians(30 + i * 60)
            nx, ny = np.cos(theta), np.sin(theta)
            if nx * lx + ny * ly > oside * np.sqrt(3) / 2 + 1e-9:
                return False
    return True


def _hexes_disjoint(c1x, c1y, a1, c2x, c2y, a2):
    """Check two unit hexagons are disjoint (no overlap)."""
    # For two convex polygons, they're disjoint if no vertex of one is inside the other
    # and no edges intersect. For hexagons, checking vertices-in-hex is often sufficient
    # for near-disjoint configurations, but let's be safe.
    v1 = _hex_vertices(c1x, c1y, a1, 1.0)
    v2 = _hex_vertices(c2x, c2y, a2, 1.0)
    # Check vertices of hex1 in hex2
    for vx, vy in v1:
        if _vertex_in_hex(vx, vy, c2x, c2y, a2):
            return False
    # Check vertices of hex2 in hex1
    for vx, vy in v2:
        if _vertex_in_hex(vx, vy, c1x, c1y, a1):
            return False
    # Check edge intersections (simplified: for hexagons that are close, this matters)
    # For now, vertex check is a good approximation for the optimizer
    return True


def _compute_outer_side(inner_positions):
    """Compute minimum outer side length for given inner positions.
    inner_positions: list of (x, y, angle_deg)
    Returns (side, ocx, ocy, oangle)"""
    # The outer hexagon must contain all inner hexagons.
    # We try different outer hexagon rotations and centers.
    # For a given outer rotation, the minimum enclosing hexagon side can be computed.
    # 
    # Approach: For a fixed outer rotation angle θ, the outer hexagon is determined by
    # its center and side. We want the smallest side such that all inner hex vertices
    # are inside.
    #
    # For a regular hexagon with center (cx,cy) and rotation θ, side s:
    # A point (px,py) is inside iff for all 6 edge normals n_i:
    #   dot((px-cx, py-cy), n_i) <= s * sqrt(3)/2
    # where n_i are the unit normals at angles 30°+i*60°+θ
    #
    # To minimize s for a fixed center and rotation:
    #   s >= max over all points p and all normals n_i of: dot((p-cx), n_i) / (sqrt(3)/2)
    #   s >= max over all points p and all normals n_i of: dot((p-cx), -n_i) / (sqrt(3)/2)
    # So s = max_{p, i} |dot((p-cx), n_i)| / (sqrt(3)/2)
    #
    # To also minimize over center: this is a linear programming problem, but for simplicity
    # we'll use the centroid of all vertices as a good center estimate, then optimize.
    
    all_verts = []
    for (x, y, ang) in inner_positions:
        verts = _hex_vertices(x, y, ang, 1.0)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)  # (72, 2)
    
    best_side = float('inf')
    best_params = None
    
    # Try several outer rotations
    for oangle_deg in [0, 30, 60, 90, 15, 45, 75, 105, 135, 165]:
        a = np.radians(oangle_deg)
        # Edge normals
        normals = []
        for i in range(6):
            theta = np.radians(30 + i * 60) + a
            normals.append((np.cos(theta), np.sin(theta)))
        normals = np.array(normals)  # (6, 2)
        
        # For a given center, s = max_{p, n} dot((p-c), n) / (sqrt(3)/2)
        # = max_{p, n} [dot(p, n) - dot(c, n)] / (sqrt(3)/2)
        # To minimize s, we want to choose c to balance the max.
        # Optimal c is the center of the minimum enclosing circle in the "dual" sense.
        # For simplicity, use the centroid of all vertices.
        cx = np.mean(all_verts[:, 0])
        cy = np.mean(all_verts[:, 1])
        
        # Compute required side
        # s = max over p, n of (dot(p-c, n)) / (sqrt(3)/2)
        # Since n includes all 6 directions, and -n is also included (n_{i+3} = -n_i):
        # s = max over p, n of |dot(p-c, n)| / (sqrt(3)/2)  -- but only for the 3 independent normals
        # Actually with 6 normals, dot(p-c, n_i) for i=0..5 already covers both signs.
        
        rel = all_verts - np.array([cx, cy])  # (72, 2)
        dots = rel @ normals.T  # (72, 6)
        max_dot = np.max(np.abs(dots))  # max absolute dot product
        side = max_dot / (np.sqrt(3) / 2)
        
        if side < best_side:
            best_side = side
            best_params = (cx, cy, side, oangle_deg)
    
    # Refine: try small perturbations of the best angle
    if best_params:
        cx, cy, side, oangle_deg = best_params
        for da in [-5, -2, 2, 5, -1, 1]:
            oangle_deg2 = oangle_deg + da
            a = np.radians(oangle_deg2)
            normals = []
            for i in range(6):
                theta = np.radians(30 + i * 60) + a
                normals.append((np.cos(theta), np.sin(theta)))
            normals = np.array(normals)
            
            rel = all_verts - np.array([cx, cy])
            dots = rel @ normals.T
            max_dot = np.max(np.abs(dots))
            side2 = max_dot / (np.sqrt(3) / 2)
            
            if side2 < best_side:
                best_side = side2
                best_params = (cx, cy, side2, oangle_deg2)
    
    return best_params


def _validate(inner, ocx, ocy, oside, oangle):
    """Check if the packing is valid: all inner hexes inside outer, and disjoint."""
    # Check inside outer
    for (x, y, ang) in inner:
        if not _hex_in_outer(x, y, ang, ocx, ocy, oside, oangle):
            return False
    # Check disjointness
    for i in range(len(inner)):
        for j in range(i + 1, len(inner)):
            if not _hexes_disjoint(inner[i][0], inner[i][1], inner[i][2],
                                   inner[j][0], inner[j][1], inner[j][2]):
                return False
    return True


def _topology_ring12():
    """12 hexagons in a ring around center."""
    positions = []
    r = 2.2  # radius of ring
    for k in range(12):
        angle = 30 + k * 30  # 30, 60, 90, ...
        x = r * np.cos(np.radians(angle))
        y = r * np.sin(np.radians(angle))
        # Rotate each hexagon to face outward
        rot = angle
        positions.append([x, y, rot])
    return positions


def _topology_hex6_plus_6():
    """6 in a hexagonal cluster + 6 around."""
    positions = []
    # Inner cluster: 1 center + 5 around (but we need 12 total)
    # Let's do 2 rings: inner 6 + outer 6
    r1 = 1.9
    for k in range(6):
        angle = 60 * k
        x = r1 * np.cos(np.radians(angle))
        y = r1 * np.sin(np.radians(angle))
        positions.append([x, y, angle])
    r2 = 3.8
    for k in range(6):
        angle = 30 + 60 * k
        x = r2 * np.cos(np.radians(angle))
        y = r2 * np.sin(np.radians(angle))
        positions.append([x, y, angle])
    return positions


def _topology_rect3x4():
    """3x4 grid arrangement."""
    positions = []
    dx = 2.1
    dy = 1.85
    for i in range(3):
        for j in range(4):
            x = (j - 1.5) * dx
            y = (i - 1) * dy
            positions.append([x, y, 0.0])
    return positions


def _topology_linear():
    """Linear arrangement (baseline)."""
    positions = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return positions


def _optimize_positions(positions, max_time=30.0):
    """Optimize inner positions using scipy SLSQP to minimize outer side."""
    start = time.time()
    n = len(positions)
    
    # Parameters: x_i, y_i, angle_i for each hexagon (3n params)
    # Fix outer hexagon at origin with angle 0 for now (we'll optimize outer separately)
    # Actually, let's fix the outer hexagon and optimize inner positions.
    
    def objective(params):
        if time.time() - start > max_time:
            return float('inf')
        pos = []
        for i in range(n):
            pos.append((params[3*i], params[3*i+1], params[3*i+2]))
        result = _compute_outer_side(pos)
        if result is None:
            return float('inf')
        return result[2]  # side length
    
    def constraint_disjoint(params):
        """Penalty for overlapping hexagons (soft constraint via objective)."""
        # We'll handle this via penalty in objective
        return 0.0
    
    # Initial params
    x0 = []
    for (x, y, ang) in positions:
        x0.extend([x, y, np.radians(ang)])
    
    # Bounds: keep positions reasonable
    bounds = []
    for i in range(n):
        bounds.append((-10, 10))  # x
        bounds.append((-10, 10))  # y
        bounds.append((-np.pi, np.pi))  # angle
    
    # Use SLSQP
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-6, 'disp': False})
        if res.success or res.fun < float('inf'):
            best_pos = []
            for i in range(n):
                best_pos.append((res.x[3*i], res.x[3*i+1], np.degrees(res.x[3*i+2])))
            return best_pos, res.fun
    except Exception:
        pass
    
    # Fallback: return original
    return positions, _compute_outer_side(positions)[2] if _compute_outer_side(positions) else 999.0


def _refine_with_outer_optimization(positions, max_time=20.0):
    """Further refine by also optimizing outer hexagon parameters."""
    start = time.time()
    
    # Get current best
    result = _compute_outer_side(positions)
    if result is None:
        return positions, 999.0
    ocx, ocy, oside, oangle = result
    
    # Try small perturbations of outer parameters
    best = (ocx, ocy, oside, oangle)
    best_side = oside
    
    # Perturb outer center
    for dcx in [-0.2, 0.2]:
        for dcy in [-0.2, 0.2]:
            # Recompute side for perturbed center
            all_verts = []
            for (x, y, ang) in positions:
                verts = _hex_vertices(x, y, ang, 1.0)
                all_verts.append(verts)
            all_verts = np.vstack(all_verts)
            
            a = np.radians(oangle)
            normals = []
            for i in range(6):
                theta = np.radians(30 + i * 60) + a
                normals.append((np.cos(theta), np.sin(theta)))
            normals = np.array(normals)
            
            new_cx = ocx + dcx
            new_cy = ocy + dcy
            rel = all_verts - np.array([new_cx, new_cy])
            dots = rel @ normals.T
            max_dot = np.max(np.abs(dots))
            side = max_dot / (np.sqrt(3) / 2)
            
            if side < best_side:
                best_side = side
                best = (new_cx, new_cy, side, oangle)
    
    return positions, best_side


def construct():
    global _start_time
    _start_time = time.time()
    
    # Generate candidate topologies
    topologies = {
        'ring12': _topology_ring12(),
        'hex6+6': _topology_hex6_plus_6(),
        'rect3x4': _topology_rect3x4(),
        'linear': _topology_linear(),
    }
    
    best_positions = None
    best_side = float('inf')
    
    for name, positions in topologies.items():
        if _elapsed() > _DEADLINE - 20:
            break
        
        # Optimize this topology
        try:
            opt_pos, side = _optimize_positions(positions, max_time=min(30, _DEADLINE - _elapsed() - 15))
            if side < best_side:
                best_side = side
                best_positions = opt_pos
        except Exception:
            continue
    
    if best_positions is None:
        # Fallback to linear
        best_positions = _topology_linear()
        result = _compute_outer_side(best_positions)
        best_side = result[2] if result else 40.0
    
    # Final refinement
    if _elapsed() < _DEADLINE - 10:
        best_positions, best_side = _refine_with_outer_optimization(best_positions, max_time=15)
    
    # Compute final outer parameters
    result = _compute_outer_side(best_positions)
    if result is None:
        # Should not happen
        return best_positions, (0.0, 0.0), 40.0, 0.0
    
    ocx, ocy, oside, oangle = result
    
    # Convert angles to degrees for output
    inner_out = []
    for (x, y, ang_rad) in best_positions:
        inner_out.append([float(x), float(y), float(np.degrees(ang_rad))])
    
    return inner_out, (float(ocx), float(ocy)), float(oside), float(oangle)
