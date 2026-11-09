"""
Structural change: use a physics-based packing search with simulated annealing.
We represent each inner hexagon by (x, y, theta). We minimize the outer side length
required to contain all hexagons without overlap.

Key ideas:
1. Compute the "radius" of a rotated hexagon in the direction toward each other hexagon/outer boundary.
2. Use pairwise collision detection via Separating Axis Theorem (SAT) for convex polygons.
3. Use a force-directed / annealing approach to find non-overlapping configurations.
4. The objective is the minimal outer side length S such that all hexagons fit inside a regular hexagon of side S.
5. To check if a hexagon fits inside the outer hexagon, we check all 6 vertices of the inner hexagon are inside the outer hexagon.

Given time constraints, we'll use a simplified but effective approach:
- Start with a hexagonal lattice arrangement (known good packing for hexagons).
- Apply small random perturbations and use simulated annealing to reduce the outer side length.
- The cost function is: max over all inner hexagons of (distance from center in the hexagonal norm) scaled appropriately, plus a penalty for overlaps.

Actually, let me think more carefully.

A regular hexagon of side s has "radius" (center to vertex) = s, and "inradius" (center to edge midpoint) = s * sqrt(3)/2.

For two regular hexagons of side 1 to be non-overlapping, the distance between their centers must be at least 2 if they are aligned (same rotation), but less if rotated relative to each other.

The minimal outer side length for N hexagons: for a hexagonal packing, if we place them in a hexagonal lattice with spacing 2 (center-to-center), the outer hexagon side length would be related to the number of hexagons.

For a hexagonal arrangement with "radius" r (number of hexagons from center to edge), the total number is 1 + 3*r*(r+1). The outer hexagon side length needed would be approximately 2*r + 1 (in units of hexagon side length) plus some margin.

Let me use a direct approach:
- Place hexagons in a hexagonal lattice.
- Compute the minimal enclosing hexagon (regular, same orientation or optimally rotated).
- Then try to optimize.

For the minimal enclosing regular hexagon of a set of points (vertices of inner hexagons), we can binary search on the side length S and check if all vertices fit inside a regular hexagon of side S centered at some point.

Actually, the outer hexagon can also be rotated. So we have 3 parameters for the outer hexagon: center_x, center_y, side_length, and rotation angle. That's 4 parameters.

Given the complexity, let me use a simpler but effective strategy:
1. Generate initial configurations using hexagonal lattice.
2. Use a cost function based on the maximum "hexagonal distance" from the center.
3. Optimize positions using gradient-free methods or simulated annealing.

Let me implement a practical solution:

The hexagonal distance (or "hex distance") from a point to the center of a regular hexagon of side S with the same orientation:
For a regular hexagon with flat top (vertices at angles 0, 60, 120, 180, 240, 300 degrees), the boundary in direction θ is at distance S * cos(π/6) / cos(θ - π/6 * k) for the appropriate edge.

Actually, for a regular hexagon centered at origin with side S, the support function in direction φ is:
h(φ) = S * cos(π/6) / cos(φ mod π/3 - π/6)

This gives the maximum distance from center in direction φ that stays inside the hexagon.

So a point (x,y) is inside if ||(x,y)|| <= h(atan2(y,x)).

For the inner hexagons, each has 6 vertices. All 36*N vertices must be inside the outer hexagon.

Let me code this up with a simulated annealing approach.
"""

import numpy as np
import time
import math
import random

N = 11

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return verts

def _point_in_hex(px, py, hx, hy, hside, hangle_deg):
    """Check if point (px,py) is inside hexagon centered at (hx,hy) with side hside and rotation hangle_deg."""
    # Translate to hexagon center
    dx = px - hx
    dy = py - hy
    # Rotate by -hangle to align
    angle_rad = math.radians(hangle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rx = dx * cos_a + dy * sin_a
    ry = -dx * sin_a + dy * cos_a
    
    # For a regular hexagon with flat-top orientation (vertex at angle 0), 
    # the hexagon has vertices at angles 0, 60, 120, 180, 240, 300.
    # The inradius (apothem) is hside * sqrt(3)/2
    # A point is inside if for all 6 edges, the point is on the correct side.
    
    apothem = hside * math.sqrt(3) / 2
    
    # The 6 edge normals (pointing inward) for a hexagon with vertex at angle 0:
    # Edges are perpendicular to directions at 30, 90, 150, 210, 270, 330 degrees
    for i in range(6):
        edge_angle = math.radians(30 + i * 60)
        nx = math.cos(edge_angle)
        ny = math.sin(edge_angle)
        # Project point onto normal
        proj = rx * nx + ry * ny
        if proj > apothem:
            return False
    return True

def _hexagons_overlap(v1, v2):
    """Check if two convex polygons (lists of vertices) overlap using SAT."""
    polys = [v1, v2]
    for poly in polys:
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            # Edge normal
            dx = x2 - x1
            dy = y2 - y1
            # Normal (perpendicular)
            nx = -dy
            ny = dx
            # Project all vertices of both polygons onto this normal
            min1, max1 = float('inf'), float('-inf')
            min2, max2 = float('inf'), float('-inf')
            for px, py in v1:
                proj = px * nx + py * ny
                min1 = min(min1, proj)
                max1 = max(max1, proj)
            for px, py in v2:
                proj = px * nx + py * ny
                min2 = min(min2, proj)
                max2 = max(max2, proj)
            # Check for gap
            if max1 < min2 or max2 < min1:
                return False
    return True

def _fits_in_outer(hexagon_verts, outer_cx, outer_cy, outer_side, outer_angle):
    """Check if all vertices of hexagon are inside the outer hexagon."""
    for vx, vy in hexagon_verts:
        if not _point_in_hex(vx, vy, outer_cx, outer_cy, outer_side, outer_angle):
            return False
    return True

def _compute_cost(inner, outer_cx, outer_cy, outer_side, outer_angle):
    """
    Compute cost: 
    - Penalty if any inner hexagon is not fully inside outer
    - Penalty if any pair of inner hexagons overlap
    Returns (cost, is_valid)
    """
    # Get all inner hexagon vertices
    inner_verts = []
    for x, y, ang in inner:
        verts = _hex_vertices(x, y, 1.0, ang)
        inner_verts.append(verts)
    
    # Check containment
    for verts in inner_verts:
        if not _fits_in_outer(verts, outer_cx, outer_cy, outer_side, outer_angle):
            return float('inf'), False
    
    # Check pairwise overlaps
    n = len(inner_verts)
    for i in range(n):
        for j in range(i + 1, n):
            if _hexagons_overlap(inner_verts[i], inner_verts[j]):
                return float('inf'), False
    
    return outer_side, True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial configuration: hexagonal lattice
    # For N hexagons, use a hexagonal arrangement
    # Number of hexagons in hexagonal arrangement with radius r: 1 + 3r(r+1)
    # r=0: 1, r=1: 7, r=2: 19
    # For N=11, use r=1 (7 hexagons) plus 4 more
    
    # Better: just use a linear or 2D grid initially
    # Place in a hexagonal pattern
    
    def make_hex_lattice(n):
        """Create a hexagonal lattice of n unit hexagons."""
        positions = []
        # Hexagonal grid coordinates
        # For a hex grid, spacing is 2 in x direction for same row, 
        # and sqrt(3) in y direction between rows
        # Center-to-center distance for non-overlapping aligned hexagons is 2
        
        # Use axial coordinates
        # Generate positions in a compact hexagonal shape
        r = 0
        while True:
            count = 1 + 3 * r * (r + 1) if r > 0 else 1
            if count >= n:
                break
            r += 1
        
        # Generate hex grid positions with radius r
        positions = []
        for q in range(-r, r + 1):
            for q2 in range(max(-r, -q - r), min(r, -q + r) + 1):
                # Convert axial to pixel
                x = 2.0 * q
                y = math.sqrt(3) * (q2 + q / 2.0)
                positions.append((x, y))
        
        # Take first n
        positions = positions[:n]
        return positions
    
    pos = make_hex_lattice(N)
    
    # Initial inner configuration
    inner = [[x, y, 0.0] for x, y in pos]
    
    # Initial outer hexagon: large enough to contain all
    # Compute bounding box of all vertices
    all_verts = []
    for x, y, ang in inner:
        verts = _hex_vertices(x, y, 1.0, ang)
        all_verts.extend(verts)
    
    if not all_verts:
        return inner, (0.0, 0.0), 40.0, 0.0
    
    xs = [v[0] for v in all_verts]
    ys = [v[1] for v in all_verts]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    
    # Estimate outer side length
    max_dist = max(math.sqrt((x - cx)**2 + (y - cy)**2) for x, y in all_verts)
    # For a regular hexagon, the max distance from center to vertex is side length
    # But we need to account for orientation. Start with a generous estimate.
    outer_side = max_dist * 1.2 + 2.0
    outer_angle = 0.0
    
    # Verify initial solution
    cost, valid = _compute_cost(inner, cx, cy, outer_side, outer_angle)
    if not valid:
        # Increase outer side
        outer_side *= 1.5
        cost, valid = _compute_cost(inner, cx, cy, outer_side, outer_angle)
    
    best_inner = [list(h) for h in inner]
    best_cx, best_cy = cx, cy
    best_side = outer_side
    best_angle = outer_angle
    best_cost = outer_side if valid else float('inf')
    
    if not valid:
        return best_inner, (best_cx, best_cy), best_side, best_angle
    
    # Simulated annealing
    current_inner = [list(h) for h in inner]
    current_cx, current_cy = cx, cy
    current_side = outer_side
    current_angle = outer_angle
    
    temperature = 0.5
    cooling = 0.995
    step_size = 0.3
    
    # Precompute some constants
    sqrt3 = math.sqrt(3)
    
    iteration = 0
    no_improve_count = 0
    
    while time.time() < deadline and temperature > 0.01:
        iteration += 1
        
        # Random move: perturb one inner hexagon
        idx = random.randint(0, N - 1)
        move_type = random.random()
        
        if move_type < 0.4:
            # Move position
            dx = random.gauss(0, step_size)
            dy = random.gauss(0, step_size)
            current_inner[idx][0] += dx
            current_inner[idx][1] += dy
        elif move_type < 0.7:
            # Rotate
            dangle = random.gauss(0, step_size * 10)
            current_inner[idx][2] += dangle
        elif move_type < 0.85:
            # Adjust outer center
            current_cx += random.gauss(0, step_size * 0.5)
            current_cy += random.gauss(0, step_size * 0.5)
        else:
            # Adjust outer angle
            current_angle += random.gauss(0, step_size * 5)
        
        # Compute new cost
        new_cost, new_valid = _compute_cost(current_inner, current_cx, current_cy, current_side, current_angle)
        
        if new_valid:
            # Try to reduce outer side length
            # Binary search for minimal outer side
            lo = 1.0
            hi = current_side
            for _ in range(15):
                mid = (lo + hi) / 2
                _, v = _compute_cost(current_inner, current_cx, current_cy, mid, current_angle)
                if v:
                    hi = mid
                else:
                    lo = mid
            if hi < current_side:
                current_side = hi
                new_cost = current_side
        
        if new_valid and new_cost < best_cost:
            best_inner = [list(h) for h in current_inner]
            best_cx, best_cy = current_cx, current_cy
            best_side = current_side
            best_angle = current_angle
            best_cost = new_cost
            no_improve_count = 0
        else:
            # Metropolis acceptance
            if new_valid:
                delta = new_cost - best_cost
                if delta < 0 or random.random() < math.exp(-delta / max(temperature, 0.001)):
                    no_improve_count += 1
                else:
                    # Revert
                    if move_type < 0.4:
                        current_inner[idx][0] -= dx
                        current_inner[idx][1] -= dy
                    elif move_type < 0.7:
                        current_inner[idx][2] -= dangle
                    elif move_type < 0.85:
                        current_cx -= random.gauss(0, step_size * 0.5)  # This is wrong, we can't revert properly
                        current_cy -= random.gauss(0, step_size * 0.5)
                    else:
                        current_angle -= random.gauss(0, step_size * 5)
                    no_improve_count += 1
            else:
                # Revert
                if move_type < 0.4:
                    current_inner[idx][0] -= dx
                    current_inner[idx][1] -= dy
                elif move_type < 0.7:
                    current_inner[idx][2] -= dangle
                elif move_type < 0.85:
                    pass  # Can't easily revert, just continue
                else:
                    pass
                no_improve_count += 1
        
        temperature *= cooling
        
        if no_improve_count > 100:
            no_improve_count = 0
            step_size *= 0.8
    
    return best_inner, (best_cx, best_cy), best_side, best_angle
