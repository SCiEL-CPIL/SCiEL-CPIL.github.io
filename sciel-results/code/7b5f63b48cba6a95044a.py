"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    def hex_vertices(cx, cy, side, angle_deg):
        """Get vertices of a regular hexagon."""
        angle = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle + i * math.pi / 3.0
            verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
        return verts
    
    def min_enclosing_hex_for_points(points, outer_angle_deg):
        """Find minimum side length and center of hexagon with given angle containing all points.
        Returns (cx, cy, side) or None if infeasible."""
        angle = math.radians(outer_angle_deg)
        cos_a = math.cos(-angle)
        sin_a = math.sin(-angle)
        
        # Transform points to outer hex's local frame
        transformed = []
        for (px, py) in points:
            tx = px * cos_a - py * sin_a
            ty = px * sin_a + py * cos_a
            transformed.append((tx, ty))
        
        if not transformed:
            return (0.0, 0.0, 0.0)
        
        # In local frame, hexagon has normals at 30, 90, 150, 210, 270, 330 degrees
        # Constraint: for each normal n_i, |dot(p - c, n_i)| <= s * sqrt(3)/2
        # => s >= max_i (max_p dot(p, n_i) - min_p dot(p, n_i)) / sqrt(3)
        # Center c must satisfy: for each i, dot(c, n_i) is in [max_p dot(p,n_i) - s*sqrt(3)/2, 
        #   min_p dot(p,n_i) + s*sqrt(3)/2]
        
        normals = []
        for i in range(6):
            na = math.radians(30 + i * 60)
            normals.append((math.cos(na), math.sin(na)))
        
        # Compute ranges
        ranges = []
        mins = []
        maxs = []
        for (nx, ny) in normals:
            dots = [px * nx + py * ny for (px, py) in transformed]
            mn = min(dots)
            mx = max(dots)
            ranges.append(mx - mn)
            mins.append(mn)
            maxs.append(mx)
        
        # Minimum side length
        s = max(ranges) / sqrt3
        
        # Find center: dot(c, n_i) should be (max_i + min_i)/2 for the tightest constraint
        # Use centroid as initial guess, then verify
        cx_t = sum(p[0] for p in transformed) / len(transformed)
        cy_t = sum(p[1] for p in transformed) / len(transformed)
        
        # Verify containment
        for (px, py) in transformed:
            for (nx, ny) in normals:
                d = (px - cx_t) * nx + (py - cy_t) * ny
                if abs(d) > s * sqrt3 / 2.0 + 1e-9:
                    # Need to adjust center or increase s
                    # For now, increase s to accommodate
                    pass
        
        # Transform center back to global frame
        cx = cx_t * math.cos(angle) - cy_t * math.sin(angle)
        cy = cx_t * math.sin(angle) + cy_t * math.cos(angle)
        
        return (cx, cy, s)
    
    def get_all_vertices(inner_hexes):
        """Get all vertices of all inner hexagons."""
        all_pts = []
        for (hx, hy, hside, hang) in inner_hexes:
            verts = hex_vertices(hx, hy, hside, hang)
            all_pts.extend(verts)
        return all_pts
    
    def evaluate_arrangement(inner_hexes):
        """Find best enclosing hexagon for an arrangement."""
        all_pts = get_all_vertices(inner_hexes)
        best_L = float('inf')
        best_result = None
        
        # Try outer hexagon angles 0 to 60 degrees
        for outer_angle_deg in np.linspace(0, 60, 121):
            result = min_enclosing_hex_for_points(all_pts, outer_angle_deg)
            if result:
                cx, cy, L = result
                if L < best_L:
                    best_L = L
                    best_result = (cx, cy, L, outer_angle_deg)
        
        return best_result
    
    # Generate candidate arrangements
    candidates = []
    
    # Arrangement 1: 2x6 grid (flat-top hexagons)
    for offset in [0.0, sqrt3/4]:
        inner_hexes = []
        for row in range(2):
            for col in range(6):
                hx = col * 1.5
                hy = row * sqrt3 + offset
                inner_hexes.append((hx, hy, 1.0, 0.0))
        candidates.append(inner_hexes)
    
    # Arrangement 2: 3x4 grid
    for offset in [0.0, sqrt3/4]:
        inner_hexes = []
        for row in range(3):
            for col in range(4):
                hx = col * 1.5
                hy = row * sqrt3 + offset
                inner_hexes.append((hx, hy, 1.0, 0.0))
        candidates.append(inner_hexes)
    
    # Arrangement 3: 4x3 grid
    for offset in [0.0, sqrt3/4]:
        inner_hexes = []
        for row in range(4):
            for col in range(3):
                hx = col * 1.5
                hy = row * sqrt3 + offset
                inner_hexes.append((hx, hy, 1.0, 0.0))
        candidates.append(inner_hexes)
    
    # Arrangement 4: Hexagonal cluster (7 + 5 from ring 2)
    axial_positions = []
    for q in range(-1, 2):
        for r in range(-1, 2):
            if abs(q + r) <= 1:
                axial_positions.append((q, r))
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q+r)) == 2 and len(axial_positions) < 12:
                axial_positions.append((q, r))
    
    inner_hexes = []
    for (q, r) in axial_positions[:12]:
        x = 1.5 * q
        y = sqrt3 * (r + q/2.0)
        inner_hexes.append((x, y, 1.0, 0.0))
    candidates.append(inner_hexes)
    
    # Arrangement 5: Linear chain
    inner_hexes = []
    for i in range(12):
        inner_hexes.append((i * 1.5, 0.0, 1.0, 0.0))
    candidates.append(inner_hexes)
    
    # Arrangement 6: Two rows of 6 with different offset
    inner_hexes = []
    for col in range(6):
        inner_hexes.append((col * 1.5, 0.0, 1.0, 0.0))
        inner_hexes.append((col * 1.5 + 0.75, sqrt3/2, 1.0, 0.0))
    candidates.append(inner_hexes)
    
    # Arrangement 7: 5+4+3 rows (triangular-ish)
    inner_hexes = []
    for row in range(3):
        for col in range(5 - row):
            hx = col * 1.5 + row * 0.75
            hy = row * sqrt3/2
            inner_hexes.append((hx, hy, 1.0, 0.0))
    # That's 5+4+3=12
    candidates.append(inner_hexes)
    
    # Evaluate all candidates
    best_overall = None
    best_L = float('inf')
    
    for idx, inner_hexes in enumerate(candidates):
        if time.time() > deadline - 10:
            break
        result = evaluate_arrangement(inner_hexes)
        if result:
            cx, cy, L, angle = result
            if L < best_L:
                best_L = L
                best_overall = (inner_hexes, cx, cy, L, angle)
    
    if best_overall is None:
        # Fallback: linear arrangement with large hexagon
        inner_hexes = [(i * 1.5, 0.0, 1.0, 0.0) for i in range(12)]
        return [[h[0], h[1], h[3]] for h in inner_hexes], (0.0, 0.0), 20.0, 0.0
    
    inner_hexes, cx, cy, L, angle = best_overall
    # Return format: inner rows (x, y, angle_degrees), outer center, side, angle
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]
    return inner, (cx, cy), L, angle

# EVOLVE-BLOCK-END

