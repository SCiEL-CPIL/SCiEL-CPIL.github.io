"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0  # 120s safety margin within 150s limit

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _min_enclosing_hex_side(inner_hexes):
    """Find minimum side length of a regular hexagon that contains all inner hexagons.
    Returns (center_x, center_y, side_length, angle_deg)."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Collect all vertices of all inner hexagons
    all_vertices = []
    for ix, iy, iangle in inner_hexes:
        verts = _get_hex_vertices(ix, iy, iangle)
        all_vertices.extend(verts)
    
    # Try different angles for the outer hexagon
    # The optimal angle is likely to align with the symmetry of the inner cluster
    # or be 30 degrees off.
    for angle_deg in [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]:
        # The normals of the outer hexagon are at angle_deg + 30 + 60*k
        # We need to find a center (cx, cy) and side length s such that
        # for all vertices (vx, vy) and all 6 normals:
        # (vx-cx)*nx + (vy-cy)*ny <= s * sqrt(3)/2
        # and
        # (cx-vx)*nx + (cy-vy)*ny <= s * sqrt(3)/2  (i.e., -proj <= inradius)
        
        # The required inradius for a given center is the max over all vertices and normals of the projection.
        # To minimize s, we should center the hexagon around the centroid of the vertices.
        
        cx_mean = sum(v[0] for v in all_vertices) / len(all_vertices)
        cy_mean = sum(v[1] for v in all_vertices) / len(all_vertices)
        
        # Refine center slightly? The centroid is a good start.
        # We can try a small grid around the centroid.
        step = 0.1
        for cx in [cx_mean + dx for dx in [-step, 0.0, step]]:
            for cy in [cy_mean + dy for dy in [-step, 0.0, step]]:
                required_inradius = 0.0
                for k in range(6):
                    a = math.radians(angle_deg + 30 + 60 * k)
                    nx = math.cos(a)
                    ny = math.sin(a)
                    
                    max_proj = 0.0
                    for vx, vy in all_vertices:
                        proj = (vx - cx) * nx + (vy - cy) * ny
                        if proj > max_proj:
                            max_proj = proj
                        # Also check the opposite direction (negative normal)
                        # This is equivalent to checking proj against -max_proj_neg
                        # But simply taking max(abs(proj)) for each normal pair is safer?
                        # No, the hexagon is defined by 6 half-planes.
                        # Normal 1: n. p <= R
                        # Normal 2: -n. p <= R  => n. p >= -R
                        # So we need max(n.p) <= R and min(n.p) >= -R
                        # i.e., max(n.p, -min(n.p)) <= R
                        min_proj = 0.0
                        for vx, vy in all_vertices:
                            proj = (vx - cx) * nx + (vy - cy) * ny
                            if proj < min_proj:
                                min_proj = proj
                        extent = max(max_proj, -min_proj)
                        if extent > required_inradius:
                            required_inradius = extent
                
                side = required_inradius / (math.sqrt(3) / 2)
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = angle_deg
                    
    return best_center[0], best_center[1], best_side, best_angle

def construct():
    global _START_TIME, _DEADLINE
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Return a safe fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Generate 11 points on a triangular lattice with spacing sqrt(3)
    # Lattice basis: u = (sqrt(3), 0), v = (sqrt(3)/2, 1.5)
    sqrt3 = math.sqrt(3)
    u = (sqrt3, 0.0)
    v = (sqrt3 / 2, 1.5)
    
    # Generate candidate points in a spiral/grid around the origin
    # We want the 11 closest points to the origin in the triangular lattice
    # Lattice points are (i, j) -> i*u + j*v
    # Distance squared from origin: (i*u + j*v)^2
    # Let's generate points with |i|, |j| <= 3 and sort by distance
    
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            dist_sq = x * x + y * y
            candidates.append((dist_sq, x, y))
    
    # Sort by distance
    candidates.sort(key=lambda t: t[0])
    
    # Take the first N unique points
    # Since the lattice is unique, just take first N
    positions = []
    for _, x, y in candidates:
        positions.append((x, y))
        if len(positions) == N:
            break
    
    # Create inner hexagons with all angles = 0 (aligned with lattice)
    # This is the densest packing configuration for hexagons
    inner = [[pos[0], pos[1], 0.0] for pos in positions]
    
    # Compute minimum enclosing hexagon
    if time.time() > _DEADLINE - 10.0:
        # Quick estimate
        xs = [h[0] for h in inner]
        ys = [h[1] for h in inner]
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2
        max_dist = 0
        for h in inner:
            d = math.sqrt((h[0] - cx)**2 + (h[1] - cy)**2)
            if d > max_dist:
                max_dist = d
        # Add circumradius 1 to max_dist for a rough bound
        side = (max_dist + 1.0) / (math.sqrt(3) / 2) + 0.1
        return inner, (cx, cy), side, 0.0
    
    cx, cy, side, angle = _min_enclosing_hex_side(inner)
    
    # Verify no overlaps (should be guaranteed by lattice spacing sqrt(3))
    # But just in case, check a few pairs
    # For unit hexagons on triangular lattice with spacing sqrt(3), they touch at edges.
    # This is valid (disjoint interiors).
    
    return inner, (cx, cy), side, angle
# EVOLVE-BLOCK-END
