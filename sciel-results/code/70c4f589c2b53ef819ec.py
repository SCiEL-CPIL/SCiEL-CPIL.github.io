"""Optimized hexagon packing using honeycomb lattice structure."""
import time
import math
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # We need to pack N unit regular hexagons (side length 1) inside a regular hexagon.
    # The inner hexagons can be translated and rotated.
    # The outer hexagon has center (cx, cy), side length S, and rotation angle theta.
    
    # Strategy: Place inner hexagons on a honeycomb lattice.
    # A regular hexagon of side 1 has:
    #   - Circumradius (center to vertex) = 1
    #   - Inradius (center to edge) = sqrt(3)/2
    #   - Distance between centers of adjacent hexagons in a honeycomb tiling = 2 * inradius = sqrt(3)
    
    # In a honeycomb lattice, the centers of hexagons are at positions:
    #   (i * 3/2, j * sqrt(3) + (i % 2) * sqrt(3)/2) for axial coordinates (i, j)
    # But since we can rotate each inner hexagon, we can align them all the same way.
    # 
    # Actually, for a honeycomb tiling of regular hexagons with side length 1:
    # The distance between centers of adjacent hexagons is sqrt(3) (since they share an edge of length 1,
    # and the distance between centers is 2 * inradius = 2 * sqrt(3)/2 = sqrt(3)).
    
    # Let's think about this differently. If we place N unit hexagons in a compact arrangement,
    # we want to minimize the enclosing regular hexagon.
    
    # For small N, let's try specific arrangements:
    # N=1: Single hexagon, outer side = 1 (if we align them perfectly, the outer hexagon can have side 1)
    #   Wait, the outer hexagon must CONTAIN the inner hexagon. If the inner hexagon has side 1,
    #   and we place it at the center with the same orientation, the outer hexagon needs side >= 1.
    #   Actually, if the inner hexagon is centered and aligned, the outer hexagon of side 1 would exactly
    #   contain it (the vertices coincide). So for N=1, side=1.
    
    # For N>1, we need to arrange multiple hexagons.
    
    # Let's use a honeycomb lattice approach:
    # Place hexagons at centers forming a compact cluster on the honeycomb lattice.
    # Then find the minimal enclosing regular hexagon.
    
    # Honeycomb lattice vectors:
    # a1 = (3/2, sqrt(3)/2)
    # a2 = (0, sqrt(3))
    # Wait, that's not quite right. Let me reconsider.
    
    # For regular hexagons of side 1 tiling the plane:
    # The centers form a triangular lattice with spacing sqrt(3).
    # Lattice vectors:
    #   v1 = (sqrt(3), 0)
    #   v2 = (sqrt(3)/2, 3/2)
    # Check: |v1| = sqrt(3), |v2| = sqrt(3/4 + 9/4) = sqrt(12/4) = sqrt(3). Good.
    # The angle between them: dot = sqrt(3)*sqrt(3)/2 + 0 = 3/2.
    # cos(theta) = (3/2) / (sqrt(3)*sqrt(3)) = (3/2)/3 = 1/2. theta = 60 degrees. Good.
    
    # So centers are at m*v1 + n*v2 for integers m, n.
    
    # Now, for a given set of inner hexagon centers, we need to find the minimal regular hexagon
    # (with some rotation) that contains all of them.
    
    # A regular hexagon with center (cx, cy), side length S, and rotation angle phi can be described
    # as the intersection of 6 half-planes. The distance from center to each edge is S * sqrt(3)/2.
    
    # For a point (x, y) to be inside the hexagon, for each of the 6 edge directions,
    # the projection of (x-cx, y-cy) onto the normal of that edge must be <= S * sqrt(3)/2.
    
    # The 6 edge normals (for rotation angle phi) are at angles:
    #   phi, phi + 60, phi + 120, phi + 180, phi + 240, phi + 300 degrees
    # Wait, the normals to the edges of a regular hexagon are spaced 60 degrees apart.
    # If the hexagon has a vertex at angle phi, then the edge normals are at phi+30, phi+90, etc.
    # Let me be more careful.
    
    # A regular hexagon with circumradius R (= side length S for a regular hexagon) and one vertex
    # at angle alpha has vertices at angles alpha + k*60 for k=0..5.
    # The edges are perpendicular to the bisectors of adjacent vertices, so the edge normals
    # are at angles alpha + 30 + k*60 for k=0..5.
    # The inradius is R * cos(30) = R * sqrt(3)/2.
    
    # So for a hexagon with center (cx,cy), circumradius R, and "vertex angle" alpha:
    # A point (x,y) is inside iff for all k=0..5:
    #   (x-cx)*cos(alpha + 30 + k*60) + (y-cy)*sin(alpha + 30 + k*60) <= R * sqrt(3)/2
    
    # To minimize R for a given set of points and center (cx,cy), we can:
    # 1. Fix the center (cx,cy)
    # 2. For a given rotation alpha, the required R is:
    #    R = max over all points p and all k of:
    #        [(p.x-cx)*cos(alpha+30+k*60) + (p.y-cy)*sin(alpha+30+k*60)] / (sqrt(3)/2)
    #    But we need all projections to be <= R*sqrt(3)/2, so R must be at least the max projection / (sqrt(3)/2).
    #    Also R must be at least the max of -projection / (sqrt(3)/2) if we consider both sides, but since
    #    we have all 6 normals (which include opposite pairs), the max over all 6 normals covers both sides.
    
    # Actually, since the 6 normals include opposite pairs (k and k+3 are opposite), the constraint
    # for normal n and -n together mean |projection| <= R*sqrt(3)/2. So:
    # R >= max over all points p and all 3 axis directions of |projection| / (sqrt(3)/2)
    
    # For a fixed center, we can optimize over alpha. But this is complex.
    
    # Simpler approach: For a given set of points, find the minimal enclosing regular hexagon.
    # This is a known geometric problem but let's use a simpler heuristic.
    
    # For small N (N=11), let's try specific configurations and compute the required outer hexagon side.
    
    # Configuration 1: Linear arrangement
    # Place hexagons in a line. Centers at (k*sqrt(3), 0) for k = -5..5 (11 hexagons).
    # The extent is from -5*sqrt(3) to 5*sqrt(3) in x, and the hexagons have extent sqrt(3)/2 in y.
    # Total x extent: 10*sqrt(3) + 2*1 = 10*sqrt(3) + 2 (adding the circumradius on each end)
    # Wait, the hexagon vertices extend to circumradius 1 from center.
    # So the leftmost point is at -5*sqrt(3) - 1, rightmost at 5*sqrt(3) + 1.
    # Top/bottom: sqrt(3)/2 + 1 (inradius + circumradius... no, the topmost point of a hexagon is at inradius if flat-top, or circumradius if pointy-top).
    
    # This is getting complicated. Let me just compute it numerically for a few configurations.
    
    def compute_min_hexagon_side(points, center):
        """Compute the minimum side length of a regular hexagon with the given center
        that contains all points. We optimize over the rotation angle."""
        cx, cy = center
        dx = [p[0] - cx for p in points]
        dy = [p[1] - cy for p in points]
        
        best_R = float('inf')
        # Try many rotation angles
        for phi_deg in range(0, 180, 1):  # 180 degrees is enough due to symmetry
            phi = math.radians(phi_deg)
            max_proj = 0
            for i in range(3):  # Only need 3 axes (opposite pairs)
                angle = phi + math.radians(30 + i * 60)
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                for j in range(len(points)):
                    proj = abs(dx[j] * cos_a + dy[j] * sin_a)
                    if proj > max_proj:
                        max_proj = proj
            R = max_proj / (math.sqrt(3) / 2)
            if R < best_R:
                best_R = R
        return best_R
    
    # Try honeycomb cluster configurations
    # v1 = (sqrt(3), 0), v2 = (sqrt(3)/2, 3/2)
    v1 = (math.sqrt(3), 0)
    v2 = (math.sqrt(3)/2, 3/2)
    
    # Generate candidate clusters on the honeycomb lattice
    # For N=11, a compact cluster would be like a hexagonal number pattern.
    # Hexagonal numbers: 1, 7, 19, ... 
    # 7 is a hexagonal cluster (center + 6 neighbors).
    # 11 = 7 + 4 more.
    
    # Let's try: center + 6 neighbors + 4 additional
    # Or try different shapes.
    
    # Generate all lattice points within a certain radius
    candidates = []
    for m in range(-3, 4):
        for n in range(-3, 4):
            x = m * v1[0] + n * v2[0]
            y = m * v1[1] + n * v2[1]
            candidates.append((x, y, m, n))
    
    # Sort by distance from origin
    candidates.sort(key=lambda c: c[0]**2 + c[1]**2)
    
    # Try different subsets of size N
    best_side = float('inf')
    best_config = None
    
    # For efficiency, try compact clusters
    # First, try the N closest points to origin
    for start in range(min(5, len(candidates) - N + 1)):
        if time.time() > deadline:
            break
        # Take N points starting from position 'start' in the sorted list
        # But this might not be optimal. Let's try a few different strategies.
        pass
    
    # Strategy: Try the N closest points to the origin
    closest = candidates[:N]
    points = [(c[0], c[1]) for c in closest]
    center = (0.0, 0.0)
    side = compute_min_hexagon_side(points, center)
    
    # Also try centering the cluster at the centroid
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    side_centroid = compute_min_hexagon_side(points, (cx, cy))
    if side_centroid < side:
        side = side_centroid
        center = (cx, cy)
    
    best_side = side
    best_config = (points, center, side)
    
    # Try a few other configurations
    # Configuration: linear
    if time.time() < deadline:
        line_points = [(k * math.sqrt(3), 0) for k in range(-(N//2), N - N//2)]
        cx = 0.0
        cy = 0.0
        side_line = compute_min_hexagon_side(line_points, (cx, cy))
        if side_line < best_side:
            best_side = side_line
            best_config = (line_points, (cx, cy), side_line)
    
    # Configuration: 2D grid-like
    if time.time() < deadline:
        # Try a 3x4 or similar arrangement
        # For N=11, try rows of 4,4,3
        pts = []
        row_y = [0, math.sqrt(3), 2*math.sqrt(3)]
        row_counts = [4, 4, 3]
        for i, (ry, rc) in enumerate(zip(row_y, row_counts)):
            for j in range(rc):
                x_offset = (j - (rc-1)/2) * math.sqrt(3)
                pts.append((x_offset, ry))
        if len(pts) == N:
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            side_grid = compute_min_hexagon_side(pts, (cx, cy))
            if side_grid < best_side:
                best_side = side_grid
                best_config = (pts, (cx, cy), side_grid)
    
    # Extract the best configuration
    points, center, side = best_config
    cx, cy = center
    
    # Create inner hexagons: each at the point, with angle 0
    inner = [[p[0], p[1], 0.0] for p in points]
    
    # The outer hexagon: center (cx, cy), side length = side, rotation angle
    # We need to find the optimal rotation for the outer hexagon
    # From compute_min_hexagon_side, we found the best phi. Let's recompute it.
    
    dx = [p[0] - cx for p in points]
    dy = [p[1] - cy for p in points]
    best_phi = 0
    best_R = float('inf')
    for phi_deg in range(0, 180, 1):
        phi = math.radians(phi_deg)
        max_proj = 0
        for i in range(3):
            angle = phi + math.radians(30 + i * 60)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            for j in range(len(points)):
                proj = abs(dx[j] * cos_a + dy[j] * sin_a)
                if proj > max_proj:
                    max_proj = proj
        R = max_proj / (math.sqrt(3) / 2)
        if R < best_R:
            best_R = R
            best_phi = phi_deg
    
    return inner, (cx, cy), best_R, best_phi

# EVOLVE-BLOCK-END
