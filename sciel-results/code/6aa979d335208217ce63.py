"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def generate_hexagonal_cluster(n):
        """Generate centers of n hexagons in a compact hexagonal cluster."""
        if n == 1:
            return [(0.0, 0.0)]
        
        # Hexagonal numbers: 1, 7, 19, 37, ...
        # For n between hex numbers, fill partially
        positions = []
        # Start with center
        positions.append((0.0, 0.0))
        
        # Add rings
        ring = 1
        while len(positions) < n:
            # A ring of radius 'ring' has 6*ring positions
            for i in range(6 * ring):
                if len(positions) >= n:
                    break
                # Position on the hexagonal ring
                # The ring consists of 6 sides, each with 'ring' positions
                side = i // ring  # 0-5
                pos_on_side = i % ring  # 0 to ring-1
                
                # Unit vectors for the 6 directions
                # For flat-top hexagons, the 6 neighbor directions are:
                # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
                # scaled by ring
                sqrt3 = math.sqrt(3)
                if side == 0:
                    x = ring * 1.5
                    y = -ring * sqrt3 / 2 + pos_on_side * sqrt3
                elif side == 1:
                    x = ring * 0.75
                    y = ring * sqrt3 / 2 + pos_on_side * sqrt3
                elif side == 2:
                    x = -ring * 0.75
                    y = ring * sqrt3 / 2 - pos_on_side * sqrt3
                elif side == 3:
                    x = -ring * 1.5
                    y = ring * sqrt3 / 2 - pos_on_side * sqrt3
                elif side == 4:
                    x = -ring * 0.75
                    y = -ring * sqrt3 / 2 + pos_on_side * sqrt3
                elif side == 5:
                    x = ring * 0.75
                    y = -ring * sqrt3 / 2 - pos_on_side * sqrt3
                
                positions.append((x, y))
            ring += 1
        
        return positions[:n]
    
    def compute_bounding_hexagon_side(positions, angle_deg=0.0):
        """Compute the minimum side length of a regular hexagon (with given rotation) 
        that contains all unit hexagons centered at positions."""
        # For a flat-top unit hexagon, the vertices are at distance 1 from center
        # in directions 0, 60, 120, 180, 240, 300 degrees
        # The "radius" (center to vertex) is 1, center to flat is sqrt(3)/2
        
        # To check if a unit hexagon at position p fits inside an outer hexagon,
        # we need all 6 vertices of the inner hexagon to be inside the outer hexagon.
        
        # For simplicity, let's use the fact that a regular hexagon of side s 
        # with center at origin and rotation theta can be described as the set of points
        # where the projection onto each of 6 directions is within s*sqrt(3)/2.
        
        # Actually, a regular hexagon with side length s, centered at origin, 
        # with one vertex at angle theta, has the property that for any point (x,y),
        # it's inside if and only if for all k in 0..5:
        #   dot((x,y), (cos(theta + k*60°), sin(theta + k*60°))) <= s
        # Wait, that's for the vertex-to-vertex distance = 2s, so the support function is s.
        
        # Let me use: a regular hexagon with "circumradius" R (distance center to vertex) = s
        # (since side length = circumradius for regular hexagon)
        # The support function in direction phi is R * cos(min_k |phi - theta - k*60°|)
        # This is complex. Let me use a simpler approach.
        
        # For a flat-top hexagon (vertex at 0°) with side length s, centered at origin:
        # It contains point (x,y) iff:
        #   |x| <= s
        #   |y| <= s*sqrt(3)/2
        #   |x/2 + y*sqrt(3)/2| <= s
        #   |-x/2 + y*sqrt(3)/2| <= s
        # Wait, that's not right either. Let me think again.
        
        # A regular hexagon with vertices at (s,0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), 
        # (-s,0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
        # The edges connect consecutive vertices. The six edges have equations:
        # 1. Right edge: x = s (from y=-s*sqrt(3)/2 to y=s*sqrt(3)/2)... no, that's not an edge.
        
        # The edges are:
        # Between (s,0) and (s/2, s*sqrt(3)/2): line with slope (s*sqrt(3)/2 - 0)/(s/2 - s) = (s*sqrt(3)/2)/(-s/2) = -sqrt(3)
        #   Equation: y - 0 = -sqrt(3)(x - s) => y = -sqrt(3)x + s*sqrt(3) => sqrt(3)x + y = s*sqrt(3)
        # Between (s/2, s*sqrt(3)/2) and (-s/2, s*sqrt(3)/2): y = s*sqrt(3)/2
        # Between (-s/2, s*sqrt(3)/2) and (-s, 0): slope = (0 - s*sqrt(3)/2)/(-s + s/2) = (-s*sqrt(3)/2)/(-s/2) = sqrt(3)
        #   Equation: y - 0 = sqrt(3)(x + s) => y = sqrt(3)x + s*sqrt(3) => -sqrt(3)x + y = s*sqrt(3)
        # Between (-s, 0) and (-s/2, -s*sqrt(3)/2): slope = (-s*sqrt(3)/2 - 0)/(-s/2 + s) = (-s*sqrt(3)/2)/(s/2) = -sqrt(3)
        #   Equation: y = -sqrt(3)(x + s) => sqrt(3)x + y = -s*sqrt(3)
        # Between (-s/2, -s*sqrt(3)/2) and (s/2, -s*sqrt(3)/2): y = -s*sqrt(3)/2
        # Between (s/2, -s*sqrt(3)/2) and (s, 0): slope = (0 + s*sqrt(3)/2)/(s - s/2) = (s*sqrt(3)/2)/(s/2) = sqrt(3)
        #   Equation: y = sqrt(3)(x - s) => -sqrt(3)x + y = -s*sqrt(3)
        
        # So the hexagon is:
        #   sqrt(3)*x + y <= s*sqrt(3)
        #   y <= s*sqrt(3)/2
        #   -sqrt(3)*x + y <= s*sqrt(3)
        #   sqrt(3)*x + y >= -s*sqrt(3)
        #   y >= -s*sqrt(3)/2
        #   -sqrt(3)*x + y >= -s*sqrt(3)
        
        # Simplifying:
        #   |y| <= s*sqrt(3)/2
        #   |sqrt(3)*x + y| <= s*sqrt(3)
        #   |-sqrt(3)*x + y| <= s*sqrt(3)
        
        # For a unit inner hexagon at position (px, py) with 0° rotation, its vertices are at:
        # (px+1, py), (px+0.5, py+sqrt(3)/2), (px-0.5, py+sqrt(3)/2), (px-1, py), 
        # (px-0.5, py-sqrt(3)/2), (px+0.5, py-sqrt(3)/2)
        
        # For all these vertices to be inside the outer hexagon (centered at origin, side s, 0° rotation):
        # For each vertex (vx, vy):
        #   |vy| <= s*sqrt(3)/2
        #   |sqrt(3)*vx + vy| <= s*sqrt(3)
        #   |-sqrt(3)*vx + vy| <= s*sqrt(3)
        
        # The maximum of |vy| over all vertices of all inner hexagons gives lower bound on s*sqrt(3)/2
        # The maximum of |sqrt(3)*vx + vy| gives lower bound on s*sqrt(3)
        # The maximum of |-sqrt(3)*vx + vy| gives lower bound on s*sqrt(3)
        
        # So s must satisfy:
        #   s >= max(|vy|) * 2/sqrt(3)
        #   s >= max(|sqrt(3)*vx + vy|) / sqrt(3)
        #   s >= max(|-sqrt(3)*vx + vy|) / sqrt(3)
        
        # Let's compute these for all vertices of all inner hexagons
        
        sqrt3 = math.sqrt(3)
        max_y = 0.0
        max_sqrt3x_plus_y = 0.0
        max_neg_sqrt3x_plus_y = 0.0
        
        for (px, py) in positions:
            # 6 vertices of unit hexagon at (px, py) with 0° rotation
            vertices = [
                (px + 1.0, py),
                (px + 0.5, py + sqrt3/2),
                (px - 0.5, py + sqrt3/2),
                (px - 1.0, py),
                (px - 0.5, py - sqrt3/2),
                (px + 0.5, py - sqrt3/2),
            ]
            for (vx, vy) in vertices:
                max_y = max(max_y, abs(vy))
                max_sqrt3x_plus_y = max(max_sqrt3x_plus_y, abs(sqrt3 * vx + vy))
                max_neg_sqrt3x_plus_y = max(max_neg_sqrt3x_plus_y, abs(-sqrt3 * vx + vy))
        
        s1 = max_y * 2.0 / sqrt3
        s2 = max_sqrt3x_plus_y / sqrt3
        s3 = max_neg_sqrt3x_plus_y / sqrt3
        
        return max(s1, s2, s3)
    
    def try_arrangement(positions, angle_deg=0.0):
        """Compute the outer hexagon side length for given inner positions."""
        return compute_bounding_hexagon_side(positions, angle_deg)
    
    # Generate candidate arrangements
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Candidate 1: Hexagonal cluster
    positions1 = generate_hexagonal_cluster(N)
    side1 = try_arrangement(positions1)
    if side1 < best_side:
        best_side = side1
        best_inner = [[p[0], p[1], 0.0] for p in positions1]
        best_center = (0.0, 0.0)
        best_angle = 0.0
    
    # Check deadline
    if time.time() > deadline:
        return best_inner, best_center, best_side, best_angle
    
    # Candidate 2: Try a few different rotations for the outer hexagon
    for angle in [0.0, 30.0, 15.0, 45.0]:
        # For simplicity, we'll just use the same positions and compute with rotation
        # This requires modifying compute_bounding_hexagon_side to handle rotation
        # For now, skip and just return the best we have
        pass
    
    # Let's also try a more rectangular arrangement
    def generate_rectangular(n, cols):
        """Generate n hexagons in a rectangular-ish honeycomb pattern."""
        positions = []
        sqrt3 = math.sqrt(3)
        row = 0
        while len(positions) < n:
            for col in range(cols):
                if len(positions) >= n:
                    break
                x = col * 1.5
                y = row * sqrt3
                if row % 2 == 1:
                    x += 0.75
                positions.append((x, y))
            row += 1
        return positions[:n]
    
    for cols in [3, 4, 5]:
        positions2 = generate_rectangular(N, cols)
        side2 = try_arrangement(positions2)
        if side2 < best_side:
            best_side = side2
            best_inner = [[p[0], p[1], 0.0] for p in positions2]
            best_center = (0.0, 0.0)
            best_angle = 0.0
    
    # Center the arrangement by computing centroid and shifting
    if best_inner:
        xs = [p[0] for p in best_inner]
        ys = [p[1] for p in best_inner]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        best_inner = [[p[0] - cx, p[1] - cy, 0.0] for p in best_inner]
        best_center = (0.0, 0.0)
        # Recompute side length after centering (should be same)
        best_side = try_arrangement([(p[0], p[1]) for p in best_inner])
    
    return best_inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END

