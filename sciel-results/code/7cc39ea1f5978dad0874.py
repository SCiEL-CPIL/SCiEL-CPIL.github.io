"""Packing unit hexagons in a regular hexagon using honeycomb lattice."""
import numpy as np
import time
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s limit
    
    # Unit regular hexagon vertices (flat-top orientation, side=1)
    # Center at origin, vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # For a flat-top hexagon with side 1:
    #   width = 2, height = sqrt(3)
    #   vertices: (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2)
    hex_vertices = np.array([
        [1.0, 0.0],
        [0.5, math.sqrt(3)/2],
        [-0.5, math.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -math.sqrt(3)/2],
        [0.5, -math.sqrt(3)/2]
    ])
    
    # Honeycomb lattice: unit hexagons tile the plane
    # The centers of adjacent hexagons are at distance 2*sqrt(3)/2 = sqrt(3)
    # In a honeycomb lattice, the basis vectors for hexagon centers are:
    #   a1 = (3, sqrt(3))  -> wait, let me recalculate
    # Actually for flat-top hexagons with side 1:
    #   The distance between centers of adjacent hexagons is 2 * (sqrt(3)/2) * ... 
    # Let me think more carefully.
    # A regular hexagon with side s has:
    #   - Inradius (apothem) = s * sqrt(3)/2
    #   - Circumradius = s
    #   - Width (flat-to-flat) = 2 * inradius = s * sqrt(3)
    #   - Height (point-to-point) = 2 * circumradius = 2s
    # 
    # For flat-top hexagons (vertices at 0, 60, ...):
    #   Width (horizontal) = 2s
    #   Height (vertical) = s * sqrt(3)
    # 
    # Adjacent hexagons in the tiling:
    #   - Horizontal neighbors: center distance = 2 * inradius = s*sqrt(3)? No.
    #   Let me just use the standard honeycomb lattice vectors.
    #
    # For a tiling of regular hexagons with side 1:
    #   The lattice of centers has basis vectors:
    #     v1 = (3/2, sqrt(3)/2)  -- to the upper-right neighbor
    #     v2 = (0, sqrt(3))       -- to the upper neighbor
    #   Wait, that's not right either. Let me derive it.
    #
    # Place one hexagon centered at origin. Its right neighbor (sharing the right edge)
    # has its center at distance 2*inradius = sqrt(3) to the right? No.
    # The shared edge is vertical for flat-top hexagons? No, for flat-top, the left and right
    # edges are vertical. The distance between centers sharing a vertical edge is 2 * inradius = sqrt(3).
    # But actually, flat-top hexagons have vertices at angles 0,60,...,300. The edges are between
    # consecutive vertices. The edge between vertex at 0 deg (1,0) and 60 deg (0.5, sqrt(3)/2) 
    # is not vertical. Let me reconsider.
    #
    # Actually, let's use pointy-top hexagons instead, or just compute the lattice properly.
    #
    # For a regular hexagon with side 1, the six edges connect consecutive vertices.
    # The normal to each edge points in the direction of the vertex midpoint.
    # For flat-top (vertices at 0, 60, 120, 180, 240, 300):
    #   Edge normals are at 30, 90, 150, 210, 270, 330 degrees.
    #   The distance from center to each edge (inradius) = sqrt(3)/2.
    #   Two hexagons sharing an edge have centers separated by 2 * inradius = sqrt(3) 
    #   in the direction of the edge normal.
    #
    # So the 6 neighbor offsets are:
    #   sqrt(3) * [cos(30), sin(30)] = sqrt(3)*[sqrt(3)/2, 1/2] = [3/2, sqrt(3)/2]
    #   sqrt(3) * [cos(90), sin(90)] = [0, sqrt(3)]
    #   sqrt(3) * [cos(150), sin(150)] = [-3/2, sqrt(3)/2]
    #   sqrt(3) * [cos(210), sin(210)] = [-3/2, -sqrt(3)/2]
    #   sqrt(3) * [cos(270), sin(270)] = [0, -sqrt(3)]
    #   sqrt(3) * [cos(330), sin(330)] = [3/2, -sqrt(3)/2]
    #
    # The lattice basis can be taken as:
    #   a1 = [3/2, sqrt(3)/2]
    #   a2 = [0, sqrt(3)]
    # Then all centers are at i*a1 + j*a2 for integers i,j.
    #
    # Let's verify: a1 + a2 = [3/2, 3*sqrt(3)/2] which should be a neighbor... 
    # Actually the 6 neighbors are ±a1, ±a2, ±(a1-a2).
    # a1 - a2 = [3/2, sqrt(3)/2 - sqrt(3)] = [3/2, -sqrt(3)/2]. Yes! That's one of the 6.
    # -a1 = [-3/2, -sqrt(3)/2], -a2 = [0, -sqrt(3)], -(a1-a2) = [-3/2, sqrt(3)/2]. 
    # So the 6 neighbors are: a1, a2, a1-a2, -a1, -a2, -(a1-a2). Correct.
    
    a1 = np.array([1.5, math.sqrt(3)/2])
    a2 = np.array([0.0, math.sqrt(3)])
    
    # Generate hexagon positions on a hexagonal region of the lattice
    # We want N = 11 hexagons. Place them in a compact hexagonal cluster.
    # A hexagonal cluster of "radius" r (in lattice steps) has 1 + 3*r*(r+1) hexagons.
    # For r=1: 1+6=7, r=2: 1+18=19. So 11 doesn't fit exactly in a perfect hex cluster.
    # We'll place them in a compact arrangement.
    
    # Let's place N hexagons in a roughly hexagonal shape.
    # We'll use a spiral or row-based approach.
    
    # Row-based: place hexagons in rows.
    # For a hexagonal packing, rows alternate in offset.
    # Let's just place them in a compact block.
    
    positions = []
    # Place in a hexagonal pattern centered at origin
    # For N=11, we can do:
    # Center: (0,0)
    # 6 neighbors
    # 4 more from the next ring
    
    # Generate lattice points in order of distance from origin
    max_radius = 5  # in lattice units
    candidates = []
    for i in range(-max_radius, max_radius+1):
        for j in range(-max_radius, max_radius+1):
            pos = i * a1 + j * a2
            dist = np.linalg.norm(pos)
            candidates.append((dist, i, j, pos))
    
    candidates.sort(key=lambda x: x[0])
    
    # Take the first N positions
    for idx in range(N):
        if idx < len(candidates):
            pos = candidates[idx][3]
            positions.append(pos)
    
    # Now find the minimal enclosing regular hexagon
    # A regular hexagon with side S, centered at (cx, cy), rotated by angle theta
    # contains a point (px, py) if the distance from (px,cy) to each of the 6 sides <= S*sqrt(3)/2
    # in the rotated frame.
    
    # For a regular hexagon centered at origin with side S and rotation 0 (flat-top):
    # The 6 edge normals are at angles 30, 90, 150, 210, 270, 330 degrees.
    # The inradius is S*sqrt(3)/2.
    # A point (x,y) is inside if for all k=0..5:
    #   x*cos(nk) + y*sin(nk) <= S*sqrt(3)/2
    # where nk = (30 + 60*k) degrees.
    
    # To find the minimal S for a given center and rotation, we need:
    #   S >= max over all points and all 6 directions of (px-cx)*cos(nk) + (py-cy)*sin(nk) / (sqrt(3)/2)
    
    def compute_min_side(positions, center, rotation_deg):
        """Compute minimal side length for regular hexagon containing all positions."""
        cx, cy = center
        rot = math.radians(rotation_deg)
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        
        max_val = 0.0
        for px, py in positions:
            # Rotate point into hexagon's local frame
            dx = px - cx
            dy = py - cy
            # Rotate by -rotation
            lx = dx * cos_r + dy * sin_r
            ly = -dx * sin_r + dy * cos_r
            
            # Check all 6 directions
            for k in range(6):
                angle = math.radians(30 + 60 * k)
                ca, sa = math.cos(angle), math.sin(angle)
                val = lx * ca + ly * sa
                if val > max_val:
                    max_val = val
        
        return max_val / (math.sqrt(3) / 2)
    
    # Try different centers and rotations
    # The optimal center is likely near the centroid of the positions
    centroid = np.mean(positions, axis=0)
    
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_rotation = 0.0
    
    # Try rotation = 0 and 30 degrees (the two natural orientations)
    # and center = centroid
    for rot in [0.0, 30.0]:
        side = compute_min_side(positions, centroid, rot)
        if side < best_side:
            best_side = side
            best_center = (float(centroid[0]), float(centroid[1]))
            best_rotation = rot
    
    # Also try a few nearby centers and rotations for refinement
    for rot in [0.0, 15.0, 30.0, 45.0, 60.0]:
        for dcx in [-0.5, 0.0, 0.5]:
            for dcy in [-0.5, 0.0, 0.5]:
                center = (centroid[0] + dcx, centroid[1] + dcy)
                side = compute_min_side(positions, center, rot)
                if side < best_side:
                    best_side = side
                    best_center = center
                    best_rotation = rot
    
    # Refine with finer search
    for _ in range(3):
        improved = False
        for rot in np.linspace(best_rotation - 5, best_rotation + 5, 11):
            for dcx in np.linspace(-0.2, 0.2, 5):
                for dcy in np.linspace(-0.2, 0.2, 5):
                    center = (best_center[0] + dcx, best_center[1] + dcy)
                    side = compute_min_side(positions, center, float(rot))
                    if side < best_side:
                        best_side = side
                        best_center = center
                        best_rotation = float(rot)
                        improved = True
        if not improved:
            break
    
    # Check time
    if time.time() > deadline:
        pass  # Use what we have
    
    # Build inner list
    inner = []
    for pos in positions:
        inner.append([float(pos[0]), float(pos[1]), 0.0])
    
    return inner, best_center, float(best_side), float(best_rotation)
# EVOLVE-BLOCK-END
