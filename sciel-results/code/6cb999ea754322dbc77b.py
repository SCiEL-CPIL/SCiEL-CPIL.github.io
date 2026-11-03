import math
import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate candidates: Compact clusters
    candidates = []

    # 1. Central hexagon (7) + 4 neighbors on the outer ring
    # Outer ring axial coords: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Neighbors of (1,0): (2,0), (2,-1), (1,1) -- wait, (1,1) is already in ring? No, (1,1) is dist 2?
    # Axial distance squared: q^2 + r^2 + qr.
    # (1,1): 1+1+1=3. dist sqrt(3). Same as (1,0).
    # So (1,1) is part of the "radius 1" ring?
    # Standard hex grid:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) are dist 1.
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1) are dist 2.
    
    # Candidate 1: 7 center + 4 from dist 2 ring.
    # To minimize bounding box, pick 4 that are close together?
    # Or spread out? Spreading out increases max projection.
    # We want to minimize max projection.
    # So pick 4 neighbors that are "inward" or clustered?
    # Actually, any 4 from the dist 2 ring will have max projection determined by the furthest one.
    # If we pick (2,0), (1,1), (0,2), (-1,2), they are in a quadrant.
    # Max projection in x-direction?
    # Let's just try a few specific sets.

    # Set 1: 3x3 block approximation
    # Axial:
    # Row 0: 0, 1, 2
    # Row 1: 0, 1, 2
    # Row 2: 0, 1, 2
    # This is 9 hexagons.
    # Add 2 more.
    # Where to add? To keep it compact.
    # Add (-1, 1) and (3, 1)? No, that stretches it.
    # Add (0, 3) and (1, 3)?
    
    # Let's use a known good packing:
    # A hexagon of side 2 has 19 hexagons.
    # We need 11.
    # Remove 8 from the side 2 hexagon.
    # Side 2 hexagon axial coords:
    # All (q,r) such that max(|q|, |r|, |q+r|) <= 2.
    
    def generate_hexagon(n):
        pts = []
        for q in range(-n, n+1):
            for r in range(max(-n, -q-n), min(n, -q+n)+1):
                pts.append((q, r))
        return pts

    full_hex_2 = generate_hexagon(2) # 19 points
    full_hex_1 = generate_hexagon(1) # 7 points
    
    # We need to remove 8 points from full_hex_2 to get 11.
    # To minimize the bounding box, we should remove points from the "corners" or "edges" that stick out?
    # Actually, removing points from the convex hull reduces the size.
    # The convex hull of the 19 points is the hexagon itself.
    # If we remove corner points, the hull shrinks.
    
    # Let's try removing the 6 corner points of the side 2 hexagon?
    # Corners: (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2).
    # Removing these 6 leaves 13 points.
    # Remove 2 more.
    # The remaining points are the "edge" points of the side 2 hexagon (excluding corners) and the inner 7.
    # Edge points (dist 2, not corner):
    # (1,1), (-1,2), (-1,0) is dist 1.
    # Let's list dist 2 points:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1).
    # Wait, (-2,1) is a corner? max(2,1,1)=2. Yes.
    # Corners are where two coordinates are at max/min?
    # (2,0): q=2, r=0, q+r=2.
    # (1,1): q=1, r=1, q+r=2. This is on the edge q+r=2.
    
    # Strategy: Start with full hex 2. Remove points to minimize max projection.
    # We can just try a few manual subsets.
    
    # Subset A: Remove 6 corners + 2 adjacent edge points.
    corners = [(2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2)]
    # Note: (-2,2) is q=-2, r=2, q+r=0. Max is 2.
    # My generate_hexagon logic:
    # q in -2..2.
    # r in max(-2, -q-2) .. min(2, -q+2).
    
    # Let's just define the 11 points explicitly for a compact shape.
    # Shape: A "thick" cross or a 3x3 grid.
    
    # 3x3 Grid in Axial:
    # This is tricky because axial is not rectangular.
    # Let's use Cube coordinates (x,y,z) where x+y+z=0.
    # Hexagon side 2: max(|x|,|y|,|z|) <= 2.
    
    def cube_to_axial(x, y, z):
        return (x, z) # q=x, r=z
    
    def axial_to_cube(q, r):
        return (q, -q-r, r)
        
    def dist_cube(x, y, z):
        return max(abs(x), abs(y), abs(z))
        
    full_hex_2_cube = []
    for x in range(-2, 3):
        for y in range(-2, 3):
            z = -x-y
            if -2 <= z <= 2:
                full_hex_2_cube.append((x, y, z))
                
    # We have 19 points. Remove 8.
    # To minimize the bounding box (which is a hexagon), we want to keep the shape as close to a smaller hexagon or circle as possible.
    # Removing corners (dist 2) is good.
    # There are 6 corners.
    # Remove them: 13 left.
    # The remaining dist 2 points are on the edges.
    # There are 12 edge points? No.
    # Total 19.
    # Center: 1.
    # Ring 1: 6.
    # Ring 2: 12.
    # Total 19.
    # Remove 6 corners of Ring 2.
    # Left: 1 (center) + 6 (ring 1) + 6 (ring 2 edges) = 13.
    # Remove 2 more from Ring 2 edges.
    # Which ones?
    # The Ring 2 edges are at dist 2.
    # If we remove any 2, the max dist is still 2 (from the remaining ring 2 points).
    # So the bounding box size is determined by the points at dist 2.
    # Can we get a smaller bounding box?
    # If we remove ALL Ring 2 points, we have 7 points. Bounding box dist 1.
    # But we need 11.
    # So we must have some dist 2 points.
    # Therefore, max projection is at least that of a dist 2 point.
    # Dist 2 point in Cartesian:
    # (2,0) axial -> x = sqrt(3)*2 = 3.464.
    # So D >= 3.464.
    # S >= 3.464 / 0.866 + 1 = 4 + 1 = 5.
    
    # So the minimum possible side length is 5?
    # Let's check if we can rotate to reduce the projection.
    # The projection of a point at distance R is R * cos(theta).
    # Max projection over all angles is R.
    # So if any point is at distance R from the center, the max projection is at least R.
    # So S >= R/APOTHEM + 1.
    # If we use points from Ring 2, R = 2 * sqrt(3) * (1/sqrt(3))?
    # Axial (2,0) -> Cartesian (2*sqrt(3), 0). Distance 2*sqrt(3) ≈ 3.464.
    # So S >= 5.
    
    # Is it possible to use a non-axial arrangement?
    # The problem allows translation and rotation of each hexagon.
    # But the "axial" candidates assume they are on a lattice.
    # We can move them slightly to reduce the max distance from the center?
    # No, if they are disjoint, the centers must be at least distance 2*APOTHEM = sqrt(3) apart?
    # No, side length 1 hexagons.
    # Distance between centers of touching hexagons is 2 * APOTHEM = sqrt(3).
    # So the minimum distance between centers is sqrt(3).
    # This is a packing problem.
    
    # The lattice packing is optimal for hexagons.
    # So the centers must be on a triangular lattice (or rotated).
    # The minimal enclosing hexagon of a set of points on a triangular lattice.
    
    # So the answer should be around 5.
    # The current score is 9.36.
    # This implies the current code is failing to find the S=5 solution.
    
    # Let's rewrite the code to specifically test the "13 points minus 2" configuration.
    
    # Construct the 11 points.
    # Keep Center (0,0).
    # Keep Ring 1 (6 points).
    # Keep 4 points from Ring 2.
    # To minimize max projection, pick 4 points from Ring 2 that are clustered?
    # No, max projection is determined by the farthest point in any direction.
    # If we pick 4 points, they are at dist 2*sqrt(3).
    # The max projection will be 2*sqrt(3) unless they are all in a hemisphere and we rotate?
    # No, the container can rotate.
    # We want to minimize the max projection of the set of points.
    # This is the "minimum width" problem? No, minimum enclosing hexagon.
    
    # For a set of points, the min enclosing hexagon side S is determined by the max of the 6 widths.
    
    # Let's just output the configuration of 11 points from the side-2 hexagon (removing 8).
    # And optimize the rotation.
    
    # Candidate:
    # Points:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,0), (1,1), (0,2), (-1,2)  <-- 4 points from ring 2
    
    pts_axial = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2)
    ]
    
    # Convert to Cartesian
    pts = []
    for q, r in pts_axial:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    pts = np.array(pts)
    
    # Center the points
    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])
    pts_centered = pts - np.array([cx, cy])
    
    # Optimize rotation beta
    def get_S(beta):
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        max_proj = 0.0
        for nx, ny in normals:
            projs = pts_centered[:, 0] * nx + pts_centered[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
            
        D = max_proj
        S = (D + APOTHEM) / APOTHEM
        return S
        
    # Minimize S
    res = minimize_scalar(get_S, bounds=(0, math.pi/6), method='bounded', options={'xatol': 1e-10})
    best_beta = res.x
    best_S = res.fun
    
    # Final coordinates
    angle_deg = math.degrees(best_beta)
    
    inner = []
    for x, y in pts:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(best_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
