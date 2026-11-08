"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # 1. Define the layout of 12 inner hexagons using axial coordinates
    # We use a compact hexagonal cluster.
    # Let's use a 3x4 parallelogram in axial coordinates for simplicity and density.
    # Axial coords (q, r)
    axial_points = []
    # 4 rows of 3? Or 3 rows of 4?
    # Let's try a "rhombus" shape:
    # r=0: q=0,1,2
    # r=1: q=-1,0,1,2
    # r=2: q=-2,-1,0,1
    # r=3: q=-3,-2,-1,0
    # This is 3+4+4+3 = 14 points. Too many.
    
    # Let's use a specific 12-point cluster:
    # Center (0,0)
    # Ring 1: 6 points
    # Ring 2: 5 points (to make 12)
    # We pick 5 from Ring 2 to keep it compact.
    # Ring 2 points: (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), (-1,0), (-1,1), (-1,2), (0,2), (1,2), (2,1)
    # Actually, let's just use a rectangular grid in axial coords:
    # q in [-1, 1], r in [-1, 2] -> 3 * 4 = 12 points.
    # This forms a parallelogram.
    for r in range(-1, 3):
        for q in range(-1, 2):
            axial_points.append((q, r))
            
    # Convert axial to Cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    centers = []
    for q, r in axial_points:
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        centers.append((x, y))
        
    # Center the layout at (0,0)
    cx_mean = np.mean([c[0] for c in centers])
    cy_mean = np.mean([c[1] for c in centers])
    centers = [(x - cx_mean, y - cy_mean) for x, y in centers]
    
    # Inner hexagons: side 1, angle 0
    inner = [[x, y, 0.0] for x, y in centers]
    
    # 2. Bisection search for minimal outer side length
    # Outer hexagon centered at (0,0), angle 0
    # We need to find min S such that all inner hexagons are inside outer hexagon of side S.
    
    def is_valid(S):
        # Check if all inner hexagons are inside outer hexagon of side S
        # Outer hexagon vertices:
        # Angle 0: vertices at 0, 60, 120, 180, 240, 300 degrees
        # Distance from center to vertex is S (circumradius = side length for regular hexagon)
        # So outer hexagon is the intersection of 6 half-planes.
        # The distance from center to each edge is apothem = S * sqrt(3) / 2.
        
        # For a point (x,y) to be inside, its projection on each normal vector must be <= apothem.
        # Normals are at 0, 60, ... degrees? No, edges are perpendicular to vertices?
        # Regular hexagon with flat top/bottom or pointy top/bottom?
        # Our inner hexagons are generated with angle 0.
        # hexagon_vertices(cx, cy, side, angle_deg):
        # angles = angle_rad + arange(6) * pi/3
        # If angle=0, vertex is at (side, 0). So pointy right.
        # The edges are between vertices.
        # The apothem direction is at 30, 90, 150, 210, 270, 330 degrees.
        
        apothem = S * np.sqrt(3) / 2.0
        
        # Normals to the edges (pointing outwards)
        # Edge 1: between (1,0) and (0.5, sqrt(3)/2). Normal at 30 deg.
        # Let's just check distance from center to all 6 vertices of each inner hexagon
        # and ensure they are inside the outer hexagon.
        
        # Faster: Check if max distance of any vertex from center, projected on the 6 normal directions, is <= apothem.
        # Or simply: The outer hexagon is convex. The inner hexagons are convex.
        # It suffices to check that all vertices of all inner hexagons are inside the outer hexagon.
        
        # Pre-compute normals for outer hexagon (pointy right)
        # Vertices of outer hexagon (side S, angle 0):
        # V0: (S, 0)
        # V1: (S/2, S*sqrt(3)/2)
        # ...
        # The supporting lines are perpendicular to the vectors from center to edge midpoints.
        # Edge midpoints are at distance apothem, angles 30, 90, 150, 210, 270, 330.
        
        angles_normals = np.radians([30, 90, 150, 210, 270, 330])
        normals = np.column_stack([np.cos(angles_normals), np.sin(angles_normals)])
        
        for x, y, ang in inner:
            # Get vertices of this inner hexagon
            # side = 1.0
            # angle = ang
            ang_rad = np.radians(ang)
            v_angles = ang_rad + np.arange(6) * np.pi / 3.0
            verts = np.column_stack([
                x + np.cos(v_angles),
                y + np.sin(v_angles)
            ])
            
            # Check if any vertex is outside
            # Project vertices on normals
            # proj = verts @ normals.T
            # If any proj > apothem, it's outside.
            projs = verts @ normals.T
            if np.any(projs > apothem):
                return False
                
        return True

    # Initial bounds
    # Max distance of any inner vertex from center
    max_dist = 0
    for x, y, ang in inner:
        ang_rad = np.radians(ang)
        v_angles = ang_rad + np.arange(6) * np.pi / 3.0
        verts_x = x + np.cos(v_angles)
        verts_y = y + np.sin(v_angles)
        dists = np.sqrt(verts_x**2 + verts_y**2)
        max_dist = max(max_dist, np.max(dists))
        
    # Side length S is at least max_dist / (sqrt(3)/2) ?
    # If the furthest point is along a normal, S = apothem / (sqrt(3)/2) = apothem * 2/sqrt(3).
    # If the furthest point is along a vertex, S = dist.
    # So lower bound is max_dist.
    # Upper bound can be 2 * max_dist.
    
    lo = max_dist
    hi = 2 * max_dist
    
    # Bisection
    best_S = hi
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if is_valid(mid):
            best_S = mid
            hi = mid
        else:
            lo = mid
            
        if time.time() - start_time > 10:
            break
            
    # Add safety margin
    best_S += 0.01
    
    return inner, (0.0, 0.0), best_S, 0.0
# EVOLVE-BLOCK-END
