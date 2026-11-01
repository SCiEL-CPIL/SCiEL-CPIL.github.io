import math
import numpy as np
from itertools import combinations

def construct():
    # Unit regular hexagons of side 1.
    # The "diameter" across opposite vertices is 2.
    # The "width" across opposite sides is sqrt(3).
    # Centers of adjacent hexagons in a tiling are separated by sqrt(3).
    
    # We want to pack 11 unit hexagons.
    # The optimal arrangement is likely a compact cluster.
    # Let's consider the hexagonal grid of centers.
    # Center at (0,0).
    # Neighbors at distance sqrt(3) in 6 directions (60 degrees apart).
    # Next neighbors at distance 2*sqrt(3) etc.
    
    # Let's define the 11 centers in a specific shape.
    # A common compact shape for 11 is a hexagon of side 2 (which has 19 cells) minus 8 corners?
    # Or simply a line of 11? No, that's long.
    # A 3x4 rectangle in hex coordinates?
    # Let's try a "flower" shape: 1 center + 6 neighbors + 4 from the second ring.
    # Total 11.
    
    # Coordinates in Cartesian for a hex grid where neighbor distance is d = sqrt(3).
    # Let d = math.sqrt(3)
    # Directions: 0, 60, 120, 180, 240, 300 degrees.
    # However, the orientation of the hexagons matters.
    # If we align the hexagons with the grid, the enclosing hexagon will also be aligned.
    
    # Let's generate the 11 centers:
    centers = []
    d = math.sqrt(3)
    
    # Ring 0
    centers.append((0.0, 0.0))
    
    # Ring 1 (6 centers)
    for i in range(6):
        theta = i * math.pi / 3
        x = d * math.cos(theta)
        y = d * math.sin(theta)
        centers.append((x, y))
        
    # Ring 2 (we need 4 more to make 11)
    # The 12 centers in ring 2 are at distance 2*d from origin? No, the hex grid step is d.
    # The vertices of the hexagon formed by ring 1 centers are at distance d.
    # The next shell of centers in a hex lattice...
    # Let's just define the coordinates explicitly for a compact cluster.
    # A 3x3 block of hexes has 9 centers? No.
    # Let's use axial coordinates for the grid of centers.
    # Center (0,0) is at axial (0,0).
    # Neighbors are at (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
    # Let's pick the 11 cells in a "cross" or "hexagon" shape.
    # The smallest hexagon that contains 11 cells?
    # Side 1: 1 cell
    # Side 2: 19 cells
    # So we can't fit 11 in a side 2 hexagon perfectly without gaps?
    # Wait, the "side" of the container hexagon is what we minimize.
    # The container is a regular hexagon.
    # The inner hexagons are unit side length.
    
    # Let's construct the 11 centers using axial coordinates (q, r) on the triangular lattice of centers.
    # Distance between axial (q,r) and (q',r') in Cartesian (with y pointing down-ish for standard hex grid):
    # x = size * (3/2 * q)
    # y = size * (sqrt(3) * r + sqrt(3)/2 * q)
    # Here size is the center-to-vertex distance of the unit hexagon? No.
    # For unit side length s=1, the distance between centers of adjacent hexagons is sqrt(3).
    # So in the formula above, if we set size such that the step is sqrt(3):
    # x = sqrt(3) * (3/2 * q) is wrong scaling.
    # Let's just use the vector directions.
    # Basis vectors:
    # v0 = (sqrt(3), 0)
    # v1 = (sqrt(3)/2, 3/2)  # This is not unit distance sqrt(3) apart?
    # |v0| = sqrt(3).
    # |v1| = sqrt(3/4 + 9/4) = sqrt(12/4) = sqrt(3).
    # Dot product: sqrt(3)*sqrt(3)/2 + 0 = 3/2.
    # Angle: cos(theta) = (3/2) / 3 = 1/2 -> 60 degrees. Correct.
    
    # So center(q,r) = q * v0 + r * v1.
    # v0 = (sqrt(3), 0)
    # v1 = (sqrt(3)/2, 3/2)
    
    # Let's select 11 axial coordinates that form a compact shape.
    # (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # That's 7 centers.
    # We need 4 more.
    # Let's take (2,0), (1,1), (0,2), (-2,1)?
    # Let's check the shape.
    # To be compact, we should fill the "corners" of the 7-point flower.
    # The 7-point flower is a hexagon of side 1 (in terms of centers).
    # The next layer adds 12 points.
    # We just need any 4. To minimize the bounding box, we should pick ones that are balanced.
    # Let's pick (2,0), (-1,2), (-2,-1)? No, let's stick to the first quadrant and opposite.
    # Let's pick:
    # (1,1), (1,-1) are already in ring 1?
    # Axial neighbors of (0,0):
    # (1,0) -> v0
    # (0,1) -> v1
    # (-1,1) -> -v0+v1
    # (-1,0) -> -v0
    # (0,-1) -> -v1
    # (1,-1) -> v0-v1
    
    # Ring 2 candidates (distance 2 steps):
    # (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2)
    # (1,1), (1,-2), (0,1) is ring 1...
    # Let's just define a set of 11 that is known to be good.
    # A 3x4 grid in rectangular coordinates doesn't map directly.
    # Let's use a "parallelogram" of 3x4? No, hex grid.
    # Let's use the 11 cells:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,0), (0,2), (-2,2), (-2,0) ? 
    # Let's check the extent.
    
    # Actually, let's just compute the minimal enclosing hexagon for a set of candidate shapes.
    # We will test a few topologies.
    
    # Topology 1: 1 center, 6 neighbors, 4 specific ring-2.
    # Topology 2: A 3x3 square in axial coords? (9 cells) + 2 more.
    # Topology 3: A line? (bad)
    
    # Let's implement a function to find the min enclosing hexagon side length L
    # for a set of centers.
    # The hexagon is centered at (cx, cy), side length L, rotation angle alpha.
    # The condition for a point (x,y) to be inside:
    # The hexagon can be defined by 6 half-planes.
    # For a hexagon with "flat" top/bottom (pointy left/right), the width in x is 2L, height is sqrt(3)L.
    # Wait, side length L.
    # Distance from center to side (apothem) is L * sqrt(3) / 2.
    # Let a = L * sqrt(3) / 2.
    # The 6 normals are at angles alpha + k*60 deg.
    # Point (x,y) is inside if for all k, dot((x-cx, y-cy), n_k) <= a.
    # We want to minimize L, which is equivalent to minimizing a.
    # We can fix the center (cx, cy) to the centroid of the points (or search over it).
    # And we can search over alpha.
    
    # Given the symmetry of the problem and the grid, the optimal alpha is likely 0, 30, 60 etc.
    # And the center is likely the center of symmetry of the cluster.
    
    # Let's refine the set of 11 points.
    # The most compact packing of 11 unit circles (which is analogous to hexagons in terms of center distance)
    # is the 1-6-4 arrangement.
    # Let's generate the Cartesian coordinates for this arrangement.
    
    v0 = np.array([math.sqrt(3), 0.0])
    v1 = np.array([math.sqrt(3)/2, 1.5])
    
    # Axial coords
    # Center
    pts = [(0,0)]
    # Ring 1
    pts += [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    # Ring 2 (pick 4)
    # To be symmetric, pick (2,0) and (-2,0)? That stretches it.
    # Pick (1,1), (0,2), (-1,1) is ring 1? No (-1,1) is ring 1.
    # Ring 2 axial coords with max(|q|,|r|,|q+r|) == 2? No, hex distance.
    # Hex distance of (q,r) is (|q|+|r|+|q+r|)/2.
    # (2,0): (2+0+2)/2 = 2.
    # (1,1): (1+1+2)/2 = 2.
    # (0,2): (0+2+2)/2 = 2.
    # (-1,2): (1+2+1)/2 = 2.
    # (-2,1): (2+1+1)/2 = 2.
    # (-2,0): 2.
    # (-1,-1): 2.
    # (0,-2): 2.
    # (1,-2): 2.
    # (2,-1): 2.
    # (2,-2) is dist 3? (2+2+0)/2 = 2. Wait. (2,-2): q=2, r=-2, q+r=0. Dist=2.
    
    # We need 4 more.
    # Let's pick the ones that form a smaller bounding box.
    # If we pick (1,1), (-1,-1), (0,2), (0,-2)?
    # Let's try picking (2,0), (-2,0), (1,1), (-1,-1).
    # Or (1,1), (-1,2), (-2,0), (0,-2)?
    
    # Let's just try all combinations of 4 from the 12 ring-2 points?
    # 12 choose 4 is 495. Manageable.
    # For each combo, compute the min enclosing hexagon.
    
    # First, convert axial to Cartesian.
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * q + math.sqrt(3)/2 * r
        y = 1.5 * r
        return np.array([x, y])
        
    ring0 = [(0,0)]
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    ring2 = [(2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2), (1,-3)] # Check dist
    # Let's verify dist for ring2 list.
    # (2,0): 2
    # (1,1): 2
    # (0,2): 2
    # (-1,2): 2
    # (-2,1): 2
    # (-2,0): 2
    # (-1,-1): 2
    # (0,-2): 2
    # (1,-2): 2
    # (2,-1): 2
    # (2,-2): 2
    # (1,-3): (1+3+2)/2 = 3. So (1,-3) is ring 3.
    # Correct ring 2 is 12 points.
    ring2 = [(2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2), (1,-1)]
    # (1,-1) is ring 1. (1-1+0)/2 = 0? No. |1|+|-1|+|0| = 2. 2/2 = 1.
    # My manual list was messy. Let's generate it programmatically.
    
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if (abs(q) + abs(r) + abs(q+r)) // 2 == 2:
                ring2.append((q,r))
                
    # Now search
    base_pts = ring0 + ring1
    base_cart = [axial_to_cartesian(q,r) for q,r in base_pts]
    
    best_L = float('inf')
    best_config = None
    
    # Optimization: The center of the enclosing hexagon should be close to the centroid of the points.
    # And the angle should be one of the principal ones.
    
    # Let's iterate over all C(12,4) subsets of ring2.
    from itertools import combinations
    
    # Pre-compute cart for ring2
    ring2_cart = [axial_to_cartesian(q,r) for q,r in ring2]
    
    for combo in combinations(ring2_cart, 4):
        pts_cart = base_cart + list(combo)
        
        # Compute centroid
        centroid = np.mean(pts_cart, axis=0)
        
        # Try angles 0 to 60 deg in steps.
        # Due to symmetry of the grid, 0 to 30 is enough?
        # Let's try 0, 10, 20, 30.
        for angle_deg in [0, 10, 20, 30]:
            angle = math.radians(angle_deg)
            
            # Find min L such that all points are inside hexagon centered at centroid with this angle.
            # We can binary search L.
            # Lower bound: max distance from centroid to any point? No, apothem distance.
            # For a point P, the required apothem a is max_k dot(P-C, n_k).
            # L = a * 2 / sqrt(3).
            
            # Normals
            normals = []
            for k in range(6):
                theta = angle + k * math.pi / 3
                normals.append(np.array([math.cos(theta), math.sin(theta)]))
            
            # Required apothem
            req_a = 0.0
            for p in pts_cart:
                v = p - centroid
                for n in normals:
                    d = np.dot(v, n)
                    if d > req_a:
                        req_a = d
            
            L = req_a * 2 / math.sqrt(3)
            
            if L < best_L:
                best_L = L
                # Store the configuration
                # inner: list of [x, y, angle_degrees]
                # The inner hexagons are all aligned with the grid (angle 0) or do they need to be rotated?
                # "each inner hexagon may translate and rotate"
                # If we pack them in a lattice, they are all aligned with the lattice.
                # The lattice is rotated by 0 relative to the container?
                # If the container angle is `angle`, and the lattice is aligned with the container,
                # then the inner hexagons should also be rotated by `angle`?
                # The task asks for `angle_degrees` for each inner hexagon.
                # And the container has a rotation.
                # Usually, aligning everything minimizes the size.
                # So inner angle = container angle.
                
                inner = []
                for p in pts_cart:
                    inner.append([p[0], p[1], angle_deg])
                
                # Center of container
                center = [centroid[0], centroid[1]]
                
                best_config = (inner, center, L, angle_deg)
                
    # If no config found (should not happen), fallback
    if best_config is None:
        # Fallback to seed-like
        inner = [[4.0 * k, 0.0, 0.0] for k in range(11)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    return best_config

# Quick sanity check execution (not part of construct)
# import time
# start = time.time()
# res = construct()
# print(time.time() - start)
# print(res[2])
