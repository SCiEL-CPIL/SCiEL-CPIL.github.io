import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0 (pointy-top, vertex at 0 deg)
    # Vertices at 0, 60, 120, 180, 240, 360 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0, outer_center=(0.0, 0.0)):
        """
        Calculate the minimum side length of a regular hexagon centered at outer_center 
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        # Normals to the sides of the outer hexagon
        # A regular hexagon with side s has apothem a = s * sqrt(3)/2.
        # The distance from center to any side is the apothem.
        # So if max projection distance from center is D, then s = 2 * D / sqrt(3).
        
        normals = []
        for k in range(6):
            # The sides are perpendicular to directions 30, 90, 150, 210, 270, 330 relative to the hexagon's "top"
            # If we rotate the hexagon by outer_angle, the normals rotate by outer_angle.
            # Standard pointy-top hexagon (vertex at 0 deg) has sides with normals at 30, 90, ...
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals) # (6, 2)
        
        max_proj = 0.0
        ocx, ocy = outer_center
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            # Translate vertices relative to outer center
            verts_rel = verts - np.array([ocx, ocy])
            projs = verts_rel @ normals.T # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    # Strategy: Discrete Tiling Search
    # 1. Generate candidate rigid cluster configurations on a honeycomb lattice.
    # 2. For each valid cluster (non-overlapping), find the optimal global rotation 
    #    and outer hexagon rotation to minimize the enclosing side length.
    
    # Honeycomb lattice basis
    # a1 = (1.5, sqrt(3)/2)
    # a2 = (0, sqrt(3))
    # Any lattice point can be represented as i*a1 + j*a2
    # Coordinates in this basis (i, j) are integers.
    # Distance squared between (i1, j1) and (i2, j2) can be calculated exactly.
    # Two hexagons at lattice points p1 and p2 overlap if they are the same point.
    # They touch if the distance is exactly 2 (side length 1, distance between centers in honeycomb for touching is 2*apothem = sqrt(3)? No.
    # Center-to-center distance for touching hexagons of side 1 is 2 * apothem = sqrt(3).
    # Wait, apothem of unit hexagon is sqrt(3)/2.
    # So touching centers are at distance sqrt(3).
    # In honeycomb lattice, nearest neighbors are at distance sqrt(3).
    # So any distinct lattice points are at least sqrt(3) apart.
    # Thus, any set of distinct lattice points forms a valid non-overlapping packing.
    
    # We need to find the "tightest" clusters of 12 lattice points.
    # A cluster is defined by the set of 12 points.
    # The "shape" of the cluster determines the bounding box.
    # We want to minimize the width of the cluster in all 6 directions.
    
    # Let's enumerate small clusters.
    # The most compact shape is a larger hexagon, but 12 doesn't form a perfect hexagon (1, 7, 19...).
    # 12 points can form:
    # - A 3x4 rectangle in the skewed lattice?
    # - A hexagon with missing corners?
    
    # Let's generate all subsets of 12 points from a small grid of lattice points
    # and keep the ones with the smallest "diameter" or bounding box area.
    
    # Grid range: i in [-4, 4], j in [-4, 4]
    # Total points: 9*9 = 81.
    # Subsets of 12 from 81 is huge.
    # Instead, let's look at connected components or specific patterns.
    
    # Alternative: Use the fact that the optimal packing likely resembles a "hexagonal prism" or similar compact shape.
    # Let's try to find the 12 points that minimize the maximum projection onto the 6 lattice directions.
    
    # Directions in lattice coordinates (i,j):
    # The 6 neighbors of (0,0) are:
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # These correspond to the 6 directions of the hexagonal grid.
    
    # For a set of points S, the "width" in direction d is max(p.d) - min(p.d) for p in S.
    # The side length of the enclosing hexagon is roughly proportional to the max width over all 6 directions.
    # Specifically, if the cluster is aligned with the lattice, the enclosing hexagon side length S is:
    # S = (max_width / sqrt(3)) * 2 ?
    # Let's verify.
    # If we have a single hexagon, width in normal direction is 2*apothem = sqrt(3).
    # Side length is 1.
    # Formula: S = width / sqrt(3).
    # Wait, apothem a = sqrt(3)/2. Width = 2a = sqrt(3).
    # S = 1.
    # So S = width / sqrt(3).
    
    # We want to minimize the maximum width over the 6 lattice directions.
    
    # Let's generate candidate clusters by taking a central point and adding neighbors in a "greedy" compact way,
    # or by enumerating small bounding boxes in the lattice.
    
    # Lattice vectors:
    v1 = np.array([1.5, math.sqrt(3)/2])
    v2 = np.array([0, math.sqrt(3)])
    
    # Generate a pool of candidate points
    pool = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            p = i * v1 + j * v2
            pool.append((i, j, p))
            
    # We will test specific "shapes" of 12 points.
    # Common compact shapes for 12:
    # 1. 3-4-3-2 (rows of 3, 4, 3, 2 in a skewed grid)
    # 2. 4-4-4 (rows of 4)
    # 3. Hexagon of radius 2 (19 points) minus 7 points.
    
    # Let's define a function to calculate the "cost" (max width) of a set of lattice points.
    def calc_cluster_cost(points):
        # points: list of (i, j)
        if len(points) != 12:
            return float('inf')
        
        # The 6 directions in (i,j) space
        # d1 = (1, 0)
        # d2 = (0, 1)
        # d3 = (-1, 1)
        # d4 = (-1, 0)
        # d5 = (0, -1)
        # d6 = (1, -1)
        
        dirs = [
            (1, 0),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (0, -1),
            (1, -1)
        ]
        
        max_width = 0
        for di, dj in dirs:
            projs = [i*di + j*dj for i, j in points]
            width = max(projs) - min(projs)
            if width > max_width:
                max_width = width
                
        return max_width

    # Generate candidate clusters
    # We'll search for clusters that fit in a small bounding box.
    # Let's try to find the 12 points with the minimal cost.
    # Since enumerating all subsets is too slow, we will use a heuristic:
    # Start with a central point and add the 11 nearest neighbors?
    # The 11 nearest neighbors to (0,0) in the hex grid are:
    # Ring 1: 6 points
    # Ring 2: 12 points. We need 5 more.
    # This forms a "hexagon with 5 extra".
    
    # Let's generate the set of 12 points consisting of the center, the 6 neighbors, and 5 from the next ring.
    # Which 5? To be compact, we should pick 5 that minimize the max width.
    
    # Center: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 points:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,1), (1,2) -- wait, let's list them properly.
    # Neighbors of Ring 1:
    # (1,0) neighbors: (2,0), (1,1), (0,1)[R1], (-1,1)[R1], (0,-1)[R1], (1,-1)[R1] -> New: (2,0), (1,1)
    # (0,1) neighbors: (1,1), (0,2), (-1,2), (-1,1)[R1], (0,0)[C], (1,0)[R1] -> New: (0,2), (-1,2)
    # (-1,1) neighbors: (-1,2), (-2,2), (-2,1), (-2,0), (-1,0)[R1], (0,1)[R1] -> New: (-2,2), (-2,1), (-2,0)
    # ... and so on.
    
    # Actually, the 12 points of Ring 2 are:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,1), (1,2)
    # Let's check distances.
    # (2,0) dist: 2*1.5 = 3? No, dist^2 = 4*1.5^2 = 9? No.
    # Lattice distance: d^2 = 3*(i^2 + j^2 + i*j) ?
    # Let's use the vector calculation.
    # (2,0) -> 2*v1 = (3, sqrt(3)). Dist^2 = 9 + 3 = 12. Dist = sqrt(12) = 2*sqrt(3).
    # (1,1) -> v1+v2 = (1.5, 1.5*sqrt(3)). Dist^2 = 2.25 + 6.75 = 9. Dist = 3.
    # (0,2) -> 2*v2 = (0, 2*sqrt(3)). Dist^2 = 12. Dist = 2*sqrt(3).
    # So Ring 2 has points at distance 3 (6 of them: (1,1), (-1,2), (-2,1), (-1,-1), (1,-2), (2,-1))
    # and points at distance 2*sqrt(3) (6 of them: (2,0), (0,2), (-2,1)?? No.
    
    # Let's just generate the 18 points in the "hexagon of radius 2" (center + ring 1 + ring 2) and pick 12.
    # Center: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2: 
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0)
    # (-1,-1), (0,-2), (1,-2), (2,-1), (2,1), (1,2) -- wait, (2,1) and (1,2) are further?
    # (2,1) -> 2v1+v2 = (3, 2sqrt(3)). Dist^2 = 9 + 12 = 21.
    # (1,2) -> v1+2v2 = (1.5, 3sqrt(3)). Dist^2 = 2.25 + 27 = 29.25.
    # So the "hexagon of radius 2" in terms of graph distance has 19 points.
    # The 12 points of Ring 2 are the ones at graph distance 2.
    # They are: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,1)?? No.
    
    # Let's list neighbors of Ring 1 again carefully.
    # (1,0) -> (2,0), (1,1)
    # (0,1) -> (0,2), (-1,2)
    # (-1,1) -> (-2,1), (-2,2) -- wait, (-2,2) is dist?
    # (-2,2) -> -2v1+2v2 = (-3, 2sqrt(3)). Dist^2 = 9+12=21.
    # So (-2,2) is not in the "compact" ring 2?
    # Graph distance 2 from (0,0) means 2 steps.
    # (0,0) -> (1,0) -> (2,0)
    # (0,0) -> (1,0) -> (1,1)
    # (0,0) -> (0,1) -> (0,2)
    # (0,0) -> (0,1) -> (-1,2)
    # (0,0) -> (-1,1) -> (-2,2)
    # (0,0) -> (-1,1) -> (-2,1)
    # (0,0) -> (-1,0) -> (-2,0)
    # (0,0) -> (-1,0) -> (-1,-1)
    # (0,0) -> (0,-1) -> (0,-2)
    # (0,0) -> (0,-1) -> (1,-2)
    # (0,0) -> (1,-1) -> (2,-1)
    # (0,0) -> (1,-1) -> (2,0)
    
    # So the 12 points of Ring 2 are:
    # (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,0)
    # Wait, (2,0) appears twice?
    # (1,0)->(2,0) and (1,-1)->(2,0). Yes.
    # So unique points:
    # (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    # That's 11 points.
    # Did I miss one?
    # (0,0) has 6 neighbors. Each neighbor has 5 new neighbors (one is center, 3 are shared with other R1 neighbors? No).
    # Total points in hexagon of radius 2 is 1 + 6 + 12 = 19.
    # So there are 12 points in Ring 2.
    # Let's list them by angle.
    # 1. (2,0)
    # 2. (1,1)
    # 3. (0,2)
    # 4. (-1,2)
    # 5. (-2,2)
    # 6. (-2,1)
    # 7. (-2,0)
    # 8. (-1,-1)
    # 9. (0,-2)
    # 10. (1,-2)
    # 11. (2,-1)
    # 12. (2,1) ??
    # Let's check (2,1). Path: (0,0)->(1,0)->(2,1)? (1,0) to (2,1) is a valid step?
    # (2,1) - (1,0) = (1,1). Is (1,1) a lattice vector? No.
    # Lattice vectors are (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
    # (1,1) is not a basis vector.
    # So (2,1) is not a neighbor of (1,0).
    # Neighbors of (1,0) are: (2,0), (1,1), (0,1), (-1,1), (0,-1), (1,-1).
    # So (2,1) is not in Ring 2.
    # What is the 12th point?
    # Let's count:
    # 1. (2,0)
    # 2. (1,1)
    # 3. (0,2)
    # 4. (-1,2)
    # 5. (-2,2)
    # 6. (-2,1)
    # 7. (-2,0)
    # 8. (-1,-1)
    # 9. (0,-2)
    # 10. (1,-2)
    # 11. (2,-1)
    # 12. (2,1) is NOT there.
    # Let's check (-2,2). Path: (0,0)->(-1,1)->(-2,2). Yes.
    # Let's check (1,2). Path: (0,0)->(0,1)->(1,2)? (1,2)-(0,1)=(1,1). Not a step.
    # Path: (0,0)->(1,0)->(1,1)->(1,2)? (1,1)->(1,2) is (0,1). Yes.
    # So (1,2) is in Ring 3?
    # (1,2) dist: v1+2v2. Dist^2 = 29.25.
    # (2,1) dist: 2v1+v2. Dist^2 = 21.
    # (2,0) dist: 2v1. Dist^2 = 12.
    # (1,1) dist: v1+v2. Dist^2 = 9.
    # (0,2) dist: 2v2. Dist^2 = 12.
    # (-1,2) dist: -v1+2v2. Dist^2 = 9.
    # (-2,2) dist: -2v1+2v2. Dist^2 = 12.
    # (-2,1) dist: -2v1+v2. Dist^2 = 9.
    # (-2,0) dist: -2v1. Dist^2 = 12.
    # (-1,-1) dist: -v1-v2. Dist^2 = 9.
    # (0,-2) dist: -2v2. Dist^2 = 12.
    # (1,-2) dist: v1-2v2. Dist^2 = 9.
    # (2,-1) dist: 2v1-v2. Dist^2 = 12.
    # (2,1) dist: 2v1+v2. Dist^2 = 21.
    
    # So the 12 points of Ring 2 are:
    # (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    # That's 11.
    # Where is the 12th?
    # Ah, (2,1) is not in Ring 2.
    # Let's check (1,2) again.
    # (0,0)->(0,1)->(1,1) is a step? (1,1)-(0,1)=(1,0). Yes.
    # So (1,1) is in Ring 2.
    # (1,1)->(1,2) is a step? (1,2)-(1,1)=(0,1). Yes.
    # So (1,2) is in Ring 3.
    
    # Let's re-enumerate neighbors of Ring 1.
    # R1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # N(1,0): (2,0), (1,1)
    # N(0,1): (0,2), (-1,2)
    # N(-1,1): (-2,2), (-2,1)
    # N(-1,0): (-2,0), (-1,-1)
    # N(0,-1): (0,-2), (1,-2)
    # N(1,-1): (2,-1), (2,0)
    
    # Unique new points:
    # (2,0), (1,1)
    # (0,2), (-1,2)
    # (-2,2), (-2,1)
    # (-2,0), (-1,-1)
    # (0,-2), (1,-2)
    # (2,-1)
    
    # Total: 2 + 2 + 2 + 2 + 2 + 1 = 11.
    # Why 11?
    # Because (2,0) is shared by (1,0) and (1,-1).
    # Are there any other shares?
    # (1,1) shared by (1,0) and (0,1)?
    # (1,1) - (1,0) = (0,1). Yes.
    # (1,1) - (0,1) = (1,0). Yes.
    # So (1,1) is shared.
    # (-1,2) shared by (0,1) and (-1,1)?
    # (-1,2) - (0,1) = (-1,1). Yes.
    # (-1,2) - (-1,1) = (0,1). Yes.
    # So (-1,2) is shared.
    # (-2,1) shared by (-1,1) and (-1,0)?
    # (-2,1) - (-1,1) = (-1,0). Yes.
    # (-2,1) - (-1,0) = (-1,1). Yes.
    # So (-2,1) is shared.
    # (-1,-1) shared by (-1,0) and (0,-1)?
    # (-1,-1) - (-1,0) = (0,-1). Yes.
    # (-1,-1) - (0,-1) = (-1,0). Yes.
    # So (-1,-1) is shared.
    # (1,-2) shared by (0,-1) and (1,-1)?
    # (1,-2) - (0,-1) = (1,-1). Yes.
    # (1,-2) - (1,-1) = (0,-1). Yes.
    # So (1,-2) is shared.
    # (2,0) shared by (1,-1) and (1,0)?
    # (2,0) - (1,-1) = (1,1). No.
    # (2,0) - (1,0) = (1,0). Yes.
    # (2,0) - (1,-1) = (1,1). No.
    # Wait, (1,-1) neighbors: (2,-1), (2,0)?
    # (2,0) - (1,-1) = (1,1). Not a basis vector.
    # So (2,0) is NOT a neighbor of (1,-1).
    # Neighbors of (1,-1):
    # (1,-1)+(1,0) = (2,-1)
    # (1,-1)+(0,1) = (1,0) [R1]
    # (1,-1)+(-1,1) = (0,0) [C]
    # (1,-1)+(-1,0) = (0,-1) [R1]
    # (1,-1)+(0,-1) = (1,-2)
    # (1,-1)+(1,-1) = (2,-2)
    
    # So (2,0) is only a neighbor of (1,0).
    # And (2,-1) is only a neighbor of (1,-1).
    # So we have:
    # (2,0) [from 1,0]
    # (1,1) [from 1,0 and 0,1]
    # (0,2) [from 0,1]
    # (-1,2) [from 0,1 and -1,1]
    # (-2,2) [from -1,1]
    # (-2,1) [from -1,1 and -1,0]
    # (-2,0) [from -1,0]
    # (-1,-1) [from -1,0 and 0,-1]
    # (0,-2) [from 0,-1]
    # (1,-2) [from 0,-1 and 1,-1]
    # (2,-1) [from 1,-1]
    # (2,-2) [from 1,-1]
    
    # Total: 12 points!
    # The 12 points are:
    # (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2)
    
    # Let's verify this set.
    # It forms a "hexagon" with one corner cut off?
    # Let's calculate the cost of this specific cluster.
    
    cluster_12 = [
        (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), 
        (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2)
    ]
    
    # We should also consider other permutations or shapes.
    # But this is the most symmetric "almost hexagon".
    # Let's also try removing 1 point from the "hexagon of radius 2" (19 points) to get 12?
    # No, we want 12 points.
    # The cluster above is 12 points.
    
    # Let's generate a few candidate clusters:
    # 1. The one above.
    # 2. A 3x4 rectangle in the lattice?
    #    Points (i,j) where i in [0,3], j in [0,4]?
    #    Let's try to find the optimal 12-point subset of the 19-point hexagon.
    
    # We will evaluate the cost of the cluster above and a few variations.
    
    candidates = []
    
    # Candidate 1: The "almost hexagon" derived above
    c1 = cluster_12
    candidates.append(c1)
    
    # Candidate 2: Shift one point to make it more symmetric?
    # The set c1 has points at (2,0) and (2,-2).
    # What if we replace (2,-2) with (2,1)?
    # (2,1) is not in Ring 2.
    
    # Let's just use the 12 points with the smallest "cost" from the 19-point hexagon.
    # We can do a quick local search on the 19 points.
    
    hex19 = [(0,0)] + [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)] + cluster_12
    
    # We want to pick 12 from these 19.
    # The center (0,0) is always good to keep.
    # The 6 Ring 1 points are always good to keep.
    # So we need to pick 5 from the 12 Ring 2 points.
    
    ring2 = cluster_12
    # We need to pick 5 from ring2.
    # To minimize max width, we should pick 5 that are "balanced".
    # The 12 points in ring2 are arranged in a ring.
    # Picking 5 consecutive points?
    # Picking 5 spread out?
    
    # Let's try picking 5 that are roughly evenly spaced.
    # Indices in ring2:
    # 0: (2,0)
    # 1: (1,1)
    # 2: (0,2)
    # 3: (-1,2)
    # 4: (-2,2)
    # 5: (-2,1)
    # 6: (-2,0)
    # 7: (-1,-1)
    # 8: (0,-2)
    # 9: (1,-2)
    # 10: (2,-1)
    # 11: (2,-2)
    
    # Try picking indices 0, 2, 4, 7, 9?
    # Or 0, 3, 6, 9, 11?
    
    # Let's just evaluate the cost of the full 12-point cluster (all ring 2 points) vs subsets.
    # Wait, the cluster I defined earlier WAS all 12 ring 2 points.
    # Center + 6 R1 + 12 R2 = 19 points.
    # I want 12 points total.
    # So I keep Center + 6 R1 = 7 points.
    # I need 5 more from R2.
    
    # Let's find the best 5 from R2.
    best_subset = None
    best_cost = float('inf')
    
    # There are C(12, 5) = 792 subsets. This is small.
    import itertools
    
    for combo in itertools.combinations(range(12), 5):
        subset = [ring2[i] for i in combo]
        full_set = [(0,0)] + [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)] + subset
        cost = calc_cluster_cost(full_set)
        if cost < best_cost:
            best_cost = cost
            best_subset = full_set
            
    candidates.append(best_subset)
    
    # Also try a different base shape: a "rectangle"
    # i in [-1, 2], j in [-2, 1] ?
    # Let's just stick with the two best candidates found so far.
    
    # For each candidate cluster, we optimize the global rotation and outer hexagon rotation.
    # The cluster is rigid. We rotate the whole cluster by alpha.
    # We also rotate the outer hexagon by beta.
    # We translate the cluster to center it?
    # The optimal translation is to center the cluster's bounding box?
    # Actually, for a fixed shape and fixed outer rotation, the optimal translation
    # is to minimize the max projection. This is equivalent to centering the shape
    # in the "midpoint" of the bounding box in each direction?
    # For a regular hexagon container, the optimal position of a rigid body is
    # to align its "center of symmetry" with the container's center, if it has one.
    # Our clusters might not be perfectly symmetric.
    
    # Let's define the optimization variables:
    # x = [alpha, beta]
    # alpha: rotation of the cluster
    # beta: rotation of the outer hexagon
    
    # Objective: min_outer_side(positions, angles, beta)
    # positions are derived from the cluster points, rotated by alpha.
    # angles are all 0 (relative to cluster), so absolute angle is alpha.
    
    def objective_cluster(params, cluster_points):
        alpha = params[0]
        beta = params[1]
        
        # Generate positions
        positions = []
        for i, j in cluster_points:
            p = i * v1 + j * v2
            # Rotate p by alpha
            cos_a = math.cos(alpha)
            sin_a = math.sin(alpha)
            rx = p[0] * cos_a - p[1] * sin_a
            ry = p[0] * sin_a + p[1] * cos_a
            positions.append((rx, ry))
            
        # Center the cluster
        cx = np.mean([p[0] for p in positions])
        cy = np.mean([p[1] for p in positions])
        positions = [(p[0]-cx, p[1]-cy) for p in positions]
        
        # Angles are all alpha
        angles = [alpha * 180.0 / math.pi] * N
        
        # Calculate min outer side
        # Outer center is (0,0)
        return calc_min_outer_side(np.array(positions), np.array(angles), beta * 180.0 / math.pi, (0.0, 0.0))

    best_global_score = float('inf')
    best_params_global = None
    best_cluster = None
    
    for cluster in candidates:
        if not cluster:
            continue
            
        # Initial guess: alpha=0, beta=0
        x0 = np.array([0.0, 0.0])
        
        try:
            res = minimize(
                objective_cluster, 
                x0, 
                args=(cluster,),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if res.fun < best_global_score:
                best_global_score = res.fun
                best_params_global = res.x
                best_cluster = cluster
        except Exception:
            pass
            
        # Try a few other initial guesses for alpha, beta
        for a_init in [math.pi/6, math.pi/3]:
            for b_init in [math.pi/6, math.pi/3]:
                x0 = np.array([a_init, b_init])
                try:
                    res = minimize(
                        objective_cluster, 
                        x0, 
                        args=(cluster,),
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
                    )
                    if res.fun < best_global_score:
                        best_global_score = res.fun
                        best_params_global = res.x
                        best_cluster = cluster
                except Exception:
                    pass

    if best_cluster is None or best_params_global is None:
        # Fallback to a simple configuration
        # Just place them in a line?
        positions = np.array([(i, 0) for i in range(12)])
        angles = np.zeros(12)
        final_side = calc_min_outer_side(positions, angles, 0.0, (0.0, 0.0))
        return [[positions[i][0], positions[i][1], 0.0] for i in range(12)], (0.0, 0.0), final_side, 0.0

    alpha = best_params_global[0]
    beta = best_params_global[1]
    
    # Construct final positions
    final_positions = []
    for i, j in best_cluster:
        p = i * v1 + j * v2
        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)
        rx = p[0] * cos_a - p[1] * sin_a
        ry = p[0] * sin_a + p[1] * cos_a
        final_positions.append((rx, ry))
        
    # Center
    cx = np.mean([p[0] for p in final_positions])
    cy = np.mean([p[1] for p in final_positions])
    final_positions = [(p[0]-cx, p[1]-cy) for p in final_positions]
    
    final_angles = [alpha * 180.0 / math.pi] * N
    final_side = best_global_score
    final_outer_angle = beta * 180.0 / math.pi
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, final_outer_angle

