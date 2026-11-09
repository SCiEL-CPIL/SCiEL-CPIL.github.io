"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import itertools
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin (evaluator limit 120s, keep 10s buffer)
    
    SQRT3 = math.sqrt(3)
    
    # Helper: Get vertices of a unit hexagon centered at (cx, cy) with rotation angle
    def get_hex_vertices(cx, cy, angle_deg):
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60)
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return verts

    # Helper: Convert axial coordinates (q, r) to Cartesian for a hexagonal grid
    # Standard flat-top hex grid:
    # x = SQRT3 * (q + r/2)
    # y = 1.5 * r
    # This assumes unit side length 1, so distance between centers is SQRT3 (touching)
    # To ensure strict disjointness (interiors disjoint), we can use a slightly larger spacing
    # or rely on the fact that touching is usually allowed in "disjoint" packing contexts 
    # unless "strictly disjoint" (no boundary contact) is required. 
    # The task says "disjoint". In geometry, disjoint sets have empty intersection.
    # Touching hexagons share an edge, so their intersection is not empty.
    # Therefore, we MUST use a spacing > SQRT3 to ensure strict disjointness.
    # However, the score is the outer side length. If we scale the whole configuration,
    # the ratio of inner to outer size remains constant if we just scale coordinates.
    # Wait, the inner hexagons are FIXED size (side length 1).
    # So we cannot just scale the lattice. We must place them with a gap.
    # Let's use a spacing factor S > 1.
    # Distance between centers = S * SQRT3.
    # Minimum distance for disjointness: > SQRT3 (for touching).
    # Let's try S = 1.01 initially for tight packing, or optimize S?
    # Actually, if we use S=1.01, the gap is tiny.
    # But the combinatorial search depends on the *relative* positions.
    # If we fix the lattice topology, the relative positions are fixed by the axial coords.
    # The absolute positions are scaled by S.
    # The outer hexagon size will scale with S.
    # So minimizing the bounding box in lattice units is equivalent to minimizing S * (bbox_size).
    # Since S is a constant multiplier for the whole cluster, we just need to find the
    # lattice configuration with the smallest bounding box in "lattice units" (where unit distance is SQRT3).
    # Then we can set S to be just large enough to make them disjoint?
    # No, S determines the physical gap.
    # If we pick a lattice config, the physical distance between centers is d_phys = d_lattice * S * SQRT3.
    # For disjointness, we need d_phys > SQRT3 (if aligned) or > 2 (if rotated 30 deg? No, min dist is SQRT3 for parallel).
    # Actually, for two unit hexagons, the minimum center distance for disjointness is SQRT3 (when edges touch).
    # So we need d_phys > SQRT3.
    # d_lattice * S * SQRT3 > SQRT3  =>  d_lattice * S > 1.
    # Since d_lattice >= 1 for distinct lattice points, we can pick S such that S * min_dist_lattice > 1.
    # To minimize outer size, we want to minimize the physical extent.
    # Physical extent = L_lattice * S * SQRT3.
    # We are free to choose S as small as possible for a given lattice config.
    # S_min = 1 / min_dist_lattice (strictly greater, so 1 + epsilon).
    # So the problem reduces to: Find a set of 11 distinct axial coordinates (q,r) such that:
    # 1. The cluster is connected/compact (to minimize bounding box).
    # 2. We calculate the "bounding radius" R_lattice of this set in lattice units.
    # 3. We calculate the min distance D_min between any two centers in lattice units.
    # 4. The required scale factor is S = 1.0 / D_min (approx).
    # 5. The final outer side length is proportional to R_lattice * S.
    # 6. Minimize R_lattice / D_min.
    
    # Actually, there's a simpler way.
    # If we fix the lattice to be "touching" (S=1), the hexagons overlap/touch.
    # The task requires disjoint.
    # If we use S=1+epsilon, the size is slightly larger.
    # But the topology is what matters most.
    # Let's assume the optimal packing is a "hexagonal number" shape or close to it.
    # For N=11, the most compact shape is a central hex + 6 surrounding + 4 more.
    # The "radius" of this cluster in lattice steps is 2.
    # The min distance is 1.
    # So S must be > 1.
    # Let's just enumerate all compact sets of 11 cells.
    
    # Generate candidate sets of 11 axial coordinates.
    # A good strategy is to start with the center (0,0) and add neighbors.
    # We can use a BFS/DFS to generate all connected sets of size 11 within a small bounding box.
    # Or, simpler: Enumerate all subsets of a small grid (e.g., -3 to 3 in q and r) 
    # that contain the origin (for symmetry) and have size 11.
    # This is C(49, 11) which is huge.
    
    # Better: Use a "growth" algorithm.
    # Start with {(0,0)}.
    # Add neighbors.
    # Prune if bounding box exceeds current best.
    
    # Let's implement a simple search for the optimal lattice shape.
    
    def axial_to_cartesian(q, r, scale=1.0):
        x = SQRT3 * scale * (q + r / 2.0)
        y = 1.5 * scale * r
        return x, y

    def get_lattice_bbox(coords):
        # Returns (max_x, max_y, min_x, min_y) in lattice units (unrotated)
        # Actually, we need the bounding box in the hexagonal coordinate system
        # to determine the outer hexagon size.
        # The outer hexagon is aligned with the lattice (angle 30 deg? or 0?).
        # Let's assume the outer hexagon is aligned with the inner lattice.
        # The "width" of a hexagonal cluster in lattice units can be measured by
        # the max projection onto the 3 main axes.
        # Axes:
        # 1. Along q-axis (angle 0 in lattice? No, q-axis is 0 deg in axial, but 0 deg in Cartesian is x-axis).
        # In Cartesian:
        # x = SQRT3 * (q + r/2)
        # y = 1.5 * r
        # The three main directions of the hex grid are:
        # 1. (1, 0) in axial -> (SQRT3, 0) in Cart. Angle 0.
        # 2. (0, 1) in axial -> (SQRT3/2, 1.5) in Cart. Angle 60.
        # 3. (-1, 1) in axial -> (-SQRT3/2, 1.5) in Cart. Angle 120.
        
        # To find the size of the containing hexagon (side S), we need the max distance
        # from the center to any vertex, projected onto the normals of the outer hexagon.
        # If the outer hexagon is aligned with the lattice, its normals are at 30, 90, 150...
        # Wait, a flat-top hexagon has normals at 90, 30, 330...
        # Let's just calculate the Cartesian coordinates for a given scale S=1 (touching),
        # find the bounding hexagon size for that cluster, and then adjust for the gap.
        
        # For a given set of axial coords, let's compute the "lattice radius" R.
        # The physical size is proportional to R.
        # The gap requirement scales the whole thing by factor 1/D_min.
        # So we minimize (R / D_min).
        
        # R can be approximated by the max distance from the centroid to any vertex in lattice units.
        # D_min is the min distance between any two centers in lattice units.
        
        xs = [q + r/2.0 for q, r in coords]
        ys = [r for q, r in coords]
        
        # Centroid
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        
        # Max distance from centroid (in lattice units, scaled by SQRT3 for x, 1.5 for y? No, let's use Euclidean in a normalized space)
        # Let's use the actual Cartesian coordinates with scale=1.
        max_r = 0
        for q, r in coords:
            x, y = axial_to_cartesian(q, r, 1.0)
            dx = x - (SQRT3 * (cx + cy/2.0)) # Centroid in Cart
            dy = y - (1.5 * cy)
            max_r = max(max_r, math.sqrt(dx*dx + dy*dy))
            
        # Min distance
        min_d = float('inf')
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                q1, r1 = coords[i]
                q2, r2 = coords[j]
                # Distance in lattice units (Euclidean in axial space is not uniform, but we can use Cartesian)
                x1, y1 = axial_to_cartesian(q1, r1, 1.0)
                x2, y2 = axial_to_cartesian(q2, r2, 1.0)
                d = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                min_d = min(min_d, d)
                
        if min_d == 0: return float('inf')
        
        return max_r / min_d

    # Search for the best set of 11 coordinates
    # We will use a randomized local search / hill climbing on the lattice.
    # Start with a compact shape.
    
    # Initial shape: 1 center, 6 neighbors, 4 more.
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Add: (2,0), (2,-1), (2,1), (1,1) -> This is a 4x3 block roughly.
    
    def generate_initial_shape():
        coords = set()
        coords.add((0,0))
        neighbors = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
        for n in neighbors:
            coords.add(n)
        # Add 4 more to make 11
        # Try to keep it compact
        extra_candidates = [
            (2,0), (2,-1), (2,1), (1,1), 
            (-2,0), (-2,1), (-2,-1), (-1,-1),
            (0,2), (0,-2), (1,2), (-1,2)
        ]
        # Just pick the first 4 that are not already in coords
        for e in extra_candidates:
            if len(coords) >= 11: break
            if e not in coords:
                coords.add(e)
        return list(coords)
        
    best_score = float('inf')
    best_coords = None
    
    # Local search: Randomly move a hexagon to an empty neighbor position
    # If it improves the score (R/D_min), keep it.
    
    current_coords = set(generate_initial_shape())
    
    # Precompute neighbors for quick lookup
    def get_neighbors(q, r):
        return [(q+1, r), (q-1, r), (q, r+1), (q, r-1), (q+1, r-1), (q-1, r+1)]
    
    iterations = 0
    max_iterations = 5000 # Should be fast
    
    while time.time() < deadline - 10.0 and iterations < max_iterations:
        iterations += 1
        # Pick a random hexagon to move
        idx = np.random.randint(0, len(current_coords))
        coord_list = list(current_coords)
        q, r = coord_list[idx]
        
        # Pick a random neighbor of (q,r) that is not occupied
        neighbors = get_neighbors(q, r)
        random.shuffle(neighbors)
        
        moved = False
        for nq, nr in neighbors:
            if (nq, nr) not in current_coords:
                # Try moving
                new_coords = set(current_coords)
                new_coords.remove((q, r))
                new_coords.add((nq, nr))
                
                score = get_lattice_bbox(list(new_coords))
                
                if score < best_score:
                    best_score = score
                    best_coords = list(new_coords)
                    current_coords = new_coords
                    moved = True
                    break
        if not moved:
            # If no move improved, maybe try a random restart?
            # Or just continue. Hill climbing can get stuck.
            # Let's add a small probability of random restart
            if random.random() < 0.01:
                current_coords = set(generate_initial_shape())
                # Re-evaluate current
                score = get_lattice_bbox(list(current_coords))
                if score < best_score:
                    best_score = score
                    best_coords = list(current_coords)

    # Use the best found lattice configuration
    if best_coords is None:
        best_coords = generate_initial_shape()
        best_score = get_lattice_bbox(best_coords)
        
    # Now, construct the final result.
    # We need to set the scale S such that the hexagons are disjoint.
    # Min distance in lattice units (Cartesian) for the best config:
    min_d_lattice = float('inf')
    for i in range(len(best_coords)):
        for j in range(i+1, len(best_coords)):
            q1, r1 = best_coords[i]
            q2, r2 = best_coords[j]
            x1, y1 = axial_to_cartesian(q1, r1, 1.0)
            x2, y2 = axial_to_cartesian(q2, r2, 1.0)
            d = math.sqrt((x1-x2)**2 + (y1-y2)**2)
            min_d_lattice = min(min_d_lattice, d)
            
    # We need physical distance > SQRT3.
    # Physical dist = d_lattice * S * SQRT3 (since axial_to_cartesian with scale 1 gives dist in units where touching is SQRT3? 
    # Let's check: axial_to_cartesian(1,0,1) = (SQRT3, 0). axial_to_cartesian(0,0,1) = (0,0). Dist = SQRT3.
    # So d_lattice is the distance in "touching units".
    # Physical dist = d_lattice * S * SQRT3? No.
    # My axial_to_cartesian function: x = SQRT3 * scale * (q + r/2).
    # If scale=1, dist between (0,0) and (1,0) is SQRT3.
    # So d_lattice (calculated above) is the distance in "touching units" * SQRT3?
    # No, the calculation above used scale=1.0, so the distances are in the same units as SQRT3.
    # So min_d_lattice is the actual Cartesian distance when scale=1.
    # We want: min_d_lattice * S > SQRT3.
    # S > SQRT3 / min_d_lattice.
    # Let S = SQRT3 / min_d_lattice * 1.001 (small gap).
    
    S = (SQRT3 / min_d_lattice) * 1.001
    
    # Now generate the final inner list
    # We need to center the cluster.
    # Calculate centroid in Cartesian with scale S.
    xs = []
    ys = []
    for q, r in best_coords:
        x, y = axial_to_cartesian(q, r, S)
        xs.append(x)
        ys.append(y)
        
    cx_avg = sum(xs) / len(xs)
    cy_avg = sum(ys) / len(ys)
    
    inner_list = []
    for i in range(len(best_coords)):
        x = xs[i] - cx_avg
        y = ys[i] - cy_avg
        # Angle: 30 degrees? 
        # The lattice is flat-top. The hexagons should be flat-top to align.
        # Flat-top hexagon vertices are at 0, 60, 120...
        # So angle_deg = 0.
        inner_list.append([x, y, 0.0])
        
    # Calculate outer side length
    # The outer hexagon should be aligned with the lattice (angle 0).
    # We need to find the smallest S_outer such that all inner vertices are inside.
    # Inner vertices are at distance 1 from their centers.
    # The "bounding radius" of the cluster of centers is R_centers.
    # The max distance from origin to any vertex is R_centers + 1.
    # For a hexagon, the apothem (distance to side) is S_outer * SQRT3 / 2.
    # The max projection of a vertex onto a normal is R_centers + 1 (approx, if vertex aligns with normal).
    # Actually, we should calculate it precisely.
    
    def calc_outer_side(inner_hexes, outer_angle_deg=0.0):
        # inner_hexes: list of (cx, cy, angle)
        # We need the max projection of any vertex onto the 6 normals of the outer hexagon.
        max_proj = 0.0
        base_angles = [30, 90, 150, 210, 270, 330] # Normals for flat-top outer hexagon
        
        for cx, cy, angle in inner_hexes:
            verts = get_hex_vertices(cx, cy, angle)
            for vx, vy in verts:
                for a in base_angles:
                    rad = math.radians(a + outer_angle_deg)
                    nx = math.cos(rad)
                    ny = math.sin(rad)
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
                        
        return 2.0 * max_proj / SQRT3
        
    side = calc_outer_side(inner_list, 0.0)
    
    # Try rotating the outer hexagon to see if we can get a smaller side?
    # The lattice has 6-fold symmetry, so rotating the outer hexagon by 30 degrees (pointy-top) might help?
    # Let's check both 0 and 30.
    side_0 = calc_outer_side(inner_list, 0.0)
    side_30 = calc_outer_side(inner_list, 30.0)
    
    if side_30 < side_0:
        side = side_30
        outer_angle = 30.0
    else:
        side = side_0
        outer_angle = 0.0
        
    return (inner_list, (0.0, 0.0), side, outer_angle)

# EVOLVE-BLOCK-END
