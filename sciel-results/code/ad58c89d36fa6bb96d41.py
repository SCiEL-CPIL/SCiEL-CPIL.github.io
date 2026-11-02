"""Strategy: Hexagonal Lattice Combinatorial Search."""
import numpy as np
import time
from scipy.optimize import linprog
import itertools

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s/150s limit
    
    # Geometry constants
    SQRT3 = np.sqrt(3)
    
    # Flat-top hexagon vertices for side length 1, centered at origin
    # Angles: 0, 60, 120, 180, 240, 300 degrees
    base_vertices = np.array([
        [1.0, 0.0],
        [0.5, SQRT3/2],
        [-0.5, SQRT3/2],
        [-1.0, 0.0],
        [-0.5, -SQRT3/2],
        [0.5, -SQRT3/2]
    ])

    def axial_to_cube(q, r):
        x = q
        z = r
        y = -x - z
        return (x, y, z)

    def cube_to_axial(x, y, z):
        return (x, z)

    def cube_round(x, y, z):
        rx = round(x)
        ry = round(y)
        rz = round(z)
        x_diff = abs(rx - x)
        y_diff = abs(ry - y)
        z_diff = abs(rz - z)
        if x_diff > y_diff and x_diff > z_diff:
            rx = -ry - rz
        elif y_diff > z_diff:
            ry = -rx - rz
        else:
            rz = -rx - ry
        return (rx, ry, rz)

    def axial_to_pip(q, r):
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        return (x, y)

    def pip_to_axial(x, y):
        q = (2.0 / 3.0) * x / SQRT3 - (1.0 / 3.0) * y / 1.5
        r = (2.0 / 3.0) * y / 1.5
        rx, ry, rz = cube_round(q, -q - r, r)
        return cube_to_axial(rx, ry, rz)

    def axial_distance(q1, r1, q2, r2):
        return (abs(q1 - q2) + abs(q1 + r1 - q2 - r2) + abs(r1 - r2)) // 2

    def get_hex_verts(center, angle_deg, side=1.0):
        theta = np.deg2rad(angle_deg)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return center + (side * base_vertices @ R.T)

    def point_in_hex_dist(pt, center, angle_deg, side=1.0):
        # Returns 0 if inside, positive distance if outside
        theta = np.deg2rad(angle_deg)
        R_inv = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
        local_pt = R_inv @ (pt - center)
        
        x, y = local_pt
        # Distance to each of the 6 sides
        # The hexagon is defined by |x| <= 1, |y| <= sqrt(3)/2, and |x| + |y|/sqrt(3) <= 1? 
        # Actually, the standard flat-top hexagon inequalities:
        # 1. x <= side
        # 2. -x <= side
        # 3. sqrt(3)/2*y + 1/2*x <= sqrt(3)/2 * side
        # 4. -sqrt(3)/2*y + 1/2*x <= sqrt(3)/2 * side
        # 5. sqrt(3)/2*y - 1/2*x <= sqrt(3)/2 * side
        # 6. -sqrt(3)/2*y - 1/2*x <= sqrt(3)/2 * side
        
        d1 = max(0.0, x - side)
        d2 = max(0.0, -x - side)
        d3 = max(0.0, 0.5*x + (SQRT3/2)*y - (SQRT3/2)*side)
        d4 = max(0.0, 0.5*x - (SQRT3/2)*y - (SQRT3/2)*side)
        d5 = max(0.0, -0.5*x + (SQRT3/2)*y - (SQRT3/2)*side)
        d6 = max(0.0, -0.5*x - (SQRT3/2)*y - (SQRT3/2)*side)
        
        # The actual Euclidean distance to the polygon boundary is the max of these 
        # only if the normals are unit vectors. 
        # Normals:
        # 1: (1,0) -> unit
        # 2: (-1,0) -> unit
        # 3: (0.5, sqrt(3)/2) -> unit
        # 4: (0.5, -sqrt(3)/2) -> unit
        # 5: (-0.5, sqrt(3)/2) -> unit
        # 6: (-0.5, -sqrt(3)/2) -> unit
        # So yes, max of these is the Euclidean distance to the boundary (if outside).
        return max(d1, d2, d3, d4, d5, d6)

    def is_valid_configuration(positions, S):
        """
        positions: list of (x, y) centers for N hexagons, all aligned (angle 0).
        S: outer side length.
        Returns True if all inner hexagons are inside outer and disjoint.
        """
        # 1. Check containment in outer hexagon (centered at 0,0, side S, angle 0)
        for (x, y) in positions:
            # Vertices of inner hexagon at (x,y)
            verts = get_hex_verts([x, y], 0.0, 1.0)
            for v in verts:
                if point_in_hex_dist(v, [0, 0], 0.0, S) > 1e-9:
                    return False
        
        # 2. Check disjointness
        # For two aligned unit hexagons, they are disjoint if the distance between centers
        # is at least the minimum distance required.
        # The minimum distance between centers of two disjoint aligned unit hexagons is sqrt(3) (side-to-side) 
        # or 2.0 (vertex-to-vertex)? 
        # Actually, if they touch side-to-side, the distance between centers is sqrt(3).
        # If they touch vertex-to-vertex, the distance is 2.0.
        # So if center distance > sqrt(3), they are strictly disjoint.
        # If center distance == sqrt(3), they touch (valid for "disjoint" usually meaning no interior overlap, 
        # but often packing problems allow touching). The prompt says "disjoint", which usually implies 
        # interiors are disjoint. Touching is allowed.
        # So we require dist >= sqrt(3) - epsilon.
        
        min_dist_sq = (SQRT3 - 1e-9)**2
        
        for i in range(N):
            for j in range(i+1, N):
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                dist_sq = (x1-x2)**2 + (y1-y2)**2
                if dist_sq < min_dist_sq:
                    return False
        return True

    def calculate_min_S(positions):
        """
        Calculate the minimum side length S of an outer hexagon (centered at 0,0, angle 0)
        that contains all inner hexagons at given positions.
        """
        max_dist = 0.0
        for (x, y) in positions:
            verts = get_hex_verts([x, y], 0.0, 1.0)
            for v in verts:
                # Distance from origin to point v in the "hex metric" relative to outer hexagon
                # We need the smallest S such that v is inside the hexagon of side S.
                # v is inside hex(S) iff max(d1..d6) <= 0 for that S.
                # d1 = x - S <= 0 => S >= x
                # d2 = -x - S <= 0 => S >= -x
                # d3 = 0.5x + sqrt(3)/2 y - S*sqrt(3)/2 <= 0 => S >= (0.5x + sqrt(3)/2 y) / (sqrt(3)/2)
                # d4 = 0.5x - sqrt(3)/2 y - S*sqrt(3)/2 <= 0 => S >= (0.5x - sqrt(3)/2 y) / (sqrt(3)/2)
                # d5 = -0.5x + sqrt(3)/2 y - S*sqrt(3)/2 <= 0 => S >= (-0.5x + sqrt(3)/2 y) / (sqrt(3)/2)
                # d6 = -0.5x - sqrt(3)/2 y - S*sqrt(3)/2 <= 0 => S >= (-0.5x - sqrt(3)/2 y) / (sqrt(3)/2)
                
                px, py = v
                s1 = px
                s2 = -px
                s3 = (0.5*px + (SQRT3/2)*py) / (SQRT3/2)
                s4 = (0.5*px - (SQRT3/2)*py) / (SQRT3/2)
                s5 = (-0.5*px + (SQRT3/2)*py) / (SQRT3/2)
                s6 = (-0.5*px - (SQRT3/2)*py) / (SQRT3/2)
                
                required_S = max(s1, s2, s3, s4, s5, s6)
                if required_S > max_dist:
                    max_dist = required_S
        return max_dist

    # Strategy:
    # 1. Generate a set of candidate axial coordinates within a reasonable radius.
    # 2. Use a heuristic/local search to find a set of N non-overlapping positions.
    # 3. Calculate the minimal S for that configuration.
    # 4. Try to optimize S by shifting the entire configuration or rotating it? 
    #    Since we fixed angle=0 for inner and outer, we can only translate the inner set.
    #    However, the outer is fixed at (0,0). We can translate the inner set.
    #    Actually, due to symmetry, the optimal configuration is likely centered.
    #    We will search for configurations on the lattice.

    # Candidate positions in axial coordinates
    # Radius R_axial such that Cartesian radius is sufficient.
    # For N=11, a hexagon of side 2 (radius 2) has 19 cells. We need 11.
    # Let's search in a range of axial coords [-3, 3]
    candidates = []
    for q in range(-4, 5):
        for r in range(-4, 5):
            # Optional: restrict to a circle in axial coords?
            candidates.append((q, r))
    
    # Convert to Cartesian
    cart_candidates = []
    for q, r in candidates:
        x, y = axial_to_pip(q, r)
        cart_candidates.append(((q, r), (x, y)))
    
    # We want to pick N positions from cart_candidates such that they are non-overlapping.
    # Non-overlapping condition in axial coords: axial_distance >= 2? 
    # No, axial distance 1 means sharing an edge. So they overlap.
    # Axial distance 2 means separated by one cell? No.
    # Let's check Cartesian distance.
    # If axial dist is 1, Cartesian dist is sqrt(3). They touch.
    # If axial dist is 2, Cartesian dist is either 2*sqrt(3) (along axis) or 2 (diagonal)?
    # Let's just use the Cartesian check in is_valid_configuration.
    
    # Since N=11 is small, we can use a greedy approach with randomization or local search.
    # But we can also just try to pack them into a smaller hexagonal cluster.
    # A hexagon of side 2 (radius 2) has 19 cells. 
    # A hexagon of side 1 (radius 1) has 7 cells.
    # So we need a shape slightly larger than radius 1.
    
    # Let's try to find a configuration of 11 hexes that fits in a minimal bounding hexagon.
    # We will iterate over possible "center" shifts and selections.
    
    # A better approach: 
    # The optimal packing of 11 hexes in a hexagon is likely a specific known pattern.
    # Let's try to construct it by adding hexes one by one to the densest spot.
    
    # We will use a simple local search:
    # 1. Start with a random valid set of 11 positions (or a heuristic one).
    # 2. Calculate S.
    # 3. Try to swap a position with a nearby empty one to reduce S.
    
    # Heuristic initial guess:
    # Take the 11 closest points to the origin from the lattice?
    # That would be the center + 6 neighbors + 4 of the next ring.
    
    # Let's generate a list of lattice points sorted by distance to origin.
    lattice_points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x, y = axial_to_pip(q, r)
            dist = x*x + y*y
            lattice_points.append((dist, q, r, x, y))
    
    lattice_points.sort()
    
    # The first 7 are the center and its 6 neighbors.
    # The next 12 are the ring at axial distance 2.
    # We need 11. So we take the first 7, and 4 from the next 12.
    # To minimize the bounding hexagon, we should pick the 4 that are "most inward" or symmetric.
    # Actually, any 4 from the ring of 12 might work, but symmetry helps.
    
    # Let's try all combinations of 4 from the 12 points at axial distance 2?
    # That's C(12,4) = 495 combinations. Very fast.
    
    # First, get the 7 core points.
    core_points = []
    ring2_points = []
    
    for d, q, r, x, y in lattice_points:
        ax_dist = axial_distance(0, 0, q, r)
        if ax_dist == 0:
            core_points.append((q, r, x, y))
        elif ax_dist == 1:
            core_points.append((q, r, x, y))
        elif ax_dist == 2:
            ring2_points.append((q, r, x, y))
            
    # core_points should have 7 items.
    # ring2_points should have 12 items.
    
    best_S = float('inf')
    best_positions = None
    
    # Iterate over all combinations of 4 points from ring2
    # There are 495 combos.
    # For each, form the set of 11 positions.
    # Check if they are valid (non-overlapping). 
    # Note: Any two points from the lattice with axial distance >= 2 are non-overlapping?
    # If axial dist is 1, they touch. We allow touching.
    # So any set of distinct lattice points is non-overlapping (interiors disjoint).
    # Wait, if they are on the lattice, they are centers. 
    # Two hexes centered at lattice points with axial distance 1 touch. 
    # This is valid.
    # So we just need to pick 11 distinct lattice points.
    
    # The constraint is that they must fit in the smallest possible outer hexagon.
    # The outer hexagon is centered at (0,0).
    
    # We can just evaluate the S for each combination.
    
    for combo in itertools.combinations(range(len(ring2_points)), 4):
        if time.time() > deadline:
            break
            
        selected_ring = [ring2_points[i] for i in combo]
        all_points = core_points + selected_ring
        
        # Calculate S
        S = calculate_min_S([(p[2], p[3]) for p in all_points])
        
        if S < best_S:
            best_S = S
            best_positions = all_points

    # If we found a configuration, return it.
    # We assume the optimal outer hexagon is aligned with the inner ones (angle 0).
    # The problem allows rotation. Does rotating the outer hexagon help?
    # If the inner configuration has 6-fold symmetry, rotating the outer doesn't change S.
    # If it doesn't, maybe. But the core+4 ring selection likely has some symmetry.
    # Let's assume angle 0 is optimal for this symmetric setup.
    
    if best_positions is not None:
        inner = []
        for p in best_positions:
            # p is (q, r, x, y)
            inner.append([p[2], p[3], 0.0])
        
        return inner, (0.0, 0.0), best_S, 0.0
    else:
        # Fallback: Return a valid but loose configuration
        # Just the first 11 from the sorted lattice
        inner = []
        for i in range(11):
            d, q, r, x, y = lattice_points[i]
            inner.append([x, y, 0.0])
        S = calculate_min_S([(p[0], p[1]) for p in inner])
        return inner, (0.0, 0.0), S, 0.0

# EVOLVE-BLOCK-END
