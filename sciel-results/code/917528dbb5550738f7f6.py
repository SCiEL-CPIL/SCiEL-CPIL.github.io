"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Place N unit regular hexagons (side length 1) inside a regular hexagon,
    minimizing the outer side length. Each inner hexagon may translate and rotate.
    
    Strategy: Use a hexagonal packing arrangement. For N hexagons, arrange them
    in a compact hexagonal cluster. The optimal packing of N unit hexagons in a
    hexagonal container is related to hexagonal numbers.
    
    For a hexagonal number N = 3k(k-1)+1, the optimal arrangement is a perfect
    hexagonal cluster of side k, which fits exactly in a hexagon of side k+1.
    
    For other N, we use the next larger hexagonal number's arrangement and place
    the hexagons in a compact cluster.
    
    The distance between centers of adjacent unit hexagons in a hexagonal packing
    is sqrt(3) (the distance between centers of two touching unit hexagons).
    
    Actually, for unit regular hexagons (side length 1), the distance between
    centers of two hexagons that are touching is sqrt(3) * 1 = sqrt(3)? 
    No - the width of a unit hexagon (distance between opposite sides) is 2 * (sqrt(3)/2) = sqrt(3).
    So two hexagons touching side-to-side have center distance sqrt(3).
    
    Wait, let me reconsider. A regular hexagon with side length s has:
    - Width (flat-to-flat): s * sqrt(3)
    - Height (point-to-point): 2s
    
    For s=1: flat-to-flat = sqrt(3), point-to-point = 2.
    
    Two unit hexagons touching on a side have centers separated by sqrt(3).
    
    In a hexagonal lattice packing, the spacing between rows is sqrt(3)/2 * sqrt(3) = 3/2? 
    No. Let me think more carefully.
    
    In a hexagonal close packing of unit hexagons:
    - The centers form a triangular lattice with nearest-neighbor distance sqrt(3).
    - The horizontal spacing between centers in a row is sqrt(3).
    - The vertical spacing between rows is sqrt(3) * sin(60°) = sqrt(3) * (sqrt(3)/2) = 3/2.
    - Adjacent rows are offset by sqrt(3)/2 horizontally.
    
    So for a hexagonal cluster of side k (meaning k hexagons on each edge of the hexagonal cluster):
    - The cluster has N = 3k(k-1) + 1 hexagons.
    - The overall bounding hexagon of the cluster has side length... 
      The distance from center to the outermost hexagon center in the radial direction is (k-1) * sqrt(3).
      Then we need to add half the "radius" of a unit hexagon in that direction.
      
    Actually, let's think about it differently. If we place N hexagons in a hexagonal cluster pattern,
    the minimum enclosing hexagon (with the same orientation) will have a certain side length.
    
    For a hexagonal cluster where hexagons are arranged with centers on a triangular lattice with
    spacing sqrt(3), the cluster of side k (k hexagons per side) has:
    - The outermost centers are at distance (k-1)*sqrt(3) from the center in the 6 main directions.
    - A unit hexagon extends sqrt(3)/2 from its center in the flat-to-flat direction, and 1 in the point-to-point direction.
    
    If we align the inner hexagons with the outer hexagon (all angles 0), then:
    - In the direction of the flat sides of the outer hexagon, the inner hexagons extend sqrt(3)/2 from their centers.
    - So the outer hexagon needs to have its flat side at distance (k-1)*sqrt(3) + sqrt(3)/2 from the center.
    - The flat-to-flat distance of the outer hexagon is 2 * [(k-1)*sqrt(3) + sqrt(3)/2] = (2k-1)*sqrt(3).
    - The side length of the outer hexagon is flat_to_flat / sqrt(3) = 2k - 1.
    
    So for a hexagonal cluster of side k, the outer hexagon side length is 2k - 1.
    
    For N = 11:
    - Hexagonal numbers: 1, 7, 19, 37, ...
    - 11 is not a hexagonal number. The next hexagonal number >= 11 is 19 (k=3).
    - But we only need to place 11 hexagons, not 19.
    - We can use a cluster of side 3 (which holds 19) but only place 11 of them.
    - The outer hexagon side would still be 2*3-1 = 5 if we use the full cluster bounds.
    - But we might be able to do better by using a more compact arrangement of just 11 hexagons.
    
    Alternative: Use a partial hexagonal cluster. Place 11 hexagons in the most compact arrangement.
    
    Let's think about what's the minimum side length for 11 unit hexagons.
    
    Lower bound: Area of 11 unit hexagons = 11 * (3*sqrt(3)/2) = 16.5*sqrt(3) ≈ 28.57.
    Area of outer hexagon with side L = (3*sqrt(3)/2) * L^2.
    So L^2 >= 16.5, L >= sqrt(16.5) ≈ 4.06.
    
    So the outer side must be at least ~4.06.
    
    Can we achieve L = 4? Area = (3*sqrt(3)/2)*16 = 24*sqrt(3) ≈ 41.57. That's enough area.
    But we need to check if 11 hexagons can actually fit in a hexagon of side 4.
    
    Let's try to construct a solution. We'll use a hexagonal lattice arrangement.
    
    For L=4, the flat-to-flat distance is 4*sqrt(3) ≈ 6.93.
    The center-to-center distance between adjacent hexagons is sqrt(3) ≈ 1.732.
    
    Let's try placing hexagons in a pattern that fits within a hexagon of side 4.
    
    I'll use a greedy/constructive approach: place hexagons in a hexagonal lattice pattern,
    starting from the center, and check if they fit within the outer hexagon.
    
    Actually, let me just try specific arrangements and compute the minimum enclosing hexagon.
    
    For simplicity, let's use a hexagonal cluster arrangement and compute the required outer side.
    
    Let's define the positions of N hexagons in a hexagonal cluster pattern, then compute
    the minimum side length of a regular hexagon (with any rotation) that contains all of them.
    
    But for now, let's use a simple approach: place hexagons in a hexagonal lattice, all with
    angle 0, and compute the bounding hexagon.
    
    Let's try N=11 with a specific arrangement:
    - Row 0 (center): 1 hexagon at (0,0)
    - Row 1: 3 hexagons
    - Row 2: 3 hexagons  
    - Row 3: 3 hexagons
    - Row 4: 1 hexagon
    Total: 11
    
    Actually, let's just use a standard hexagonal number arrangement and trim.
    
    For k=3 (hexagonal number 19), the positions are:
    For i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                if abs(i) + abs(j) + abs(k) <= 2 and i+j+k==0:
                    position...
    
    This is getting complex. Let me just implement a simple heuristic:
    1. Generate candidate positions on a hexagonal lattice.
    2. Select N positions that are most central.
    3. Compute the minimum enclosing hexagon side length.
    4. Return the arrangement.
    
    For the minimum enclosing hexagon, if all hexagons have the same orientation, we can compute
    the support function in the 6 directions and find the minimum side length.
    
    Let me implement this.
    """
    # Generate positions on a hexagonal lattice
    # Use axial coordinates: (q, r) -> Cartesian (x, y)
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    
    sqrt3 = np.sqrt(3)
    
    def axial_to_cartesian(q, r):
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        return x, y
    
    # Generate a large set of candidate positions in axial coordinates
    max_coord = 5
    candidates = []
    for q in range(-max_coord, max_coord + 1):
        for r in range(-max_coord, max_coord + 1):
            x, y = axial_to_cartesian(q, r)
            candidates.append((x, y, q, r))
    
    # Sort by distance from origin (most central first)
    candidates.sort(key=lambda c: c[0]**2 + c[1]**2)
    
    # Select N most central positions
    selected = candidates[:N]
    
    # Convert to Cartesian coordinates
    positions = [(x, y) for x, y, q, r in selected]
    
    # All hexagons have angle 0
    angles = [0.0] * N
    
    # Now compute the minimum side length of a regular hexagon (with any rotation)
    # that contains all N unit hexagons.
    # 
    # For a regular hexagon with center (cx, cy), side length L, and rotation theta,
    # a point (x, y) is inside if for all 6 directions, the projection is within L*sqrt(3)/2.
    # 
    # Actually, the condition for a point to be inside a regular hexagon with center c,
    # side length L, and rotation theta is:
    # For each of the 6 outward normals n_i (at angles theta + i*60 degrees),
    # the dot product (p - c) . n_i <= L * sqrt(3) / 2
    # 
    # The width of the hexagon in direction n_i is L * sqrt(3) (flat-to-flat distance).
    # So the half-width is L * sqrt(3) / 2.
    #
    # To find the minimum L for a given center and rotation, we compute:
    # L_min = max over all points p and all 6 directions i of:
    #   (p - c) . n_i / (sqrt(3) / 2)
    # where we take the maximum of the absolute values.
    #
    # Actually, for each direction i, we need:
    #   max_p (p - c) . n_i <= L * sqrt(3) / 2
    #   min_p (p - c) . n_i >= -L * sqrt(3) / 2
    # So L >= 2 * max_i max( max_p (p-c).n_i, -min_p (p-c).n_i ) / sqrt(3)
    #
    # But this is for the hexagon containing the centers. We also need to account for
    # the size of the unit hexagons themselves.
    #
    # A unit hexagon with center at p and rotation 0 extends:
    # - In the direction of its flat sides (which are at angles 0, 60, 120, 180, 240, 300 from horizontal
    #   if we define it that way), the extent is sqrt(3)/2 from the center.
    # - In the direction of its vertices, the extent is 1 from the center.
    #
    # This is getting complicated. Let me simplify by assuming all inner hexagons have the same
    # orientation as the outer hexagon, and compute the required outer side length.
    #
    # For a unit hexagon with side 1 and rotation 0, the support function in direction theta is:
    # h(theta) = max over vertices of (v . n)
    # where n = (cos(theta), sin(theta))
    #
    # The vertices of a unit hexagon centered at origin with rotation 0 are at:
    # (cos(60k), sin(60k)) for k=0..5, i.e., at angles 0, 60, 120, 180, 240, 300.
    #
    # The support function is h(theta) = max_k cos(theta - 60k).
    #
    # For the outer hexagon to contain all inner hexagons, we need:
    # For each inner hexagon at position p_i, and for each direction theta:
    #   (p_i . n(theta)) + h_inner(theta) <= L * h_outer(theta)
    # where h_outer(theta) = L * sqrt(3) / 2 * ... 
    #
    # This is too complex. Let me just use a numerical approach:
    # 1. Compute the center of mass of the inner hexagons.
    # 2. For a given rotation theta of the outer hexagon, compute the minimum L needed.
    # 3. Minimize over theta.
    
    # Compute center of mass
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = sum(xs) / N
    cy = sum(ys) / N
    
    # For a unit hexagon with rotation 0, the support function in direction theta is:
    def hex_support(theta, L=1.0):
        # Vertices at angles 0, 60, 120, 180, 240, 300
        max_val = 0
        for k in range(6):
            angle = k * np.pi / 3
            val = L * np.cos(theta - angle)
            if val > max_val:
                max_val = val
        return max_val
    
    # For the outer hexagon with side length L_out and rotation theta_out,
    # the support function in direction theta is:
    # h_out(theta) = L_out * sqrt(3) / 2 * |cos(theta - theta_out + 30°)| ... 
    # No, that's not right either.
    #
    # Let me just use a simpler approach: for a given outer hexagon orientation,
    # compute the maximum extent of the inner hexagons in each of the 6 normal directions
    # of the outer hexagon, and set L accordingly.
    
    best_L = float('inf')
    best_theta = 0.0
    
    for theta_deg in range(0, 30, 1):
        theta = np.radians(theta_deg)
        # The 6 outward normals of the outer hexagon
        normals = []
        for k in range(6):
            angle = theta + k * np.pi / 3
            normals.append((np.cos(angle), np.sin(angle)))
        
        # For each normal direction, compute the maximum extent of all inner hexagons
        max_extent = 0
        for nx, ny in normals:
            # For each inner hexagon, compute its support in this direction
            max_proj = -float('inf')
            for px, py in positions:
                # The inner hexagon at (px, py) with rotation 0 has support in direction (nx, ny):
                # (px, py) . (nx, ny) + hex_support_inner(theta_dir)
                theta_dir = np.arctan2(ny, nx)
                support_inner = hex_support(theta_dir, 1.0)
                proj = px * nx + py * ny + support_inner
                if proj > max_proj:
                    max_proj = proj
            # Also check the opposite direction
            min_proj = float('inf')
            for px, py in positions:
                theta_dir = np.arctan2(-ny, -nx)
                support_inner = hex_support(theta_dir, 1.0)
                proj = px * nx + py * ny - support_inner
                if proj < min_proj:
                    min_proj = proj
            
            # The total extent in this direction is max_proj - min_proj
            # But we need it relative to the center
            # Actually, let's compute relative to (cx, cy)
            max_proj_rel = -float('inf')
            min_proj_rel = float('inf')
            for px, py in positions:
                dx = px - cx
                dy = py - cy
                theta_dir = np.arctan2(ny, nx)
                support_inner = hex_support(theta_dir, 1.0)
                proj = dx * nx + dy * ny + support_inner
                if proj > max_proj_rel:
                    max_proj_rel = proj
                
                theta_dir2 = np.arctan2(-ny, -nx)
                support_inner2 = hex_support(theta_dir2, 1.0)
                proj2 = dx * nx + dy * ny - support_inner2
                if proj2 < min_proj_rel:
                    min_proj_rel = proj2
            
            extent = max(max_proj_rel, -min_proj_rel)
            if extent > max_extent:
                max_extent = extent
        
        # The half-width of the outer hexagon in the normal direction is L * sqrt(3) / 2
        # So L >= 2 * max_extent / sqrt(3)
        L = 2 * max_extent / sqrt3
        if L < best_L:
            best_L = L
            best_theta = theta
    
    # Create the inner hexagon list
    inner = []
    for i, (px, py) in enumerate(positions):
        inner.append([px, py, angles[i]])
    
    return inner, (cx, cy), best_L, np.degrees(best_theta)
# EVOLVE-BLOCK-END
