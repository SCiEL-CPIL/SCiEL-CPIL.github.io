"""
Packing 11 unit regular hexagons inside a regular hexagon.

Strategy:
1. Enumerate a set of compact lattice-based topologies for the 11 hexagons.
   Each topology is defined by relative positions in the hexagonal lattice
   (using axial coordinates) and a set of allowed rotations for each hexagon
   (0, 60, 120, 180, 240, 300 degrees - though for regular hexagons, only
   0 and 60 matter due to symmetry, but we can allow arbitrary angles in
   optimization if needed. However, the problem says "each inner hexagon may
   translate and rotate", so we should allow arbitrary rotation angles.
   
   Wait, the current program uses fixed positions and angles. The score is
   the outer side length. We need to minimize the outer side length such that
   all 11 unit hexagons fit inside.

   The current program places them in a line with side 40.0, which is huge.
   We need a much tighter packing.

   Let's think about the geometry.
   A regular hexagon of side length s has a circumradius R = s.
   The distance from center to vertex is s.
   The distance from center to edge (apothem) is s * sqrt(3)/2.

   If we pack hexagons in a hexagonal lattice, they touch each other.
   The distance between centers of touching hexagons is 2 * apothem = sqrt(3) * s_inner.
   Here s_inner = 1, so distance between centers is sqrt(3).

   We need to fit 11 such hexagons.
   A central hexagon surrounded by 6 others makes 7.
   Then 4 more.
   Total 11.

   Let's define the positions in a local coordinate system where the "lattice"
   spacing is sqrt(3).
   
   Axial coordinates (q, r) for a hexagon lattice:
   Center: (0,0)
   Neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
   
   Let's try a cluster:
   1. (0,0)
   2. (1,0)
   3. (0,1)
   4. (-1,1)
   5. (-1,0)
   6. (0,-1)
   7. (1,-1)
   This is the 7-hexagon flower.
   
   Now add 4 more.
   Possible next layer positions (distance 2 in lattice steps, but not all are adjacent to the flower without gaps? No, the flower is compact).
   The next shell around the flower has 12 positions.
   We need to pick 4 that keep the shape compact.
   
   Let's list the 12 outer positions:
   (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-1) -- wait, let's be precise.
   
   Axial coords (q, r):
   (2,0)
   (1,1)
   (0,2)
   (-1,2)
   (-2,2)
   (-2,1)
   (-2,0)
   (-1,-1)
   (0,-2)
   (1,-2)
   (2,-1)
   (2,-1) is a duplicate in my list? No.
   
   Let's generate neighbors of the outer ring of the flower.
   Flower outer ring: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
   Neighbors of (1,0) not in flower: (2,0), (1,1), (2,-1).
   Neighbors of (0,1) not in flower: (1,1), (0,2), (-1,2).
   Neighbors of (-1,1) not in flower: (0,2), (-1,2), (-2,2), (-2,1).
   Neighbors of (-1,0) not in flower: (-2,1), (-2,0), (-2,-1)?? No, (-1,0) neighbors are (0,0)[in], (-1,1)[in], (-2,1), (-2,0), (-1,-1), (0,-1)[in]. So (-2,1), (-2,0), (-1,-1).
   Neighbors of (0,-1) not in flower: (-1,-1), (-1,-2), (0,-2), (1,-2).
   Neighbors of (1,-1) not in flower: (0,-2), (1,-2), (2,-1), (2,-2)?? No, (1,-1) neighbors: (0,-1)[in], (1,0)[in], (2,0), (2,-1), (1,-2), (0,-2). So (2,0), (2,-1), (1,-2), (0,-2).
   
   Unique outer positions:
   (2,0)
   (1,1)
   (0,2)
   (-1,2)
   (-2,2)
   (-2,1)
   (-2,0)
   (-1,-1)
   (0,-2)
   (1,-2)
   (2,-1)
   (2,-2) is not a neighbor of the flower? (1,-1) has neighbor (2,-1). (2,0) has neighbor (2,-1).
   Let's check (2,-2). Distance from (1,-1) is 1? (2,-2) - (1,-1) = (1,-1). Yes. But is (2,-2) adjacent to the flower?
   (1,-1) is in the flower. So (2,-2) is a neighbor of (1,-1).
   So (2,-2) is also an outer position.
   
   List:
   1. (2,0)
   2. (1,1)
   3. (0,2)
   4. (-1,2)
   5. (-2,2)
   6. (-2,1)
   7. (-2,0)
   8. (-1,-1)
   9. (0,-2)
   10. (1,-2)
   11. (2,-1)
   12. (2,-2)
   
   Wait, is (2,-2) valid?
   Axial distance from center (0,0) to (2,-2) is max(|2|, |-2|, |2-(-2)|) = 4? No, axial distance is (|q| + |r| + |q+r|)/2.
   (2,-2): (2+2+0)/2 = 2.
   (2,0): (2+0+2)/2 = 2.
   (1,1): (1+1+2)/2 = 2.
   (0,2): (0+2+2)/2 = 2.
   (-1,2): (1+2+1)/2 = 2.
   (-2,2): (2+2+0)/2 = 2.
   (-2,1): (2+1+1)/2 = 2.
   (-2,0): (2+0+2)/2 = 2.
   (-1,-1): (1+1+2)/2 = 2.
   (0,-2): (0+2+2)/2 = 2.
   (1,-2): (1+2+1)/2 = 2.
   (2,-1): (2+1+1)/2 = 2.
   (2,-2): (2+2+0)/2 = 2.
   
   So there are 12 positions in the second ring.
   We need to choose 4 of them to minimize the bounding box (in terms of the enclosing hexagon side length).
   
   The enclosing hexagon must contain all 11 hexagons.
   If we assume the outer hexagon is aligned with the lattice (rotation 0), we can calculate the required side length.
   
   Let's convert axial (q, r) to Cartesian (x, y) for the centers.
   Let the distance between centers be D = sqrt(3).
   x = D * (q + r/2)
   y = D * (sqrt(3)/2 * r)
   
   We need to find the minimum side length S of a regular hexagon (centered at some point, with some rotation) that contains all the vertices of the 11 inner hexagons.
   Actually, it's easier to say: The inner hexagons must be inside the outer hexagon.
   This is equivalent to: The maximum distance from the outer hexagon center to any vertex of any inner hexagon, projected onto the 6 normal directions of the outer hexagon, must be <= S * apothem_outer?
   No.
   
   Let's use the property: A point P is inside a regular hexagon of side S centered at C with rotation theta if and only if:
   For k in 0..5:
   Dot(P - C, n_k) <= S * sqrt(3)/2
   where n_k are the unit normal vectors to the sides.
   
   We want to minimize S.
   For a fixed set of inner hexagon positions and rotations, and a fixed outer hexagon rotation, the minimum S is determined by the "widest" projection of the union of inner hexagons.
   
   Let U be the union of the 11 inner hexagons.
   The minimum S is such that U is contained in the outer hexagon.
   This is equivalent to finding the minimum circumradius of a hexagon? No, it's the minimum side length of a circumscribing hexagon.
   
   For a convex set U, the minimum side length S of a circumscribing regular hexagon (with fixed orientation) is:
   S = max_{k=0..5} ( max_{p in U} Dot(p, n_k) - min_{p in U} Dot(p, n_k) ) / (2 * sqrt(3)/2) ?
   No, the outer hexagon is centered.
   If the outer hexagon is centered at C, then for each direction n_k, the maximum extent of U in that direction must be <= S * sqrt(3)/2.
   So S >= max_{k} ( max_{p in U} Dot(p-C, n_k) ).
   But we can also shift C to minimize this maximum.
   This is the Chebyshev center problem for a hexagon?
   Actually, for a fixed orientation, the optimal center C is the midpoint of the bounding box of U in the dual sense?
   
   Let's simplify.
   We will define the topologies (sets of axial coordinates).
   For each topology, we will assume the inner hexagons are aligned with the outer hexagon (rotation 0).
   We will calculate the required S.
   Then we will try a few rotations of the entire cluster relative to the outer hexagon to see if it improves.
   
   Let's implement a function to calculate the minimum S for a given set of points (vertices of inner hexagons) and outer hexagon orientation.
   
   Steps:
   1. Generate a few candidate topologies (sets of 11 axial coords).
   2. For each topology:
      a. Calculate Cartesian coordinates of the 11 centers.
      b. Generate the 6 vertices for each center (assuming rotation 0 for inner hexagons initially).
      c. Collect all 66 vertices.
      d. For a given outer hexagon rotation theta:
         - Calculate the 6 normal vectors.
         - Find the center C that minimizes the maximum distance to the boundaries.
           This is equivalent to finding the minimum radius R such that all points are within distance R from C in the hexagonal norm.
           Or simpler:
           For each normal n_k, let m_k = max_{p} Dot(p, n_k) and M_k = min_{p} Dot(p, n_k).
           The width in direction n_k is m_k - M_k.
           The side length S must satisfy S * sqrt(3) >= m_k - M_k for all k?
           No, the outer hexagon has width 2 * S * sqrt(3)/2 = S * sqrt(3) across the flats.
           So S * sqrt(3) >= m_k - M_k.
           S >= (m_k - M_k) / sqrt(3).
           So S_min = max_k ( (m_k - M_k) / sqrt(3) ).
           And the center C should be at the midpoint of the projections?
           C_k = (m_k + M_k) / 2.
           For a regular hexagon, the center is the same for all directions.
           So we need to find C such that C is close to all C_k.
           Actually, if we fix the orientation, the optimal C is the average of the midpoints? Or we can just compute the bounding box in the 6 directions and take the max width.
           The center that minimizes the max distance is the center of the minimum enclosing circle? No, it's the center of the hexagon.
           For a fixed orientation, the minimal S is determined by the maximum width of the set of points in the 6 normal directions.
           S = max_k ( width_k / sqrt(3) ).
           The center C can be chosen as the centroid of the points, or more precisely, the point that equalizes the margins.
           However, for the purpose of finding a good packing, we can just compute S for a specific C (e.g., centroid) and then optimize C.
           
   3. We can also allow the inner hexagons to rotate. But let's start with aligned.

   Let's write the code to evaluate a topology.
"""

import numpy as np
import time
import itertools

N = 11

# EVOLVE-BLOCK-START

def hexagon_vertices(center, side, rotation):
    """Generate 6 vertices of a regular hexagon."""
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, 7) + np.deg2rad(rotation)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def min_enclosing_hexagon_side(points, rotation):
    """
    Calculate the minimum side length of a regular hexagon with given rotation
    that encloses all points.
    points: Nx2 array.
    rotation: in degrees.
    Returns: side length and center (cx, cy).
    """
    theta = np.deg2rad(rotation)
    # Normal vectors to the sides
    # Side normals are at angles theta + 30 + 60*k
    # Wait, a regular hexagon with a vertex at angle theta has sides with normals at theta + 30 + 60*k?
    # Let's check.
    # If vertex is at 0, side is from 0 to 60. Midpoint angle is 30. Normal is 30.
    # So normals are at 30 + 60*k.
    # If we rotate the hexagon by theta, the vertices are at theta + 60*k.
    # The normals are at theta + 30 + 60*k.
    
    angles = np.deg2rad(30 + theta) + np.deg2rad(60) * np.arange(6)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Project points onto normals
    # proj: 6 x N
    proj = points @ normals.T
    
    mins = np.min(proj, axis=1)
    maxs = np.max(proj, axis=1)
    widths = maxs - mins
    
    # The width of the hexagon across the flats is S * sqrt(3).
    # So S * sqrt(3) >= width_k for all k.
    S = np.max(widths) / np.sqrt(3)
    
    # The center C should be such that the projection of C onto normal k is (min_k + max_k) / 2.
    # Let C = (cx, cy).
    # cx * cos(a_k) + cy * sin(a_k) = mid_k
    # This is a system of 6 equations for 2 unknowns. We can solve it in a least squares sense.
    mids = (mins + maxs) / 2
    
    # normals is 6x2. We want C such that normals @ C = mids.
    C, residuals, rank, s = np.linalg.lstsq(normals, mids, rcond=None)
    cx, cy = C
    
    return S, (cx, cy)

def evaluate_topology(axial_coords, inner_rotations, outer_rotation):
    """
    axial_coords: list of (q, r)
    inner_rotations: list of rotation angles for each hexagon
    outer_rotation: rotation of the outer hexagon
    Returns: side length, inner_positions, outer_center
    """
    D = np.sqrt(3)
    centers = []
    for q, r in axial_coords:
        x = D * (q + r / 2.0)
        y = D * (np.sqrt(3) / 2.0 * r)
        centers.append((x, y))
    
    all_points = []
    for i, (cx, cy) in enumerate(centers):
        verts = hexagon_vertices((cx, cy), 1.0, inner_rotations[i])
        all_points.append(verts)
    
    points = np.vstack(all_points)
    S, center = min_enclosing_hexagon_side(points, outer_rotation)
    
    # Construct the return format
    inner = [[centers[i][0], centers[i][1], inner_rotations[i]] for i in range(len(centers))]
    return S, inner, center

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Define some topologies
    # Topology 1: 7-hexagon flower + 4 from the next ring
    # Flower:
    flower = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    # Outer ring candidates
    outer_ring = [
        (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), 
        (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2)
    ]
    
    best_S = float('inf')
    best_inner = None
    best_center = (0,0)
    best_outer_rot = 0
    
    # Try all combinations of 4 from the outer ring
    # There are C(12,4) = 495 combinations.
    # For each, try a few outer rotations.
    
    combos = list(itertools.combinations(range(12), 4))
    
    for combo in combos:
        if time.time() > deadline:
            break
            
        extra = [outer_ring[i] for i in combo]
        axial_coords = flower + extra
        
        # Try a few outer rotations
        for outer_rot in [0, 15, 30, 45, 60]:
            if time.time() > deadline:
                break
                
            # Inner rotations: assume all 0
            inner_rots = [0] * 11
            S, inner, center = evaluate_topology(axial_coords, inner_rots, outer_rot)
            
            if S < best_S:
                best_S = S
                best_inner = inner
                best_center = center
                best_outer_rot = outer_rot
    
    # If we have time, try to optimize the inner rotations for the best topology found?
    # Or try other topologies?
    
    # Let's also try a linear arrangement? No, hexagonal is better.
    
    # What if we use a different base topology?
    # E.g., a 2x2 block?
    # Or a 3x3 grid?
    
    # Let's try a 3x3 grid of hexagons (9 hexagons) + 2 more.
    # 3x3 grid in axial coords:
    # (0,0), (1,0), (2,0)
    # (0,1), (1,1), (2,1)
    # (0,2), (1,2), (2,2)
    # This is 9.
    # Add (0,-1) and (3,0)?
    
    grid33 = [
        (0,0), (1,0), (2,0),
        (0,1), (1,1), (2,1),
        (0,2), (1,2), (2,2)
    ]
    
    # Add 2 more to minimize bounding box.
    # Candidates near the grid:
    # (-1,0), (-1,1), (-1,2)
    # (3,0), (3,1), (3,2)
    # (0,-1), (1,-1), (2,-1)
    # (0,3), (1,3), (2,3)
    
    extra_candidates = [
        (-1,0), (-1,1), (-1,2),
        (3,0), (3,1), (3,2),
        (0,-1), (1,-1), (2,-1),
        (0,3), (1,3), (2,3)
    ]
    
    for combo in itertools.combinations(range(len(extra_candidates)), 2):
        if time.time() > deadline:
            break
            
        extra = [extra_candidates[i] for i in combo]
        axial_coords = grid33 + extra
        
        for outer_rot in [0, 15, 30, 45, 60]:
            if time.time() > deadline:
                break
                
            inner_rots = [0] * 11
            S, inner, center = evaluate_topology(axial_coords, inner_rots, outer_rot)
            
            if S < best_S:
                best_S = S
                best_inner = inner
                best_center = center
                best_outer_rot = outer_rot

    # If best_inner is None, fall back to a simple line
    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_S, best_outer_rot

# EVOLVE-BLOCK-END
