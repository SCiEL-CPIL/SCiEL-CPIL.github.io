"""
Strategy: Global Deterministic Hexagonal Lattice Packing with Convex Outer Minimization.
Instead of stochastic local search with overlap penalties, we exploit the fact that
regular hexagons pack most densely on a hexagonal lattice (all aligned). We construct
the densest 12-hexagon cluster (1 center, 6 ring 1, 5 ring 2) and then solve for the
minimal enclosing regular hexagon using:
1. A ternary search over the outer hexagon's rotation angle (0 to 60 degrees).
2. For each angle, a convex optimization (minimizing max projection) to find the 
   optimal center of the outer hexagon relative to the cluster.
This avoids the pitfalls of SA (getting stuck in local minima, slow convergence, 
penalty tuning) and directly optimizes the true objective for the best structural 
candidate.
"""
import numpy as np
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # 1. Generate the densest 12-hexagon cluster on a hexagonal lattice
    #    Hexagons are aligned (angle 0).
    #    Coordinates in axial-like system converted to Cartesian.
    #    Let's use standard Cartesian for a "flat-top" or "pointy-top" lattice.
    #    For aligned hexagons, the distance between centers is 2 * apothem = sqrt(3) * side?
    #    No, for side length 1, the distance between touching centers is 2 * apothem = sqrt(3).
    #    Wait, apothem = sqrt(3)/2. Distance between parallel sides is sqrt(3).
    #    So center-to-center distance for touching is sqrt(3).
    
    # Let's define the lattice vectors for a hexagonal packing of hexagons.
    # If hexagons are oriented with a flat side up (angle 0 in our convention where vertices are at 0, 60...),
    # the normals to sides are at 30, 90, 150...
    # The distance to a neighbor in direction 0 (along x-axis) is 2 * apothem = sqrt(3).
    # The distance to a neighbor in direction 60 is also sqrt(3).
    
    # Let's generate positions.
    # Center at (0,0).
    positions = [(0.0, 0.0)]
    
    # Ring 1: 6 neighbors
    # Angles 0, 60, 120, 180, 240, 300
    # Distance d = sqrt(3)
    d1 = math.sqrt(3)
    for k in range(6):
        angle = math.radians(60 * k)
        x = d1 * math.cos(angle)
        y = d1 * math.sin(angle)
        positions.append((x, y))
        
    # Ring 2: We need 5 more.
    # In a hex lattice, the second ring has 12 positions.
    # We pick 5 that are symmetric or compact.
    # The 12 positions in ring 2 are at:
    # (2*d1, 0), (d1, d1*sqrt(3)), (0, 2*d1), (-d1, d1*sqrt(3)), ... etc.
    # Let's list them properly.
    # Axial coordinates (q, r) where x = d1 * (q + r/2), y = d1 * sqrt(3)/2 * r
    # Wait, standard hex grid coords:
    # x = size * 3/2 * q
    # y = size * sqrt(3) * (r + q/2)
    # For touching hexagons, size=1, distance is sqrt(3).
    # Let's just use the 12 ring 2 spots and pick 5.
    # Spots:
    # (2, 0), (1, 1), (0, 2), (-1, 1), (-2, 0), (-1, -1), (0, -2), (1, -1) -> this is 8? No.
    # Ring 2 has 12 spots.
    # (2,0), (2,-1), (1,-2), (0,-2), (-1,-2), (-2,-1), (-2,0), (-2,1), (-1,2), (0,2), (1,2), (2,1)
    # Let's pick 5 that form a compact shape.
    # Maybe (2,0), (1,1), (0,2), (-1,1), (-2,0)? This is 5.
    # Or symmetric? 5 is odd, so perfect symmetry is hard.
    # Let's pick 5 that minimize the bounding box.
    # (2,0), (0,2), (-2,0), (-1,-1), (1,-1) ?
    
    # Let's generate all 12 ring 2 positions and select the 5 that minimize the area.
    # Actually, for 12 hexagons, the optimal packing is likely the "hexagonal number" shape minus 3.
    # Hexagonal numbers: 1, 7, 19.
    # 12 is 7 + 5.
    # So 1 center + 6 ring 1 + 5 from ring 2.
    # To minimize the enclosing hexagon, we should pick the 5 ring-2 hexagons that are most "central" or symmetric.
    # Let's try all combinations of 5 from 12? C(12,5) = 792. That's small.
    # But we also need to consider the outer angle.
    
    # Let's hardcode a good set of 5.
    # The 12 ring 2 positions in Cartesian (with d1 = sqrt(3)):
    # (2*d1, 0)
    # (d1, d1*sqrt(3))  -> (sqrt(3), 3)
    # (0, 2*d1*sqrt(3)/2 * 2?) No.
    # Let's use the axial to cartesian formula:
    # x = d1 * (q + r/2)
    # y = d1 * (sqrt(3)/2) * r
    # Ring 2 axial coords:
    # (2, 0), (2, -1), (1, -2), (0, -2), (-1, -2), (-2, -1), (-2, 0), (-2, 1), (-1, 2), (0, 2), (1, 2), (2, 1)
    
    ring2_axial = [
        (2, 0), (2, -1), (1, -2), (0, -2), (-1, -2), (-2, -1),
        (-2, 0), (-2, 1), (-1, 2), (0, 2), (1, 2), (2, 1)
    ]
    
    ring2_cart = []
    for q, r in ring2_axial:
        x = d1 * (q + r/2.0)
        y = d1 * (math.sqrt(3)/2.0) * r
        ring2_cart.append((x, y))
        
    # We need to choose 5 from these 12.
    # To minimize the enclosing hexagon, we should balance the cluster.
    # Let's try a symmetric selection if possible, or near-symmetric.
    # Since 5 is odd, we can't have full 6-fold symmetry.
    # Let's pick 5 that cover 5 of the 6 "directions".
    # E.g., exclude one direction.
    # Directions correspond to the 6 "corners" of the ring 2 hexagon.
    # The 12 points form a hexagon.
    # Let's just pick 5 consecutive ones? Or spread out?
    # Spread out is better for compactness.
    # Let's pick indices 0, 2, 4, 6, 8 (every other one)? That's 5 points.
    # Or 0, 3, 5, 8, 10?
    
    # Let's try a few specific sets and pick the best one.
    # Set A: 0, 2, 4, 6, 8
    # Set B: 0, 2, 5, 7, 9
    # Set C: 0, 3, 5, 8, 10
    
    candidate_sets = [
        [0, 2, 4, 6, 8],
        [0, 2, 5, 7, 9],
        [0, 3, 5, 8, 10],
        [0, 1, 3, 6, 9],
        [0, 2, 4, 7, 9]
    ]
    
    best_cluster = None
    best_side = float('inf')
    
    def min_enclosing_hexagon_side(points, outer_angle_deg):
        """
        Given a set of points (vertices of inner hexagons) and an outer angle,
        find the minimum side length of a regular hexagon with that angle
        that contains all points. The center of the outer hexagon is optimized.
        """
        # points: Nx2 array
        # outer_angle_deg: angle of outer hexagon
        
        # The outer hexagon is defined by 6 half-planes.
        # For a regular hexagon of side S and angle theta, the apothem is a = S * sqrt(3) / 2.
        # The normals to the sides are at angles theta + 30, theta + 90, ...
        # The constraint is n_i . (p - c) <= a for all i, p.
        # This is equivalent to n_i . p - n_i . c <= a.
        # We want to minimize S, which is proportional to a.
        # So we want to minimize a such that there exists a center c satisfying:
        # n_i . c >= n_i . p - a for all i, p.
        # This is a linear programming problem:
        # Minimize a
        # Subject to:
        # n_i . c + a >= max_p(n_i . p)  for all i
        # -n_i . c + a >= max_p(-n_i . p) for all i (since the hexagon is symmetric, we need to contain points in both directions)
        # Actually, the hexagon is { p | |n_i . (p-c)| <= a }? No, that's for a rhombus?
        # A regular hexagon is the intersection of 6 half-planes.
        # The constraints are n_i . (p - c) <= a for i=1..6.
        # Note that n_{i+3} = -n_i.
        # So we have n_i . (p - c) <= a and -n_i . (p - c) <= a, which means |n_i . (p - c)| <= a.
        # So for each axis i (3 axes), the projection of all points onto n_i must lie within [c_i - a, c_i + a].
        # The width of the projection is w_i = max_p(n_i . p) - min_p(n_i . p).
        # We need 2a >= w_i for all i.
        # So a >= max_i(w_i / 2).
        # To minimize a, we should center c such that the projections are centered?
        # Wait, if we can choose c, then for each axis i, the interval [min_p(n_i . p), max_p(n_i . p)]
        # must be contained in [n_i . c - a, n_i . c + a].
        # This requires 2a >= max_p(n_i . p) - min_p(n_i . p).
        # The value of a is determined by the maximum width over the 3 axes.
        # The center c can be chosen freely to satisfy the constraints?
        # Yes, as long as 2a >= w_i for all i, we can always find a c_i such that the interval is contained.
        # Specifically, c_i can be the midpoint of the projection interval.
        # So the minimum a is max_i(w_i / 2).
        # And S = 2a / sqrt(3).
        
        base_normals = np.radians(np.arange(30, 360, 60))
        outer_rad = np.radians(outer_angle_deg)
        normals = base_normals + outer_rad
        
        # We only need 3 unique axes (opposing normals are redundant for width)
        # But let's just compute width for all 6, it's the same.
        max_width = 0.0
        for na in normals[:3]:  # 30, 90, 150
            nx = np.cos(na)
            ny = np.sin(na)
            projs = points[:, 0] * nx + points[:, 1] * ny
            width = np.max(projs) - np.min(projs)
            if width > max_width:
                max_width = width
                
        a = max_width / 2.0
        S = 2 * a / math.sqrt(3)
        
        # Now we need to find the center c.
        # For each axis, c_i is the midpoint of the projection.
        # c is the intersection of these midpoints?
        # Actually, c is a single point (cx, cy).
        # The projection of c onto n_i is n_i . c.
        # We want n_i . c = (max_p(n_i . p) + min_p(n_i . p)) / 2 for all i?
        # This is a system of linear equations:
        # n_1 . c = m_1
        # n_2 . c = m_2
        # n_3 . c = m_3
        # Where m_i is the midpoint of the projection.
        # This system may be overdetermined (3 equations, 2 unknowns).
        # We should solve it in a least-squares sense.
        
        mids = np.zeros(3)
        for i, na in enumerate(normals[:3]):
            nx = np.cos(na)
            ny = np.sin(na)
            projs = points[:, 0] * nx + points[:, 1] * ny
            mids[i] = (np.max(projs) + np.min(projs)) / 2.0
            
        # Solve A c = m
        A = np.array([
            [np.cos(normals[0]), np.sin(normals[0])],
            [np.cos(normals[1]), np.sin(normals[1])],
            [np.cos(normals[2]), np.sin(normals[2])]
        ])
        
        c, residuals, rank, sv = np.linalg.lstsq(A, mids, rcond=None)
        
        return S, c[0], c[1]

    # For each candidate set of 5 ring-2 hexagons, compute the vertices and find min S
    for cset in candidate_sets:
        cluster_positions = positions.copy()
        for idx in cset:
            cluster_positions.append(ring2_cart[idx])
            
        # Generate all vertices
        all_verts = []
        for cx, cy in cluster_positions:
            # Hexagon vertices for side 1, angle 0
            angles = np.radians(np.arange(0, 360, 60))
            verts = np.column_stack([
                cx + np.cos(angles),
                cy + np.sin(angles)
            ])
            all_verts.append(verts)
            
        points = np.vstack(all_verts)
        
        # Ternary search for outer angle
        # The function S(theta) is likely unimodal or has few local minima.
        # Let's scan 0 to 60 degrees.
        best_theta = 0.0
        best_S = float('inf')
        
        # Coarse scan
        for theta in np.arange(0, 60, 1.0):
            S, cx, cy = min_enclosing_hexagon_side(points, theta)
            if S < best_S:
                best_S = S
                best_theta = theta
                
        # Fine scan
        for theta in np.arange(best_theta - 1.0, best_theta + 1.0, 0.01):
            S, cx, cy = min_enclosing_hexagon_side(points, theta)
            if S < best_S:
                best_S = S
                best_theta = theta
                
        if best_S < best_side:
            best_side = best_S
            best_cluster = cluster_positions
            best_angle = best_theta
            # Store center? We need to recompute it for the final result.
            
    # Finalize with the best cluster
    # Recompute center and side for the best cluster and angle
    all_verts = []
    for cx, cy in best_cluster:
        angles = np.radians(np.arange(0, 360, 60))
        verts = np.column_stack([
            cx + np.cos(angles),
            cy + np.sin(angles)
        ])
        all_verts.append(verts)
        
    points = np.vstack(all_verts)
    
    side, center_x, center_y = min_enclosing_hexagon_side(points, best_angle)
    
    # Construct inner list
    # The inner positions are relative to the global coordinate system.
    # The outer center is (center_x, center_y).
    # The inner positions are (cx, cy) as computed.
    # We need to return inner positions and outer center.
    # The interface expects inner positions in the same coordinate system as the outer center.
    # So we just return the (cx, cy, angle) for each inner hexagon.
    
    inner = []
    for cx, cy in best_cluster:
        inner.append((cx, cy, 0.0))  # All aligned
        
    center = (center_x, center_y)
    angle = best_angle
    
    # Validation
    # Check if all inner hexagons are inside the outer one.
    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        dx = px - cx
        dy = py - cy
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        
        apothem = side * np.sqrt(3) / 2
        normal_angles = np.radians(np.arange(30, 360, 60))
        
        for na in normal_angles:
            nx = np.cos(na)
            ny = np.sin(na)
            dist = local_x * nx + local_y * ny
            if dist > apothem + 1e-9:
                return False
        return True

    valid = True
    for cx, cy, ang in inner:
        angles = np.radians(np.arange(0, 360, 60))
        verts = np.column_stack([
            cx + np.cos(angles),
            cy + np.sin(angles)
        ])
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, center[0], center[1], side, angle):
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback: expand side slightly
        side *= 1.01

    return inner, center, side, angle

# EVOLVE-BLOCK-END
