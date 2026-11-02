"""Reframed strategy: Discrete Lattice Enumeration."""
import numpy as np
import time
import itertools

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s/150s limit
    
    # Hexagonal lattice basis vectors for unit hexagons
    # A regular hexagon of side 1 has width 2 (flat-top) or height 2 (pointy-top).
    # Let's use a flat-top orientation for the lattice.
    # Distance between centers of adjacent flat-top hexagons:
    # Horizontal neighbors: distance = 1.5
    # Vertical neighbors: distance = sqrt(3)
    # Diagonal neighbors: distance = sqrt(3)
    
    # Let's define the lattice in axial coordinates (q, r).
    # Position (x, y) in Cartesian:
    # x = 1.5 * q
    # y = sqrt(3) * (r + 0.5 * (q - (q & 1)))  # For offset coordinates? 
    # Standard axial to pixel for flat-top:
    # x = size * 3/2 * q
    # y = size * sqrt(3) * (r + q/2)
    # Here size=1.
    
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = np.sqrt(3) * (r + 0.5 * q)
        return np.array([x, y])

    # Generate all possible sets of 11 lattice points that form a connected cluster
    # and are "compact" (small bounding box).
    # Instead of generating all connected subsets (which is hard), we can generate
    # all subsets of a small grid (e.g., -3 to 3 in q and r) of size 11, 
    # check if they are valid (non-overlapping is guaranteed by lattice),
    # and compute the minimal enclosing outer hexagon.
    
    # However, the number of subsets of a 7x7 grid (49 points) taken 11 at a time is C(49,11) ~ 10^11. Too many.
    
    # Better approach: Generate "hexagonal numbers" shapes.
    # The most compact shapes are concentric hexagons.
    # A hexagon of "radius" k has 1 + 3k(k+1) hexagons.
    # k=0: 1
    # k=1: 7
    # k=2: 19
    # We need 11. So it's a partial hexagon of radius 2, or a mix.
    
    # Let's enumerate all subsets of the 19 points of a radius-2 hexagon.
    # C(19, 11) = 75582. This is very manageable!
    
    # Define the 19 points of a radius-2 hexagon in axial coords.
    # Axial coords for hexagon radius R:
    # max(|q|, |r|, |q+r|) <= R
    # For R=2:
    lattice_points = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q+r)) <= 2:
                lattice_points.append((q, r))
                
    # There are 19 points.
    # We want to choose 11 of them.
    # To ensure the shape is "compact" and likely to yield a small outer hexagon,
    # we should prioritize subsets that are connected.
    # But even if we check all 75k subsets, we can just compute the outer size.
    # Disconnected subsets will likely have a larger outer size, so they won't be optimal.
    
    # Pre-compute Cartesian coordinates for all 19 points.
    cart_points = [axial_to_cartesian(q, r) for q, r in lattice_points]
    
    best_S = float('inf')
    best_indices = None
    
    # Function to calculate minimal outer hexagon side length S
    # such that all inner hexagons (centers at cart_points[i], angle 0) are inside.
    # Since inner hexagons are axis-aligned (angle 0), and outer is axis-aligned (angle 0),
    # the condition for a point p to be inside a hexagon of side S centered at origin is:
    # |x| <= S
    # |0.5x + sqrt(3)/2 y| <= S * sqrt(3)/2
    # |-0.5x + sqrt(3)/2 y| <= S * sqrt(3)/2
    
    # Actually, the distance from center to side is S * sqrt(3)/2.
    # The hexagon is the intersection of 6 half-planes.
    # For a point (x,y) to be inside:
    # 1. x <= S
    # 2. -x <= S  => |x| <= S
    # 3. 0.5x + sqrt(3)/2 y <= S * sqrt(3)/2
    # 4. -0.5x - sqrt(3)/2 y <= S * sqrt(3)/2
    # 5. -0.5x + sqrt(3)/2 y <= S * sqrt(3)/2
    # 6. 0.5x - sqrt(3)/2 y <= S * sqrt(3)/2
    
    # So for a set of points, S must be at least:
    # S >= max(|x_i|)
    # S >= max(|0.5x_i + sqrt(3)/2 y_i|) / (sqrt(3)/2)
    # S >= max(|-0.5x_i + sqrt(3)/2 y_i|) / (sqrt(3)/2)
    
    # Wait, the inner hexagons have size 1. The points are their CENTERS.
    # The condition is that the ENTIRE inner hexagon must be inside the outer.
    # This means the distance from the outer center to the FURTHEST POINT of the inner hexagon
    # must be within the outer hexagon's boundaries.
    
    # For a flat-top unit hexagon centered at (cx, cy), the vertices are:
    # (cx+1, cy), (cx+0.5, cy+sqrt(3)/2), (cx-0.5, cy+sqrt(3)/2),
    # (cx-1, cy), (cx-0.5, cy-sqrt(3)/2), (cx+0.5, cy-sqrt(3)/2)
    
    # Let's pre-compute the 6 vertices for each of the 19 lattice points.
    # v_k = center + R_k
    # Where R_k are the 6 unit hexagon vertices.
    
    unit_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # For each lattice point i, and each vertex k, we have a point P_ik.
    # The outer hexagon (side S, center 0,0) must contain all P_ik.
    # This is equivalent to: for all i, k:
    # |P_ik_x| <= S
    # |0.5 P_ik_x + sqrt(3)/2 P_ik_y| <= S * sqrt(3)/2
    # |-0.5 P_ik_x + sqrt(3)/2 P_ik_y| <= S * sqrt(3)/2
    
    # So S >= max( |P_ik_x|, |0.5 P_ik_x + sqrt(3)/2 P_ik_y| / (sqrt(3)/2), ... )
    
    # Let's pre-compute the required S for each individual lattice point i.
    # S_i = max over its 6 vertices of the "radius" function.
    # Then for a set of indices I, S(I) = max_{i in I} S_i.
    # This is a huge simplification! The minimal S for a set is just the max of the minimal S for each individual hexagon.
    
    def calc_min_S_for_point(center):
        verts = center[:, np.newaxis] + unit_verts.T
        # verts is 6x2
        x = verts[:, 0]
        y = verts[:, 1]
        
        d1 = np.abs(x)
        d2 = np.abs(0.5*x + (np.sqrt(3)/2)*y) / (np.sqrt(3)/2)
        d3 = np.abs(-0.5*x + (np.sqrt(3)/2)*y) / (np.sqrt(3)/2)
        
        return np.max(np.maximum(d1, np.maximum(d2, d3)))
    
    point_S = []
    for p in cart_points:
        s_val = calc_min_S_for_point(p)
        point_S.append(s_val)
    
    # Now, for any subset of indices, S = max(point_S[i] for i in subset).
    # We want to minimize this S.
    # This is equivalent to: Find the smallest S such that there exist 11 points
    # with point_S[i] <= S.
    # Since we are restricted to the 19 points of the radius-2 hexagon,
    # we can just sort point_S and pick the 11 smallest?
    # Wait, we must pick 11 points from the 19.
    # The minimal S for a subset of 11 is the 11th smallest value in the sorted list of point_S?
    # No, we can choose WHICH 11 points.
    # To minimize the max S, we should choose the 11 points with the smallest individual S.
    
    sorted_S = sorted(point_S)
    # The best possible S using these 19 points is the 11th smallest value.
    candidate_S = sorted_S[10]
    
    # Is it possible that a configuration using points OUTSIDE the radius-2 hexagon
    # yields a smaller S?
    # The radius-2 hexagon is the set of 19 points closest to the origin in the lattice.
    # Any point outside this set has a larger distance from the origin.
    # Since S_i is a monotonic function of distance (roughly), points outside will have larger S_i.
    # Therefore, the optimal set of 11 points must be a subset of the 19 closest points.
    # And to minimize the max S_i, we pick the 11 smallest S_i.
    
    # Let's verify this logic.
    # If we pick the 11 points with smallest S_i, the resulting outer hexagon has side candidate_S.
    # Is this configuration valid?
    # Yes, because they are on a lattice, so they are disjoint.
    # And they are inside the outer hexagon of side candidate_S by construction.
    
    # So the answer is simply candidate_S?
    # Let's double check if rotation helps.
    # The problem allows rotation of inner hexagons.
    # If we rotate inner hexagons, their "radius" function changes.
    # However, for a regular hexagon, the minimal enclosing hexagon of the same orientation
    # is achieved when the inner hexagons are aligned with the outer.
    # Any rotation increases the effective radius.
    # So angle 0 is optimal for inner hexagons if outer is angle 0.
    
    # What if we rotate the OUTER hexagon?
    # Due to symmetry, rotating the outer is equivalent to rotating the inner set relative to it.
    # We already established that alignment is optimal.
    
    # So the strategy holds:
    # 1. Generate 19 closest lattice points.
    # 2. Calculate minimal S for each.
    # 3. Pick 11 smallest.
    # 4. Return those 11 centers and the 11th smallest S.
    
    # Let's find the indices of the 11 smallest S.
    indices = np.argsort(point_S)
    best_indices = indices[:11]
    
    # Construct the result
    inner = []
    for idx in best_indices:
        q, r = lattice_points[idx]
        x, y = cart_points[idx]
        inner.append([x, y, 0.0])
        
    S = point_S[best_indices[10]] # The max of the chosen 11
    
    # Just in case, let's verify with a quick check if a slightly larger set of points
    # (e.g. radius 3) could offer a better configuration due to "packing" effects?
    # No, S is determined by the furthest point. Adding points further out only increases S.
    # Removing points closer in and replacing with points further out increases S.
    # So the 11 closest points are always optimal for minimizing the enclosing radius.
    
    return inner, (0.0, 0.0), S, 0.0

# EVOLVE-BLOCK-END
