"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # 1. Determine the shape of the hexagonal cluster of N unit hexagons.
    # A centered hexagonal number H_k = 3k(k+1) + 1 fits in a hexagon of side k+1.
    # We need to find the smallest k such that H_k >= N.
    # Then we fill the positions in that hexagonal grid, starting from the center outwards.
    # This creates a "compact" cluster that is as close to a regular hexagon as possible.
    
    def get_hex_grid_positions(N):
        # Find smallest k such that 3k(k+1)+1 >= N
        # k=0: 1
        # k=1: 7
        # k=2: 19
        # k=3: 37
        # For N=11, k=2 (since 7 < 11 <= 19). The cluster will fit in a hexagon of side 3.
        k = 0
        while 3*k*(k+1) + 1 < N:
            k += 1
        
        # Generate all positions in a hexagon of radius k (side length k+1 in terms of hex count)
        # Using axial coordinates (q, r) where distance from center is max(|q|, |r|, |q+r|) <= k
        positions = []
        for q in range(-k, k+1):
            for r in range(-k, k+1):
                s = -q - r
                if abs(s) <= k:
                    positions.append((q, r))
        
        # Sort positions by distance from center (0,0) to fill the most central spots first
        # This ensures the cluster is as compact and symmetric as possible.
        def dist_sq(p):
            q, r = p
            return q*q + q*r + r*r
            
        positions.sort(key=dist_sq)
        
        # Take the first N positions
        selected = positions[:N]
        
        # Convert axial coordinates to Cartesian coordinates for flat-top hexagons
        # Axial (q, r) to Cartesian (x, y) for unit hexagons (side length 1):
        # x = 1.5 * q
        # y = sqrt(3) * (r + q/2)
        # Note: This assumes "pointy-top" or "flat-top"?
        # Standard flat-top hexagon tiling:
        # Center at (0,0).
        # Neighbors at (3/2, sqrt(3)/2), (3/2, -sqrt(3)/2), (0, sqrt(3)), etc.
        # Let's use the standard mapping:
        # x = 1.5 * q
        # y = sqrt(3) * (r + 0.5 * q)
        
        cart_coords = []
        for q, r in selected:
            x = 1.5 * q
            y = np.sqrt(3) * (r + 0.5 * q)
            cart_coords.append((x, y))
            
        return cart_coords, k

    # 2. Calculate the minimal outer side length S for this cluster.
    # The cluster of unit hexagons has a certain bounding box or extent.
    # We want to place this cluster inside a regular hexagon of side S.
    # Due to symmetry, we can align the cluster with the outer hexagon.
    # The "radius" of the cluster (distance from center to the farthest vertex) determines S.
    
    def get_hex_verts(center, angle_deg, side=1.0):
        theta = np.deg2rad(angle_deg)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        base_vertices = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1.0, 0.0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])
        return center + (side * base_vertices @ R.T)

    positions, k = get_hex_grid_positions(N)
    
    # The cluster is formed by hexagons at these positions with angle 0.
    # We need to find the minimal S such that all these hexagons fit inside an outer hexagon of side S centered at (0,0) with angle 0.
    
    # To find the minimal S, we can check the constraints for each vertex of each inner hexagon.
    # An inner hexagon at (x_i, y_i) with angle 0 has vertices v_j.
    # The outer hexagon (side S, angle 0) is defined by 6 half-planes:
    # 1. x <= S
    # 2. 0.5x + sqrt(3)/2 y <= S*sqrt(3)/2
    # 3. -0.5x + sqrt(3)/2 y <= S*sqrt(3)/2
    # 4. -x <= S
    # 5. -0.5x - sqrt(3)/2 y <= S*sqrt(3)/2
    # 6. 0.5x - sqrt(3)/2 y <= S*sqrt(3)/2
    
    # For a point (px, py) to be inside, it must satisfy:
    # S >= max(px, (0.5px + sqrt(3)/2 py)/sqrt(3), (-0.5px + sqrt(3)/2 py)/sqrt(3), -px, ...)
    # Actually, simpler:
    # The distance from the center to the boundary in direction (cos theta, sin theta) for a hexagon of side S is:
    # R(theta) = S * cos(pi/6) / cos((theta mod pi/3) - pi/6)  ... this is for pointy top?
    # Let's just use the half-plane inequalities directly.
    
    min_S = 1.0
    for (x_i, y_i) in positions:
        # Get vertices of the inner hexagon at (x_i, y_i) with angle 0
        verts = get_hex_verts((x_i, y_i), 0.0)
        for (px, py) in verts:
            # Calculate required S for this point to be inside
            # Inequality 1: px <= S  => S >= px
            # Inequality 4: -px <= S => S >= -px
            # Inequality 2: 0.5px + sqrt(3)/2 py <= S*sqrt(3)/2 => S >= (0.5px + sqrt(3)/2 py) / (sqrt(3)/2) = (px/sqrt(3) + py)
            # Inequality 5: -0.5px - sqrt(3)/2 py <= S*sqrt(3)/2 => S >= (-0.5px - sqrt(3)/2 py) / (sqrt(3)/2) = (-px/sqrt(3) - py)
            # Inequality 3: -0.5px + sqrt(3)/2 py <= S*sqrt(3)/2 => S >= (-0.5px + sqrt(3)/2 py) / (sqrt(3)/2) = (-px/sqrt(3) + py)
            # Inequality 6: 0.5px - sqrt(3)/2 py <= S*sqrt(3)/2 => S >= (0.5px - sqrt(3)/2 py) / (sqrt(3)/2) = (px/sqrt(3) - py)
            
            s1 = px
            s2 = px/np.sqrt(3) + py
            s3 = -px/np.sqrt(3) + py
            s4 = -px
            s5 = -px/np.sqrt(3) - py
            s6 = px/np.sqrt(3) - py
            
            req_S = max(s1, s2, s3, s4, s5, s6)
            if req_S > min_S:
                min_S = req_S

    # 3. Refine the solution.
    # The simple grid alignment might not be optimal if we allow rotation of the inner cluster relative to the outer hexagon.
    # However, for a symmetric cluster like a centered hexagon, aligning them is usually optimal or near-optimal.
    # Let's try rotating the inner cluster by small angles to see if we can reduce S.
    # Since the cluster is symmetric (for N=7, 19, etc.), rotation by 60 degrees is equivalent.
    # For N=11, the cluster is not perfectly symmetric (it's a subset of the k=2 hexagon).
    # The "missing" 8 spots in the k=2 hexagon (19-11=8) are the ones with the largest distance.
    # Let's check which positions were selected.
    # k=2 positions:
    # dist 0: (0,0) -> 1
    # dist 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -> 6
    # dist 2: (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1) -> 12
    # Total 19. We need 11.
    # Sorted by dist:
    # 0: (0,0)
    # 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # 2: (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    # The 11th spot is the first one from the dist=2 group.
    # The tie-breaking in sort depends on the order of iteration.
    # To get the most compact shape, we should pick the 11 spots that minimize the bounding hexagon.
    # The current simple sort might pick an asymmetric set.
    # Let's try to optimize the selection of positions from the k=2 hexagon to minimize the bounding S.
    
    # Generate all subsets of size N from the k=2 hexagon positions?
    # Number of positions is 19. We need to choose 11.
    # C(19, 11) = C(19, 8) = 75582. This is small enough to brute force!
    
    all_positions_k2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if abs(s) <= 2:
                all_positions_k2.append((q, r))
                
    def calc_min_S_for_subset(subset_coords):
        min_S = 1.0
        for (x_i, y_i) in subset_coords:
            verts = get_hex_verts((x_i, y_i), 0.0)
            for (px, py) in verts:
                s1 = px
                s2 = px/np.sqrt(3) + py
                s3 = -px/np.sqrt(3) + py
                s4 = -px
                s5 = -px/np.sqrt(3) - py
                s6 = px/np.sqrt(3) - py
                req_S = max(s1, s2, s3, s4, s5, s6)
                if req_S > min_S:
                    min_S = req_S
        return min_S

    # Convert all positions to cartesian
    all_cart = []
    for q, r in all_positions_k2:
        x = 1.5 * q
        y = np.sqrt(3) * (r + 0.5 * q)
        all_cart.append((x, y))
        
    best_subset_S = min_S # Use the first heuristic result as initial best
    best_subset_indices = None
    
    # Brute force subsets of size N from 19 positions
    # To speed up, we can prune. But 75k iterations is fast in Python if vectorized or simple.
    # Let's just loop.
    
    from itertools import combinations
    
    # To ensure we include the center (0,0) for compactness? Not necessarily, but likely.
    # Let's just brute force all.
    
    # Optimization: Precompute vertices for all 19 positions?
    # No, just compute S for each subset.
    
    # 75582 subsets * 11 hexagons * 6 vertices * 6 checks = ~30 million operations.
    # This might take a few seconds in pure Python. Let's see if we can vectorize or prune.
    # Pruning: If a subset contains a point that is "far out", it will likely have a large S.
    # The points with the largest distance from origin in the k=2 hexagon are the "corners".
    # Actually, let's just run it. 150s is plenty.
    
    # We can also try rotating the whole cluster.
    # But first, find the best subset with angle 0.
    
    # To make it faster, we can note that the bounding box is determined by the extreme points.
    # Let's just iterate.
    
    # To avoid timeouts, we will limit the brute force if it takes too long.
    # But 75k is small.
    
    best_S_final = 1e9
    best_subset = None
    
    # We can also try rotating the cluster relative to the outer hexagon.
    # The outer hexagon is fixed at angle 0.
    # We can rotate the inner cluster by angle alpha.
    # Due to symmetry of the outer hexagon, we only need to check alpha in [0, 30 degrees].
    
    # Let's do a grid search over alpha and subset?
    # That might be too much.
    # Let's stick to alpha=0 first, find best subset.
    
    # To speed up, we can precompute the "constraint values" for each position?
    # The constraint is max over vertices of the hexagon at that position.
    # For a hexagon at (x,y) with angle 0, the max constraint value is the max of the 6 linear functions of (x,y).
    # Actually, the max over vertices of a convex function (like the distance to the boundary of the outer hexagon) is achieved at the vertices.
    # The function f(x,y) = max( x, x/sqrt(3)+y, ... ) is the support function of the outer hexagon?
    # Yes. The minimal S for a set of points is the max over all points of the support function value.
    # So we just need to calculate the support function value for each center (x,y) of the inner hexagons?
    # No, the inner hexagons have size.
    # The support function of the Minkowski sum of the inner hexagons and the outer hexagon?
    # Actually, the condition is: Inner Hexagon_i + (0,0) subset Outer Hexagon_S.
    # This is equivalent to: max_{v in Inner Hex_i} h(v) <= S, where h(v) is the support function of the outer hexagon unit.
    # The support function of a regular hexagon of side 1 (angle 0) is:
    # h(x,y) = max( x, x/sqrt(3)+y, -x/sqrt(3)+y, -x, -x/sqrt(3)-y, x/sqrt(3)-y )
    # Wait, the support function h(u) = max_{x in K} u·x.
    # Here we want the max over x in K_inner of h_outer(x).
    # Since K_inner is a convex polygon, the max of a convex function (h_outer is convex) over K_inner is at the vertices.
    # So we just need to check the vertices.
    
    # Can we precompute the max constraint for each of the 19 positions?
    # Yes.
    pos_constraints = []
    for (x, y) in all_cart:
        verts = get_hex_verts((x, y), 0.0)
        max_c = 0
        for (px, py) in verts:
            s1 = px
            s2 = px/np.sqrt(3) + py
            s3 = -px/np.sqrt(3) + py
            s4 = -px
            s5 = -px/np.sqrt(3) - py
            s6 = px/np.sqrt(3) - py
            c = max(s1, s2, s3, s4, s5, s6)
            if c > max_c:
                max_c = c
        pos_constraints.append(max_c)
        
    # Now, for any subset, the minimal S is the max of pos_constraints of the selected positions.
    # We want to choose N positions from 19 to minimize the max constraint.
    # This is a "bottleneck" problem.
    # Sort positions by their constraint value.
    # The minimal possible max constraint is the N-th smallest constraint value?
    # No, because we can choose any N.
    # To minimize the max, we should pick the N positions with the smallest constraint values!
    # Is that true?
    # Yes. If we pick the N positions with the smallest individual max-constraints, the max of the set is the largest of those N, which is the N-th smallest value.
    # Any other set of N positions would have a max value >= the N-th smallest value (pigeonhole principle / order statistics).
    # So the optimal subset is simply the N positions with the smallest "bounding radius" in the outer hexagon metric.
    
    # This reduces the problem to O(M log M) where M=19.
    
    sorted_indices = np.argsort(pos_constraints)
    best_subset_indices = sorted_indices[:N]
    best_S_final = pos_constraints[best_subset_indices[-1]]
    
    # Now, we have the best subset for angle 0.
    # Can we do better by rotating the inner cluster?
    # The pos_constraints depend on the angle of the inner hexagons relative to the outer.
    # Let's try a few angles.
    
    best_global_S = best_S_final
    best_global_angles = 0.0
    best_global_subset = best_subset_indices
    
    # Try angles 0, 15, 30 degrees.
    for angle_deg in [0, 15, 30]:
        # Recalculate constraints for this angle
        pos_constraints_angle = []
        for (x, y) in all_cart:
            verts = get_hex_verts((x, y), angle_deg)
            max_c = 0
            for (px, py) in verts:
                s1 = px
                s2 = px/np.sqrt(3) + py
                s3 = -px/np.sqrt(3) + py
                s4 = -px
                s5 = -px/np.sqrt(3) - py
                s6 = px/np.sqrt(3) - py
                c = max(s1, s2, s3, s4, s5, s6)
                if c > max_c:
                    max_c = c
            pos_constraints_angle.append(max_c)
            
        sorted_idx = np.argsort(pos_constraints_angle)
        candidate_subset = sorted_idx[:N]
        candidate_S = pos_constraints_angle[candidate_subset[-1]]
        
        if candidate_S < best_global_S:
            best_global_S = candidate_S
            best_global_angles = angle_deg
            best_global_subset = candidate_subset
            
    # Final check:
    # The subset is the N positions with smallest constraints for the best angle.
    # We construct the final answer.
    
    inner = []
    for idx in best_global_subset:
        x, y = all_cart[idx]
        inner.append([x, y, best_global_angles])
        
    # Return
    # inner: list of [x, y, angle_degrees]
    # center: (0.0, 0.0)
    # side: best_global_S
    # angle: 0.0 (outer hexagon angle)
    
    return inner, (0.0, 0.0), best_global_S, 0.0

# EVOLVE-BLOCK-END
