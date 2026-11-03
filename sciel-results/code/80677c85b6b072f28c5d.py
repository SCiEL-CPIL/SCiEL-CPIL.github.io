import math
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # A regular hexagon of side length s has:
    # - Inradius (apothem) = s * sqrt(3)/2
    # - Circumradius = s
    # The hexagon is defined by 6 half-planes.
    # For a hexagon centered at (cx, cy) with side length S and rotation theta:
    # The vertices are at distance S from center, at angles theta + k*pi/3 for k=0..5.
    # The walls are perpendicular to the radii to the vertices? No, the walls are perpendicular to the apothems.
    # Let's define the hexagon by its 6 outward normals.
    # If rotation is 0, the flat sides are horizontal? Or pointy top?
    # Standard "flat-top" hexagon: normals are at angles 0, 60, 120, 180, 240, 300.
    # Wait, if a side is horizontal (flat top/bottom), the normal is vertical (90, 270).
    # Let's use the definition where the hexagon is the intersection of 6 half-planes.
    # Let the outward normals be n_k = (cos(alpha_k), sin(alpha_k)) where alpha_k = theta + k * pi/3.
    # The distance from center to each wall is the inradius R_in = S * sqrt(3) / 2.
    # So the constraint for a point p to be inside is: dot(p - center, n_k) <= R_in for all k.
    
    # We want to pack N=11 unit hexagons (side length 1) inside a larger hexagon (side length S).
    # Each inner hexagon has side length 1.
    # Inradius of inner hex = sqrt(3)/2.
    # Circumradius of inner hex = 1.
    # The inner hexagons can be rotated.
    
    # Objective: Minimize S.
    # Variables:
    # - Positions (x_i, y_i) for i=0..N-1
    # - Rotations phi_i for i=0..N-1
    # - Outer center (cx, cy) - we can fix this to (0,0) by symmetry?
    #   Due to translational symmetry of the objective (minimizing size), we can fix the outer center to (0,0) without loss of generality, 
    #   provided we allow the inner hexagons to move anywhere.
    #   However, fixing the outer center might be restrictive if the optimal packing is not centered? 
    #   Actually, the outer hexagon is just a container. We can fix its center to (0,0).
    # - Outer rotation theta.
    # - Outer side length S.
    
    # Constraints:
    # 1. Non-overlap of inner hexagons.
    #    Two convex polygons P1 and P2 do not overlap if and only if there exists a separating axis.
    #    For two regular hexagons, the separating axes are the normals to their edges.
    #    This is complex to implement directly in a smooth optimizer.
    #    Alternative: Use a distance metric. The minimum distance between two convex polygons must be >= 0.
    #    Or, use the fact that for regular hexagons, if the distance between centers is greater than the sum of their "effective radii" in that direction, they don't overlap.
    #    A simpler, robust approach for small N:
    #    Check overlap using the Separating Axis Theorem (SAT).
    #    SAT is non-smooth.
    
    #    Let's try a different formulation.
    #    Since N=11 is small, we can use a grid-based or heuristic search combined with local optimization.
    #    Or, we can model the non-overlap constraints using the distance between the convex hulls.
    #    
    #    Let's use a penalty method or a constrained optimization with a non-smooth constraint handled by subgradients or smoothing.
    #    SciPy's SLSQP or trust-constr can handle constraints, but they must be smooth.
    #    
    #    Idea: Approximate the non-overlap constraint by requiring the distance between the centers to be at least the sum of the circumradii?
    #    No, that's too loose.
    #    The minimum distance between two regular hexagons of side 1 depends on their relative rotation.
    #    If aligned, the minimum center distance for non-overlap is 2 * inradius? No.
    #    If aligned, they touch when the distance between centers is 2 * inradius = sqrt(3) ~ 1.732?
    #    Wait. Two aligned hexagons side by side.
    #    The distance between parallel edges is 0. The center-to-center distance is 2 * inradius = sqrt(3).
    #    If rotated, the required distance changes.
    #    
    #    Let's compute the exact minimum center distance for two unit regular hexagons with relative rotation delta.
    #    This function d_min(delta) is the minimum distance between centers such that they do not overlap.
    #    Then the constraint is ||p_i - p_j|| >= d_min(phi_i - phi_j).
    #    This is still non-smooth (d_min is piecewise linear or smooth? It's the support function sum).
    #    
    #    Let's precompute d_min(delta) for delta in [0, pi/3] (symmetry).
    #    d_min(delta) = max over all separating axes of (support1 + support2).
    #    
    #    Actually, let's just use a generic SAT check in a penalty function.
    #    Penalty = sum over all pairs (max(0, overlap_depth_ij)^2) + sum over all i (max(0, dist_to_wall_i - R_in_outer)^2)
    #    We minimize the penalty. If penalty is 0, we have a valid packing.
    #    Then we minimize S subject to penalty=0.
    #    
    #    Better: Minimize S.
    #    Constraints:
    #    1. For each inner hex i, it must be inside the outer hex.
    #       This means for each of the 6 walls of the outer hex, the maximum projection of the inner hex onto the wall normal must be <= R_in_outer.
    #       max_{v in inner_i} dot(v, n_wall) <= cx*n_wall_x + cy*n_wall_y + R_in_outer.
    #       Since inner_i is convex, max is at a vertex.
    #       So we check the 6 vertices of inner_i.
    #    2. For each pair (i, j), inner_i and inner_j must not overlap.
    #       Use SAT. If they overlap, add a penalty.
    #    
    #    Let's implement a smooth approximation of the overlap penalty.
    #    For two convex polygons, the overlap depth can be computed.
    #    But for optimization, it's better to use a differentiable surrogate.
    #    
    #    Alternative Strategy:
    #    Since N=11 is small, we can use a random restart + local search approach.
    #    1. Randomly initialize positions and rotations.
    #    2. Use a gradient-free optimizer (like Nelder-Mead or Powell) to minimize a cost function:
    #       Cost = S + lambda * (violation_penalty)
    #    3. Or, fix S and check if a valid packing exists. Binary search on S.
    #    
    #    Let's try Binary Search on S.
    #    For a given S, can we place 11 unit hexagons inside a hexagon of side S?
    #    This is a feasibility problem.
    #    We can use a local search to try to find a valid configuration.
    #    
    #    Given the time limit, a robust heuristic is best.
    #    
    #    Heuristic:
    #    1. Start with a known good packing or random.
    #    2. Use a "repulsion" force model.
    #       - Forces pushing inner hexagons apart if they overlap.
    #       - Forces pushing inner hexagons away from the outer walls if they intersect.
    #       - Minimize the outer side length S?
    #    
    #    Let's stick to the optimization of S directly with penalties.
    #    
    #    Variables: x = [x_0, y_0, phi_0, ..., x_10, y_10, phi_10, theta]
    #    S is determined by the constraints? No, S is a variable we want to minimize.
    #    Let's include S in the variables.
    #    x = [x_0..x_10, y_0..y_10, phi_0..phi_10, theta, S]
    #    
    #    Objective: S
    #    Constraints:
    #    1. Inner hex i inside outer hex:
    #       For each wall k (0..5) of outer hex:
    #       max_{v in inner_i} dot(v - center_outer, n_k) <= R_in_outer
    #       This is equivalent to:
    #       max_{v in inner_i} dot(v, n_k) <= dot(center_outer, n_k) + R_in_outer
    #       Since center_outer is (0,0) (fixed), this is:
    #       max_{v in inner_i} dot(v, n_k) <= R_in_outer
    #       The max is achieved at a vertex of inner_i.
    #       So for each vertex v of inner_i, dot(v, n_k) <= R_in_outer.
    #       This gives 6*6 = 36 linear constraints per inner hex, total 11*36 = 396 constraints.
    #       Wait, the vertices of inner_i depend on phi_i.
    #       v_m = p_i + R_circ_inner * (cos(phi_i + m*pi/3), sin(phi_i + m*pi/3)) for m=0..5.
    #       R_circ_inner = 1.
    #       So dot(p_i + v_m_offset, n_k) <= R_in_outer.
    #       This is a smooth constraint in p_i, phi_i, theta, S.
    #       R_in_outer = S * sqrt(3) / 2.
    #       n_k = (cos(theta + k*pi/3), sin(theta + k*pi/3)).
    #       
    #    2. Non-overlap of inner hexagons:
    #       This is the hard part.
    #       We can approximate the non-overlap by requiring the distance between centers to be >= d_min(phi_i - phi_j).
    #       d_min(delta) is the minimum center distance for non-overlap of two unit hexagons with relative rotation delta.
    #       We can precompute d_min(delta) for a grid of delta values and interpolate.
    #       Then the constraint is ||p_i - p_j||^2 >= d_min(phi_i - phi_j)^2.
    #       This is a smooth constraint if d_min is smooth.
    #       Is d_min smooth? It is the maximum of linear functions (support functions), so it is convex and piecewise linear, hence non-smooth at the breakpoints.
    #       We can smooth it with a log-sum-exp or just use it as is and rely on the optimizer's robustness.
    #       Or, use a penalty method for non-overlap.
    #    
    #    Let's use a penalty method for the non-overlap to make it a smooth objective.
    #    Objective: S + lambda * sum_{i<j} max(0, d_min(phi_i - phi_j) - ||p_i - p_j||)^2
    #    Constraints:
    #    1. Inner hexes inside outer hex (smooth, linear in S and trig in phi, theta).
    #    
    #    Let's implement this.
    
    def get_hex_vertices(center, side, rot):
        """Returns 6 vertices of a regular hexagon."""
        # Vertices are at distance 'side' from center? No, circumradius is 'side'.
        # Yes, for a regular hexagon, side length = circumradius.
        angles = rot + np.arange(6) * np.pi / 3
        return center + side * np.column_stack([np.cos(angles), np.sin(angles)])
    
    def d_min_table(delta):
        """
        Precompute minimum center distance for non-overlap of two unit hexagons
        with relative rotation delta.
        delta in [0, pi/3].
        """
        # Sample points
        deltas = np.linspace(0, np.pi/3, 100)
        d_mins = []
        for d in deltas:
            # Hex 1 at (0,0) rot 0
            # Hex 2 at (r, 0) rot d
            # We want min r such that they don't overlap.
            # Use SAT.
            # Axes for Hex 1: normals at 0, 60, 120, 180, 240, 300.
            # Axes for Hex 2: normals at d, d+60, ...
            # Union of axes.
            axes1 = np.column_stack([np.cos(np.arange(6)*np.pi/3), np.sin(np.arange(6)*np.pi/3)])
            axes2 = np.column_stack([np.cos(d + np.arange(6)*np.pi/3), np.sin(d + np.arange(6)*np.pi/3)])
            axes = np.vstack([axes1, axes2])
            
            v1 = get_hex_vertices(np.array([0.0, 0.0]), 1.0, 0.0)
            # v2 depends on r. v2 = get_hex_vertices(np.array([r, 0.0]), 1.0, d)
            # Projection of v1 on axis a: [min1, max1]
            # Projection of v2 on axis a: [min2, max2]
            # Overlap if intervals intersect.
            # min2 = r * a[0] + min_offset2
            # max2 = r * a[0] + max_offset2
            # No overlap if max1 < min2 or max2 < min1.
            # Since r > 0 and we assume separation along x, we look for max1 < min2.
            # min2 = r * a_x + min_off2
            # max1 = max_off1
            # So r * a_x > max_off1 - min_off2
            # r > (max_off1 - min_off2) / a_x  if a_x > 0
            # r > (min_off2 - max_off1) / a_x  if a_x < 0 ? No.
            # Let's just do a binary search on r for each delta.
            
            def overlaps(r):
                v2 = get_hex_vertices(np.array([r, 0.0]), 1.0, d)
                for a in axes:
                    proj1 = np.dot(v1, a)
                    proj2 = np.dot(v2, a)
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    if max1 >= min2 and max2 >= min1:
                        # Check if they actually overlap (not just touch)
                        # For SAT, touching is okay? We want non-overlap.
                        # If they just touch, it's valid.
                        # So if intervals intersect, they might overlap.
                        # We need to check if there is a separating axis.
                        # If for ALL axes, intervals intersect, they overlap.
                        # So if we find an axis where they do NOT intersect, they don't overlap.
                        pass
                # If we get here, all axes had intersecting intervals?
                # Wait, the logic is:
                # If there exists an axis where projections do NOT overlap, then polygons do NOT overlap.
                # So if for ALL axes, projections DO overlap, then polygons DO overlap.
                # So to check if they overlap:
                # for all a in axes:
                #   if not (max1 >= min2 and max2 >= min1):
                #      return False (they don't overlap)
                # return True (they overlap)
                
                for a in axes:
                    proj1 = np.dot(v1, a)
                    proj2 = np.dot(v2, a)
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    if not (max1 >= min2 and max2 >= min1):
                        return False
                return True
            
            # Binary search for min r
            lo, hi = 0.0, 10.0
            for _ in range(50):
                mid = (lo + hi) / 2
                if overlaps(mid):
                    lo = mid
                else:
                    hi = mid
            d_mins.append(lo)
        return deltas, np.array(d_mins)
    
    # Precompute d_min table
    print("Precomputing d_min table...")
    delta_grid, d_min_grid = d_min_table(0.0)
    
    def get_d_min(delta):
        """Interpolate d_min for a given relative rotation delta."""
        # Normalize delta to [0, pi/3]
        delta = delta % (np.pi/3)
        # Due to symmetry, d_min(delta) = d_min(pi/3 - delta)?
        # Yes, relative rotation is symmetric.
        # Actually, the table is for delta in [0, pi/3].
        # We can just interpolate.
        return np.interp(delta, delta_grid, d_min_grid)
    
    def objective_and_grad(x):
        # x = [x_0..x_10, y_0..y_10, phi_0..phi_10, theta, S]
        # Indices:
        # 0..10: x
        # 11..21: y
        # 22..32: phi
        # 33: theta
        # 34: S
        S = x[34]
        theta = x[33]
        R_in_outer = S * np.sqrt(3) / 2
        
        # Objective: S + penalty
        penalty = 0.0
        
        # 1. Inside outer hex constraint
        # For each inner hex i, and each outer wall k:
        # max_{v in inner_i} dot(v, n_k) <= R_in_outer
        # n_k = (cos(theta + k*pi/3), sin(theta + k*pi/3))
        # v_m = p_i + (cos(phi_i + m*pi/3), sin(phi_i + m*pi/3))
        # dot(v_m, n_k) = p_i_x * n_k_x + p_i_y * n_k_y + cos(phi_i + m*pi/3)*n_k_x + sin(phi_i + m*pi/3)*n_k_y
        # We need this to be <= R_in_outer for all m, k.
        # Violation = max(0, dot(v_m, n_k) - R_in_outer)
        # Sum of squares of violations.
        
        # 2. Non-overlap penalty
        # For each pair (i, j):
        # dist = ||p_i - p_j||
        # d_req = get_d_min(phi_i - phi_j)
        # violation = max(0, d_req - dist)
        # penalty += violation^2
        
        # This is a pure function, no gradient needed if we use Nelder-Mead or Powell.
        # Given the complexity, let's use Powell or Nelder-Mead.
        
        # Compute penalty
        for i in range(N):
            xi = x[i]
            yi = x[i + N]
            phi_i = x[i + 2*N]
            p_i = np.array([xi, yi])
            
            # Check walls
            for k in range(6):
                alpha_k = theta + k * np.pi / 3
                n_k = np.array([np.cos(alpha_k), np.sin(alpha_k)])
                # Max projection of inner hex i on n_k
                # The max is achieved at the vertex closest to n_k.
                # We can compute the max over the 6 vertices.
                max_proj = -np.inf
                for m in range(6):
                    alpha_m = phi_i + m * np.pi / 3
                    v_offset = np.array([np.cos(alpha_m), np.sin(alpha_m)])
                    proj = np.dot(p_i + v_offset, n_k)
                    if proj > max_proj:
                        max_proj = proj
                if max_proj > R_in_outer:
                    penalty += (max_proj - R_in_outer)**2
        
        for i in range(N):
            for j in range(i+1, N):
                pi = np.array([x[i], x[i+N]])
                pj = np.array([x[j], x[j+N]])
                dist = np.linalg.norm(pi - pj)
                delta = x[i+2*N] - x[j+2*N]
                d_req = get_d_min(delta)
                if dist < d_req:
                    penalty += (d_req - dist)**2
        
        return S + 10.0 * penalty
    
    # Optimization
    # Initial guess
    # Place hexagons in a grid pattern.
    x0 = np.zeros(3*N + 2)
    # S initial guess
    x0[34] = 10.0
    # Theta initial guess
    x0[33] = 0.0
    
    # Positions
    # Arrange in a hexagonal grid
    # Center at (0,0)
    # (0,0)
    # (1.732, 0), (-1.732, 0)
    # (0.866, 1.5), (-0.866, 1.5), (0.866, -1.5), (-0.866, -1.5)
    # (2.598, 1.5), (-2.598, 1.5), (2.598, -1.5), (-2.598, -1.5)
    # This is 1 + 2 + 4 + 4 = 11?
    # Ring 0: 1
    # Ring 1: 6
    # Ring 2: 12 (we only need 4 more)
    
    positions = []
    # Center
    positions.append((0.0, 0.0))
    # Ring 1
    for k in range(6):
        angle = k * np.pi / 3
        r = 2 * np.sqrt(3)/2 # 2 * inradius? No, center distance for touching is 2*inradius = sqrt(3) ~ 1.732
        # Wait, if they touch, distance is 2 * inradius = sqrt(3).
        # But we can pack tighter if rotated?
        # For initial guess, use touching distance.
        r = np.sqrt(3)
        positions.append((r * np.cos(angle), r * np.sin(angle)))
    # Ring 2 (pick 4)
    # The 6 neighbors of the ring 1 hexagons that are not already in ring 1.
    # Let's just place them further out.
    for k in range(6):
        angle = k * np.pi / 3 + np.pi/6 # Offset
        r = 2 * np.sqrt(3)
        if len(positions) < 11:
            positions.append((r * np.cos(angle), r * np.sin(angle)))
    
    # Fill remaining
    while len(positions) < 11:
        positions.append((0.0, 0.0)) # Will be pushed apart by penalty
        
    for i in range(N):
        x0[i] = positions[i][0]
        x0[i+N] = positions[i][1]
        x0[i+2*N] = 0.0 # Rotation
    
    # Optimize
    best_x = x0
    best_obj = objective_and_grad(x0)
    
    # Use Nelder-Mead
    # It's derivative-free and robust.
    try:
        res = minimize(objective_and_grad, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.fun < best_obj:
            best_x = res.x
            best_obj = res.fun
    except Exception as e:
        print(f"Nelder-Mead failed: {e}")
    
    # Check time
    if time.time() > deadline:
        # Return best found
        S = best_x[34]
        theta = best_x[33]
        inner = []
        for i in range(N):
            inner.append([best_x[i], best_x[i+N], np.degrees(best_x[i+2*N])])
        return inner, (0.0, 0.0), S, np.degrees(theta)
    
    # Try a few random restarts if time permits
    for restart in range(5):
        if time.time() > deadline - 10:
            break
            
        x_rand = x0.copy()
        # Add noise
        x_rand[:2*N] += np.random.normal(0, 1.0, 2*N)
        x_rand[2*N:3*N] += np.random.uniform(-np.pi/6, np.pi/6, N)
        x_rand[33] = np.random.uniform(0, np.pi/3)
        x_rand[34] = 10.0
        
        try:
            res = minimize(objective_and_grad, x_rand, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_obj:
                best_x = res.x
                best_obj = res.fun
        except Exception as e:
            pass
    
    S = best_x[34]
    theta = best_x[33]
    inner = []
    for i in range(N):
        inner.append([best_x[i], best_x[i+N], np.degrees(best_x[i+2*N])])
    return inner, (0.0, 0.0), S, np.degrees(theta)
# EVOLVE-BLOCK-END
