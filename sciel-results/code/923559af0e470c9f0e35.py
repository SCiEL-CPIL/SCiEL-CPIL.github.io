"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin (evaluator limit 150s, keep 15s buffer)
    
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

    # Helper: Axial to Cartesian coordinates for hex grid
    # Using pointy-top hexagons for grid, but the unit hexagons themselves are flat-top?
    # The task says "Unit regular hexagons". Usually standard orientation is flat-top or pointy-top.
    # The seed program used vertices at 0, 60, 120... which is flat-top (vertices at right, top-right, etc)
    # Wait, vertex at 0 deg is (1,0). Vertex at 60 is (0.5, sqrt(3)/2).
    # This is a flat-top hexagon if we consider the flat edge at top? No, vertex at top-right.
    # Let's stick to the seed's definition: vertices at angles 0, 60, 120, 180, 240, 300.
    # This is a "pointy-right" or standard mathematical orientation.
    # Grid spacing for non-overlapping hexagons of this orientation:
    # Distance between centers of touching hexagons is 2 * apothem = sqrt(3).
    # Axial coordinates (q, r) -> Cartesian:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    # Let's verify distance between (0,0) and (1,0):
    # (0,0) -> (0,0)
    # (1,0) -> (sqrt(3), 0). Distance sqrt(3). Correct.
    # (0,1) -> (sqrt(3)/2, 1.5). Distance from (0,0) is sqrt(3/4 + 2.25) = sqrt(3). Correct.
    
    def axial_to_cartesian(q, r):
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    # Helper: Calculate minimum outer hexagon side length containing all given hexagons
    # The outer hexagon can be rotated. We find the best rotation.
    def calc_outer_side(hex_centers, hex_angles, outer_angle_deg):
        """
        Calculate the side length of the smallest regular hexagon with rotation outer_angle_deg
        that contains all inner hexagons.
        """
        # The outer hexagon is defined by its apothem 'a' and rotation.
        # A point (x,y) is inside a hexagon centered at origin with apothem 'a' and rotation 'phi'
        # if for all 6 normal directions n_i, dot((x,y), n_i) <= a.
        # The normal directions for a hexagon with vertices at 0, 60... are at 30, 90, 150...
        # So if outer hex is rotated by phi, normals are at phi + 30, phi + 90, ...
        
        base_normal_angles = [30, 90, 150, 210, 270, 330]
        normals = []
        for a in base_normal_angles:
            rad = math.radians(a + outer_angle_deg)
            normals.append((math.cos(rad), math.sin(rad)))
        
        max_proj = 0.0
        for cx, cy, angle in zip(hex_centers, hex_angles, hex_angles): # dummy angle usage, we need vertices
            pass
            
        # We need to project vertices of each inner hexagon
        # To be fast, we can precompute vertices for the cluster once per global rotation?
        # No, inner hexagons have fixed relative rotations (0 in grid search).
        # So vertices are fixed relative to cluster center.
        
        # Let's assume all inner hexagons have angle 0 relative to the grid.
        # We compute their absolute positions relative to cluster center.
        
        # This function is called inside the optimization loop, so it must be fast.
        # We will pass the list of absolute vertices (relative to cluster centroid) to avoid recalculation.
        
        # Actually, let's structure the search differently.
        # 1. Generate a set of axial coordinates (grid shape).
        # 2. Convert to Cartesian relative to centroid.
        # 3. For a given global rotation R, the vertices of hex i are Rot(R) * (v_i_local) + Rot(R) * c_i
        #    where v_i_local are vertices of unit hex at 0, and c_i is center of hex i relative to centroid.
        #    Wait, if we rotate the whole cluster, the relative positions c_i rotate too.
        #    And the local vertices v_i_local rotate too.
        #    Since all inner hexagons are identical and axis-aligned in the grid, their local vertices are the same.
        #    Let V be the set of 6 vertices of a unit hex at origin.
        #    Let C be the set of N center vectors (relative to cluster centroid).
        #    For a global rotation theta, the absolute vertex positions are:
        #    P = { Rot(theta) * (c + v) | c in C, v in V }
        #    We need the smallest hexagon (apothem A) such that all P are inside.
        #    Max projection of P onto any of the 6 outer normals <= A.
        
        # To optimize, we can iterate over the 6 outer normal angles.
        # For a fixed outer rotation phi, the max projection is:
        # max_{c in C, v in V} dot( Rot(theta) * (c+v), n(phi) )
        # = max_{c in C, v in V} dot( c+v, Rot(-theta) * n(phi) )
        # Let m = Rot(-theta) * n(phi).
        # Then max_proj = max_{c in C, v in V} dot(c+v, m)
        # = max_{c in C} ( dot(c, m) + max_{v in V} dot(v, m) )
        
        # max_{v in V} dot(v, m) is the support function of the unit hexagon in direction m.
        # For a unit hexagon (vertices at 0,60..), the support function is:
        # h(m) = max( cos(alpha), cos(alpha-60), ... ) where alpha is angle of m.
        # Actually, for a regular hexagon with circumradius 1, h(m) = cos(alpha mod 60 - 30) ?
        # Let's check. Vertex at 0: (1,0). m=(1,0). dot=1. cos(0-30)=cos(-30)=0.866. No.
        # The support function of a hexagon with vertices at k*60 is:
        # h(m) = max_k cos(angle(m) - k*60).
        # This is periodic with period 60.
        # Let beta = angle(m) % 60.
        # h(m) = max( cos(beta), cos(beta-60), cos(beta+60) ... )
        # Since cos is symmetric, h(m) = cos(min distance to multiple of 60).
        # Specifically, if we take beta in [0, 60), the max is at the nearest multiple.
        # If beta < 30, nearest is 0 -> cos(beta).
        # If beta > 30, nearest is 60 -> cos(60-beta) = cos(beta-60).
        # So h(m) = cos(abs(beta - 30) - 30) ? No.
        # h(m) = cos( min(beta, 60-beta) )?
        # Let's test beta=0: cos(0)=1. Correct (vertex).
        # beta=30: cos(30)=0.866. Correct (apothem).
        # beta=60: cos(0)=1. Correct.
        
        # So, for a given direction m, max_{v in V} dot(v, m) = cos( min(beta, 60-beta) * pi/180 )
        # where beta is the angle of m in degrees, modulo 60.
        
        # This allows very fast evaluation!
        # We don't need to project all vertices.
        
        # Algorithm:
        # 1. Compute C (centers relative to centroid) for the grid shape.
        # 2. For a candidate global rotation theta and outer rotation phi:
        #    For each of the 6 outer normals n_k (angle phi + 30 + k*60):
        #       m_k = Rot(-theta) * n_k.
        #       angle_m_k = angle(m_k)
        #       support_v = cos( min(angle_m_k % 60, 60 - (angle_m_k % 60)) * pi/180 )
        #       max_c_proj = max_{c in C} dot(c, m_k)
        #       total_proj = max_c_proj + support_v
        #    A = max(total_proj for k in 0..5)
        #    Side = 2 * A / SQRT3
        
        # We can optimize further:
        # The function f(theta, phi) is piecewise smooth.
        # We can fix the relative rotation delta = phi - theta?
        # No, theta rotates the cluster, phi rotates the container.
        # The container is symmetric under 60 deg rotation.
        # The cluster has some symmetry.
        
        # Let's just search over theta (cluster rotation) and phi (container rotation).
        # Since container is symmetric, we can fix phi=0 and search theta in [0, 60).
        # Wait, if we fix phi=0, the container normals are fixed at 30, 90, etc.
        # Then we rotate the cluster by theta.
        # This covers all relative orientations.
        
        # So, fix outer hexagon at phi=0.
        # Normals are at 30, 90, 150, 210, 270, 330.
        # For a cluster rotation theta:
        # For each normal n_k (angle alpha_k):
        #    m_k = Rot(-theta) * n_k.
        #    angle_m_k = alpha_k - theta.
        #    support_v = cos( min((angle_m_k % 60 + 60) % 60, 60 - ((angle_m_k % 60 + 60) % 60)) * pi/180 )
        #    max_c_proj = max_{c in C} dot(c, m_k)
        #    A_k = max_c_proj + support_v
        # A = max(A_k)
        # Side = 2 * A / SQRT3
        
        # We need to minimize Side over theta.
        
        # Precompute C for the current grid shape.
        # C is a list of (x, y) tuples.
        
        # To make this fast in Python, we can use NumPy.
        # C is an (N, 2) array.
        # m_k is a (2,) vector.
        # dot(C, m_k) is a vector of N values. max is scalar.
        
        # We will implement this evaluation function.
        
        # Return side length
        return -1 # Placeholder, will be implemented in the main loop

    # Better: Implement the evaluation inside the search loop for speed, or as a nested function with precomputed C.
    
    def evaluate_shape(centers_cartesian):
        """
        centers_cartesian: list of (x, y) tuples, relative to centroid.
        Returns minimum outer side length.
        """
        C = np.array(centers_cartesian)
        # Precompute max_c_proj for each of the 6 normals for a given theta?
        # No, theta changes m_k.
        
        # We will search theta in [0, 60).
        # The function is smooth except when a vertex of an inner hex aligns with a normal?
        # No, the support function is smooth.
        # The max_c_proj is piecewise linear (max of linear functions).
        # So the total function is piecewise smooth.
        
        # Let's sample theta finely.
        # 60 degrees / 0.1 deg = 600 samples.
        # For each sample, 6 normals.
        # 6 * 600 = 3600 dot products of size N=11.
        # Very fast.
        
        best_side = float('inf')
        # Use a coarse grid first, then refine?
        # 0.1 deg should be enough.
        
        # Precompute cos/sin for theta?
        # m_k = (cos(alpha_k - theta), sin(alpha_k - theta))
        
        base_angles = [30, 90, 150, 210, 270, 330]
        
        for t_deg in np.arange(0, 60, 0.1):
            t_rad = math.radians(t_deg)
            max_A = 0.0
            for alpha_deg in base_angles:
                alpha_rad = math.radians(alpha_deg)
                # m_k angle = alpha_rad - t_rad
                m_angle = alpha_rad - t_rad
                mx = math.cos(m_angle)
                my = math.sin(m_angle)
                
                # dot(C, m_k)
                # C is (N, 2). m is (2,).
                # proj = C[:, 0] * mx + C[:, 1] * my
                proj = C[:, 0] * mx + C[:, 1] * my
                max_c = np.max(proj)
                
                # support_v
                # angle_m_deg = m_angle * 180 / pi
                # beta = angle_m_deg % 60
                # support = cos(min(beta, 60-beta) * pi/180)
                # We can compute this without modulo if we are careful, or just use modulo.
                m_deg = m_angle * 180.0 / math.pi
                beta = m_deg % 60.0
                # Ensure beta in [0, 60)
                if beta < 0: beta += 60.0
                min_dist = min(beta, 60.0 - beta)
                support = math.cos(math.radians(min_dist))
                
                A_k = max_c + support
                if A_k > max_A:
                    max_A = A_k
            
            side = 2.0 * max_A / SQRT3
            if side < best_side:
                best_side = side
                
        return best_side

    # Search for the best grid shape
    # We use a random walk on the hex grid to generate connected sets of 11 cells.
    # Start with one cell, add a neighbor at each step.
    # Keep a set of visited cells to avoid duplicates.
    
    best_side_global = float('inf')
    best_shape = None
    best_theta = 0.0
    
    # To avoid re-evaluating the same shape, we can use a hash of the sorted axial coordinates.
    # But the random walk might generate duplicates.
    # Let's just run a fixed number of iterations.
    
    num_iterations = 20000
    iter_count = 0
    
    while iter_count < num_iterations and time.time() < deadline - 5.0:
        # Generate a random shape
        # Start at (0,0)
        cells = [(0,0)]
        cell_set = {(0,0)}
        
        # Random walk
        for _ in range(N - 1):
            last_q, last_r = cells[-1]
            # 6 neighbors
            neighbors = [
                (last_q + 1, last_r),
                (last_q - 1, last_r),
                (last_q, last_r + 1),
                (last_q, last_r - 1),
                (last_q + 1, last_r - 1),
                (last_q - 1, last_r + 1)
            ]
            # Filter out visited
            valid_neighbors = [n for n in neighbors if n not in cell_set]
            if not valid_neighbors:
                # Dead end, restart? Or pick a random cell to expand from?
                # For simplicity, restart this shape generation
                break
            next_cell = random.choice(valid_neighbors)
            cells.append(next_cell)
            cell_set.add(next_cell)
            
        if len(cells) < N:
            continue
            
        # Convert to Cartesian
        centers = []
        for q, r in cells:
            x, y = axial_to_cartesian(q, r)
            centers.append((x, y))
            
        # Center the shape
        cx_avg = sum(c[0] for c in centers) / N
        cy_avg = sum(c[1] for c in centers) / N
        centers_relative = [(x - cx_avg, y - cy_avg) for x, y in centers]
        
        # Evaluate
        side = evaluate_shape(centers_relative)
        
        if side < best_side_global:
            best_side_global = side
            best_shape = centers_relative
            # We could also store the best theta here, but evaluate_shape finds it internally.
            # For the final output, we need the specific theta that gave the best side.
            # Let's modify evaluate_shape to also return the best theta.
            
        iter_count += 1
        
        # Adaptive: if we found a good one, increase iterations?
        # For now, fixed.
        
    # Refine the best shape found
    # We can try to optimize the positions slightly off-grid?
    # The task allows translation and rotation.
    # The grid search assumes axis-aligned hexagons.
    # However, the global rotation handles the orientation.
    # Are there better packings where hexagons are not on a hex grid?
    # For N=11, the optimal packing might involve some hexagons rotated relative to others?
    # Or just shifted?
    # The "disjoint" constraint is strict.
    # The grid packing is a valid solution.
    # To improve, we could take the best grid shape and run a local Nelder-Mead on the centers
    # to allow slight offsets that might reduce the bounding box?
    # But if they are on a grid, they are touching. Moving them apart increases the bounding box.
    # Moving them closer causes overlap.
    # So grid positions are likely local minima for the "touching" constraint.
    # However, if the shape is not "tight" (e.g. a hole), filling it or rearranging might help.
    # The random walk generates "polyhexes", which are usually tight.
    
    # Let's try a local refinement on the best grid shape to see if we can squeeze it.
    # We will allow small perturbations of the centers.
    # We must check for overlap.
    
    if best_shape is not None:
        # Local refinement
        # Use Nelder-Mead on the centers of the best shape.
        # Objective: Side length + Penalty for overlap.
        
        def objective_refinement(params):
            # params: 2N values (x1, y1, ..., xN, yN)
            centers = [(params[i*2], params[i*2+1]) for i in range(N)]
            
            # Check overlap
            # For each pair, check if they overlap.
            # Since they are close to grid positions, we can use a fast check.
            # If overlap, return large penalty.
            # We need to check if the hexagons overlap.
            # Two hexagons of side 1 overlap if their center distance is < sqrt(3) ?
            # No, that's only if they are aligned.
            # If rotated, the condition is different.
            # But in our grid, they are aligned.
            # If we perturb, they might rotate? No, we only perturb centers.
            # So they remain aligned (angle 0).
            # For aligned hexagons, they overlap iff distance < sqrt(3).
            # Wait, is that true?
            # Two regular hexagons of side 1, aligned.
            # Distance between centers of touching hexagons is sqrt(3).
            # If distance < sqrt(3), they overlap.
            # Yes, because the width across flats is sqrt(3).
            # So for aligned hexagons, non-overlap <=> dist >= sqrt(3).
            
            overlap_penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = centers[i][0] - centers[j][0]
                    dy = centers[i][1] - centers[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < SQRT3 - 1e-6:
                        # Overlap
                        # Penalty proportional to violation
                        violation = SQRT3 - dist
                        overlap_penalty += violation * violation * 100.0 + 10.0
                        
            if overlap_penalty > 0:
                # Also calculate side to guide the search away from overlaps?
                # Or just return penalty.
                # Let's return penalty + side
                pass
            
            # Calculate side
            # Re-use logic from evaluate_shape but with fixed theta?
            # No, we should optimize theta too?
            # If we move centers, the optimal theta might change.
            # But for small perturbations, theta is likely stable.
            # Let's fix theta to the one found by the grid search for speed,
            # or re-optimize theta (600 steps) which is fast.
            
            # Let's re-optimize theta.
            C = np.array(centers)
            # Center the configuration?
            # The objective function for side length is translation invariant.
            # So we can center it.
            cx_avg = np.mean(C[:, 0])
            cy_avg = np.mean(C[:, 1])
            C = C - np.array([cx_avg, cy_avg])
            
            best_side = float('inf')
            base_angles = [30, 90, 150, 210, 270, 330]
            for t_deg in np.arange(0, 60, 0.5): # Coarser for refinement
                t_rad = math.radians(t_deg)
                max_A = 0.0
                for alpha_deg in base_angles:
                    alpha_rad = math.radians(alpha_deg)
                    m_angle = alpha_rad - t_rad
                    mx = math.cos(m_angle)
                    my = math.sin(m_angle)
                    proj = C[:, 0] * mx + C[:, 1] * my
                    max_c = np.max(proj)
                    m_deg = m_angle * 180.0 / math.pi
                    beta = m_deg % 60.0
                    if beta < 0: beta += 60.0
                    min_dist = min(beta, 60.0 - beta)
                    support = math.cos(math.radians(min_dist))
                    A_k = max_c + support
                    if A_k > max_A:
                        max_A = A_k
                side = 2.0 * max_A / SQRT3
                if side < best_side:
                    best_side = side
                    
            return overlap_penalty + best_side

        # Initial params from best_shape
        initial_params = []
        for x, y in best_shape:
            initial_params.extend([x, y])
            
        from scipy.optimize import minimize
        try:
            res = minimize(objective_refinement, initial_params, method='Nelder-Mead', 
                           options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_side_global:
                # Update best
                # Extract centers
                centers = [(res.x[i*2], res.x[i*2+1]) for i in range(N)]
                # Check validity
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        dx = centers[i][0] - centers[j][0]
                        dy = centers[i][1] - centers[j][1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < SQRT3 - 1e-5:
                            valid = False
                            break
                    if not valid: break
                if valid:
                    # Recalculate side
                    C = np.array(centers)
                    cx_avg = np.mean(C[:, 0])
                    cy_avg = np.mean(C[:, 1])
                    C = C - np.array([cx_avg, cy_avg])
                    best_side = float('inf')
                    base_angles = [30, 90, 150, 210, 270, 330]
                    best_t = 0
                    for t_deg in np.arange(0, 60, 0.1):
                        t_rad = math.radians(t_deg)
                        max_A = 0.0
                        for alpha_deg in base_angles:
                            alpha_rad = math.radians(alpha_deg)
                            m_angle = alpha_rad - t_rad
                            mx = math.cos(m_angle)
                            my = math.sin(m_angle)
                            proj = C[:, 0] * mx + C[:, 1] * my
                            max_c = np.max(proj)
                            m_deg = m_angle * 180.0 / math.pi
                            beta = m_deg % 60.0
                            if beta < 0: beta += 60.0
                            min_dist = min(beta, 60.0 - beta)
                            support = math.cos(math.radians(min_dist))
                            A_k = max_c + support
                            if A_k > max_A:
                                max_A = A_k
                        side = 2.0 * max_A / SQRT3
                        if side < best_side:
                            best_side = side
                            best_t = t_deg
                    if best_side < best_side_global:
                        best_side_global = best_side
                        best_shape = [(x - cx_avg, y - cy_avg) for x, y in centers]
                        # Store best_t? We need it for the output.
                        # Let's re-calculate the best theta for the final shape in the output section.
        except Exception:
            pass

    # Final Output Construction
    # We have best_shape (centers relative to centroid, angle 0)
    # We need to find the best outer rotation (theta) for this shape.
    
    C = np.array(best_shape)
    cx_avg = np.mean(C[:, 0])
    cy_avg = np.mean(C[:, 1])
    C = C - np.array([cx_avg, cy_avg])
    
    best_side = float('inf')
    best_t = 0.0
    base_angles = [30, 90, 150, 210, 270, 330]
    for t_deg in np.arange(0, 60, 0.05): # Fine search
        t_rad = math.radians(t_deg)
        max_A = 0.0
        for alpha_deg in base_angles:
            alpha_rad = math.radians(alpha_deg)
            m_angle = alpha_rad - t_rad
            mx = math.cos(m_angle)
            my = math.sin(m_angle)
            proj = C[:, 0] * mx + C[:, 1] * my
            max_c = np.max(proj)
            m_deg = m_angle * 180.0 / math.pi
            beta = m_deg % 60.0
            if beta < 0: beta += 60.0
            min_dist = min(beta, 60.0 - beta)
            support = math.cos(math.radians(min_dist))
            A_k = max_c + support
            if A_k > max_A:
                max_A = A_k
        side = 2.0 * max_A / SQRT3
        if side < best_side:
            best_side = side
            best_t = t_deg

    # Construct inner list
    # The inner hexagons are at C, with angle 0 (relative to the grid).
    # But the "angle_degrees" in the output is the rotation of the inner hexagon.
    # In our model, the inner hexagons are aligned with the grid (angle 0).
    # The "global rotation" best_t is the rotation of the *container* relative to the grid?
    # No, in our evaluation:
    # We fixed the container normals at 30, 90... (phi=0).
    # We rotated the cluster by t.
    # So the cluster is rotated by t relative to the container.
    # This means the inner hexagons (which are part of the cluster) are rotated by t relative to the container.
    # So the angle of each inner hexagon should be best_t.
    
    inner_list = []
    for i in range(N):
        inner_list.append([C[i, 0], C[i, 1], best_t])
        
    # Outer hexagon
    # Center (0,0)
    # Side best_side
    # Rotation 0.0 (since we fixed phi=0)
    
    return (inner_list, (0.0, 0.0), best_side, 0.0)

# EVOLVE-BLOCK-END
