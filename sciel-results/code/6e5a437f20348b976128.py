import math
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 15s margin for safety

    # Precompute unit hexagon vertices for angle 0 (flat-top)
    # Vertices at 0, 60, ..., 300 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    
    # Precompute normals for the 6 sides of a unit hexagon
    # Edge 0: (1,0) to (0.5, sqrt(3)/2). Normal points outward: (sqrt(3)/2, 0.5) -> 30 deg
    unit_normals_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(30, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0)
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        # Normals for the outer hexagon sides
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def find_best_outer_angle(positions, angles, coarse=False):
        """Find the optimal rotation of the outer hexagon that minimizes its side length."""
        best_side = float('inf')
        best_angle = 0.0
        
        if coarse:
            # Very fast check for use inside optimizers
            test_angles = np.linspace(0, 60, 13) # 5 degree steps
        else:
            # Accurate check for final result
            test_angles = np.linspace(0, 60, 121) # 0.5 degree steps

        for test_angle in test_angles:
            side = calc_min_outer_side(positions, angles, test_angle)
            if side < best_side:
                best_side = side
                best_angle = test_angle
        
        if not coarse:
            # Refine with finer search around the best
            fine_angles = np.linspace(max(0, best_angle - 0.5), min(60, best_angle + 0.5), 21)
            for test_angle in fine_angles:
                side = calc_min_outer_side(positions, angles, test_angle)
                if side < best_side:
                    best_side = side
                    best_angle = test_angle
        
        return best_side, best_angle

    def sat_gap(poly1, poly2):
        """
        Returns the minimum separation distance between two convex polygons.
        If they overlap, returns a negative value (approximate penetration depth).
        If they are separated, returns the positive distance.
        """
        min_gap = float('inf')
        
        # Axes from poly1 edges
        for i in range(6):
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            norm = np.linalg.norm(normal1)
            if norm < 1e-9: continue
            axis = normal1 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Gap on this axis
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                # Overlapping on this axis
                min_gap = min(min_gap, gap)
            else:
                # Separated on this axis -> disjoint
                return gap
                
        # Axes from poly2 edges
        for i in range(6):
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            norm = np.linalg.norm(normal2)
            if norm < 1e-9: continue
            axis = normal2 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
                
        return min_gap

    def is_valid(positions, angles, tol=1e-5):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True

    def solve_lp_for_fixed_angles(angles, outer_angle=0.0):
        """
        Solve a linear program to find optimal positions for fixed inner angles
        and fixed outer angle, minimizing the outer hexagon side length.
        
        Variables: x_0, y_0, ..., x_{N-1}, y_{N-1}, t
        where t is the outer side length (scaled).
        
        Objective: minimize t
        
        Constraints:
        1. For each inner hexagon i and each outer normal direction k:
           max over vertices v of (v @ n_k) <= t * sqrt(3)/2
           This expands to: for each vertex v of hexagon i (relative to center),
           (center_i + v) @ n_k <= t * sqrt(3)/2
           => center_i @ n_k - t * sqrt(3)/2 <= -v @ n_k
        
        2. Non-overlap: For each pair (i, j), the Minkowski difference of hex_i and hex_j
           must not contain the origin. This is equivalent to: for each direction n,
           the projection of (center_i - center_j) onto n must be >= support(hex_i - hex_j, n)
           
           For two unit hexagons with angles a_i and a_j, the support function of their
           Minkowski difference in direction n is:
           h_{A-B}(n) = h_A(n) + h_A(-n) where h_A(n) = max_{v in A} v @ n
           
           Actually, for non-overlap: (center_i - center_j) @ n >= h_{hex_i}(n) + h_{hex_j}(-n)
           for all n. But this must hold for all n, which is infinite constraints.
           
           However, for convex polygons, it suffices to check the edge normals of both
           polygons. The critical directions are the normals to the edges of hex_i and hex_j.
           
           For hexagon i with angle a_i, the edge normals are at angles a_i + 30 + k*60.
           For hexagon j with angle a_j, the edge normals are at angles a_j + 30 + k*60.
           
           So we have 12 critical directions per pair.
           
           For each such direction n:
           (center_i - center_j) @ n >= h_i(n) + h_j(-n)
           where h_i(n) = max over vertices v of hex_i of (v @ n)
           
           Note: h_j(-n) = max_{v in hex_j} (v @ (-n)) = -min_{v in hex_j} (v @ n)
           But also h_j(-n) = h_j rotated by 180 = same as h_j for a centrally symmetric shape!
           Wait, a hexagon is centrally symmetric, so h(n) = h(-n).
           So the constraint is:
           (center_i - center_j) @ n >= h_i(n) + h_j(n)
           
           But this is for the Minkowski sum, not difference. Let me reconsider.
           
           Two convex sets A and B are disjoint iff there exists a direction n such that
           max_{a in A} a@n < min_{b in B} b@n, i.e., h_A(n) < -h_B(-n) = -h_B(n) [by central symmetry]
           Wait, that's not right either.
           
           Actually: A = center_i + H_i, B = center_j + H_j where H_i, H_j are unit hexagons.
           A and B are disjoint iff (center_i - center_j) is not in H_i - H_j.
           H_i - H_j = {a - b : a in H_i, b in H_j}
           The support function of H_i - H_j in direction n is h_{H_i - H_j}(n) = h_{H_i}(n) + h_{-H_j}(n) = h_{H_i}(n) + h_{H_j}(-n)
           Since H_j is centrally symmetric, h_{H_j}(-n) = h_{H_j}(n).
           So h_{H_i - H_j}(n) = h_{H_i}(n) + h_{H_j}(n).
           
           (center_i - center_j) is not in H_i - H_j iff for all n:
           (center_i - center_j) @ n < h_{H_i - H_j}(n)  OR  (center_i - center_j) @ n > h_{H_i - H_j}(-n)
           Wait, this is getting complicated. Let's use the separating axis theorem properly.
           
           Two convex polygons are disjoint iff there exists a direction n (from the set of edge normals of either polygon)
           such that the projections are disjoint.
           
           For our case, we want to ENSURE they are disjoint. So we need to enforce that for ALL critical directions n,
           the projections are disjoint. But that's a disjunction, which is not linear.
           
           Alternative approach: Use the fact that two convex polygons are disjoint iff the minimum distance
           between them is positive. But that's not linear either.
           
           Better approach: For a FIXED contact graph (which pairs are touching), we can linearize.
           But we don't know the contact graph a priori.
           
           Simpler approach for LP: Use a relaxed version where we enforce separation in a set of directions.
           If we enforce separation in enough directions, it's likely sufficient.
           
           Let's use the 12 critical directions per pair (6 from each hexagon's edge normals).
           For each direction n:
           (center_i - center_j) @ n >= h_i(n) + h_j(n)  OR  (center_i - center_j) @ n <= -h_i(n) - h_j(n)
           
           This is still a disjunction. To make it linear, we can introduce binary variables, but that makes it an ILP.
           
           Alternative: Use a penalty-based approach or just use the non-overlap constraint as:
           For each pair (i,j) and each of the 12 critical directions n:
           |(center_i - center_j) @ n| >= h_i(n) + h_j(n)
           
           This is still non-linear (absolute value).
           
           Let's try a different formulation. We can use the fact that for two convex polygons,
           they are disjoint iff the minimum of the support function of their Minkowski difference
           in any direction is positive.
           
           Actually, the simplest linear formulation for non-overlap of two convex polygons is:
           For each edge normal direction n of polygon i:
           (center_j - center_i) @ n >= h_i(n) + h_j(n)  OR  (center_j - center_i) @ n <= -h_i(n) - h_j(n)
           
           This is still a disjunction.
           
           Given the complexity, let's use a different strategy:
           1. Start with a good initial configuration from the grid search.
           2. Use SLSQP with the non-overlap constraints (which are non-smooth but workable).
           3. Then, for the final polishing, use the LP formulation with a FIXED set of active constraints.
           
           For the LP, we can assume that the active constraints are the ones that are tight
           in the SLSQP solution. We can identify these and then solve the LP with only those constraints.
           
           Let's implement a simpler LP: minimize t subject to the outer hexagon constraints only,
           and then check if the solution is valid. If not, add constraints and re-solve.
        """
        # This is a complex formulation. Let's implement a simpler version:
        # Minimize t subject to the outer hexagon containing all inner hexagons.
        # We'll ignore the non-overlap constraints in the LP and check validity after.
        
        # Variables: x_0, y_0, ..., x_{N-1}, y_{N-1}, t
        # Total variables: 2*N + 1
        
        # Objective: minimize t
        c = np.zeros(2*N + 1)
        c[2*N] = 1.0  # Minimize t
        
        # Constraints: For each inner hexagon i and each outer normal direction k:
        # (center_i + v) @ n_k <= t * sqrt(3)/2
        # => center_i @ n_k - t * sqrt(3)/2 <= -v @ n_k
        
        theta_rad = math.radians(outer_angle)
        outer_normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            outer_normals.append([math.cos(ang), math.sin(ang)])
        outer_normals = np.array(outer_normals)
        
        A_ub = []
        b_ub = []
        
        for i in range(N):
            ang_i = angles[i]
            verts_i = get_hex_verts(0, 0, 1.0, ang_i)  # Relative to center
            
            for k in range(6):
                n_k = outer_normals[k]
                # For each vertex v of hexagon i:
                for v in verts_i:
                    # (center_i + v) @ n_k <= t * sqrt(3)/2
                    # center_i @ n_k - t * sqrt(3)/2 <= -v @ n_k
                    row = np.zeros(2*N + 1)
                    row[2*i] = n_k[0]
                    row[2*i+1] = n_k[1]
                    row[2*N] = -math.sqrt(3)/2
                    A_ub.append(row)
                    b_ub.append(-v @ n_k)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds: x_i, y_i in [-10, 10], t in [0, inf)
        bounds = [(-10, 10)] * (2*N) + [(0, None)]
        
        # Solve LP
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            positions = res.x[:2*N].reshape(N, 2)
            t = res.x[2*N]
            side = t * 2.0 / math.sqrt(3)
            return positions, side
        else:
            return None, float('inf')

    # --- PHASE 1: DISCRETE GRID SEARCH FOR TOPOLOGY ---
    sqrt3 = math.sqrt(3)
    
    best_valid_config = None
    best_valid_score = float('inf')
    
    # Standard dense packing lattice vectors
    base_v1 = np.array([1.5, sqrt3/2])
    base_v2 = np.array([0, sqrt3])
    
    # Try a few scales and rotations. Keep it small to avoid timeout.
    scales = [0.98, 1.0, 1.02]
    rotations = [0, 30]
    
    for sx in scales:
        for sy in scales:
            v1 = base_v1 * np.array([sx, 1.0])
            v2 = base_v2 * np.array([1.0, sy])
            
            points = []
            for i in range(-3, 4):
                for j in range(-3, 4):
                    p = i * v1 + j * v2
                    points.append(p)
            
            points.sort(key=lambda p: np.dot(p, p))
            selected = np.array(points[:12])
            centroid = np.mean(selected, axis=0)
            selected -= centroid
            
            for rot_deg in rotations:
                rot_rad = math.radians(rot_deg)
                rot_mat = np.array([[math.cos(rot_rad), -math.sin(rot_rad)], 
                                    [math.sin(rot_rad), math.cos(rot_rad)]])
                rotated_selected = selected @ rot_mat.T
                angles = np.full(N, rot_deg)
                
                if is_valid(rotated_selected, angles, tol=1e-6):
                    # Use coarse angle search for speed in grid phase
                    score, _ = find_best_outer_angle(rotated_selected, angles, coarse=True)
                    if score < best_valid_score:
                        best_valid_score = score
                        best_valid_config = (rotated_selected.copy(), angles.copy())
                
                if time.time() > deadline - 30:
                    break
            if time.time() > deadline - 30:
                break
        if time.time() > deadline - 30:
            break

    # Fallback if grid search failed or timed out
    if best_valid_config is None:
        # Simple fallback: honeycomb subset
        lattice_points = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * 1.5
                y = i * sqrt3 / 2.0 + j * sqrt3
                lattice_points.append((x, y))
        lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
        init_positions = np.array(lattice_points[:12])
        init_positions -= np.mean(init_positions, axis=0)
        init_angles = np.zeros(N)
        best_valid_config = (init_positions, init_angles)
        best_valid_score, _ = find_best_outer_angle(init_positions, init_angles, coarse=True)

    init_positions, init_angles = best_valid_config
    x0 = np.concatenate([init_positions.flatten(), init_angles])
    
    # --- PHASE 2: SLSQP REFINEMENT ---
    # Use a FAST objective for SLSQP (coarse outer angle) to allow more iterations
    def objective_fast(params):
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        side, _ = find_best_outer_angle(positions, angles, coarse=True)
        return side

    def make_constraint(i, j):
        def constraint(params):
            pos_i = params[2*i:2*i+2]
            pos_j = params[2*j:2*j+2]
            ang_i = params[2*N + i]
            ang_j = params[2*N + j]
            
            poly_i = get_hex_verts(pos_i[0], pos_i[1], 1.0, ang_i)
            poly_j = get_hex_verts(pos_j[0], pos_j[1], 1.0, ang_j)
            
            return sat_gap(poly_i, poly_j)
        return constraint

    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
    
    max_radius = 10.0
    bounds = []
    for i in range(N):
        bounds.append((-max_radius, max_radius))
        bounds.append((-max_radius, max_radius))
    for i in range(N):
        bounds.append((0, 360))

    best_score = best_valid_score
    best_result = x0.copy()
    best_outer_angle = 0.0

    if time.time() < deadline - 20:
        try:
            # Reduced maxiter to ensure it finishes
            res = minimize(
                objective_fast, 
                x0, 
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-8, 'disp': False}
            )
            
            if res.success or res.fun < best_score:
                pos = res.x[:2*N].reshape(N, 2)
                ang = res.x[2*N:]
                
                if is_valid(pos, ang, tol=1e-4):
                    score, outer_ang = find_best_outer_angle(pos, ang, coarse=False)
                    if score < best_score:
                        best_score = score
                        best_result = res.x.copy()
                        best_outer_angle = outer_ang
        except Exception as e:
            print(f"SLSQP error: {e}")

    # --- PHASE 3: LP REFINEMENT WITH FIXED ANGLES ---
    if time.time() < deadline - 10:
        # Try to improve using LP with fixed angles from SLSQP result
        final_positions = best_result[:2*N].reshape(N, 2)
        final_angles = best_result[2*N:]
        
        # Try a few outer angles
        for outer_angle in [0, 15, 30, 45]:
            lp_positions, lp_side = solve_lp_for_fixed_angles(final_angles, outer_angle)
            
            if lp_positions is not None:
                # Check if the LP solution is valid (non-overlapping)
                if is_valid(lp_positions, final_angles, tol=1e-5):
                    # Calculate the actual outer side length
                    actual_side, actual_outer_angle = find_best_outer_angle(lp_positions, final_angles, coarse=False)
                    if actual_side < best_score:
                        best_score = actual_side
                        best_outer_angle = actual_outer_angle
                        best_result = np.concatenate([lp_positions.flatten(), final_angles])

    # --- PHASE 4: LOCAL POLISHING (Nelder-Mead) ---
    if time.time() < deadline - 5:
        def obj_penalty(params):
            positions = params[:2*N].reshape(N, 2)
            angles = params[2*N:]
            side_len, _ = find_best_outer_angle(positions, angles, coarse=True)
            penalty = 0.0
            polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
            for i in range(N):
                for j in range(i+1, N):
                    gap = sat_gap(polys[i], polys[j])
                    if gap < 0:
                        penalty += 1000.0 * gap**2
            return side_len + penalty

        try:
            res2 = minimize(
                obj_penalty, 
                best_result, 
                method='Nelder-Mead',
                options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True, 'disp': False}
            )
            
            pos = res2.x[:2*N].reshape(N, 2)
            ang = res2.x[2*N:]
            
            if is_valid(pos, ang, tol=1e-5):
                score2, outer_ang2 = find_best_outer_angle(pos, ang, coarse=False)
                if score2 < best_score:
                    best_score = score2
                    best_result = res2.x.copy()
                    best_outer_angle = outer_ang2
        except Exception:
            pass

    # Finalize
    final_positions = best_result[:2*N].reshape(N, 2)
    final_angles = best_result[2*N:]
    
    # One last accurate check for outer angle
    final_side, final_outer_angle = find_best_outer_angle(final_positions, final_angles, coarse=False)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, final_outer_angle

if __name__ == "__main__":
    construct()
