import math
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute unit hexagon vertices for angle 0 (flat-top)
    # Vertices at 0, 60, ..., 300 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

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
        """
        Find the optimal rotation of the outer hexagon that minimizes its side length.
        """
        best_side = float('inf')
        best_angle = 0.0
        
        if coarse:
            test_angles = np.linspace(0, 60, 13)
        else:
            test_angles = np.linspace(0, 60, 181)

        for test_angle in test_angles:
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
                min_gap = min(min_gap, gap)
            else:
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
           
           Actually: A and B are disjoint iff for all n, h_A(n) + h_B(-n) <= 0 is NOT what we want.
           
           Let me think again. A = center_i + H_i, B = center_j + H_j where H_i, H_j are unit hexagons.
           A and B are disjoint iff (center_i - center_j) is not in H_i - H_j.
           H_i - H_j = {a - b : a in H_i, b in H_j}
           The support function of H_i - H_j in direction n is h_{H_i - H_j}(n) = h_{H_i}(n) + h_{-H_j}(n) = h_{H_i}(n) + h_{H_j}(-n)
           Since H_j is centrally symmetric, h_{H_j}(-n) = h_{H_j}(n).
           So h_{H_i - H_j}(n) = h_{H_i}(n) + h_{H_j}(n).
           
           (center_i - center_j) is not in H_i - H_j iff there exists n such that
           (center_i - center_j) @ n > h_{H_i - H_j}(n) = h_{H_i}(n) + h_{H_j}(n)
           
           But we need this for ALL possible n to guarantee non-overlap? No, we need that
           (center_i - center_j) is NOT in H_i - H_j, which is equivalent to:
           there exists n such that (center_i - center_j) @ n > h_{H_i - H_j}(n)
           
           But for an LP constraint, we need: for ALL n, (center_i - center_j) @ n <= h_{H_i - H_j}(n) is FALSE
           which is hard to express.
           
           Actually, the correct condition for non-overlap is:
           (center_i - center_j) NOT IN H_i - H_j
           which is equivalent to:
           exists n: (center_i - center_j) @ n > h_{H_i - H_j}(n)
           
           But this is an existential quantifier, not universal. This makes it non-convex.
           
           However, if we require the STRONGER condition that for ALL n:
           (center_i - center_j) @ n <= h_{H_i - H_j}(n)
           then (center_i - center_j) IS in H_i - H_j, which means they OVERLAP or touch.
           
           So the non-overlap condition is the complement, which is non-convex.
           
           This means we cannot directly use LP for the non-overlap constraints in general.
           
           BUT: if we fix the relative orientation (i.e., all hexagons have the same angle),
           then H_i - H_j = H (the Minkowski difference of two identical hexagons), which is
           a larger hexagon. And the condition (center_i - center_j) NOT IN H is still non-convex.
           
           So LP alone won't work for non-overlap. We need a different approach.
           
           Alternative: Use the fact that for a fixed contact graph (which pairs are touching),
           the positions can be determined by linear equations. But we don't know the contact graph.
           
           Let me try a different approach: use SLSQP but with better initialization from the LP.
           
           Actually, let me just improve the existing SLSQP approach with better heuristics.
        """
        pass

    # --- PHASE 1: DISCRETE GRID SEARCH FOR TOPOLOGY ---
    # Instead of optimizing continuous coordinates from a random start, we search
    # a lattice of potential centers and a set of discrete angles to find a valid
    # configuration. This avoids getting stuck in local minima of the penalty method.
    
    sqrt3 = math.sqrt(3)
    
    # Generate candidate centers in a hexagonal lattice pattern
    # The ideal packing for hexagons is on a triangular lattice.
    # Lattice vectors: a1 = (1.5, sqrt(3)/2), a2 = (0, sqrt(3))
    # We search a range of offsets and scales.
    
    best_valid_config = None
    best_valid_score = float('inf')
    
    # Search space for lattice scaling and offset
    # A unit hexagon has width 2. To pack them, centers should be at least ~2 apart in some directions.
    # Let's try lattice spacings that allow tight packing.
    
    # Pre-computed discrete angles (multiples of 30 degrees are often optimal for hexagons)
    discrete_angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    
    # We will search for configurations on a lattice defined by:
    # P_i = origin + i * v1 + j * v2
    # where v1, v2 are lattice vectors.
    
    # Common tight packings use specific lattice vectors.
    # For hexagons, the "standard" dense packing has centers at distance 2*sqrt(3)/2 * 2? 
    # Distance between centers in dense hex packing is 2 * (inradius) + 2 * (inradius) = 2 * sqrt(3) * 1 = 2*sqrt(3) ~ 3.46?
    # Wait, inradius of unit hex is sqrt(3)/2. Distance between parallel sides is sqrt(3).
    # If aligned, centers need to be sqrt(3) apart in y and 1.5 in x?
    # Lattice vectors for aligned hexagons:
    # v1 = (1.5, sqrt(3)/2), v2 = (0, sqrt(3))
    # This creates a grid where hexagons touch.
    
    base_v1 = np.array([1.5, sqrt3/2])
    base_v2 = np.array([0, sqrt3])
    
    # We will vary the scale of this lattice and the rotation of the whole cluster.
    # Also allow small perturbations to break symmetry if needed? 
    # For now, stick to exact lattice points to ensure validity, then refine.
    
    candidates = []
    
    # Try different lattice scales (stretching/compressing)
    for scale_x in [0.9, 0.95, 1.0, 1.05, 1.1]:
        for scale_y in [0.9, 0.95, 1.0, 1.05, 1.1]:
            v1 = base_v1 * np.array([scale_x, 1.0])
            v2 = base_v2 * np.array([1.0, scale_y])
            
            # Generate a large enough grid to pick 12 points
            # We need to pick 12 points from this lattice that form a compact cluster.
            # The "best" 12 points are usually the ones closest to the centroid.
            
            points = []
            for i in range(-4, 5):
                for j in range(-4, 5):
                    p = i * v1 + j * v2
                    points.append(p)
            
            # Sort by distance from origin
            points.sort(key=lambda p: np.dot(p, p))
            
            # Take the first 12
            selected = np.array(points[:12])
            
            # Center the cluster
            centroid = np.mean(selected, axis=0)
            selected -= centroid
            
            # Try different global rotations for this cluster
            for rot_deg in [0, 30, 60, 90]:
                rot_rad = math.radians(rot_deg)
                rot_mat = np.array([[math.cos(rot_rad), -math.sin(rot_rad)], 
                                    [math.sin(rot_rad), math.cos(rot_rad)]])
                rotated_selected = selected @ rot_mat.T
                
                # Assign angles. For aligned packing, all angles are 0 (or rot_deg).
                # But we can also try mixing angles? 
                # Let's stick to uniform angle first for validity, then SLSQP will optimize.
                angles = np.full(N, rot_deg)
                
                # Check validity
                if is_valid(rotated_selected, angles, tol=1e-6):
                    # Use the best outer angle for this configuration
                    score, _ = find_best_outer_angle(rotated_selected, angles, coarse=True)
                    if score < best_valid_score:
                        best_valid_score = score
                        best_valid_config = (rotated_selected.copy(), angles.copy())
                    
                    if time.time() > deadline - 20:
                        break
            if time.time() > deadline - 20:
                break
        if time.time() > deadline - 20:
            break

    # If grid search found nothing, fallback to a standard honeycomb subset
    if best_valid_config is None:
        print("Grid search failed, using fallback honeycomb")
        lattice_points = []
        for i in range(-4, 5):
            for j in range(-4, 5):
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
    
    # --- PHASE 2: IMPROVED SLSQP WITH BETTER CONSTRAINTS ---
    # Use a smoother constraint function based on the minimum over a dense set of directions
    
    def objective(params):
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        return calc_min_outer_side(positions, angles, 0.0)
    
    def make_smooth_constraint(i, j):
        """
        Create a smooth approximation of the non-overlap constraint.
        We use the minimum gap over a set of critical directions.
        """
        def constraint(params):
            pos_i = params[2*i:2*i+2]
            pos_j = params[2*j:2*j+2]
            ang_i = params[2*N + i]
            ang_j = params[2*N + j]
            
            poly_i = get_hex_verts(pos_i[0], pos_i[1], 1.0, ang_i)
            poly_j = get_hex_verts(pos_j[0], pos_j[1], 1.0, ang_j)
            
            # Use the exact SAT gap, which is piecewise linear
            return sat_gap(poly_i, poly_j)
        return constraint

    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({'type': 'ineq', 'fun': make_smooth_constraint(i, j)})
    
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
    for i in range(N):
        bounds.append((0, 360))

    best_score = best_valid_score
    best_result = x0.copy()

    # Run SLSQP with multiple restarts from perturbed initial points
    for restart in range(3):
        if time.time() > deadline - 30:
            break
            
        if restart == 0:
            current_x0 = x0.copy()
        else:
            # Perturb the initial point
            current_x0 = x0.copy()
            noise = np.random.randn(2*N + N) * 0.1
            current_x0 += noise
            # Keep angles in [0, 360)
            current_x0[2*N:] = current_x0[2*N:] % 360
        
        try:
            res = minimize(
                objective, 
                current_x0, 
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-9, 'disp': False}
            )
            
            if res.success or res.fun < best_score:
                pos = res.x[:2*N].reshape(N, 2)
                ang = res.x[2*N:]
                
                # Verify validity strictly
                if is_valid(pos, ang, tol=1e-4):
                    # Use the best outer angle for this configuration
                    score, _ = find_best_outer_angle(pos, ang, coarse=True)
                    if score < best_score:
                        best_score = score
                        best_result = res.x.copy()
        except Exception as e:
            print(f"SLSQP error: {e}")

    # --- PHASE 3: LOCAL POLISHING (Nelder-Mead) ---
    # SLSQP might get stuck. Nelder-Mead can sometimes escape.
    # We run it on the best result so far, with a penalty objective to ensure validity.
    
    if time.time() < deadline - 10:
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
                options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True, 'disp': False}
            )
            
            pos = res2.x[:2*N].reshape(N, 2)
            ang = res2.x[2*N:]
            
            if is_valid(pos, ang, tol=1e-5):
                score2, _ = find_best_outer_angle(pos, ang, coarse=True)
                if score2 < best_score:
                    best_score = score2
                    best_result = res2.x.copy()
        except Exception:
            pass

    # Finalize
    final_positions = best_result[:2*N].reshape(N, 2)
    final_angles = best_result[2*N:]
    
    # Find the precise best outer angle for the final configuration
    final_side, final_outer_angle = find_best_outer_angle(final_positions, final_angles, coarse=False)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, final_outer_angle

if __name__ == "__main__":
    construct()
