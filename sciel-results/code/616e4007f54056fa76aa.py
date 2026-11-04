import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
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
            test_angles = np.linspace(0, 60, 61)
        else:
            test_angles = np.linspace(0, 60, 361)

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

    def smooth_penalty(params, beta=50.0):
        """
        Vectorized smooth penalty for overlaps.
        Uses a soft-min approximation for the SAT gap.
        """
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        # Objective: minimize outer side length
        side_len = calc_min_outer_side(positions, angles, 0.0)
        
        # Penalty: sum of squared soft-negative gaps
        penalty = 0.0
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        
        for i in range(N):
            for j in range(i+1, N):
                gap = sat_gap(polys[i], polys[j])
                if gap < 0:
                    # Soft penalty: smooth approximation of max(0, -gap)^2
                    # Using log-sum-exp style or just quadratic with damping
                    # Simple quadratic is non-smooth at 0, but L-BFGS can handle it with care
                    # Better: use a smooth hinge like -log(1+exp(-beta*gap)) ? No, that's for log-barrier.
                    # Let's use a smooth quadratic: 0.5 * (gap - |gap|)^2 is still non-diff at 0.
                    # We will rely on the fact that L-BFGS uses finite diffs or we can provide a smooth approx.
                    # For simplicity and speed, we use a high-weight quadratic penalty.
                    penalty += 100.0 * gap * gap
        return side_len + penalty

    def objective_lbfgs(params):
        # L-BFGS-B requires a smooth function.
        # We use a "soft" penalty that is differentiable everywhere.
        # Approximation of max(0, x)^2 is not differentiable at 0.
        # However, in practice, L-BFGS-B often works if the penalty is strong enough
        # and the initial guess is close to feasible.
        # To be safer, we can use a smooth approximation like:
        # smooth_max(0, x) = (x + sqrt(x^2 + eps))/2
        # Then square it.
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        side_len = calc_min_outer_side(positions, angles, 0.0)
        
        penalty = 0.0
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        
        for i in range(N):
            for j in range(i+1, N):
                gap = sat_gap(polys[i], polys[j])
                if gap < 0:
                    # Smooth hinge
                    eps = 1e-4
                    soft_gap = (gap - math.sqrt(gap*gap + eps)) / 2.0  # This is negative
                    # We want to penalize negative gaps
                    # If gap is -0.1, soft_gap is approx -0.1
                    # Penalty should be positive
                    # Let's just use gap^2 for now, it's robust enough for L-BFGS if we start valid
                    penalty += 1000.0 * gap * gap
        return side_len + penalty

    # --- PHASE 1: GENERATE DIVERSE INITIAL CONFIGURATIONS ---
    
    sqrt3 = math.sqrt(3)
    
    # We will generate several distinct topologies
    # 1. Honeycomb subset (standard)
    # 2. Ring/Flower (1 center, 6 around, 5 more?)
    # 3. Linear strips
    # 4. Random valid-ish starts
    
    candidates = []
    
    # Topology 1: Honeycomb
    # Lattice vectors
    v1 = np.array([1.5, sqrt3/2])
    v2 = np.array([0, sqrt3])
    pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * v1 + j * v2
            pts.append(p)
    pts.sort(key=lambda p: p[0]**2 + p[1]**2)
    init1 = np.array(pts[:12])
    init1 -= np.mean(init1, axis=0)
    ang1 = np.zeros(N)
    candidates.append((init1, ang1))
    
    # Topology 2: "Flower" - 1 center, 6 around, 5 more in second ring?
    # Center at (0,0)
    # 6 neighbors at distance 2*sqrt(3)/2 * ? 
    # Distance between centers of touching hexagons is sqrt(3) * 2? No.
    # Inradius is sqrt(3)/2. Distance between parallel sides is sqrt(3).
    # If aligned, center distance is 1.5 * 2 = 3? No.
    # Let's stick to the lattice which is known to be tight.
    # Try a rotated lattice?
    init2 = init1.copy()
    rot_rad = math.radians(30)
    rot_mat = np.array([[math.cos(rot_rad), -math.sin(rot_rad)], [math.sin(rot_rad), math.cos(rot_rad)]])
    init2 = init2 @ rot_mat.T
    ang2 = np.full(N, 30.0)
    candidates.append((init2, ang2))
    
    # Topology 3: Two rows of 6?
    # Row 1: y=0, x = -4.5, -3, -1.5, 0, 1.5, 3, 4.5 ? 
    # Width of hex is 2. Spacing 1.5? They would overlap.
    # Spacing must be at least 2?
    # Let's try a grid of 3x4 with adjusted spacing.
    # This might not be valid, but L-BFGS can fix it if penalty is high.
    
    # Random starts: Generate random positions and angles, then project to valid?
    # Too hard. Let's rely on the lattice variants.
    
    # Try different subsets of the lattice?
    # The "closest 12" is usually optimal for symmetric containers.
    
    # Let's add a "compressed" lattice start
    scale = 0.95
    init3 = init1 * scale
    ang3 = np.zeros(N)
    candidates.append((init3, ang3))
    
    # --- PHASE 2: OPTIMIZE EACH CANDIDATE ---
    
    best_score = float('inf')
    best_result = None
    best_outer_angle = 0.0
    
    # Bounds for positions and angles
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
    for i in range(N):
        bounds.append((0, 60)) # Angles modulo 60 due to symmetry? 
        # Actually, hexagon symmetry is 60 deg. So 0-60 is enough for unique orientations.
        
    for idx, (pos_init, ang_init) in enumerate(candidates):
        if time.time() > deadline - 20:
            break
            
        x0 = np.concatenate([pos_init.flatten(), ang_init])
        
        # Use L-BFGS-B with penalty
        # This is much faster than SLSQP with constraints.
        try:
            # We will run a few iterations of L-BFGS-B
            # Since the penalty is non-smooth at 0, we might need to be careful.
            # But L-BFGS-B is robust.
            res = minimize(
                objective_lbfgs,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            
            pos_res = res.x[:2*N].reshape(N, 2)
            ang_res = res.x[2*N:]
            
            # Check validity
            if is_valid(pos_res, ang_res, tol=1e-5):
                # Find best outer angle
                score, out_ang = find_best_outer_angle(pos_res, ang_res, coarse=True)
                if score < best_score:
                    best_score = score
                    best_result = res.x.copy()
                    best_outer_angle = out_ang
            else:
                # If invalid, try to repair by pushing apart?
                # Or just discard. L-BFGS with penalty usually stays valid if started valid.
                # If started invalid (e.g. compressed lattice), it might get stuck.
                # Let's try a Nelder-Mead polish on the best valid one found so far?
                pass
                
        except Exception as e:
            # print(f"Optimization failed for candidate {idx}: {e}")
            pass

    # If no valid result found, fallback
    if best_result is None:
        # Use the first candidate and force it to be valid by expanding?
        # Or just return the initial lattice which is valid.
        pos_init, ang_init = candidates[0]
        x0 = np.concatenate([pos_init.flatten(), ang_init])
        best_result = x0
        best_score, best_outer_angle = find_best_outer_angle(pos_init, ang_init, coarse=True)

    # --- PHASE 3: POLISH WITH NELLER-MEAD ---
    # Nelder-Mead doesn't need gradients and can handle non-smoothness better.
    if time.time() < deadline - 10:
        def obj_nm(params):
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
            res_nm = minimize(
                obj_nm,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True, 'disp': False}
            )
            
            pos_nm = res_nm.x[:2*N].reshape(N, 2)
            ang_nm = res_nm.x[2*N:]
            
            if is_valid(pos_nm, ang_nm, tol=1e-5):
                score_nm, out_ang_nm = find_best_outer_angle(pos_nm, ang_nm, coarse=True)
                if score_nm < best_score:
                    best_score = score_nm
                    best_result = res_nm.x.copy()
                    best_outer_angle = out_ang_nm
        except Exception:
            pass

    # --- PHASE 4: FINAL PRECISION ---
    if time.time() < deadline - 5:
        # One more L-BFGS step with tighter tolerance
        try:
            res_final = minimize(
                objective_lbfgs,
                best_result,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10, 'disp': False}
            )
            pos_f = res_final.x[:2*N].reshape(N, 2)
            ang_f = res_final.x[2*N:]
            if is_valid(pos_f, ang_f, tol=1e-6):
                score_f, out_ang_f = find_best_outer_angle(pos_f, ang_f, coarse=False)
                if score_f < best_score:
                    best_score = score_f
                    best_result = res_final.x.copy()
                    best_outer_angle = out_ang_f
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
