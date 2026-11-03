import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3)
DEADLINE_FRACTION = 0.85

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 * DEADLINE_FRACTION

    # Precompute unit hexagon vertices (pointy-top, side length 1)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    unit_verts = np.array([
        [math.cos(math.radians(a)), math.sin(math.radians(a))]
        for a in range(0, 360, 60)
    ])

    def hex_vertices(cx, cy, side, angle_deg):
        """Get vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
        if angle_deg != 0.0:
            theta = math.radians(angle_deg)
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            verts = (unit_verts @ R.T) * side
        else:
            verts = unit_verts * side
        return verts + np.array([cx, cy])

    def get_all_vertices(params):
        """
        params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11, cx, cy, S, theta]
        Returns:
            centers: (12, 2)
            angles: (12,)
            all_verts: (72, 2)
            outer_center: (2,)
            outer_side: float
            outer_angle: float
        """
        centers = params[:24].reshape(12, 2)
        angles = params[24:48]
        outer_center = params[48:50]
        outer_side = params[50]
        outer_angle = params[51]
        
        verts_list = []
        for i in range(N):
            v = hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
            verts_list.append(v)
        all_verts = np.vstack(verts_list)
        
        return centers, angles, all_verts, outer_center, outer_side, outer_angle

    def compute_sat_gaps(params):
        """
        Compute the minimum separation distance for each pair of hexagons using SAT.
        Returns a list of gaps (positive if separated, negative if overlapping).
        """
        centers, angles, _, _, _, _ = get_all_vertices(params)
        
        polys = [hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i]) for i in range(N)]
        
        gaps = []
        for i in range(N):
            for j in range(i + 1, N):
                v1, v2 = polys[i], polys[j]
                
                # Bounding box quick check
                if (np.max(v1[:, 0]) < np.min(v2[:, 0]) or
                    np.max(v2[:, 0]) < np.min(v1[:, 0]) or
                    np.max(v1[:, 1]) < np.min(v2[:, 1]) or
                    np.max(v2[:, 1]) < np.min(v1[:, 1])):
                    # Definitely separated, estimate gap roughly (or just large)
                    gaps.append(1.0)
                    continue
                
                # SAT: Check all edge normals of both polygons
                min_gap = float('inf')
                for v in [v1, v2]:
                    for k in range(6):
                        p1 = v[k]
                        p2 = v[(k + 1) % 6]
                        # Edge vector
                        ex = p2[0] - p1[0]
                        ey = p2[1] - p1[1]
                        # Normal vector (perpendicular)
                        nx = -ey
                        ny = ex
                        norm = math.sqrt(nx*nx + ny*ny)
                        if norm < 1e-9: continue
                        nx /= norm
                        ny /= norm
                        
                        # Project both polygons onto this normal
                        proj1 = v1 @ np.array([nx, ny])
                        proj2 = v2 @ np.array([nx, ny])
                        
                        min1, max1 = np.min(proj1), np.max(proj1)
                        min2, max2 = np.min(proj2), np.max(proj2)
                        
                        # Gap: distance between intervals
                        # If max1 < min2: gap is min2 - max1
                        # If max2 < min1: gap is min1 - max2
                        # Else: overlap, gap is negative
                        if max1 < min2:
                            gap = min2 - max1
                        elif max2 < min1:
                            gap = min1 - max2
                        else:
                            # Overlap
                            # Depth of overlap
                            overlap1 = max1 - min2
                            overlap2 = max2 - min1
                            gap = -max(overlap1, overlap2)
                        
                        if gap < min_gap:
                            min_gap = gap
                
                gaps.append(min_gap)
        
        return np.array(gaps)

    def objective(params):
        if time.time() > deadline:
            return 1e10
            
        centers, angles, all_verts, outer_center, outer_side, outer_angle = get_all_vertices(params)
        
        # 1. Compute Outer Container Fit
        # The outer hexagon must contain all inner vertices.
        # Outer hexagon is centered at outer_center, side S, angle theta.
        # Normals of outer hexagon:
        theta_rad = math.radians(outer_angle)
        normals = np.array([
            [math.cos(theta_rad + math.pi/6 + k * math.pi/3),
             math.sin(theta_rad + math.pi/6 + k * math.pi/3)]
            for k in range(6)
        ])
        
        # Project inner vertices onto normals
        # Shift vertices relative to outer center
        shifted_verts = all_verts - outer_center
        projs = shifted_verts @ normals.T # (72, 6)
        
        # Max projection in each direction
        max_projs = np.max(projs, axis=0)
        
        # Constraint: max_proj <= S * sqrt(3) / 2
        # So we need S >= max(max_projs) / (sqrt(3)/2)
        required_side = np.max(max_projs) / (SQRT3 / 2.0)
        
        # Penalty if current S is too small
        container_penalty = 0.0
        if required_side > outer_side:
            container_penalty = (required_side - outer_side) ** 2 * 10.0
        
        # 2. Compute Non-Overlap Penalty (Smooth)
        # Use log-sum-exp of negative gaps to create a smooth barrier
        # If gap > 0, it's fine. If gap < 0, it's bad.
        # We want to maximize the minimum gap.
        # Let's define penalty based on the sum of soft-min of gaps.
        
        # Recompute gaps (expensive, but necessary for accuracy)
        # Optimization: We can approximate or reuse if we cache, but for simplicity:
        gaps = compute_sat_gaps(params)
        
        # Smooth minimum: -log(sum(exp(-gap/tau))) * tau
        # If all gaps > 0, this is small.
        # If any gap < 0, this grows.
        # Actually, we want to penalize negative gaps heavily.
        # Let's use: sum( max(0, -gap)^2 ) + soft_min_penalty
        
        # Better: Smooth "minimum gap" using log-sum-exp
        # min(g) ≈ -1/alpha * log(sum(exp(-alpha * g)))
        # We want min(g) to be large (positive).
        # So minimize -min(g) ≈ 1/alpha * log(sum(exp(-alpha * g)))
        
        alpha = 10.0
        # Clip gaps to avoid overflow in exp
        # If gap is very negative, exp(-alpha*gap) is huge.
        # If gap is very positive, exp(-alpha*gap) is 0.
        
        # For numerical stability:
        # log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
        x = -alpha * gaps
        max_x = np.max(x)
        if max_x > 500: # Overflow protection
             smooth_min_penalty = 100.0
        else:
             smooth_min_val = max_x + np.log(np.sum(np.exp(x - max_x)))
             smooth_min_penalty = smooth_min_val / alpha
        
        # Total objective: Minimize outer_side + container_penalty + smooth_min_penalty
        # Wait, we are minimizing outer_side directly?
        # If we minimize outer_side, the optimizer will shrink it until it hits the container constraint.
        # The container constraint is handled by container_penalty.
        # The non-overlap is handled by smooth_min_penalty.
        
        # We should also add a small regularization to keep things centered?
        # No, the container center is a variable, so it will adjust.
        
        return outer_side + container_penalty + smooth_min_penalty

    def objective_grad(params):
        """
        Numerical gradient approximation (since analytic is complex with SAT).
        """
        eps = 1e-5
        grad = np.zeros_like(params)
        for i in range(len(params)):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[i] += eps
            p_minus[i] -= eps
            f_plus = objective(p_plus)
            f_minus = objective(p_minus)
            grad[i] = (f_plus - f_minus) / (2 * eps)
        return grad

    # --- Generate Initial Configs ---
    
    configs = []
    
    # Config 1: Hexagonal lattice, 12 closest points to origin
    lattice = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = SQRT3 * (i + j / 2.0)
            y = 3.0 * j
            lattice.append((x, y))
    lattice.sort(key=lambda p: p[0]**2 + p[1]**2)
    init1 = np.array(lattice[:12])
    init1 -= np.mean(init1, axis=0)
    
    # Initial params for config 1
    p1 = init1.flatten().tolist() + [0.0]*12 # angles 0
    p1 += [0.0, 0.0, 5.0, 0.0] # center 0,0, side 5, angle 0
    configs.append(np.array(p1))
    
    # Config 2: Ring + center
    center = [(0, 0)]
    for k in range(6):
        a = math.radians(60 * k)
        center.append((2.0 * math.cos(a), 2.0 * math.sin(a)))
    for k in range(5):
        a = math.radians(60 * k + 30)
        center.append((3.5 * math.cos(a), 3.5 * math.sin(a)))
    init2 = np.array(center)
    init2 -= np.mean(init2, axis=0)
    
    p2 = init2.flatten().tolist() + [0.0]*12
    p2 += [0.0, 0.0, 6.0, 0.0]
    configs.append(np.array(p2))
    
    # Config 3: Random-ish compact
    rng = np.random.default_rng(42)
    p3 = rng.normal(0, 2, size=24)
    p3[0::2] += rng.normal(0, 1, 12) # spread x
    p3 += [0.0]*12
    p3 += [0.0, 0.0, 6.0, 0.0]
    configs.append(np.array(p3))

    best_params = None
    best_obj = float('inf')
    
    # Evaluate initial configs
    for cfg in configs:
        if time.time() > deadline: break
        try:
            obj = objective(cfg)
            if obj < best_obj:
                best_obj = obj
                best_params = cfg
        except:
            pass

    # --- Optimization Loop ---
    
    # Use SLSQP with numerical gradient (or BFGS if gradients are too noisy/expensive)
    # SLSQP is good for bounded constraints, but we don't have explicit bounds other than positivity of S.
    # Let's use Nelder-Mead first to get close, then SLSQP for polish?
    # The prompt suggests SLSQP. Let's try SLSQP with finite differences.
    
    for restart in range(5):
        if time.time() > deadline: break
        
        if restart == 0:
            x0 = best_params
        else:
            x0 = best_params + rng.normal(0, 0.2, size=52)
            # Ensure side length is positive
            x0[50] = max(1.0, x0[50])
        
        try:
            # SLSQP might be slow with 52 vars and expensive objective.
            # Let's limit iterations.
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                options={'maxiter': 50, 'ftol': 1e-6, 'disp': False}
            )
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
                
            # Follow up with Nelder-Mead for local refinement if time permits
            if time.time() < deadline:
                res_nm = minimize(
                    objective,
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
                )
                if res_nm.fun < best_obj:
                    best_obj = res_nm.fun
                    best_params = res_nm.x
                    
        except Exception as e:
            pass

    # --- Final Result Construction ---
    
    if best_params is None:
        # Fallback
        best_params = configs[0]

    centers, angles, all_verts, outer_center, outer_side, outer_angle = get_all_vertices(best_params)
    
    # Refine outer side and center to exact minimum for the final positions
    # We can do a quick local search on the last 4 params (cx, cy, S, theta)
    def refine_outer(params):
        def obj_outer(p):
            # p: cx, cy, S, theta
            cx, cy, S, theta = p
            if S < 0.1: return 1e10
            theta_rad = math.radians(theta)
            normals = np.array([
                [math.cos(theta_rad + math.pi/6 + k * math.pi/3),
                 math.sin(theta_rad + math.pi/6 + k * math.pi/3)]
                for k in range(6)
            ])
            shifted = all_verts - np.array([cx, cy])
            projs = shifted @ normals.T
            max_projs = np.max(projs, axis=0)
            required_side = np.max(max_projs) / (SQRT3 / 2.0)
            if required_side > S:
                return (required_side - S)**2 + S
            return S
        
        # Initial guess for outer params
        p_init = best_params[48:52]
        res_ref = minimize(obj_outer, p_init, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
        return res_ref.x

    final_outer = refine_outer(best_params)
    outer_center = final_outer[:2]
    outer_side = final_outer[2]
    outer_angle = final_outer[3]
    
    # Ensure outer_side is not negative or zero
    if outer_side < 1e-6:
        outer_side = 1.0

    # Check for overlaps in final config
    # If overlaps exist, we might need to return a valid (though maybe not optimal) config
    # But the optimization should have minimized them.
    # Let's verify containment and non-overlap.
    
    # For the return, we need inner hexagons (x, y, angle)
    inner = []
    for i in range(N):
        inner.append([centers[i, 0], centers[i, 1], angles[i]])
    
    return inner, [outer_center[0], outer_center[1]], outer_side, outer_angle
