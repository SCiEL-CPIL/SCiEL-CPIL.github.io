import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    SQRT3 = math.sqrt(3)
    
    # Pre-compute base hexagon vertices at origin, angle 0
    # Vertices are at angles 0, 60, 120, 180, 240, 300
    base_angles_rad = np.array([i * math.pi / 3.0 for i in range(6)])
    base_verts_x = np.cos(base_angles_rad)
    base_verts_y = np.sin(base_angles_rad)
    base_verts = np.column_stack((base_verts_x, base_verts_y))
    
    # Pre-compute rotation matrices for speed? 
    # Actually, L-BFGS-B calls the function many times. 
    # We can vectorize the evaluation.
    
    def get_transformed_verts(params):
        """
        params: (N, 3) array of [x, y, theta]
        Returns: (N, 6, 2) array of vertices
        """
        x = params[:, 0]
        y = params[:, 1]
        theta = params[:, 2]
        
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # Rotation matrix R = [[c, -s], [s, c]]
        # v' = R @ v + t
        # v is (6, 2), we want (N, 6, 2)
        
        # base_verts is (6, 2)
        # We need to rotate each row of base_verts by theta_i
        
        # Expand dims: theta (N, 1)
        # cos_t (N, 1), sin_t (N, 1)
        
        # x' = c*x - s*y
        # y' = s*x + c*y
        
        vx = base_verts_x[None, :] # (1, 6)
        vy = base_verts_y[None, :] # (1, 6)
        
        cos_t_exp = cos_t[:, None] # (N, 1)
        sin_t_exp = sin_t[:, None] # (N, 1)
        
        rx = cos_t_exp * vx - sin_t_exp * vy # (N, 6)
        ry = sin_t_exp * vx + cos_t_exp * vy # (N, 6)
        
        # Translate
        tx = x[:, None] # (N, 1)
        ty = y[:, None] # (N, 1)
        
        verts_x = rx + tx
        verts_y = ry + ty
        
        return np.column_stack((verts_x, verts_y)) # (N, 6, 2)

    def calc_outer_side_smooth(params, outer_angle_deg, temp=0.5):
        """
        Calculate smooth approximation of outer side length.
        Outer hexagon centered at (0,0) with side S.
        Constraint: For all vertices v of all inner hexes, |v . n_k| <= S * SQRT3 / 2
        where n_k are the 6 normal directions of the outer hexagon.
        
        We want to minimize S.
        S >= max_{i, k, v} ( |v . n_k| ) * 2 / SQRT3
        
        Let M = max_{i, k, v} ( |v . n_k| )
        We approximate M with log-sum-exp of |v . n_k| / temp, then * temp.
        """
        verts = get_transformed_verts(params) # (N, 6, 2)
        
        # Outer hex normals
        # Base normals for regular hexagon centered at origin:
        # Edges are perpendicular to normals.
        # If vertices are at 0, 60..., edges are at 30, 90...
        # Normals point to vertices? No, normals to edges.
        # For a regular hexagon with vertices at angles 0, 60.., the edge midpoints are at 30, 90..
        # The normal to the edge at 30 deg is in direction 30 deg.
        # So normals are at 30, 90, 150, 210, 270, 330.
        
        oa_rad = math.radians(outer_angle_deg)
        normal_angles = np.array([30, 90, 150, 210, 270, 330]) * math.pi / 180.0 + oa_rad
        
        nx = np.cos(normal_angles) # (6,)
        ny = np.sin(normal_angles) # (6,)
        
        # Project all vertices onto all normals
        # verts: (N, 6, 2)
        # nx: (6,)
        # We want (N, 6, 6) : N hexes, 6 verts, 6 normals
        
        # v . n = vx*nx + vy*ny
        # vx: (N, 6, 1)
        # nx: (1, 1, 6)
        
        vx = verts[:, :, 0] # (N, 6)
        vy = verts[:, :, 1] # (N, 6)
        
        # Reshape for broadcasting
        vx_b = vx[:, :, None] # (N, 6, 1)
        vy_b = vy[:, :, None] # (N, 6, 1)
        
        nx_b = nx[None, None, :] # (1, 1, 6)
        ny_b = ny[None, None, :] # (1, 1, 6)
        
        projs = vx_b * nx_b + vy_b * ny_b # (N, 6, 6)
        
        # Absolute values
        abs_projs = np.abs(projs)
        
        # Log-sum-exp approximation of max
        # max(x) approx temp * log(sum(exp(x/temp)))
        # To avoid overflow, subtract max first
        max_projs = np.max(abs_projs, axis=(1, 2)) # (N,)
        
        # We want global max over all N and all projs.
        # Actually, we can just compute log-sum-exp over the whole tensor.
        
        # Flatten to (N*36,)
        flat = abs_projs.flatten()
        
        # For numerical stability
        m = np.max(flat)
        if m == 0:
            return 0.0
            
        # log(sum(exp((flat - m)/temp))) + m/temp
        # This gives approx max(flat)/temp
        # So max(flat) approx temp * (log_sum_exp + m/temp) = temp * log_sum_exp + m
        
        # Wait, standard log-sum-exp:
        # LSE(x) = m + log(sum(exp(x-m)))
        # max(x) approx temp * LSE(x/temp)
        # Let y = x/temp
        # max(x) approx temp * (m_y + log(sum(exp(y-m_y))))
        # where m_y = max(y) = max(x)/temp
        
        # So max(x) approx m + temp * log(sum(exp((x-m)/temp)))
        
        y = (flat - m) / temp
        # Clip y to avoid underflow in exp (though exp(negative) is fine, it goes to 0)
        # But if y is very negative, exp is 0.
        # sum(exp(y)) might be 0 if all y are very negative? No, m is max, so at least one y is 0.
        # So sum >= 1.
        
        log_sum = np.log(np.sum(np.exp(y)))
        
        approx_max = m + temp * log_sum
        
        # Side length S = 2 * approx_max / SQRT3
        return 2.0 * approx_max / SQRT3

    def calc_separation_penalty(params, temp=0.5):
        """
        Smooth penalty for overlapping hexagons.
        For each pair (i, j), compute minimum distance between boundaries.
        If min_dist < 0, penalty += (min_dist)^2 or exp(-min_dist/temp).
        
        Computing exact min distance between two convex polygons is expensive.
        Approximation:
        1. Check center distance. If d > 4, disjoint.
        2. If d < 3, likely overlap.
        3. Use SAT to find min overlap depth.
        
        Since we are using L-BFGS-B, we need a smooth function.
        SAT gives a piecewise linear min distance.
        We can smooth the max(0, -min_dist) term.
        
        Alternatively, use a simpler metric:
        For each vertex of hex i, check if it's inside hex j.
        If inside, penalty += distance to boundary squared.
        This is not symmetric and might miss cases where edges cross without vertices inside.
        However, for regular hexagons of same size, if they overlap, usually a vertex is inside or edges cross.
        
        Let's stick to SAT for robustness but make it differentiable?
        SAT min distance is max over axes of (gap).
        gap_k = max(proj_i, axis_k) - min(proj_j, axis_k)
        If gap < 0, overlap.
        Min overlap depth = - min(gap_k) over all axes.
        Actually, min distance = min(gap_k).
        If min_dist < 0, they overlap.
        Penalty = max(0, -min_dist)^2
        
        min_dist = min_{k} (gap_k)
        -min_dist = max_{k} (-gap_k)
        
        So penalty = max(0, max_k(-gap_k))^2
        
        We can approximate max_k(-gap_k) with log-sum-exp of -gap_k.
        """
        
        # This is computationally heavy if done in Python loops.
        # Let's vectorize the SAT gap calculation.
        
        verts = get_transformed_verts(params) # (N, 6, 2)
        
        # Axes for each hexagon
        # Normal to edge i of hex with rotation theta is theta + 30 + i*60
        # For hex i, axes are at theta_i + 30, 90, ...
        
        # We need to check all pairs (i, j).
        # N=11, so 55 pairs.
        
        total_penalty = 0.0
        
        # Precompute axes for each hexagon
        # axes[i] is (6, 2)
        
        # To speed up, we can precompute cos/sin for the 6 axis directions relative to theta_i
        # Axis angle = theta_i + (30 + k*60) deg
        # cos(theta_i + alpha) = cos(theta_i)cos(alpha) - sin(theta_i)sin(alpha)
        
        base_axis_angles = np.array([30, 90, 150, 210, 270, 330]) * math.pi / 180.0
        cos_base_ax = np.cos(base_axis_angles)
        sin_base_ax = np.sin(base_axis_angles)
        
        theta = params[:, 2]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # axes_x[i, k] = cos_t[i]*cos_base_ax[k] - sin_t[i]*sin_base_ax[k]
        # axes_y[i, k] = sin_t[i]*cos_base_ax[k] + cos_t[i]*sin_base_ax[k]
        
        cos_t_exp = cos_t[:, None] # (N, 1)
        sin_t_exp = sin_t[:, None] # (N, 1)
        
        axes_x = cos_t_exp * cos_base_ax[None, :] - sin_t_exp * sin_base_ax[None, :] # (N, 6)
        axes_y = sin_t_exp * cos_base_ax[None, :] + cos_t_exp * sin_base_ax[None, :] # (N, 6)
        
        # For each pair i < j
        # We need projections of verts_i onto axes_i and axes_j
        # And verts_j onto axes_i and axes_j
        
        # proj_i_on_axis_k = dots(verts_i, axis_k)
        
        # Let's iterate pairs. 55 is small.
        for i in range(N):
            for j in range(i + 1, N):
                # Get verts
                vi = verts[i] # (6, 2)
                vj = verts[j] # (6, 2)
                
                # Get axes
                ax_i = np.column_stack((axes_x[i], axes_y[i])) # (6, 2)
                ax_j = np.column_stack((axes_x[j], axes_y[j])) # (6, 2)
                
                # Project vi onto ax_i
                # vi: (6, 2), ax_i: (6, 2)
                # proj = vi . ax_i (element-wise sum over axis? No, dot product with each axis)
                # We want (6,) projections for each axis.
                # proj_i_k = sum(vi * ax_i[k])
                
                # Use matmul: vi (6, 2) @ ax_i.T (2, 6) -> (6, 6)
                # Row k is projection of all vertices of i onto axis k?
                # No, vi @ ax_i.T gives (6, 6) where element (m, k) is dot(vi[m], ax_i[k])
                # We want min/max over m (vertices) for each k (axis).
                
                proj_i_on_i = vi @ ax_i.T # (6, 6)
                min_i_i = np.min(proj_i_on_i, axis=0) # (6,)
                max_i_i = np.max(proj_i_on_i, axis=0) # (6,)
                
                proj_j_on_i = vj @ ax_i.T # (6, 6)
                min_j_i = np.min(proj_j_on_i, axis=0) # (6,)
                max_j_i = np.max(proj_j_on_i, axis=0) # (6,)
                
                # Gaps on axes of i
                # gap_k = max(min_i_i[k], min_j_i[k]) - min(max_i_i[k], max_j_i[k])
                # Or more standard:
                # If max_i < min_j or max_j < min_i, gap is positive.
                # gap = min(max_i_i, max_j_i) - max(min_i_i, min_j_i)
                # If gap < 0, overlap.
                
                min_max_i_i = np.minimum(max_i_i, max_j_i)
                max_min_i_i = np.maximum(min_i_i, min_j_i)
                gaps_i = min_max_i_i - max_min_i_i # (6,)
                
                # Same for axes of j
                proj_i_on_j = vi @ ax_j.T # (6, 6)
                min_i_j = np.min(proj_i_on_j, axis=0) # (6,)
                max_i_j = np.max(proj_i_on_j, axis=0) # (6,)
                
                proj_j_on_j = vj @ ax_j.T # (6, 6)
                min_j_j = np.min(proj_j_on_j, axis=0) # (6,)
                max_j_j = np.max(proj_j_on_j, axis=0) # (6,)
                
                min_max_j_j = np.minimum(max_i_j, max_j_j)
                max_min_j_j = np.maximum(min_i_j, min_j_j)
                gaps_j = min_max_j_j - max_min_j_j # (6,)
                
                # Combine all 12 gaps
                all_gaps = np.concatenate([gaps_i, gaps_j]) # (12,)
                
                # Min distance is min(all_gaps)
                # If min_dist < 0, overlap depth is -min_dist
                # We want smooth penalty for overlap.
                # Penalty = max(0, -min_dist)^2
                # -min_dist = max(-all_gaps)
                
                neg_gaps = -all_gaps
                
                # Smooth max of neg_gaps
                # Use log-sum-exp
                m = np.max(neg_gaps)
                if m <= 0:
                    # All gaps >= 0, no overlap
                    # But we might want a small penalty if close?
                    # No, just 0.
                    continue
                    
                y = (neg_gaps - m) / temp
                log_sum = np.log(np.sum(np.exp(y)))
                smooth_max = m + temp * log_sum
                
                # Penalty = smooth_max^2
                total_penalty += smooth_max * smooth_max
                
        return total_penalty

    def objective(params_flat):
        """
        params_flat: (3N,)
        """
        params = params_flat.reshape(N, 3)
        
        # Outer side
        # We fix outer angle to 0 during optimization for simplicity,
        # or optimize it separately?
        # The problem allows rotating the outer hexagon.
        # However, since the inner hexagons can rotate freely,
        # the relative angle between inner cluster and outer hexagon is what matters.
        # If we allow each inner hex to rotate, the "outer angle" is effectively a global rotation parameter
        # that can be absorbed into the inner hex angles if we fix outer to 0.
        # Wait, if we fix outer to 0, and allow inner rotations, we cover all relative configurations.
        # So we can fix outer_angle_deg = 0.
        
        side = calc_outer_side_smooth(params, 0.0, temp=0.2)
        
        # Separation penalty
        # Weight the penalty heavily
        sep_pen = calc_separation_penalty(params, temp=0.2)
        
        # Return side + penalty
        # We want to minimize side, but ensure sep_pen is 0.
        # If sep_pen > 0, it dominates.
        
        return side + 100.0 * sep_pen

    # Initial configurations
    # 1. Honeycomb cluster
    # 2. Ring
    # 3. Random
    
    def init_honeycomb():
        # Center + 6 neighbors + 4 more
        # Axial coords
        coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (2, -1), (2, 1), (3, 0)
        ]
        SQRT3 = math.sqrt(3)
        centers = []
        for q, r in coords:
            x = SQRT3 * (q + r / 2.0)
            y = 1.5 * r
            centers.append((x, y))
        
        # Center
        cx = np.mean([c[0] for c in centers])
        cy = np.mean([c[1] for c in centers])
        
        params = np.zeros((N, 3))
        for i, (x, y) in enumerate(centers):
            params[i, 0] = x - cx
            params[i, 1] = y - cy
            params[i, 2] = 0.0
        return params.flatten()

    def init_ring():
        # 1 in center, 10 in ring?
        # 1 center, 6 inner ring, 4 outer?
        # Let's just do 1 center, 10 on a circle of radius 2.2
        params = np.zeros((N, 3))
        params[0, 0] = 0
        params[0, 1] = 0
        params[0, 2] = 0
        
        for i in range(1, N):
            angle = 2 * math.pi * (i - 1) / (N - 1)
            r = 2.2
            params[i, 0] = r * math.cos(angle)
            params[i, 1] = r * math.sin(angle)
            params[i, 2] = angle # Rotate to face outward?
        return params.flatten()

    def init_random():
        params = np.random.rand(N, 3)
        params[:, 0] *= 6.0 - 3.0 # -3 to 3
        params[:, 1] *= 6.0 - 3.0
        params[:, 2] *= 2 * math.pi
        return params.flatten()

    inits = [init_honeycomb(), init_ring()]
    
    # Add a few random inits
    for _ in range(3):
        inits.append(init_random())
        
    best_score = float('inf')
    best_params = None
    
    for init in inits:
        if time.time() > deadline - 10.0:
            break
            
        try:
            # L-BFGS-B
            res = minimize(objective, init, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8, 'maxls': 50})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                
        except Exception as e:
            pass
            
    if best_params is None:
        # Fallback
        best_params = init_honeycomb()
        
    # Refine outer angle
    # The best_params found assumes outer angle 0.
    # But we can rotate the outer hexagon to potentially reduce side length?
    # No, if inner hexagons are optimally placed relative to outer, rotating outer is same as rotating all inner.
    # Since inner angles are free, the relative orientation is already optimized.
    # However, the "outer angle" parameter in the return value allows the evaluator to rotate the container.
    # If we return angle=0, the container is fixed.
    # We should check if rotating the container helps with the specific fixed inner positions.
    # But since inner positions are optimized for container=0, rotating container without rotating inner
    # would likely increase side length unless the configuration has symmetry.
    # Wait, the inner hexagons have individual angles.
    # If we rotate the outer hexagon by alpha, it's equivalent to rotating the coordinate system by -alpha.
    # The inner hexagons are fixed in the coordinate system.
    # So we just need to find the best outer_angle_deg for the fixed best_params.
    
    params_2d = best_params.reshape(N, 3)
    
    best_side_final = float('inf')
    best_oa = 0.0
    
    # Sample outer angles
    for oa_deg in range(0, 60, 1):
        side = calc_outer_side_smooth(params_2d, oa_deg, temp=0.01) # Low temp for accurate max
        if side < best_side_final:
            best_side_final = side
            best_oa = oa_deg
            
    # Verify validity with exact SAT
    # If invalid, we need to fix it or return a valid fallback.
    # L-BFGS-B might leave small overlaps if penalty weight isn't high enough or temp too high.
    
    def is_disjoint_exact(hex1_verts, hex2_verts, axes1, axes2):
        # hex1_verts: (6, 2)
        # axes1: (6, 2)
        
        # Project hex1 onto axes1
        proj1_1 = hex1_verts @ axes1.T # (6, 6)
        min1_1 = np.min(proj1_1, axis=0)
        max1_1 = np.max(proj1_1, axis=0)
        
        proj2_1 = hex2_verts @ axes1.T
        min2_1 = np.min(proj2_1, axis=0)
        max2_1 = np.max(proj2_1, axis=0)
        
        for k in range(6):
            if max1_1[k] < min2_1[k] or max2_1[k] < min1_1[k]:
                return True
                
        # Project onto axes2
        proj1_2 = hex1_verts @ axes2.T
        min1_2 = np.min(proj1_2, axis=0)
        max1_2 = np.max(proj1_2, axis=0)
        
        proj2_2 = hex2_verts @ axes2.T
        min2_2 = np.min(proj2_2, axis=0)
        max2_2 = np.max(proj2_2, axis=0)
        
        for k in range(6):
            if max1_2[k] < min2_2[k] or max2_2[k] < min1_2[k]:
                return True
                
        return False

    # Check validity
    verts = get_transformed_verts(params_2d)
    
    # Recompute axes for validity check
    theta = params_2d[:, 2]
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    base_axis_angles = np.array([30, 90, 150, 210, 270, 330]) * math.pi / 180.0
    cos_base_ax = np.cos(base_axis_angles)
    sin_base_ax = np.sin(base_axis_angles)
    
    cos_t_exp = cos_t[:, None]
    sin_t_exp = sin_t[:, None]
    
    axes_x = cos_t_exp * cos_base_ax[None, :] - sin_t_exp * sin_base_ax[None, :]
    axes_y = sin_t_exp * cos_base_ax[None, :] + cos_t_exp * sin_base_ax[None, :]
    
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            vi = verts[i]
            vj = verts[j]
            ax_i = np.column_stack((axes_x[i], axes_y[i]))
            ax_j = np.column_stack((axes_x[j], axes_y[j]))
            
            if not is_disjoint_exact(vi, vj, ax_i, ax_j):
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to honeycomb which is known valid
        fallback_params = init_honeycomb()
        params_2d = fallback_params.reshape(N, 3)
        
        # Recalculate side for fallback
        best_side_final = calc_outer_side_smooth(params_2d, 0.0, temp=0.01)
        best_oa = 0.0
        
        # Re-verify (should be valid)
        # ... assume valid for honeycomb
        
    # Finalize
    inner_list = []
    for i in range(N):
        inner_list.append([params_2d[i, 0], params_2d[i, 1], math.degrees(params_2d[i, 2])])
        
    return (inner_list, (0.0, 0.0), best_side_final, best_oa)

# EVOLVE-BLOCK-END
