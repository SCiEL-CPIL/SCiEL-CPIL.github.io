import math
import time
import random
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False

    def hex_vertices(cx, cy, S, angle_deg):
        """Return 6 vertices of regular hexagon (pointy-top by default)."""
        a = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = a + math.radians(60 * i)
            verts.append((cx + S * math.cos(theta), cy + S * math.sin(theta)))
        return verts

    def signed_distance_to_hexagon(p, cx, cy, S, angle_deg):
        """
        Calculate signed distance from point p to the hexagon defined by (cx, cy, S, angle_deg).
        Positive if outside, negative if inside.
        This is the minimum distance to any edge, with sign determined by winding/normal.
        """
        a = math.radians(angle_deg)
        px, py = p
        min_dist = float('inf')
        min_sign = 1.0
        
        for i in range(6):
            theta1 = a + math.radians(60 * i)
            theta2 = a + math.radians(60 * (i + 1))
            v1 = (cx + S * math.cos(theta1), cy + S * math.sin(theta1))
            v2 = (cx + S * math.cos(theta2), cy + S * math.sin(theta2))
            
            # Vector from v1 to v2
            ex = v2[0] - v1[0]
            ey = v2[1] - v1[1]
            # Vector from v1 to p
            px1 = px - v1[0]
            py1 = py - v1[1]
            
            # Project p onto edge v1-v2
            edge_len_sq = ex*ex + ey*ey
            if edge_len_sq == 0:
                continue
            
            t = (px1*ex + py1*ey) / edge_len_sq
            t = max(0.0, min(1.0, t))
            
            closest_x = v1[0] + t * ex
            closest_y = v1[1] + t * ey
            
            dist_sq = (px - closest_x)**2 + (py - closest_y)**2
            dist = math.sqrt(dist_sq)
            
            if dist < min_dist:
                min_dist = dist
                # Determine sign: normal pointing outward
                # Edge direction (ex, ey). Outward normal is (ey, -ex) or (-ey, ex)?
                # For a convex polygon defined CCW, outward normal is (dy, -dx) rotated?
                # Let's use the center to determine side.
                # Vector from center to midpoint of edge
                mid_x = (v1[0] + v2[0]) / 2.0
                mid_y = (v1[1] + v2[1]) / 2.0
                # Vector from center to p
                vx = px - cx
                vy = py - cy
                
                # Normal vector perpendicular to edge
                # Edge vector is (ex, ey). Normal is (-ey, ex) or (ey, -ex).
                # Let's check which one points away from center.
                # Dot product of normal with (mid - center) should be positive for outward.
                nx1 = -ey
                ny1 = ex
                dot1 = nx1 * (mid_x - cx) + ny1 * (mid_y - cy)
                
                if dot1 < 0:
                    nx1 = -nx1
                    ny1 = -ny1
                
                # Now check where p is relative to the edge plane
                # Vector from closest point to p
                vx_p = px - closest_x
                vy_p = py - closest_y
                dot_p = nx1 * vx_p + ny1 * vy_p
                
                # If dot_p > 0, p is on the outward side (outside)
                # If dot_p < 0, p is on the inward side (inside)
                # Wait, if p is inside, the distance to the boundary is positive in magnitude, but we want signed distance.
                # Standard convention: positive outside, negative inside.
                # If p is inside, the vector from boundary to p points inward (opposite to outward normal).
                # So dot_p will be negative.
                # If p is outside, dot_p will be positive.
                min_sign = 1.0 if dot_p > 0 else -1.0
                # Actually, if the closest point is a vertex, the logic holds.
                # But we need to be careful: if p is inside, the closest point on boundary is the edge, 
                # and the vector from that edge point to p points INTO the hexagon.
                # The outward normal points OUT. So dot(outward_normal, p - closest) < 0.
                # So sign is -1.
                
                # Refine sign logic:
                # We want d > 0 if outside, d < 0 if inside.
                # The distance is always positive.
                # So signed_dist = dist * sign.
                # If p is inside, sign should be -1.
                # If p is outside, sign should be +1.
                
                # Let's re-verify:
                # Outward normal N.
                # If p is outside, (p - closest) is roughly in direction N. Dot > 0. Sign +1.
                # If p is inside, (p - closest) is roughly in direction -N. Dot < 0. Sign -1.
                
                # This works.
        
        return min_dist * min_sign

    def compute_S_and_penalties(inner, ocx, ocy, S, oangle):
        """
        Compute the required S for containment and the overlap penalty.
        Returns: (required_S, overlap_penalty, containment_penalty)
        """
        # 1. Compute required S for containment
        # For each inner hex vertex, project onto the 6 normals of the outer hex.
        # The outer hex at (ocx, ocy) with side S and angle oangle has apothem a = S * sqrt(3)/2.
        # The support function in direction n is a.
        # So for a point p to be inside, (p - center) . n <= a for all 6 normals n.
        # a = S * sqrt(3)/2 => S = 2 * max_proj / sqrt(3)
        
        apothem_factor = math.sqrt(3) / 2.0
        max_proj = 0.0
        
        inner_verts_all = []
        for i in range(N):
            ix, iy, iangle = inner[i]
            iv = hex_vertices(ix, iy, 1.0, iangle)
            inner_verts_all.append(iv)
        
        # Normals of outer hex
        a_rad = math.radians(oangle)
        normals = []
        for i in range(6):
            theta = a_rad + math.radians(30 + 60 * i)
            normals.append((math.cos(theta), math.sin(theta)))
        
        for iv in inner_verts_all:
            for v in iv:
                vx = v[0] - ocx
                vy = v[1] - ocy
                for nx, ny in normals:
                    proj = vx * nx + vy * ny
                    if proj > max_proj:
                        max_proj = proj
        
        required_S = max_proj / apothem_factor if max_proj > 0 else 0.0
        
        # 2. Compute overlap penalty
        # Sum of squared overlaps for all pairs
        # Overlap = max(0, -min_signed_distance)
        # We use signed distance from vertices of A to B and B to A.
        # If any vertex is inside, distance is negative.
        # We want a smooth penalty.
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                vi = inner_verts_all[i]
                vj = inner_verts_all[j]
                # Check vertices of i against j
                for p in vi:
                    # Distance to hexagon j
                    # We need the parameters of hexagon j
                    jx, jy, jangle = inner[j]
                    d = signed_distance_to_hexagon(p, jx, jy, 1.0, jangle)
                    if d < 0:
                        overlap_penalty += d * d
                # Check vertices of j against i
                for p in vj:
                    ix, iy, iangle = inner[i]
                    d = signed_distance_to_hexagon(p, ix, iy, 1.0, iangle)
                    if d < 0:
                        overlap_penalty += d * d
        
        # 3. Containment penalty
        # If we use required_S, containment is satisfied by definition of S.
        # But if we fix S and check if points are inside, we need penalty.
        # In this strategy, we optimize S as part of the variable? 
        # No, S is derived from positions. 
        # So the objective is simply required_S + overlap_penalty * weight.
        # This ensures that if there is overlap, we pay a penalty, 
        # and the optimizer will try to reduce overlap while minimizing S.
        
        return required_S, overlap_penalty, 0.0

    def objective(params):
        """
        Objective function for Nelder-Mead.
        params: [x1, y1, a1, ..., xN, yN, aN, ocx, ocy, oangle]
        """
        if time.time() > deadline:
            return 1e6
        
        inner = []
        for i in range(N):
            x = params[2*i]
            y = params[2*i + 1]
            a = params[2*N + i]
            inner.append([x, y, a])
        
        ocx = params[3*N]
        ocy = params[3*N + 1]
        oangle = params[3*N + 2]
        
        required_S, overlap_penalty, _ = compute_S_and_penalties(inner, ocx, ocy, 0.0, oangle)
        
        # Weight for overlap penalty
        # We want the penalty to be significant enough to push out overlaps,
        # but not so large that it dominates the S minimization when S is large.
        # A fixed weight might work if scales are similar.
        # Overlap is in distance^2. S is in distance.
        # Let's use weight = 10.0
        weight = 10.0
        return required_S + weight * overlap_penalty

    def run_local_search(inner_init, ocx_init, ocy_init, oangle_init, seed):
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        
        rng = random.Random(seed)
        
        # Initial params
        params = []
        for i in range(N):
            params.append(inner_init[i][0])
            params.append(inner_init[i][1])
        for i in range(N):
            params.append(inner_init[i][2])
        params.append(ocx_init)
        params.append(ocy_init)
        params.append(oangle_init)
        
        # Bounds for Nelder-Mead? No, NM doesn't use bounds directly.
        # We rely on the penalty to keep things reasonable.
        # However, if points go to infinity, S goes to infinity.
        
        # To handle angle periodicity, we can wrap angles in [0, 360) or [-180, 180).
        # NM handles continuous spaces. Angles are continuous.
        
        try:
            result = minimize(
                objective,
                x0=params,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-4,
                    'fatol': 1e-4,
                    'adaptive': True
                }
            )
            
            if result.fun < 100.0: # Sanity check
                res_params = result.x
                inner_res = []
                for i in range(N):
                    inner_res.append([res_params[2*i], res_params[2*i+1], res_params[2*N+i]])
                
                ocx_res = res_params[3*N]
                ocy_res = res_params[3*N + 1]
                oangle_res = res_params[3*N + 2]
                
                # Calculate final S and check validity
                required_S, overlap_penalty, _ = compute_S_and_penalties(inner_res, ocx_res, ocy_res, 0.0, oangle_res)
                
                # If overlap is negligible, accept
                if overlap_penalty < 1e-6:
                    if required_S < best_S:
                        best_S = required_S
                        best_inner = inner_res
                        best_center = (ocx_res, ocy_res)
                        best_angle = oangle_res
                        best_valid = True
        except Exception:
            pass

    # --- Phase 1: Find good outer orientation for a known packing ---
    # Use a hexagonal lattice packing as a starting point.
    # This gives a valid configuration with some S.
    # We optimize the outer angle to minimize S for this fixed inner configuration.
    
    sp = math.sqrt(3)
    hex_config = [[0.0, 0.0, 0.0]]
    for i in range(6):
        theta = math.radians(60 * i)
        hex_config.append([sp * math.cos(theta), sp * math.sin(theta), 0.0])
    for i in [0, 2, 4]:
        theta = math.radians(60 * i + 30)
        hex_config.append([2 * sp * math.cos(theta), 2 * sp * math.sin(theta), 0.0])
    for i in [1, 3, 5]:
        theta = math.radians(60 * i)
        hex_config.append([2 * sp * math.cos(theta), 2 * sp * math.sin(theta), 0.0])
    
    # Ensure we have N hexagons
    while len(hex_config) < N:
        hex_config.append([0.0, 0.0, 0.0])
    hex_config = hex_config[:N]
    
    # Optimize outer angle for this fixed inner config
    def outer_angle_obj(angle):
        S, _, _ = compute_S_and_penalties(hex_config, 0.0, 0.0, 0.0, angle)
        return S
    
    best_angle = 0.0
    best_S_fixed = float('inf')
    for init_a in range(0, 360, 30):
        res = minimize(outer_angle_obj, [float(init_a)], method='Nelder-Mead', options={'maxiter': 50})
        if res.fun < best_S_fixed:
            best_S_fixed = res.fun
            best_angle = res.x[0]
    
    # --- Phase 2: Joint Optimization ---
    # Start from the hex config with the optimized outer angle.
    # Run multiple Nelder-Mead instances with slight perturbations.
    
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Perturb the initial configuration
        perturbed_inner = []
        for i in range(N):
            x = hex_config[i][0] + random.gauss(0, 0.2)
            y = hex_config[i][1] + random.gauss(0, 0.2)
            a = hex_config[i][2] + random.gauss(0, 5)
            perturbed_inner.append([x, y, a])
        
        ocx = random.gauss(0, 0.1)
        ocy = random.gauss(0, 0.1)
        oangle = best_angle + random.gauss(0, 5)
        
        run_local_search(perturbed_inner, ocx, ocy, oangle, r * 100 + 1)
        
        # If we haven't found a valid solution yet, try a more aggressive restart
        if not best_valid and r % 2 == 1:
            # Try random initial positions
            random_inner = []
            for i in range(N):
                random_inner.append([random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(0, 60)])
            run_local_search(random_inner, 0.0, 0.0, random.uniform(0, 60), r * 100 + 50)

    # Fallback if no valid solution found
    if not best_valid or best_inner is None:
        # Use the hex config with best angle
        inner_final = hex_config
        S_final, _, _ = compute_S_and_penalties(inner_final, 0.0, 0.0, 0.0, best_angle)
        return inner_final, (0.0, 0.0), S_final, best_angle

    return best_inner, best_center, best_S, best_angle
# EVOLVE-BLOCK-END
