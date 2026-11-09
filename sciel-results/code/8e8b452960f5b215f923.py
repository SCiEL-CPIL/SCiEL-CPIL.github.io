"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
import math
import time
import random
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    sqrt3 = math.sqrt(3)
    inradius = sqrt3 / 2.0

    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of regular hexagon with given center, side, rotation."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return verts

    def get_normals(angle_deg):
        """Return 6 outward normals of a regular hexagon with given rotation."""
        angle_rad = math.radians(angle_deg)
        normals = []
        for k in range(6):
            theta = angle_rad + math.pi / 6 + k * math.pi / 3
            normals.append((math.cos(theta), math.sin(theta)))
        return normals

    def compute_min_side_exact(positions, outer_center, outer_angle):
        """
        Compute minimum outer side to contain all inner hexagons.
        """
        cx, cy = outer_center
        max_side = 0.0
        angle_rad = math.radians(outer_angle)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)

        normals = get_normals(outer_angle)

        for ix, iy, iang in positions:
            inner_verts = hexagon_vertices(ix, iy, 1.0, iang)
            for vx, vy in inner_verts:
                dx = vx - cx
                dy = vy - cy
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a

                for nx, ny in normals:
                    proj = rx * nx + ry * ny
                    required_side = proj / inradius
                    if required_side > max_side:
                        max_side = required_side
        return max_side

    def compute_min_side_smooth(positions, outer_center, outer_angle, alpha=50.0):
        """
        Compute a smooth approximation of the minimum outer side.
        Uses log-sum-exp over all vertex-normal projections.
        """
        cx, cy = outer_center
        angle_rad = math.radians(outer_angle)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)

        normals = get_normals(outer_angle)

        max_log_term = -1e10
        log_sum = 0.0
        
        # We need to be careful with the max for numerical stability
        # Let's collect all required sides first
        req_sides = []
        for ix, iy, iang in positions:
            inner_verts = hexagon_vertices(ix, iy, 1.0, iang)
            for vx, vy in inner_verts:
                dx = vx - cx
                dy = vy - cy
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a

                for nx, ny in normals:
                    proj = rx * nx + ry * ny
                    req_sides.append(proj / inradius)
        
        if not req_sides:
            return 0.0
            
        max_val = max(req_sides)
        # Soft max: (1/alpha) * log(sum(exp(alpha * x)))
        # To avoid overflow, subtract max_val
        exp_sum = 0.0
        for val in req_sides:
            exp_sum += math.exp(alpha * (val - max_val))
        
        return max_val + (1.0 / alpha) * math.log(exp_sum)

    def hex_distance(p1, p2):
        """
        Exact minimum distance between two unit regular hexagons.
        Returns 0 if they overlap, positive if separated.
        """
        verts1 = hexagon_vertices(p1[0], p1[1], 1.0, p1[2])
        verts2 = hexagon_vertices(p2[0], p2[1], 1.0, p2[2])

        axes = []
        for i in range(6):
            for verts in (verts1, verts2):
                x1, y1 = verts[i]
                x2, y2 = verts[(i + 1) % 6]
                dx = x2 - x1
                dy = y2 - y1
                length = math.sqrt(dx * dx + dy * dy)
                if length > 1e-12:
                    axes.append((-dy / length, dx / length))

        dx_c = p2[0] - p1[0]
        dy_c = p2[1] - p1[1]
        dist_c = math.sqrt(dx_c * dx_c + dy_c * dy_c)
        if dist_c > 1e-12:
            axes.append((dx_c / dist_c, dy_c / dist_c))

        min_dist = float('inf')
        
        for nx, ny in axes:
            min1 = min(vx * nx + vy * ny for vx, vy in verts1)
            max1 = max(vx * nx + vy * ny for vx, vy in verts1)
            min2 = min(vx * nx + vy * ny for vx, vy in verts2)
            max2 = max(vx * nx + vy * ny for vx, vy in verts2)

            # Gap is positive if separated
            gap = max(min2 - max1, min1 - max2)
            if gap > 0:
                # They are separated. The distance is the maximum gap over all separating axes?
                # No, for convex polygons, if they are separated, the distance is the maximum gap?
                # Actually, the distance is the minimum distance between any point in P1 and any point in P2.
                # If there is a separating axis, the distance is the gap along that axis ONLY IF the projection intervals are disjoint.
                # Wait, SAT finds a separating axis. If one exists, the polygons are disjoint.
                # The distance is NOT necessarily the gap on the SAT axis.
                # However, for the penalty, we just need a measure of "how far apart".
                # If they are disjoint, penalty is 0.
                # If they overlap, we need a penetration depth.
                # The standard SAT returns a separating axis if they are disjoint.
                # If no separating axis is found, they overlap.
                # To get penetration depth, we can look for the axis with the *minimum* overlap (or maximum negative gap).
                
                # Let's refine:
                # If gap > 0, they are disjoint. We can return early with 0 penalty?
                # But for the penalty function in optimization, we want a smooth transition.
                # If we return 0 for any gap > 0, it's discontinuous.
                # Better: Return the signed distance.
                # If disjoint, signed distance = min_gap (positive).
                # If overlapping, signed distance = -max_overlap (negative).
                
                # Let's track the minimum positive gap (distance) or maximum negative gap (penetration).
                # Actually, if ANY axis has gap > 0, they are disjoint.
                # The distance is the minimum gap over all axes? No.
                # The distance is the minimum distance between the two sets.
                # For convex polygons, the distance is the minimum gap over all axes that separate them?
                # No, it's the minimum gap over ALL axes?
                # If they are disjoint, there exists at least one separating axis.
                # The distance is the minimum gap among the separating axes?
                # Actually, the distance is the minimum gap over all axes?
                # Let's test: Two squares. Distance is along the normal to the closest faces.
                # So yes, we should find the minimum gap over all candidate axes.
                
                if gap < min_dist:
                    min_dist = gap
                    
        if min_dist > 0:
            return min_dist
        else:
            # They overlap. We need the penetration depth.
            # Penetration depth is the minimum translation distance to separate them.
            # This is the minimum overlap over all axes?
            # If they overlap, for every axis, the projections overlap.
            # The penetration depth is the minimum "overlap" width over all axes.
            # Overlap width = min(max1, max2) - max(min1, min2).
            # We want the minimum overlap width.
            min_overlap = float('inf')
            for nx, ny in axes:
                min1 = min(vx * nx + vy * ny for vx, vy in verts1)
                max1 = max(vx * nx + vy * ny for vx, vy in verts1)
                min2 = min(vx * nx + vy * ny for vx, vy in verts2)
                max2 = max(vx * nx + vy * ny for vx, vy in verts2)
                
                overlap = min(max1, max2) - max(min1, min2)
                if overlap < min_overlap:
                    min_overlap = overlap
            
            # min_overlap is positive. Penetration depth is -min_overlap?
            # No, if they overlap, the "gap" is negative.
            # Let's return -min_overlap as the signed distance.
            return -min_overlap

    def compute_overlap_penalty(positions):
        """
        Sum of squared penetration depths for all pairs.
        """
        penalty = 0.0
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                # Quick reject
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                d = math.sqrt(dx * dx + dy * dy)
                if d >= 2.0:
                    continue
                
                dist = hex_distance(positions[i], positions[j])
                if dist < 0:
                    # Overlapping
                    penalty += (dist * dist)
        return penalty

    def make_honeycomb_init():
        """Create 11 hexagons in a honeycomb-like pattern."""
        positions = []
        positions.append([0.0, 0.0, 0.0])
        ring_dist = sqrt3 + 0.05
        for k in range(6):
            angle = math.radians(60 * k)
            x = ring_dist * math.cos(angle)
            y = ring_dist * math.sin(angle)
            positions.append([x, y, 0.0])
        gap_angles = [30, 150, 210, 330]
        gap_dist = ring_dist * 2 * math.cos(math.pi / 6) + 0.05
        for angle_deg in gap_angles:
            angle = math.radians(angle_deg)
            x = gap_dist * math.cos(angle)
            y = gap_dist * math.sin(angle)
            positions.append([x, y, 0.0])
        return positions

    # Objective function for Nelder-Mead
    def objective_flat(flat_params, outer_angle, alpha):
        # flat_params: [x1, y1, a1, x2, y2, a2, ...]
        positions = []
        for i in range(N):
            x = flat_params[i * 3]
            y = flat_params[i * 3 + 1]
            a = flat_params[i * 3 + 2]
            positions.append([x, y, a])
        
        # Smooth side length
        side = compute_min_side_smooth(positions, (0.0, 0.0), outer_angle, alpha)
        
        # Overlap penalty
        pen = compute_overlap_penalty(positions)
        
        # Weight the penalty heavily
        return side + 10.0 * pen

    # SA Objective
    def sa_objective(positions, outer_angle, alpha):
        side = compute_min_side_smooth(positions, (0.0, 0.0), outer_angle, alpha)
        pen = compute_overlap_penalty(positions)
        return side + 10.0 * pen

    # Run SA first
    best_inner_sa = None
    best_score_sa = float('inf')
    
    inits = [make_honeycomb_init()]
    # Add a few random inits
    for _ in range(3):
        rand_pos = []
        for i in range(N):
            r = random.uniform(0, 2.0)
            a = random.uniform(0, 360)
            rand_pos.append([r * math.cos(math.radians(a)), r * math.sin(math.radians(a)), random.uniform(0, 60)])
        inits.append(rand_pos)

    for init_pos in inits:
        if time.time() > deadline - 20.0:
            break
            
        current_pos = [p[:] for p in init_pos]
        current_temp = 1.0
        cooling_rate = 0.99
        min_temp = 0.01
        current_score = sa_objective(current_pos, 0.0, 50.0)
        
        best_in_run = [p[:] for p in current_pos]
        best_in_run_score = current_score
        
        max_steps = 2000
        steps = 0
        
        while current_temp > min_temp and steps < max_steps and time.time() < deadline - 15.0:
            idx = random.randint(0, N - 1)
            old_pos = current_pos[idx][:]
            new_pos = old_pos[:]
            
            step_size = 0.1 * current_temp + 0.01
            new_pos[0] += random.gauss(0, step_size)
            new_pos[1] += random.gauss(0, step_size)
            new_pos[2] += random.gauss(0, 5.0)
            
            current_pos[idx] = new_pos
            new_score = sa_objective(current_pos, 0.0, 50.0)
            
            delta = new_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / current_temp):
                current_score = new_score
                if current_score < best_in_run_score:
                    best_in_run_score = current_score
                    best_in_run = [p[:] for p in current_pos]
            else:
                current_pos[idx] = old_pos
                
            current_temp *= cooling_rate
            steps += 1
            
        if best_in_run_score < best_score_sa:
            best_score_sa = best_in_run_score
            best_inner_sa = [p[:] for p in best_in_run]

    if best_inner_sa is None:
        best_inner_sa = inits[0]

    # Now run Nelder-Mead on the best SA result
    # We will try a few outer angles
    best_final_side = float('inf')
    best_final_inner = None
    best_final_angle = 0.0

    # Flatten the initial positions
    init_flat = []
    for p in best_inner_sa:
        init_flat.extend(p)

    # Try a few angles for NM
    angle_candidates = [0.0, 15.0, 30.0, 45.0]
    
    for outer_angle in angle_candidates:
        if time.time() > deadline - 10.0:
            break
            
        # Use a moderate alpha for NM to keep it smooth
        alpha_nm = 50.0
        
        res = minimize(
            objective_flat, 
            init_flat, 
            args=(outer_angle, alpha_nm),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
        )
        
        if res.success or res.fun < best_final_side:
            # Reconstruct positions
            final_pos = []
            for i in range(N):
                final_pos.append([res.x[i*3], res.x[i*3+1], res.x[i*3+2]])
            
            # Calculate exact side length
            exact_side = compute_min_side_exact(final_pos, (0.0, 0.0), outer_angle)
            pen = compute_overlap_penalty(final_pos)
            
            if pen < 1e-6: # Valid
                if exact_side < best_final_side:
                    best_final_side = exact_side
                    best_final_inner = final_pos
                    best_final_angle = outer_angle

    # If NM failed to find a valid solution, fall back to SA result
    if best_final_inner is None:
        # Check SA result validity
        pen = compute_overlap_penalty(best_inner_sa)
        if pen < 1e-6:
            best_final_inner = best_inner_sa
            best_final_angle = 0.0
            # Find best angle for this inner
            for a in range(0, 60, 5):
                s = compute_min_side_exact(best_inner_sa, (0.0, 0.0), a)
                if s < best_final_side:
                    best_final_side = s
                    best_final_angle = a
        else:
            # Emergency fallback: safe spiral
            safe_inner = []
            for i in range(N):
                if i == 0:
                    safe_inner.append([0.0, 0.0, 0.0])
                else:
                    r = 2.1 * (i // 6 + 1)
                    angle = math.radians(60 * (i % 6))
                    x = r * math.cos(angle)
                    y = r * math.sin(angle)
                    safe_inner.append([x, y, 0.0])
            best_final_inner = safe_inner
            best_final_angle = 0.0
            best_final_side = compute_min_side_exact(safe_inner, (0.0, 0.0), 0.0)

    return best_final_inner, (0.0, 0.0), best_final_side, best_final_angle
# EVOLVE-BLOCK-END
