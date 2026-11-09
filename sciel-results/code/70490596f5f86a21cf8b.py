"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
import math
import time
import random

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    sqrt3 = math.sqrt(3)

    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of regular hexagon with given center, side, rotation."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return verts

    def compute_min_side(inner_positions, outer_center, outer_angle):
        """
        Compute minimum outer side to contain all inner hexagons.
        The outer hexagon has center (cx, cy), side length S, and rotation outer_angle.
        A point (px, py) is inside the outer hexagon iff for all 6 outward normals n_k,
        the projection of (px-cx, py-cy) rotated by -outer_angle onto n_k is <= S * sqrt(3)/2.
        So the required side is max over all points and all normals of proj / (sqrt(3)/2).
        """
        cx, cy = outer_center
        max_side = 0.0
        angle_rad = math.radians(outer_angle)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)

        # Pre-compute the 6 outward normals of a unit hexagon (inradius = sqrt(3)/2)
        # The normals point at angles 30, 90, 150, 210, 270, 330 degrees
        normals = []
        for k in range(6):
            theta = math.radians(30 + k * 60)
            normals.append((math.cos(theta), math.sin(theta)))

        for ix, iy, iang in inner_positions:
            inner_verts = hexagon_vertices(ix, iy, 1.0, iang)
            for vx, vy in inner_verts:
                dx = vx - cx
                dy = vy - cy
                # Rotate to outer hexagon's local frame
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a

                for nx, ny in normals:
                    proj = rx * nx + ry * ny
                    # Required side: proj / (sqrt(3)/2)
                    required_side = proj / (sqrt3 / 2)
                    if required_side > max_side:
                        max_side = required_side
        return max_side + 1e-10

    def hex_distance(p1, p2):
        """
        Exact minimum distance between two unit regular hexagons.
        Returns 0 if they overlap, positive if separated.
        Uses Separating Axis Theorem on the convex polygons.
        """
        # Get vertices of both hexagons
        verts1 = hexagon_vertices(p1[0], p1[1], 1.0, p1[2])
        verts2 = hexagon_vertices(p2[0], p2[1], 1.0, p2[2])

        # Collect all edge normals (candidate separating axes)
        axes = []
        for i in range(6):
            for verts in (verts1, verts2):
                x1, y1 = verts[i]
                x2, y2 = verts[(i + 1) % 6]
                dx = x2 - x1
                dy = y2 - y1
                # Normal to this edge
                length = math.sqrt(dx * dx + dy * dy)
                if length > 1e-12:
                    axes.append((-dy / length, dx / length))

        # Also add the direction between centers as a candidate axis
        dx_c = p2[0] - p1[0]
        dy_c = p2[1] - p1[1]
        dist_c = math.sqrt(dx_c * dx_c + dy_c * dy_c)
        if dist_c > 1e-12:
            axes.append((dx_c / dist_c, dy_c / dist_c))

        # For each axis, project both polygons and check for separation
        for nx, ny in axes:
            min1 = min(vx * nx + vy * ny for vx, vy in verts1)
            max1 = max(vx * nx + vy * ny for vx, vy in verts1)
            min2 = min(vx * nx + vy * ny for vx, vy in verts2)
            max2 = max(vx * nx + vy * ny for vx, vy in verts2)

            # If there's a gap, the distance is the gap
            gap = max(min2 - max1, min1 - max2)
            if gap > 0:
                return gap

        # No separating axis found -> they overlap
        return 0.0

    def check_overlap(p1, p2):
        """
        Return overlap penalty: 0 if disjoint, positive if overlapping.
        Uses exact SAT-based distance. If distance is 0, they overlap.
        We need a measure of how much they overlap. For the penalty,
        if they are disjoint (distance > 0), penalty is 0.
        If they overlap, we estimate the penetration depth.
        """
        # First quick check: if center distance > 2.0, definitely disjoint
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d = math.sqrt(dx * dx + dy * dy)
        if d >= 2.0:
            return 0.0

        # Use SAT to find minimum distance
        dist = hex_distance(p1, p2)
        if dist > 0:
            return 0.0
        else:
            # They overlap. Estimate penetration depth.
            # A rough estimate: if centers are close, penetration is significant.
            # Use (2.0 - d) as a proxy for penetration depth, clamped to be positive.
            # This is not exact but drives the optimizer to separate them.
            return max(0.0, 2.0 - d)

    def total_penalty(inner_positions):
        """Sum of overlap penalties for all pairs."""
        penalty = 0.0
        n = len(inner_positions)
        for i in range(n):
            for j in range(i + 1, n):
                penalty += check_overlap(inner_positions[i], inner_positions[j])
        return penalty

    # ---- Rigid cluster formulation ----
    # Central hex at origin (angle 0)
    # 6 ring hexagons at distance ring_r, angles 0,60,120,180,240,300
    # 4 gap hexagons at variable angles/distances
    # Free params: [ring_r, gap0_angle, gap0_dist, gap1_angle, gap1_dist, gap2_angle, gap2_dist, gap3_angle, gap3_dist]
    # That's 9 params. Let's fix ring hexagons' angles to 0,60,...,300 and make their distance a param.
    # Gap hexagons: 4 of them, each with angle and distance = 8 params + ring_r = 9 total.
    
    def build_cluster(params):
        """
        Build 11 hexagon positions from 9 parameters:
        [ring_r, g0_ang, g0_dist, g1_ang, g1_dist, g2_ang, g2_dist, g3_ang, g3_dist]
        Angles in degrees. Returns list of [x, y, angle_deg].
        """
        ring_r = params[0]
        positions = [[0.0, 0.0, 0.0]]  # central hex
        
        # 6 ring hexagons
        for k in range(6):
            ang = math.radians(60 * k)
            x = ring_r * math.cos(ang)
            y = ring_r * math.sin(ang)
            positions.append([x, y, 0.0])
        
        # 4 gap hexagons
        for g in range(4):
            ang_deg = params[1 + 2*g]
            dist = params[2 + 2*g]
            ang = math.radians(ang_deg)
            x = dist * math.cos(ang)
            y = dist * math.sin(ang)
            positions.append([x, y, 0.0])
        
        return positions

    def objective(params, outer_angle=0.0):
        """
        Objective: outer side length + large penalty for overlaps.
        Uses smooth log-sum-exp for overlap penalty to aid Nelder-Mead.
        """
        positions = build_cluster(params)
        
        # Compute overlap penalties using smooth approximation
        # For each pair, if distance < 2.0, there may be overlap
        # Use a smooth penalty: sum of max(0, overlap_depth)^2 approximated
        penalty = 0.0
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                d = math.sqrt(dx * dx + dy * dy)
                if d < 2.5:
                    # Rough overlap estimate: if d < 2, they overlap
                    # Use (2.0 - d) as penetration proxy, squared for smoothness
                    if d < 2.0:
                        pen = (2.0 - d) ** 2
                        penalty += pen
        
        side = compute_min_side(positions, (0.0, 0.0), outer_angle)
        return side + 100.0 * penalty

    def nelder_mead(f, x0, tol=1e-8, max_iter=5000, step=0.5):
        """
        Nelder-Mead simplex optimization.
        f: objective function taking a list/array
        x0: initial point (list)
        Returns best point found.
        """
        n = len(x0)
        # Initialize simplex
        simplex = [list(x0)]
        for i in range(n):
            pt = list(x0)
            pt[i] = pt[i] + step if pt[i] != 0 else step
            simplex.append(pt)
        
        values = [f(pt) for pt in simplex]
        
        for it in range(max_iter):
            # Sort simplex by function value
            order = sorted(range(n + 1), key=lambda i: values[i])
            simplex = [simplex[i] for i in order]
            values = [values[i] for i in order]
            
            # Check convergence
            if values[-1] - values[0] < tol:
                break
            
            # Centroid of all but worst point
            centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
            
            # Reflection
            xr = [centroid[j] + step * (centroid[j] - simplex[-1][j]) for j in range(n)]
            fr = f(xr)
            
            if fr < values[0]:
                # Expansion
                xe = [centroid[j] + 2 * step * (centroid[j] - simplex[-1][j]) for j in range(n)]
                fe = f(xe)
                if fe < fr:
                    simplex[-1] = xe
                    values[-1] = fe
                else:
                    simplex[-1] = xr
                    values[-1] = fr
            elif fr < values[-2]:
                simplex[-1] = xr
                values[-1] = fr
            else:
                # Contraction
                xc = [centroid[j] + 0.5 * step * (simplex[-1][j] - centroid[j]) for j in range(n)]
                fc = f(xc)
                if fc < values[-1]:
                    simplex[-1] = xc
                    values[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        values[i] = f(simplex[i])
        
        # Return best point
        best_idx = min(range(n + 1), key=lambda i: values[i])
        return simplex[best_idx]

    # ---- Optimize rigid cluster ----
    # Initial guess: ring_r = sqrt(3) + 0.1, gaps at 30,150,210,330 deg with dist ~ 2*ring_r*cos(30deg)
    ring_r0 = sqrt3 + 0.1
    gap_dist0 = 2 * ring_r0 * math.cos(math.pi / 6) + 0.1
    initial_params = [ring_r0, 30.0, gap_dist0, 150.0, gap_dist0, 210.0, gap_dist0, 330.0, gap_dist0]
    
    best_params = list(initial_params)
    best_val = objective(initial_params)
    
    # Nelder-Mead from initial guess
    if time.time() < deadline - 20.0:
        nm_result = nelder_mead(objective, initial_params, tol=1e-7, max_iter=3000, step=0.3)
        nm_val = objective(nm_result)
        if nm_val < best_val:
            best_val = nm_val
            best_params = nm_result
    
    # Simulated annealing on the 9-dim parameter space
    current_params = list(best_params)
    current_val = best_val
    best_params = list(current_params)
    best_val = current_val
    
    temp = 1.0
    cooling = 0.998
    min_temp = 0.01
    steps = 0
    max_steps = 50000
    
    while temp > min_temp and steps < max_steps and time.time() < deadline - 15.0:
        # Perturb 1-3 random params
        n_perturb = random.randint(1, 3)
        for _ in range(n_perturb):
            idx = random.randint(0, 8)
            if idx == 0:  # ring_r
                current_params[idx] += random.gauss(0, 0.1)
                current_params[idx] = max(1.8, current_params[idx])  # keep reasonable
            elif idx % 2 == 1:  # angle
                current_params[idx] += random.gauss(0, 10.0)
            else:  # distance
                current_params[idx] += random.gauss(0, 0.15)
                current_params[idx] = max(1.5, current_params[idx])
        
        new_val = objective(current_params)
        delta = new_val - current_val
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_params = list(current_params)
        else:
            # Revert
            current_params = list(best_params) if random.random() < 0.1 else current_params
            # Actually, let's just keep the current and move on; SA handles reversion implicitly
            # Revert properly:
            # We need to restore. Let's store old params before perturbation.
            # For simplicity, if rejected, revert to previous state.
            # Let me restructure: store old_params before perturbation.
        
        temp *= cooling
        steps += 1
    
    # Actually, let me redo SA properly with proper reversion
    # Reset
    current_params = list(best_params)
    current_val = best_val
    
    temp = 1.0
    cooling = 0.998
    min_temp = 0.01
    steps = 0
    max_steps = 50000
    
    while temp > min_temp and steps < max_steps and time.time() < deadline - 15.0:
        old_params = list(current_params)
        
        # Perturb 1-3 random params
        n_perturb = random.randint(1, 3)
        for _ in range(n_perturb):
            idx = random.randint(0, 8)
            if idx == 0:  # ring_r
                current_params[idx] += random.gauss(0, 0.1)
                current_params[idx] = max(1.8, current_params[idx])
            elif idx % 2 == 1:  # angle
                current_params[idx] += random.gauss(0, 10.0)
            else:  # distance
                current_params[idx] += random.gauss(0, 0.15)
                current_params[idx] = max(1.5, current_params[idx])
        
        new_val = objective(current_params)
        delta = new_val - current_val
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_params = list(current_params)
        else:
            current_params = old_params
        
        temp *= cooling
        steps += 1
    
    # Multi-start: try different initial gap angle configurations
    gap_angle_sets = [
        [30, 150, 210, 330],
        [90, 210, 270, 30],
        [0, 120, 180, 300],
        [60, 180, 240, 0],
        [30, 90, 210, 270],
        [150, 210, 270, 330],
        [45, 165, 225, 345],
        [0, 60, 180, 240],
    ]
    
    for gap_angles in gap_angle_sets:
        if time.time() > deadline - 20.0:
            break
        
        # Set initial params with these gap angles
        params = [ring_r0]
        for ang in gap_angles:
            params.append(float(ang))
            params.append(gap_dist0)
        
        # Nelder-Mead
        if time.time() < deadline - 15.0:
            nm_res = nelder_mead(objective, params, tol=1e-7, max_iter=2000, step=0.3)
            nm_v = objective(nm_res)
            if nm_v < best_val:
                best_val = nm_v
                best_params = nm_res
        
        # Brief SA
        if time.time() < deadline - 10.0:
            cur = list(best_params)
            cur_val = best_val
            t = 0.5
            for s in range(5000):
                if time.time() > deadline - 10.0:
                    break
                old = list(cur)
                for _ in range(random.randint(1, 2)):
                    idx = random.randint(0, 8)
                    if idx == 0:
                        cur[idx] += random.gauss(0, 0.08)
                        cur[idx] = max(1.8, cur[idx])
                    elif idx % 2 == 1:
                        cur[idx] += random.gauss(0, 8.0)
                    else:
                        cur[idx] += random.gauss(0, 0.12)
                        cur[idx] = max(1.5, cur[idx])
                
                nv = objective(cur)
                d = nv - cur_val
                if d < 0 or random.random() < math.exp(-d / t):
                    cur_val = nv
                    if cur_val < best_val:
                        best_val = cur_val
                        best_params = list(cur)
                else:
                    cur = old
                t *= 0.997
    
    # Build final configuration
    best_inner = build_cluster(best_params)
    
    # Optimize outer angle
    best_side = float('inf')
    best_angle = 0.0
    angle_candidates = list(range(0, 60, 3))
    for outer_angle in angle_candidates:
        if time.time() > deadline - 5.0:
            break
        side = compute_min_side(best_inner, (0.0, 0.0), outer_angle)
        if side < best_side:
            best_side = side
            best_angle = outer_angle
    
    # Fine-tune outer angle
    if time.time() < deadline - 3.0:
        for _ in range(300):
            if time.time() > deadline - 3.0:
                break
            trial_angle = best_angle + random.gauss(0, 0.8)
            trial_side = compute_min_side(best_inner, (0.0, 0.0), trial_angle)
            if trial_side < best_side:
                best_side = trial_side
                best_angle = trial_angle
    
    # Verify no overlap
    final_penalty = total_penalty(best_inner)
    if final_penalty > 1e-6:
        # Fallback to safe spiral
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
        
        best_side = float('inf')
        best_angle = 0.0
        for outer_angle in range(0, 60, 5):
            if time.time() > deadline - 2.0:
                break
            side = compute_min_side(safe_inner, (0.0, 0.0), outer_angle)
            if side < best_side:
                best_side = side
                best_angle = outer_angle
        best_inner = safe_inner
    
    return best_inner, (0.0, 0.0), best_side, best_angle
# EVOLVE-BLOCK-END
