import math
import time
import random
import numpy as np

N = 12

def construct():
    deadline = time.time() + 140.0

    def hex_vertices(cx, cy, side, angle_deg):
        a = math.radians(angle_deg)
        vs = []
        for i in range(6):
            t = a + i * math.pi / 3
            vs.append([cx + side * math.cos(t), cy + side * math.sin(t)])
        return np.array(vs)

    def min_enclosing_hex(centers, angles):
        """Find the minimum side length of a regular hexagon containing all inner hexagons.
        The outer hexagon is free to translate and rotate."""
        # Precompute all vertices of all inner hexagons
        all_pts = []
        for i in range(len(centers)):
            vs = hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            all_pts.append(vs)
        all_pts = np.vstack(all_pts)  # (72, 2)
        
        # For a regular hexagon with center (cx,cy), side s, rotation theta:
        # The 6 face normals are at angles theta + pi/6 + k*pi/3
        # Constraint: for each normal n_k, max_i (p_i . n_k) - (c . n_k) <= s * sqrt(3)/2
        # And: max_i (p_i . n_k) - (c . n_k) >= -(s * sqrt(3)/2)  [symmetric]
        # So: c . n_k + s*sqrt(3)/2 >= max_i(p_i . n_k)
        #     c . n_k - s*sqrt(3)/2 <= min_i(p_i . n_k)
        # => s >= (max_i(p_i . n_k) - min_i(p_i . n_k)) / sqrt(3)
        # And c . n_k must be in [min + s*sqrt(3)/2, max - s*sqrt(3)/2]
        # For feasibility: max - min <= s * sqrt(3) for all k
        # Minimum s for a given theta: s = max_k (range_k) / sqrt(3)
        # where range_k = max_i(p_i . n_k) - min_i(p_i . n_k)
        
        # But we also need to check that there exists a center c satisfying all constraints.
        # For a regular hexagon, if s >= max_k range_k / sqrt(3), then the center can be
        # chosen as the centroid of the projections. Let's verify.
        # Actually, for a regular hexagon, the constraints on c are:
        #   c . n_k >= max_k - s*sqrt(3)/2  for each k
        #   c . n_k <= min_k + s*sqrt(3)/2  for each k
        # Since n_k and n_{k+3} are opposite, these are consistent.
        # The minimum s is max_k range_k / sqrt(3), and c can be the centroid.
        
        best_s = float('inf')
        best_cx, best_cy = 0.0, 0.0
        best_theta = 0.0
        
        # Search over outer rotation
        for theta_deg in np.linspace(0, 60, 121):
            theta = math.radians(theta_deg)
            ranges = []
            for k in range(6):
                na = theta + math.pi / 6 + k * math.pi / 3
                nx, ny = math.cos(na), math.sin(na)
                projs = all_pts[:, 0] * nx + all_pts[:, 1] * ny
                ranges.append(projs.max() - projs.min())
            s = max(ranges) / math.sqrt(3)
            if s < best_s:
                best_s = s
                best_theta = theta_deg
                # Compute center: average of the midpoints of projections
                cx_acc, cy_acc = 0.0, 0.0
                for k in range(6):
                    na = theta + math.pi / 6 + k * math.pi / 3
                    nx, ny = math.cos(na), math.sin(na)
                    projs = all_pts[:, 0] * nx + all_pts[:, 1] * ny
                    mid = (projs.max() + projs.min()) / 2.0
                    cx_acc += mid * nx
                    cy_acc += mid * ny
                best_cx = cx_acc / 6.0
                best_cy = cy_acc / 6.0
        
        # Refine around best theta
        for theta_deg in np.linspace(best_theta - 3, best_theta + 3, 61):
            theta = math.radians(theta_deg)
            ranges = []
            for k in range(6):
                na = theta + math.pi / 6 + k * math.pi / 3
                nx, ny = math.cos(na), math.sin(na)
                projs = all_pts[:, 0] * nx + all_pts[:, 1] * ny
                ranges.append(projs.max() - projs.min())
            s = max(ranges) / math.sqrt(3)
            if s < best_s:
                best_s = s
                best_theta = theta_deg
                cx_acc, cy_acc = 0.0, 0.0
                for k in range(6):
                    na = theta + math.pi / 6 + k * math.pi / 3
                    nx, ny = math.cos(na), math.sin(na)
                    projs = all_pts[:, 0] * nx + all_pts[:, 1] * ny
                    mid = (projs.max() + projs.min()) / 2.0
                    cx_acc += mid * nx
                    cy_acc += mid * ny
                best_cx = cx_acc / 6.0
                best_cy = cy_acc / 6.0
        
        return best_s, best_cx, best_cy, best_theta

    def overlap_penalty(centers, angles, eps=0.05):
        """Smooth penalty for overlaps. Returns 0 if all hexagons are disjoint (with gap >= 0)."""
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                # Compute minimum distance between two hexagons
                # Using SAT: find the minimum penetration depth
                v1 = hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
                v2 = hex_vertices(centers[j][0], centers[j][1], 1.0, angles[j])
                
                min_pen = 0.0
                # Check all 12 axes
                for v in [v1, v2]:
                    for k in range(6):
                        p1 = v[k]
                        p2 = v[(k + 1) % 6]
                        ex, ey = p2[0] - p1[0], p2[1] - p1[1]
                        nx, ny = -ey, ex
                        # Project both
                        proj1 = v1[:, 0] * nx + v1[:, 1] * ny
                        proj2 = v2[:, 0] * nx + v2[:, 1] * ny
                        min1, max1 = proj1.min(), proj1.max()
                        min2, max2 = proj2.min(), proj2.max()
                        # Overlap on this axis
                        overlap = min(max1, max2) - max(min1, min2)
                        if overlap > min_pen:
                            min_pen = overlap
                
                if min_pen > 0:
                    # Smooth penalty: quadratic for small overlaps, linear for large
                    if min_pen < eps:
                        penalty += (min_pen / eps) ** 2
                    else:
                        penalty += 1.0 + (min_pen - eps) / eps
        
        return penalty

    def objective(centers, angles):
        """Total cost = outer side + overlap penalty."""
        s, _, _, _ = min_enclosing_hex(centers, angles)
        pen = overlap_penalty(centers, angles)
        return s + 100.0 * pen, s, pen

    def lattice_start(scale=1.0, rot_deg=0.0, jitter=0.0):
        """Generate a hexagonal lattice starting configuration."""
        d = math.sqrt(3) * scale
        rot = math.radians(rot_deg)
        cr, sr = math.cos(rot), math.sin(rot)
        
        # Hexagonal lattice basis
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
        
        # Apply rotation
        R = np.array([[cr, -sr], [sr, cr]])
        u = R @ u
        v = R @ v
        
        # Ring 0: center
        positions = [np.array([0.0, 0.0])]
        
        # Ring 1: 6 neighbors
        ring1 = [u, v, v - u, -u, -v, u - v]
        for p in ring1:
            positions.append(p.copy())
        
        # Ring 2: pick 5 from the 12 outer positions
        ring2 = []
        for p1 in ring1:
            for b in [u, v, v - u, -u, -v, u - v]:
                p2 = p1 + b
                # Check not in ring 0 or ring 1
                is_r0 = np.linalg.norm(p2) < 1e-9
                is_r1 = any(np.linalg.norm(p2 - r) < 1e-9 for r in ring1)
                if not is_r0 and not is_r1:
                    # Check duplicate
                    dup = any(np.linalg.norm(p2 - r) < 1e-9 for r in ring2)
                    if not dup:
                        ring2.append(p2)
        
        # Sort by distance and pick closest 5
        ring2.sort(key=lambda p: np.linalg.norm(p))
        for p in ring2[:5]:
            positions.append(p.copy())
        
        # Add jitter
        if jitter > 0:
            for i in range(len(positions)):
                positions[i] += jitter * np.array([random.gauss(0, 1), random.gauss(0, 1)])
        
        # All angles = rot_deg (aligned with lattice)
        angles = [rot_deg % 60.0] * N
        centers = [[p[0], p[1]] for p in positions]
        
        return centers, angles

    def anneal(centers, angles, start_temp, end_temp, n_steps, time_limit):
        """Simulated annealing to minimize the objective."""
        t0 = time.time()
        cur_cost, cur_s, cur_pen = objective(centers, angles)
        best_cost = cur_cost
        best_centers = [c[:] for c in centers]
        best_angles = angles[:]
        
        # Step sizes
        pos_step = 0.3
        ang_step = 5.0
        
        for step in range(n_steps):
            if time.time() > time_limit:
                break
            if time.time() > deadline - 5:
                break
            
            temp = start_temp * (end_temp / start_temp) ** (step / max(n_steps - 1, 1))
            
            # Propose move
            new_centers = [c[:] for c in centers]
            new_angles = angles[:]
            
            # Randomly perturb one or two hexagons
            n_perturb = random.choice([1, 1, 2])
            for _ in range(n_perturb):
                idx = random.randint(0, N - 1)
                new_centers[idx][0] += pos_step * random.gauss(0, 1)
                new_centers[idx][1] += pos_step * random.gauss(0, 1)
                new_angles[idx] = (new_angles[idx] + ang_step * random.gauss(0, 1)) % 60.0
            
            new_cost, new_s, new_pen = objective(new_centers, new_angles)
            
            # Accept/reject
            if new_cost < cur_cost or random.random() < math.exp(-(new_cost - cur_cost) / max(temp, 1e-10)):
                centers = new_centers
                angles = new_angles
                cur_cost, cur_s, cur_pen = new_cost, new_s, new_pen
                
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_centers = [c[:] for c in centers]
                    best_angles = angles[:]
            
            # Reduce step size
            pos_step *= 0.9995
            ang_step *= 0.9995
        
        return best_centers, best_angles, best_cost

    # Multiple restarts
    best_overall_cost = float('inf')
    best_overall_centers = None
    best_overall_angles = None
    
    # Initial lattice
    centers0, angles0 = lattice_start(scale=1.0, rot_deg=30.0)
    cost0, s0, pen0 = objective(centers0, angles0)
    best_overall_cost = cost0
    best_overall_centers = [c[:] for c in centers0]
    best_overall_angles = angles0[:]
    
    n_restarts = 8
    for r in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        # Generate perturbed start
        scale = 1.0 + random.uniform(-0.05, 0.05)
        rot = 30.0 + random.uniform(-15, 15)
        jitter = random.uniform(0.0, 0.3)
        centers_r, angles_r = lattice_start(scale=scale, rot_deg=rot, jitter=jitter)
        
        # Anneal
        t_start = time.time()
        time_left = deadline - 15 - t_start
        n_steps = min(5000, int(time_left / 0.01))  # ~10ms per step estimate
        start_temp = 0.5
        end_temp = 0.001
        
        c_res, a_res, cost_res = anneal(centers_r, angles_r, start_temp, end_temp, n_steps, deadline - 15)
        
        if cost_res < best_overall_cost:
            best_overall_cost = cost_res
            best_overall_centers = c_res
            best_overall_angles = a_res
    
    # Final: compute the enclosing hexagon for the best configuration
    final_s, final_cx, final_cy, final_theta = min_enclosing_hex(best_overall_centers, best_overall_angles)
    
    # Verify validity
    valid = True
    verts = []
    for i in range(N):
        verts.append(hex_vertices(best_overall_centers[i][0], best_overall_centers[i][1], 1.0, best_overall_angles[i]))
    for i in range(N):
        for j in range(i + 1, N):
            v1, v2 = verts[i], verts[j]
            disjoint = False
            for v in [v1, v2]:
                for k in range(6):
                    p1, p2 = v[k], v[(k + 1) % 6]
                    ex, ey = p2[0] - p1[0], p2[1] - p1[1]
                    nx, ny = -ey, ex
                    proj1 = v1[:, 0] * nx + v1[:, 1] * ny
                    proj2 = v2[:, 0] * nx + v2[:, 1] * ny
                    if proj1.max() < proj2.min() + 1e-9 or proj2.max() < proj1.min() + 1e-9:
                        disjoint = True
                        break
                if disjoint:
                    break
            if not disjoint:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to clean lattice
        centers0, angles0 = lattice_start(scale=1.0, rot_deg=30.0)
        final_s, final_cx, final_cy, final_theta = min_enclosing_hex(centers0, angles0)
        best_overall_centers = centers0
        best_overall_angles = angles0
    
    inner = [[c[0], c[1], a] for c, a in zip(best_overall_centers, best_overall_angles)]
    outer_center = (final_cx, final_cy)
    outer_side = final_s
    outer_angle = final_theta
    
    return inner, outer_center, outer_side, outer_angle
