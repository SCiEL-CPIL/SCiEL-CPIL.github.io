import math
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0 (pointy-top)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def poly_dist(p1, p2, eps=1e-10):
        """Minimum distance between two convex hexagons using support functions.
        Returns negative if overlapping (penetration depth), positive if separated."""
        min_dist = float('inf')
        # Check all 12 edge normals (6 from each polygon)
        for poly, other in [(p1, p2), (p2, p1)]:
            for i in range(6):
                v1 = poly[i]
                v2 = poly[(i+1) % 6]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                normal /= norm
                # Project both polygons onto this normal
                proj1 = p1 @ normal
                proj2 = other @ normal
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                # Gap: positive means separated
                gap = max(min1 - max2, min2 - max1)
                if gap < 0:
                    min_dist = min(min_dist, gap)
                else:
                    # Found a separating axis - polygons are disjoint
                    return gap
        return min_dist

    def calc_outer_side(centers, angles, outer_angle=0.0):
        """Calculate minimum side length of regular hexagon containing all inner hexagons."""
        theta = math.radians(outer_angle)
        max_proj = 0.0
        for k in range(6):
            na = theta + math.pi/6 + k * math.pi/3
            nx, ny = math.cos(na), math.sin(na)
            for i in range(N):
                cx, cy = centers[i]
                a = math.radians(angles[i])
                ca, sa = math.cos(a), math.sin(a)
                # Max projection of hex i onto direction (nx, ny)
                best = 0.0
                for k2 in range(6):
                    va = a + k2 * math.pi/3
                    proj = math.cos(va)*nx + math.sin(va)*ny
                    if proj > best:
                        best = proj
                proj_val = cx*nx + cy*ny + best
                if proj_val > max_proj:
                    max_proj = proj_val
        return 2.0 * max_proj / math.sqrt(3)

    def is_valid(centers, angles, tol=1e-6):
        """Check if all hexagons are pairwise disjoint."""
        polys = [get_hex_verts(centers[i][0], centers[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if poly_dist(polys[i], polys[j]) < -tol:
                    return False
        return True

    def build_positions_orbits(orbit_params):
        """
        orbit_params: list of tuples (r, theta_base, angle_offset)
        Returns positions (12, 2) and angles (12,)
        """
        positions = []
        angles = []
        for r, theta_base, angle_offset in orbit_params:
            for k in range(6):
                theta = theta_base + k * math.pi / 3
                x = r * math.cos(theta)
                y = r * math.sin(theta)
                positions.append([x, y])
                ang = angle_offset + theta # Lock rotation to radial direction + offset
                angles.append(ang)
        return np.array(positions), np.array(angles)

    def objective_sym(params, orbit_sizes):
        """
        params: flattened parameters for all orbits.
        For each orbit, we have 3 params: r, theta_base, angle_offset.
        """
        positions = []
        angles = []
        idx = 0
        for size in orbit_sizes:
            if size == 6:
                r = params[idx]
                theta_base = params[idx+1]
                angle_offset = params[idx+2]
                idx += 3
                for k in range(6):
                    theta = theta_base + k * math.pi / 3
                    x = r * math.cos(theta)
                    y = r * math.sin(theta)
                    positions.append([x, y])
                    angles.append(angle_offset + theta)
            elif size == 3:
                # Orbit of 3: Points on axes?
                r = params[idx]
                theta_base = params[idx+1]
                angle_offset = params[idx+2]
                idx += 3
                for k in range(3):
                    theta = theta_base + k * 2 * math.pi / 3
                    x = r * math.cos(theta)
                    y = r * math.sin(theta)
                    positions.append([x, y])
                    angles.append(angle_offset + theta)
            else:
                raise ValueError("Invalid orbit size")
        
        positions = np.array(positions)
        angles = np.array(angles)
        
        # Penalty for overlap
        if not is_valid(positions, angles):
            return 1000.0 # Large penalty
        
        return calc_outer_side(positions, angles, 0.0)

    def smooth_min_gap(centers, angles, beta=100.0):
        """Smooth approximation of minimum gap between all pairs."""
        polys = [get_hex_verts(centers[i][0], centers[i][1], 1.0, angles[i]) for i in range(N)]
        log_sum = 0.0
        max_log = -1e30
        for i in range(N):
            for j in range(i+1, N):
                gap = poly_dist(polys[i], polys[j])
                log_val = -beta * gap
                if log_val > max_log:
                    max_log = log_val
        for i in range(N):
            for j in range(i+1, N):
                gap = poly_dist(polys[i], polys[j])
                log_val = -beta * gap
                log_sum += math.exp(log_val - max_log)
        return -(max_log + math.log(log_sum)) / beta

    def objective_slsqp(params):
        centers = params[:2*N].reshape(N, 2)
        angles = params[2*N:3*N]
        outer_angle = params[3*N]
        return calc_outer_side(centers, angles, outer_angle)

    def make_constraint(i, j):
        def constraint(params):
            ci = params[2*i:2*i+2]
            cj = params[2*j:2*j+2]
            ai = params[2*N + i]
            aj = params[2*N + j]
            pi = get_hex_verts(ci[0], ci[1], 1.0, ai)
            pj = get_hex_verts(cj[0], cj[1], 1.0, aj)
            return poly_dist(pi, pj)
        return constraint

    def make_smooth_constraint(i, j, beta=100.0):
        def constraint(params):
            ci = params[2*i:2*i+2]
            cj = params[2*j:2*j+2]
            ai = params[2*N + i]
            aj = params[2*N + j]
            pi = get_hex_verts(ci[0], ci[1], 1.0, ai)
            pj = get_hex_verts(cj[0], cj[1], 1.0, aj)
            return max(0.0, poly_dist(pi, pj))
        return constraint

    # Phase 1: Symmetry-constrained optimization
    best_score = float('inf')
    best_result = None

    # Try different symmetric configurations
    configs = [
        [6, 6],
        [6, 3, 3],
        [3, 3, 3, 3]
    ]

    for sizes in configs:
        num_params = 3 * len(sizes)
        
        # Initial guess
        # Spread them out
        if sizes == [6, 6]:
            # Two rings
            x0 = [2.0, 0.0, 0.0, 5.0, 30.0, 0.0] # r1, th1, a1, r2, th2, a2
        elif sizes == [6, 3, 3]:
            x0 = [2.0, 0.0, 0.0, 4.0, 0.0, 0.0, 4.0, 60.0, 0.0]
        else: # [3,3,3,3]
            x0 = [2.0, 0.0, 0.0, 2.0, 60.0, 0.0, 2.0, 120.0, 0.0, 2.0, 180.0, 0.0]

        # Bounds
        # r > 0, theta any, angle any
        bounds = []
        for _ in range(len(sizes)):
            bounds.append((0.1, 10.0)) # r
            bounds.append((0.0, 360.0)) # theta
            bounds.append((0.0, 360.0)) # angle

        # Convert angles to radians for optimization?
        # SLSQP handles radians better usually, or we can just work in radians.
        # Let's work in radians.
        
        # Re-initialize x0 in radians
        x0_rad = []
        for i, val in enumerate(x0):
            if i % 3 == 1 or i % 3 == 2: # theta or angle
                x0_rad.append(math.radians(val))
            else:
                x0_rad.append(val)
        
        # Bounds in radians
        bounds_rad = []
        for _ in range(len(sizes)):
            bounds_rad.append((0.1, 10.0))
            bounds_rad.append((0.0, 2*math.pi))
            bounds_rad.append((0.0, 2*math.pi))

        def obj_func(p):
            if time.time() > deadline - 5:
                return 1000.0
            return objective_sym(p, sizes)

        try:
            res = minimize(
                obj_func, 
                x0_rad, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if res.fun < best_score and res.fun < 100.0:
                best_score = res.fun
                best_result = (res.x, sizes)
        except Exception:
            pass

    # If no symmetric solution found or score is bad, fall back to general optimization
    if best_result is None or best_score > 5.0:
        # General optimization with penalty
        # Initial guess: Honeycomb subset
        sqrt3 = math.sqrt(3)
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
        
        x0 = np.concatenate([init_positions.flatten(), init_angles])
        
        def obj_general(params):
            if time.time() > deadline - 5:
                return 1000.0
            positions = params[:2*N].reshape(N, 2)
            angles = params[2*N:]
            
            if not is_valid(positions, angles):
                # Penalty proportional to overlap depth?
                # Simple penalty: count of overlapping pairs * 10
                polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
                pen = 0
                for i in range(N):
                    for j in range(i+1, N):
                        if poly_dist(polys[i], polys[j]) < 0:
                            pen += 10.0
                return calc_outer_side(positions, angles, 0.0) + pen
            
            return calc_outer_side(positions, angles, 0.0)

        try:
            res = minimize(
                obj_general, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-6, 'adaptive': True}
            )
            
            if res.fun < best_score and res.fun < 100.0:
                best_score = res.fun
                best_result = (res.x, 'general')
        except Exception:
            pass

    # Finalize result
    if best_result is not None:
        params, type_str = best_result
        
        if type_str == 'general':
            final_positions = params[:2*N].reshape(N, 2)
            final_angles = params[2*N:]
        else:
            final_positions, final_angles = build_positions_orbits([(p[i*3], p[i*3+1], p[i*3+2]) for i in range(len(type_str))])
        
        # Verify validity one last time
        if not is_valid(final_positions, final_angles):
            # Fallback to a very safe, large configuration
            final_positions = np.array([
                [0, 0], [2.1, 0], [-2.1, 0],
                [1.05, 1.81], [-1.05, 1.81], [0, 3.62],
                [1.05, -1.81], [-1.05, -1.81], [0, -3.62],
                [3.15, 1.81], [-3.15, 1.81], [3.15, -1.81]
            ])
            final_angles = np.zeros(N)
            
        final_side = calc_outer_side(final_positions, final_angles, 0.0)
    else:
        # Emergency fallback
        final_positions = np.array([
            [0, 0], [2.1, 0], [-2.1, 0],
            [1.05, 1.81], [-1.05, 1.81], [0, 3.62],
            [1.05, -1.81], [-1.05, -1.81], [0, -3.62],
            [3.15, 1.81], [-3.15, 1.81], [3.15, -1.81]
        ])
        final_angles = np.zeros(N)
        final_side = calc_outer_side(final_positions, final_angles, 0.0)

    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0
