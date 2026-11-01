"""Rewritten program for packing 12 unit hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of a regular hexagon."""
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def hexagon_support(cx, cy, side, angle_deg, direction):
        """Compute support function of hexagon in given direction (unit vector)."""
        angle_rad = np.radians(angle_deg)
        # The support function of a regular hexagon with side s in direction u
        # is s * max over vertices of dot(v_i - center, u)
        # For a regular hexagon, the support function is:
        # h(u) = s * max_k cos(angle_between(u, normal_k)) where normals are at 30+60k degrees
        # Actually, for a regular hexagon centered at origin, the support function in direction u
        # is s * max(1, sqrt(3)) * |cos(angle)| ... let's compute directly.
        # The 6 vertices are at angles angle_deg + 60k, distance = s (circumradius = side for hexagon)
        # Wait, for a regular hexagon, circumradius = side length.
        # So vertices are at distance s from center.
        # Support = max_i dot(v_i, u) = s * max_i cos(angle_i - angle_u)
        # where angle_i = angle_rad + k*pi/3
        
        # More efficiently: the support function of a regular hexagon with circumradius R=s
        # in direction making angle theta with the hexagon's reference axis is:
        # h(theta) = R * max_k cos(theta - k*pi/3)
        # But we can also use the fact that for a regular hexagon, the support function
        # equals R * |cos(min(|theta mod (pi/3)|, pi/3 - |theta mod (pi/3)|))|
        
        # Let's just compute it directly using the 6 vertex directions
        vertex_angles = angle_rad + np.arange(6) * np.pi / 3
        u_angle = np.arctan2(direction[1], direction[0])
        cos_vals = np.cos(vertex_angles - u_angle)
        return side * np.max(cos_vals)

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px, py) is inside hexagon centered at (cx, cy)."""
        dx = px - cx
        dy = py - cy
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a

        apothem = side * np.sqrt(3) / 2
        normal_angles = np.radians(np.arange(30, 360, 60))

        for na in normal_angles:
            nx = np.cos(na)
            ny = np.sin(na)
            dist = local_x * nx + local_y * ny
            if dist > apothem + 1e-9:
                return False
        return True

    def min_distance_to_hexagon_edge(px, py, cx, cy, side, angle_deg):
        """Compute minimum distance from point to hexagon boundary (negative if outside)."""
        dx = px - cx
        dy = py - cy
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a

        apothem = side * np.sqrt(3) / 2
        normal_angles = np.radians(np.arange(30, 360, 60))
        
        min_dist = float('inf')
        for na in normal_angles:
            nx = np.cos(na)
            ny = np.sin(na)
            dist = local_x * nx + local_y * ny
            # Distance to this edge plane
            edge_dist = apothem - dist
            if edge_dist < min_dist:
                min_dist = edge_dist
        return min_dist

    def polygons_disjoint_sat(v1, v2):
        """Check if two convex polygons are disjoint using SAT."""
        axes = []
        for i in range(len(v1)):
            p1 = v1[i]
            p2 = v1[(i + 1) % len(v1)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)

        for i in range(len(v2)):
            p1 = v2[i]
            p2 = v2[(i + 1) % len(v2)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)

        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)

            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True
        return False

    def sat_penetration_depth(v1, v2):
        """Compute penetration depth between two convex polygons using SAT.
        Returns 0 if disjoint, negative value indicating overlap depth."""
        axes = []
        for i in range(len(v1)):
            p1 = v1[i]
            p2 = v1[(i + 1) % len(v1)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)

        for i in range(len(v2)):
            p1 = v2[i]
            p2 = v2[(i + 1) % len(v2)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)

        min_overlap = float('inf')
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)

            if max1 < min2 or max2 < min1:
                return 0.0  # Disjoint
            
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < min_overlap:
                min_overlap = overlap
        
        return -min_overlap  # Negative means penetration

    def all_inside_outer(inner_positions, outer_center, outer_side, outer_angle):
        """Check all inner hexagons are inside the outer hexagon."""
        for i in range(len(inner_positions)):
            cx, cy, angle = inner_positions[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                    return False
        return True

    def containment_violation(inner_positions, outer_center, outer_side, outer_angle):
        """Compute total containment violation (0 if all inside)."""
        total_violation = 0.0
        for i in range(len(inner_positions)):
            cx, cy, angle = inner_positions[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            for vx, vy in verts:
                dx = vx - outer_center[0]
                dy = vy - outer_center[1]
                angle_rad = np.radians(outer_angle)
                cos_a = np.cos(-angle_rad)
                sin_a = np.sin(-angle_rad)
                local_x = dx * cos_a - dy * sin_a
                local_y = dx * sin_a + dy * cos_a

                apothem = outer_side * np.sqrt(3) / 2
                normal_angles = np.radians(np.arange(30, 360, 60))
                
                for na in normal_angles:
                    nx = np.cos(na)
                    ny = np.sin(na)
                    dist = local_x * nx + local_y * ny
                    violation = dist - apothem
                    if violation > 0:
                        total_violation += violation
        return total_violation

    def pairwise_disjoint(inner_positions):
        """Check all pairs of inner hexagons are disjoint."""
        for i in range(len(inner_positions)):
            for j in range(i + 1, len(inner_positions)):
                ci = (inner_positions[i][0], inner_positions[i][1])
                cj = (inner_positions[j][0], inner_positions[j][1])
                ai = inner_positions[i][2]
                aj = inner_positions[j][2]
                v1 = hexagon_vertices(ci[0], ci[1], 1.0, ai)
                v2 = hexagon_vertices(cj[0], cj[1], 1.0, aj)
                if not polygons_disjoint_sat(v1, v2):
                    return False
        return True

    def total_penetration(inner_positions):
        """Compute total pairwise penetration depth (0 if all disjoint)."""
        total = 0.0
        for i in range(len(inner_positions)):
            for j in range(i + 1, len(inner_positions)):
                ci = (inner_positions[i][0], inner_positions[i][1])
                cj = (inner_positions[j][0], inner_positions[j][1])
                ai = inner_positions[i][2]
                aj = inner_positions[j][2]
                v1 = hexagon_vertices(ci[0], ci[1], 1.0, ai)
                v2 = hexagon_vertices(cj[0], cj[1], 1.0, aj)
                pen = sat_penetration_depth(v1, v2)
                if pen < 0:
                    total += -pen
        return total

    def is_valid(inner, oc, os, oa):
        """Full validity check."""
        return all_inside_outer(inner, oc, os, oa) and pairwise_disjoint(inner)

    def required_outer_side(inner_positions, outer_center, outer_angle):
        """Find minimum outer side length that contains all inner hexagons."""
        # Binary search on side length
        lo = 1.0
        hi = 20.0
        for _ in range(30):
            mid = (lo + hi) / 2
            if all_inside_outer(inner_positions, outer_center, mid, outer_angle):
                hi = mid
            else:
                lo = mid
        return hi

    def make_params(inner, oc, os, oa):
        """Convert to flat parameter array."""
        params = []
        for pos in inner:
            params.extend([pos[0], pos[1], pos[2]])
        params.extend([oc[0], oc[1], os, oa])
        return np.array(params)

    def parse_params(params):
        """Convert flat parameter array back to structured form."""
        inner = []
        for i in range(N):
            inner.append((params[3 * i], params[3 * i + 1], params[3 * i + 2]))
        oc = (params[3 * N], params[3 * N + 1])
        os = params[3 * N + 2]
        oa = params[3 * N + 3]
        return inner, oc, os, oa

    def make_params_no_outer(inner, oc, oa):
        """Convert to flat parameter array without outer side (it's computed)."""
        params = []
        for pos in inner:
            params.extend([pos[0], pos[1], pos[2]])
        params.extend([oc[0], oc[1], oa])
        return np.array(params)

    def parse_params_no_outer(params):
        """Convert flat parameter array back to structured form without outer side."""
        inner = []
        for i in range(N):
            inner.append((params[3 * i], params[3 * i + 1], params[3 * i + 2]))
        oc = (params[3 * N], params[3 * N + 1])
        oa = params[3 * N + 2]
        return inner, oc, oa

    def objective(params):
        """Objective function: minimize outer side length with penalties."""
        n_inner = N
        inner_positions = []
        for i in range(n_inner):
            x = params[3 * i]
            y = params[3 * i + 1]
            a = params[3 * i + 2]
            inner_positions.append((x, y, a))

        ox = params[3 * n_inner]
        oy = params[3 * n_inner + 1]
        os = params[3 * n_inner + 2]
        oa = params[3 * n_inner + 3]

        outer_center = (ox, oy)
        outer_side = max(os, 0.5)
        outer_angle = oa

        # Compute containment violation
        cont_viol = containment_violation(inner_positions, outer_center, outer_side, outer_angle)
        # Compute pairwise penetration
        pen_depth = total_penetration(inner_positions)

        if cont_viol < 1e-6 and pen_depth < 1e-6:
            return outer_side

        penalty = cont_viol * 20.0 + pen_depth * 20.0
        return outer_side + penalty

    def min_outer_side_for_config(inner_positions, outer_center, outer_angle):
        """Find minimum outer side length that contains all inner hexagons."""
        lo = 1.0
        hi = 15.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if all_inside_outer(inner_positions, outer_center, mid, outer_angle):
                hi = mid
            else:
                lo = mid
        return hi

    def objective_caging(params):
        """Caging formulation: outer side is computed from inner positions.
        Minimize the required outer side length with penalties for overlap."""
        n_inner = N
        inner_positions = []
        for i in range(n_inner):
            x = params[3 * i]
            y = params[3 * i + 1]
            a = params[3 * i + 2]
            inner_positions.append((x, y, a))

        ox = params[3 * n_inner]
        oy = params[3 * n_inner + 1]
        oa = params[3 * n_inner + 2]

        outer_center = (ox, oy)
        outer_angle = oa

        # Compute pairwise penetration
        pen_depth = total_penetration(inner_positions)

        if pen_depth < 1e-6:
            # Valid packing, compute required outer side
            os = required_outer_side(inner_positions, outer_center, outer_angle)
            return os
        else:
            # Invalid packing, add penalty
            os = required_outer_side(inner_positions, outer_center, outer_angle)
            return os + pen_depth * 100.0

    # Generate diverse initial configurations with various rotations
    def get_initial_configurations():
        configs = []
        sqrt3 = np.sqrt(3)

        # For each base arrangement, try several outer angles
        outer_angles = [0.0, 30.0, 15.0, 45.0, 60.0]

        # Base arrangement A: 4x3 grid with hexagonal offset (aligned hexagons)
        for outer_angle in outer_angles:
            positions_a = []
            inner_angle = outer_angle  # align inner with outer
            for row in range(3):
                for col in range(4):
                    x = (col - 1.5) * 3.0
                    y = (row - 1) * 2.598
                    if row % 2 == 1:
                        x += 1.5
                    positions_a.append([x, y, inner_angle])
            configs.append((positions_a, outer_angle))

        # Base arrangement B: Hexagonal ring packing
        for outer_angle in outer_angles:
            positions_b = [[0.0, 0.0, outer_angle]]
            for k in range(6):
                angle = k * np.pi / 3
                x = 2.0 * np.cos(angle)
                y = 2.0 * np.sin(angle)
                positions_b.append([x, y, outer_angle])
            for k in range(5):
                angle = k * np.pi / 3 + np.pi / 6
                x = 3.0 * np.cos(angle)
                y = 3.0 * np.sin(angle)
                positions_b.append([x, y, outer_angle])
            configs.append((positions_b[:12], outer_angle))

        # Base arrangement C: Two rows of 6
        for outer_angle in [0.0, 30.0]:
            positions_c = []
            for i in range(6):
                positions_c.append([(i - 2.5) * 2.0, -1.5, outer_angle])
                positions_c.append([(i - 2.5) * 2.0 + 1.0, 1.5, outer_angle])
            configs.append((positions_c, outer_angle))

        # Base arrangement D: 3x4 grid with tighter spacing
        for outer_angle in [0.0, 30.0]:
            positions_d = []
            for row in range(3):
                for col in range(4):
                    x = (col - 1.5) * 2.1
                    y = (row - 1) * 2.6
                    if row % 2 == 1:
                        x += 1.05
                    positions_d.append([x, y, outer_angle])
            configs.append((positions_d, outer_angle))

        return configs

    def simulated_annealing_caging(initial_positions, initial_outer_angle, max_iters=5000, 
                                    t_start=2.0, t_end=0.01, cooling=0.9995):
        """Simulated annealing with caging formulation.
        Outer side is computed, not optimized directly."""
        np.random.seed(42)
        
        positions = [list(p) for p in initial_positions]
        outer_center = [0.0, 0.0]
        outer_angle = initial_outer_angle
        
        # Compute initial energy
        inner_tuples = [(p[0], p[1], p[2]) for p in positions]
        pen = total_penetration(inner_tuples)
        os_init = required_outer_side(inner_tuples, outer_center, outer_angle)
        current_energy = os_init + pen * 100.0
        
        best_positions = [list(p) for p in positions]
        best_outer_center = list(outer_center)
        best_outer_angle = outer_angle
        best_energy = current_energy
        
        temperature = t_start
        accepted = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 20:
                break
            
            # Generate neighbor
            new_positions = [list(p) for p in positions]
            new_outer_center = list(outer_center)
            new_outer_angle = outer_angle
            
            # Random move: perturb one hexagon
            idx = np.random.randint(0, N)
            dx = np.random.normal(0, 0.5)
            dy = np.random.normal(0, 0.5)
            da = np.random.normal(0, 10)
            new_positions[idx][0] += dx
            new_positions[idx][1] += dy
            new_positions[idx][2] += da
            
            # Occasionally perturb outer center/angle
            if np.random.random() < 0.1:
                new_outer_center[0] += np.random.normal(0, 0.3)
                new_outer_center[1] += np.random.normal(0, 0.3)
            if np.random.random() < 0.05:
                new_outer_angle += np.random.normal(0, 5)
            
            # Compute new energy
            new_inner_tuples = [(p[0], p[1], p[2]) for p in new_positions]
            new_pen = total_penetration(new_inner_tuples)
            new_os = required_outer_side(new_inner_tuples, new_outer_center, new_outer_angle)
            new_energy = new_os + new_pen * 100.0
            
            # Acceptance criterion
            delta_e = new_energy - current_energy
            if delta_e < 0 or np.random.random() < np.exp(-delta_e / temperature):
                positions = new_positions
                outer_center = new_outer_center
                outer_angle = new_outer_angle
                current_energy = new_energy
                accepted += 1
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_positions = [list(p) for p in positions]
                    best_outer_center = list(outer_center)
                    best_outer_angle = outer_angle
            
            temperature *= cooling
        
        return best_positions, best_outer_center, best_outer_angle, best_energy

    best_params = None
    best_score = float('inf')

    configs = get_initial_configurations()

    # Sort configs by initial outer side length (tighter first)
    def config_initial_score(config_tuple):
        positions, outer_angle = config_tuple
        inner_tuples = [(p[0], p[1], p[2]) for p in positions]
        if not pairwise_disjoint(inner_tuples):
            return 100.0
        os = min_outer_side_for_config(inner_tuples, [0.0, 0.0], outer_angle)
        return os

    # Evaluate and sort
    scored_configs = []
    for idx, ct in enumerate(configs):
        if time.time() > deadline - 30:
            break
        s = config_initial_score(ct)
        scored_configs.append((s, idx, ct))
    scored_configs.sort(key=lambda x: x[0])

    # Phase 1: Simulated annealing with caging formulation on top configs
    sa_configs = scored_configs[:8]  # Try top 8 configs
    
    for score, idx, config_tuple in sa_configs:
        if time.time() > deadline - 40:
            break
        
        positions, outer_angle = config_tuple
        inner_tuples = [(p[0], p[1], p[2]) for p in positions]
        
        # Run simulated annealing
        sa_positions, sa_center, sa_angle, sa_energy = simulated_annealing_caging(
            positions, outer_angle, max_iters=3000, t_start=1.5, t_end=0.01, cooling=0.9995
        )
        
        # Check if SA found a valid packing
        sa_inner = [(p[0], p[1], p[2]) for p in sa_positions]
        if pairwise_disjoint(sa_inner):
            sa_os = required_outer_side(sa_inner, sa_center, sa_angle)
            if sa_os < best_score:
                best_score = sa_os
                best_params = make_params(sa_positions, sa_center, sa_os, sa_angle)
        
        # Phase 2: Polish with Nelder-Mead using caging objective
        if time.time() > deadline - 25:
            break
        
        # Use caging objective for polishing
        caging_params = make_params_no_outer(sa_positions, sa_center, sa_angle)
        
        try:
            res = minimize(objective_caging, caging_params, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 0.002, 'fatol': 0.002, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                # Convert back to full params
                res_inner, res_oc, res_oa = parse_params_no_outer(res.x)
                res_os = required_outer_side(res_inner, res_oc, res_oa)
                best_params = make_params(res_inner, res_oc, res_os, res_oa)
        except Exception:
            pass
    
    # Fallback: try original Nelder-Mead on best config if SA didn't find anything
    if best_params is None and time.time() < deadline - 20:
        for score, idx, config_tuple in scored_configs[:3]:
            if time.time() > deadline - 15:
                break
            
            positions, outer_angle = config_tuple
            inner_tuples = [(p[0], p[1], p[2]) for p in positions]
            
            if not pairwise_disjoint(inner_tuples):
                continue
            
            outer_side = min_outer_side_for_config(inner_tuples, [0.0, 0.0], outer_angle)
            outer_center = [0.0, 0.0]
            
            params = make_params(positions, outer_center, outer_side, outer_angle)
            
            try:
                res = minimize(objective, params, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 0.003, 'fatol': 0.003, 'adaptive': True})
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass

    # Fallback
    if best_params is None:
        positions_a = []
        for row in range(3):
            for col in range(4):
                x = (col - 1.5) * 3.0
                y = (row - 1) * 2.598
                if row % 2 == 1:
                    x += 1.5
                positions_a.append([x, y, 0.0])
        inner_tuples = [(p[0], p[1], p[2]) for p in positions_a]
        outer_center = [0.0, 0.0]
        outer_angle = 30.0
        outer_side = 8.0
        while not all_inside_outer(inner_tuples, outer_center, outer_side, outer_angle):
            outer_side += 0.5
        best_params = make_params(positions_a, outer_center, outer_side, outer_angle)
        best_score = outer_side

    # Final validation and return
    inner, oc, os, oa = parse_params(best_params)

    if not is_valid(inner, oc, os, oa):
        # Try to fix by increasing outer side
        while not all_inside_outer(inner, oc, os, oa):
            os += 0.05
            if os > 20:
                break
        # Check disjointness
        if not pairwise_disjoint(inner):
            # Last resort: use fallback config
            positions_a = []
            for row in range(3):
                for col in range(4):
                    x = (col - 1.5) * 3.0
                    y = (row - 1) * 2.598
                    if row % 2 == 1:
                        x += 1.5
                    positions_a.append([x, y, 0.0])
            inner = [(p[0], p[1], p[2]) for p in positions_a]
            oc = (0.0, 0.0)
            oa = 30.0
            os = 8.0
            while not all_inside_outer(inner, oc, os, oa):
                os += 0.5

    return inner, oc, os, oa
# EVOLVE-BLOCK-END
