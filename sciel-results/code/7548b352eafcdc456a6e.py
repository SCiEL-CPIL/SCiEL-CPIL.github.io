import math
import itertools
import time
import numpy as np
from scipy.optimize import minimize_scalar, minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 15.0
    
    d = math.sqrt(3)
    
    # Initial configuration: 7 central + 4 from ring 2
    # Use axial coordinates for ring 2
    ring2_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2_centers.append((x, y))
    
    # Pick 4 from ring 2 that are well-spread
    # Use every 3rd one for symmetry
    combo_indices = [0, 3, 6, 9] if len(ring2_centers) >= 10 else list(range(min(4, len(ring2_centers))))
    
    init_centers = [(0.0, 0.0)]
    for i in range(6):
        angle = i * math.pi / 3
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        init_centers.append((x, y))
    for idx in combo_indices:
        init_centers.append(ring2_centers[idx])
    
    # Vertices of unit hexagon relative to center (pointy-top)
    verts_rel = np.array([(math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)) for i in range(6)])
    
    def hex_vertices(cx, cy, theta):
        """Get 6 vertices of a unit hexagon at (cx,cy) rotated by theta."""
        c, s = math.cos(theta), math.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        return np.array([[cx, cy]]) + verts_rel @ rot.T
    
    def outer_hex_vertices(S, theta):
        """Get 6 vertices of outer hexagon with side S, center at origin, rotated by theta."""
        # For a regular hexagon with side length S, the circumradius is S
        # Vertices at angles theta + k*pi/3
        verts = []
        for k in range(6):
            angle = theta + k * math.pi / 3
            verts.append([S * math.cos(angle), S * math.sin(angle)])
        return np.array(verts)
    
    def compute_objective(params):
        """
        params: [x1,y1,th1, x2,y2,th2, ..., x11,y11,th11, S, theta_outer]
        Returns: S + penalty for overlaps and containment violations
        """
        if time.time() > deadline - safety_margin:
            return float('inf')
        
        n_hex = 11
        S = params[3 * n_hex]
        theta_outer = params[3 * n_hex + 1]
        
        # Check containment: all inner vertices must be inside outer hexagon
        outer_verts = outer_hex_vertices(S, theta_outer)
        
        # For a convex polygon, point is inside if it's on the same side of each edge
        # Outer hexagon edges
        max_violation = 0.0
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            # Ensure normal points inward (toward center)
            center_test = -p1
            if np.dot(normal, center_test) < 0:
                normal = -normal
            
            # For each inner hexagon, check its vertices
            for h in range(n_hex):
                cx = params[3 * h]
                cy = params[3 * h + 1]
                th = params[3 * h + 2]
                inner_v = hex_vertices(cx, cy, th)
                # Distance from each vertex to the edge line
                dists = np.dot(inner_v - p1, normal) / np.linalg.norm(normal)
                min_dist = np.min(dists)
                if min_dist < 0:
                    max_violation = max(max_violation, -min_dist)
        
        # Check overlaps between inner hexagons
        overlap_penalty = 0.0
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                cx_i = params[3 * i]
                cy_i = params[3 * i + 1]
                th_i = params[3 * i + 2]
                cx_j = params[3 * j]
                cy_j = params[3 * j + 1]
                th_j = params[3 * j + 2]
                
                # Distance between centers
                dist = math.sqrt((cx_i - cx_j)**2 + (cy_i - cy_j)**2)
                
                # For two unit hexagons, minimum distance for no overlap depends on relative rotation
                # Conservative: require center distance >= 2*d (hexagon diameter)
                # Actually, for same orientation, min distance is 2*d
                # For different rotations, it can be less
                # Use a conservative bound: if dist < 2*d - epsilon, there might be overlap
                # More precisely: the inradius of a unit hexagon is d/2
                # Two hexagons don't overlap if the distance between centers > 2 * (d/2) + margin for rotation
                # Actually for regular hexagons, the minimum center distance for non-overlap
                # with arbitrary rotation is 2 * (inradius) = d
                # Wait, inradius = d/2, so two inradii = d
                # But that's for the same orientation. For different rotations, it can be tighter.
                # Let's use: if dist < d, definitely overlapping
                # If dist >= 2*d, definitely not overlapping
                # In between, need more careful check
                
                if dist < d * 0.99:
                    # Definitely overlapping
                    overlap_penalty += (d * 0.99 - dist) ** 2
                elif dist < d * 1.1:
                    # Potentially overlapping, do a more careful check
                    # Check if any vertex of one is inside the other
                    verts_i = hex_vertices(cx_i, cy_i, th_i)
                    verts_j = hex_vertices(cx_j, cy_j, th_j)
                    
                    # Simple check: if any vertex of i is inside j or vice versa
                    # For a convex hexagon, point inside if same side of all edges
                    def point_in_hex(px, py, verts):
                        for k in range(6):
                            p1 = verts[k]
                            p2 = verts[(k + 1) % 6]
                            edge = p2 - p1
                            normal = np.array([-edge[1], edge[0]])
                            center_test = -p1
                            if np.dot(normal, center_test) < 0:
                                normal = -normal
                            if np.dot(np.array([px, py]) - p1, normal) < 0:
                                return False
                        return True
                    
                    for vx, vy in verts_i:
                        if point_in_hex(vx, vy, verts_j):
                            overlap_penalty += 0.1
                            break
                    else:
                        for vx, vy in verts_j:
                            if point_in_hex(vx, vy, verts_i):
                                overlap_penalty += 0.1
                                break
        
        return S + 10.0 * max_violation + 10.0 * overlap_penalty
    
    # Initial parameters
    init_params = []
    for cx, cy in init_centers:
        init_params.extend([cx, cy, 0.0])
    init_params.extend([3.5, 0.0])  # Initial S estimate, theta_outer
    
    # Optimize
    best_result = None
    best_S = float('inf')
    
    # Multiple restarts with different initial configurations
    restarts = 5
    for r in range(restarts):
        if time.time() > deadline - safety_margin:
            break
        
        # Perturb initial configuration
        np.random.seed(42 + r)
        if r == 0:
            params0 = init_params.copy()
        else:
            params0 = []
            for i in range(11):
                cx = init_centers[i][0] + np.random.uniform(-0.3, 0.3)
                cy = init_centers[i][1] + np.random.uniform(-0.3, 0.3)
                th = np.random.uniform(-math.pi/6, math.pi/6)
                params0.extend([cx, cy, th])
            params0.extend([3.2 + 0.3 * r, np.random.uniform(-math.pi/6, math.pi/6)])
        
        # Use L-BFGS-B for optimization
        try:
            result = minimize(compute_objective, params0, method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-6})
            
            if result.fun < best_S:
                # Extract S from the result
                S_val = result.x[3 * 11]
                # Check if the solution is actually valid (low penalty)
                # Re-evaluate to get clean S
                test_params = result.x.copy()
                test_S = test_params[3 * 11]
                # The objective includes penalties, so if fun is close to S, it's good
                if result.fun < test_S * 1.05:  # Penalty is small
                    best_S = test_S
                    best_result = result.x
        except Exception:
            continue
    
    # If optimization didn't find a good solution, fall back to the original approach
    if best_result is None:
        # Fall back to original combinatorial approach
        centers = [(0.0, 0.0)]
        for i in range(6):
            angle = i * math.pi / 3
            x = d * math.cos(angle)
            y = d * math.sin(angle)
            centers.append((x, y))
        
        ring2_centers = []
        for q in range(-2, 3):
            for r in range(-2, 3):
                s = -q - r
                if max(abs(q), abs(r), abs(s)) == 2:
                    x = d * (q + r / 2.0)
                    y = d * (math.sqrt(3) / 2.0) * r
                    ring2_centers.append((x, y))
        
        combo_indices = [0, 3, 6, 9] if len(ring2_centers) >= 10 else list(range(min(4, len(ring2_centers))))
        for idx in combo_indices:
            centers.append(ring2_centers[idx])
        
        verts_rel = []
        for i in range(6):
            angle = i * math.pi / 3
            vx = math.cos(angle)
            vy = math.sin(angle)
            verts_rel.append((vx, vy))
        
        points = []
        for cx, cy in centers:
            for vx, vy in verts_rel:
                points.append((cx + vx, cy + vy))
        points_np = np.array(points)
        
        def calc_S(theta):
            max_proj = 0.0
            for k in range(6):
                angle_n = theta + k * math.pi / 3
                nx = math.cos(angle_n)
                ny = math.sin(angle_n)
                dots = np.abs(points_np[:, 0] * nx + points_np[:, 1] * ny)
                current_max = np.max(dots)
                if current_max > max_proj:
                    max_proj = current_max
            return (2.0 / math.sqrt(3)) * max_proj
        
        best_theta_coarse = 0.0
        min_S_coarse = float('inf')
        for i in range(200):
            theta = (i / 200.0) * (math.pi / 3)
            S_val = calc_S(theta)
            if S_val < min_S_coarse:
                min_S_coarse = S_val
                best_theta_coarse = theta
        
        window = (math.pi / 3) / 200.0
        lower_bound = max(0.0, best_theta_coarse - window)
        upper_bound = min(math.pi / 3, best_theta_coarse + window)
        
        if lower_bound >= upper_bound:
            lower_bound = 0.0
            upper_bound = math.pi / 3
        
        res = minimize_scalar(calc_S, bounds=(lower_bound, upper_bound), method='bounded')
        
        if res.success:
            best_S = res.fun
            best_theta = res.x
        else:
            best_S = min_S_coarse
            best_theta = best_theta_coarse
        
        inner_rows = []
        for cx, cy in centers:
            inner_rows.append([cx, cy, 0.0])
        
        return inner_rows, (0.0, 0.0), best_S, math.degrees(best_theta)
    
    # Extract solution from optimization
    inner_rows = []
    for i in range(11):
        cx = best_result[3 * i]
        cy = best_result[3 * i + 1]
        th_deg = math.degrees(best_result[3 * i + 2])
        inner_rows.append([cx, cy, th_deg])
    
    S_val = best_result[3 * 11]
    theta_outer_deg = math.degrees(best_result[3 * 11 + 1])
    
    return inner_rows, (0.0, 0.0), S_val, theta_outer_deg
# EVOLVE-BLOCK-END
