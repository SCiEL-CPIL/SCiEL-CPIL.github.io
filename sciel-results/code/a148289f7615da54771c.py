import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)

    def project(vertices, nx, ny):
        projs = np.dot(vertices, [nx, ny])
        return np.min(projs), np.max(projs)

    def are_disjoint_sat(v1, v2, eps=1e-9):
        """
        Returns True if v1 and v2 are disjoint (including touching).
        """
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            edge = p2 - p1
            nx = -edge[1]
            ny = edge[0]
            min1, max1 = project(v1, nx, ny)
            min2, max2 = project(v2, nx, ny)
            if max1 < min2 + eps or max2 < min1 + eps:
                return True
        
        for i in range(6):
            p1 = v2[i]
            p2 = v2[(i + 1) % 6]
            edge = p2 - p1
            nx = -edge[1]
            ny = edge[0]
            min1, max1 = project(v1, nx, ny)
            min2, max2 = project(v2, nx, ny)
            if max1 < min2 + eps or max2 < min1 + eps:
                return True
        
        return False

    def compute_outer_side(centers, angles, ocx, ocy, otheta_deg):
        """
        Compute the minimum side length of a regular hexagon centered at (ocx, ocy)
        with rotation otheta_deg that contains all inner hexagons.
        """
        max_proj = 0.0
        otheta_rad = math.radians(otheta_deg)
        for k in range(6):
            # Normals to the sides of the outer hexagon
            # For a regular hexagon, the distance from center to side (apothem) is side * sqrt(3)/2
            # The normal angles are offset by 30 deg from the vertex angles
            normal_angle = otheta_rad + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            
            max_proj_normal = 0.0
            for i in range(len(centers)):
                cx, cy = centers[i]
                angle_rad = math.radians(angles[i])
                # Vertices of inner hexagon i
                for vi in range(6):
                    vert_angle = angle_rad + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = (vx - ocx) * nx + (vy - ocy) * ny
                    if proj > max_proj_normal:
                        max_proj_normal = proj
            
            if max_proj_normal > max_proj:
                max_proj = max_proj_normal
        
        # max_proj is the max distance from center to any vertex in any normal direction.
        # For a regular hexagon, the distance from center to a side (apothem) is side * sqrt(3)/2.
        # The max projection onto a normal direction corresponds to the apothem if the outer hexagon is aligned.
        # Actually, if we define the outer hexagon such that its sides are perpendicular to these normals,
        # the distance from center to the side is max_proj.
        # Apothem = side * sqrt(3) / 2
        # side = 2 * apothem / sqrt(3)
        return 2.0 * max_proj / math.sqrt(3)

    def check_validity(centers, angles):
        """
        Check if the configuration of inner hexagons is valid (disjoint).
        """
        verts = []
        for i in range(N):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        
        for i in range(N):
            for j in range(i + 1, N):
                if not are_disjoint_sat(verts[i], verts[j]):
                    return False
        return True

    def objective_slsqp(params):
        """
        Objective for SLSQP stage.
        Minimizes outer side length.
        """
        if time.time() > deadline - 20:
            return 1e9
            
        # Unpack
        centers = []
        angles = []
        for i in range(N):
            centers.append([params[3*i], params[3*i+1]])
            angles.append(params[3*i+2])
        
        ocx = params[3*N]
        ocy = params[3*N+1]
        otheta = params[3*N+2]
        
        side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        
        # Soft penalty for center distance to guide towards feasible region
        # Min center dist for hexagons is sqrt(3)
        min_dist = math.sqrt(3)
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        return side + 5.0 * penalty

    def objective_refine(params):
        """
        Objective for refinement stage (Nelder-Mead).
        Uses exact SAT check for penalty.
        """
        if time.time() > deadline - 10:
            return 1e9
            
        centers = []
        angles = []
        for i in range(N):
            centers.append([params[3*i], params[3*i+1]])
            angles.append(params[3*i+2])
        
        ocx = params[3*N]
        ocy = params[3*N+1]
        otheta = params[3*N+2]
        
        side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        
        # Exact validity check
        if not check_validity(centers, angles):
            # Calculate a penalty based on how much they overlap
            # Simple heuristic: sum of squared violations of center distance
            min_dist = math.sqrt(3)
            penalty = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = centers[i][0] - centers[j][0]
                    dy = centers[i][1] - centers[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < min_dist:
                        penalty += (min_dist - dist) ** 2
            return side + 100.0 * penalty
            
        return side

    def generate_initial_config():
        """
        Generate a valid initial configuration based on hexagonal tiling.
        """
        d = math.sqrt(3) + 0.01
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
        
        centers = [np.array([0.0, 0.0])]
        ring1_vectors = [
            u, v, v - u, -u, -v, u - v
        ]
        for vec in ring1_vectors:
            centers.append(vec.copy())
            
        ring2_candidates = []
        basis = [u, v, v - u, -u, -v, u - v]
        for i in range(6):
            p1 = ring1_vectors[i]
            for b in basis:
                p2 = p1 + b
                is_r0 = np.allclose(p2, np.array([0.0, 0.0]), atol=0.01)
                is_r1 = False
                for p1j in ring1_vectors:
                    if np.allclose(p2, p1j, atol=0.01):
                        is_r1 = True
                        break
                if not is_r0 and not is_r1:
                    duplicate = False
                    for pc in ring2_candidates:
                        if np.allclose(p2, pc, atol=0.01):
                            duplicate = True
                            break
                    if not duplicate:
                        ring2_candidates.append(p2)
                    
        ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
        for i in range(5):
            centers.append(ring2_candidates[i].copy())
            
        params = []
        for c in centers:
            params.append(c[0])
            params.append(c[1])
            params.append(30.0) # Angle in degrees
        params.extend([0.0, 0.0, 0.0]) # Outer center and angle
        return params

    def polish_outer(centers, angles, initial_ocx, initial_ocy, initial_otheta):
        """
        High-precision polish of outer hexagon parameters.
        """
        best_side = compute_outer_side(centers, angles, initial_ocx, initial_ocy, initial_otheta)
        best_ocx, best_ocy, best_otheta = initial_ocx, initial_ocy, initial_otheta
        
        # Multi-scale search
        scales = [0.5, 0.1, 0.01, 0.001]
        for scale in scales:
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for dx in [-scale, 0, scale]:
                    for dy in [-scale, 0, scale]:
                        for da_deg in [-scale*10, 0, scale*10]:
                            otheta = initial_otheta + da_deg
                            side = compute_outer_side(centers, angles, best_ocx + dx, best_ocy + dy, otheta)
                            if side < best_side - 1e-12:
                                best_side = side
                                best_ocx += dx
                                best_ocy += dy
                                best_otheta = otheta
                                improved = True
        
        return best_ocx, best_ocy, best_otheta, best_side

    # --- Main Execution ---
    
    # Stage 1: SLSQP to get close to a good solution
    x0 = generate_initial_config()
    
    # Constraints for SLSQP: Center distance >= sqrt(3) - epsilon
    # This is a necessary condition for disjointness.
    constraints = []
    min_dist_sq = (math.sqrt(3) - 0.01) ** 2
    for i in range(N):
        for j in range(i + 1, N):
            def dist_constraint(x, i=i, j=j):
                xi = x[3*i]
                yi = x[3*i+1]
                xj = x[3*j]
                yj = x[3*j+1]
                return (xi - xj)**2 + (yi - yj)**2 - min_dist_sq
            constraints.append({'type': 'ineq', 'fun': dist_constraint})
    
    bounds = []
    for i in range(N):
        bounds.extend([(-5, 5), (-5, 5), (0, 60)])
    bounds.extend([(-2, 2), (-2, 2), (0, 60)])
    
    try:
        res_slsqp = minimize(objective_slsqp, x0, method='SLSQP', bounds=bounds, 
                             constraints=constraints, options={'maxiter': 100, 'ftol': 1e-8})
        x1 = res_slsqp.x
    except Exception:
        x1 = x0

    # Stage 2: Nelder-Mead Refinement with Exact Validity
    # We run this for a few iterations or until time runs out
    # Nelder-Mead is robust to non-smooth penalties if the penalty is large enough
    try:
        res_nm = minimize(objective_refine, x1, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-8})
        x2 = res_nm.x
    except Exception:
        x2 = x1

    # Extract best parameters
    centers = []
    angles = []
    for i in range(N):
        centers.append([x2[3*i], x2[3*i+1]])
        angles.append(x2[3*i+2])
    
    ocx = x2[3*N]
    ocy = x2[3*N+1]
    otheta = x2[3*N+2]
    
    # Check validity of the result
    if not check_validity(centers, angles):
        # If invalid, fall back to the initial valid configuration
        # and just polish the outer hexagon
        centers = []
        angles = []
        for i in range(N):
            centers.append([x0[3*i], x0[3*i+1]])
            angles.append(x0[3*i+2])
        ocx, ocy, otheta = 0.0, 0.0, 0.0

    # Stage 3: Polish Outer Hexagon
    best_ocx, best_ocy, best_otheta, best_side = polish_outer(centers, angles, ocx, ocy, otheta)
    
    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    outer_center = (best_ocx, best_ocy)
    outer_side = best_side
    outer_angle = best_otheta
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
