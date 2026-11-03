import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0 (pointy-top, vertex at 0 deg)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0) 
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals) # (6, 2)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def check_disjoint_fast(poly1, poly2):
        """
        Fast check for disjointness using SAT.
        Returns True if disjoint (or touching), False if overlapping.
        """
        # Check all edges of both polygons
        # For convex polygons, if there is a separating axis, they are disjoint.
        # Axes are normals to edges.
        
        def project_polygon(poly, axis):
            proj = poly @ axis
            return np.min(proj), np.max(proj)
            
        axes = []
        for i in range(6):
            # Edge 1
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            axes.append(normal1)
            # Edge 2
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            axes.append(normal2)
            
        for axis in axes:
            norm = np.linalg.norm(axis)
            if norm < 1e-9: continue
            axis = axis / norm
            min1, max1 = project_polygon(poly1, axis)
            min2, max2 = project_polygon(poly2, axis)
            if max1 < min2 or max2 < min1:
                return True # Separating axis found, disjoint
                
        return False # No separating axis, overlapping

    def overlap_penalty(params):
        """
        Calculate penalty based on violation of non-overlap constraints.
        Returns 0 if valid, positive value if invalid.
        """
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        
        total_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if not check_disjoint_fast(polys[i], polys[j]):
                    # Estimate overlap depth for smooth gradient?
                    # For SLSQP, we can just use a hard constraint or a large penalty.
                    # Since SLSQP handles constraints, we will define a function that returns
                    # the minimum distance (negative if overlapping).
                    # But calculating min distance is hard.
                    # Let's stick to a penalty method with a large multiplier for the objective,
                    # or use 'ineq' constraints if we can compute a smooth measure.
                    # For now, a simple penalty: if overlapping, add a large constant.
                    # To make it smooth for the optimizer, we can approximate the penetration depth.
                    # A simple heuristic: check if center of i is in j or vice versa, or use bounding box overlap area?
                    # Let's use a robust penalty: if overlapping, return a large value.
                    # To help the optimizer, we can try to push them apart.
                    # For simplicity in this step, we'll use a penalty in the objective function.
                    total_penalty += 100.0
        return total_penalty

    # Initial guess: Honeycomb lattice
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

    best_score = calc_min_outer_side(init_positions, init_angles, 0.0)
    best_params = x0.copy()

    # Optimization Strategy:
    # We want to minimize the outer side length.
    # Constraint: All hexagons must be disjoint.
    
    def objective(params):
        if time.time() > deadline - 5:
            return 1e6
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        # Check validity
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if not check_disjoint_fast(polys[i], polys[j]):
                    penalty += 10.0
        if penalty > 0:
            return penalty + calc_min_outer_side(positions, angles, 0.0)
        return calc_min_outer_side(positions, angles, 0.0)

    # Use Nelder-Mead for robustness against non-smooth penalty landscape initially
    # Then try SLSQP if we have a smooth constraint formulation?
    # Given the time limit, Nelder-Mead is safer for non-smooth penalties.
    
    try:
        res = minimize(
            objective, 
            x0, 
            method='Nelder-Mead', 
            options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        
        if res.fun < best_score:
            # Verify
            new_positions = res.x[:2*N].reshape(N, 2)
            new_angles = res.x[2*N:]
            polys = [get_hex_verts(new_positions[i][0], new_positions[i][1], 1.0, new_angles[i]) for i in range(N)]
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if not check_disjoint_fast(polys[i], polys[j]):
                        valid = False
                        break
                if not valid: break
            
            if valid:
                best_score = res.fun
                best_params = res.x
    except Exception:
        pass

    # Local refinement on the best found
    if best_params is not None:
        # Try a few random perturbations to escape local minima
        current_best = best_score
        current_params = best_params
        
        for _ in range(10):
            if time.time() > deadline - 10:
                break
                
            # Perturb
            noise_pos = np.random.normal(0, 0.1, 2*N)
            noise_ang = np.random.normal(0, 5, N) * math.pi / 180
            x_perturbed = current_params.copy()
            x_perturbed[:2*N] += noise_pos
            x_perturbed[2*N:] += noise_ang
            
            try:
                res = minimize(
                    objective, 
                    x_perturbed, 
                    method='Nelder-Mead', 
                    options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7}
                )
                if res.fun < current_best:
                    new_positions = res.x[:2*N].reshape(N, 2)
                    new_angles = res.x[2*N:]
                    polys = [get_hex_verts(new_positions[i][0], new_positions[i][1], 1.0, new_angles[i]) for i in range(N)]
                    valid = True
                    for i in range(N):
                        for j in range(i+1, N):
                            if not check_disjoint_fast(polys[i], polys[j]):
                                valid = False
                                break
                    if valid:
                        current_best = res.fun
                        current_params = res.x
            except Exception:
                continue

        best_score = current_best
        best_params = current_params

    final_positions = best_params[:2*N].reshape(N, 2)
    final_angles = best_params[2*N:]
    final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0
