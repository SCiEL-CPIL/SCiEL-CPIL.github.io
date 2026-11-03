import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute unit hexagon vertices for angle 0 (flat-top, vertices at 0, 60, ... degrees)
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
        normals = np.array(normals)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def check_disjoint_fast(poly1, poly2):
        """
        Fast check for disjointness using SAT.
        Returns True if disjoint (or touching), False if overlapping.
        """
        axes = []
        for i in range(6):
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            axes.append(normal1)
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            axes.append(normal2)
            
        for axis in axes:
            norm = np.linalg.norm(axis)
            if norm < 1e-9: continue
            axis = axis / norm
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            if max1 < min2 or max2 < min1:
                return True
                
        return False

    def is_valid(positions, angles):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if not check_disjoint_fast(polys[i], polys[j]):
                    return False
        return True

    # Strategy: Directly optimize the 36 degrees of freedom (12 positions, 12 angles)
    # using a penalty-based objective function and a robust local optimizer (Nelder-Mead).
    # This avoids the restrictive symmetry assumptions of the previous approach.
    
    # Initial guess: Honeycomb subset
    # The honeycomb lattice is a natural packing. We take the 12 closest points to the origin.
    sqrt3 = math.sqrt(3)
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * 1.5
            y = i * sqrt3 / 2.0 + j * sqrt3
            lattice_points.append((x, y))
    
    # Sort by distance to origin
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_positions = np.array(lattice_points[:12])
    # Center the configuration
    init_positions -= np.mean(init_positions, axis=0)
    init_angles = np.zeros(N)
    
    x0 = np.concatenate([init_positions.flatten(), init_angles])
    
    def obj_general(params):
        if time.time() > deadline - 5:
            return 1000.0
            
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        # Compute the outer side length
        side_len = calc_min_outer_side(positions, angles, 0.0)
        
        # Check for overlaps and add penalty
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if not check_disjoint_fast(polys[i], polys[j]):
                    penalty += 10.0
        
        return side_len + penalty

    best_score = float('inf')
    best_result = None

    # Run Nelder-Mead optimization
    try:
        res = minimize(
            obj_general, 
            x0, 
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        
        if res.fun < best_score:
            best_score = res.fun
            best_result = res.x
    except Exception:
        pass

    # If the result is still invalid or very bad, try a few more restarts with random perturbations
    if best_result is None or best_score > 5.0:
        for restart in range(5):
            if time.time() > deadline - 10:
                break
                
            # Perturb the initial guess
            x0_perturbed = x0.copy()
            x0_perturbed += np.random.normal(0, 0.5, size=x0.shape)
            
            try:
                res = minimize(
                    obj_general, 
                    x0_perturbed, 
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-6, 'adaptive': True}
                )
                
                if res.fun < best_score:
                    best_score = res.fun
                    best_result = res.x
            except Exception:
                pass

    # Finalize result
    if best_result is not None:
        final_positions = best_result[:2*N].reshape(N, 2)
        final_angles = best_result[2*N:]
        
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
            
        final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    else:
        # Emergency fallback
        final_positions = np.array([
            [0, 0], [2.1, 0], [-2.1, 0],
            [1.05, 1.81], [-1.05, 1.81], [0, 3.62],
            [1.05, -1.81], [-1.05, -1.81], [0, -3.62],
            [3.15, 1.81], [-3.15, 1.81], [3.15, -1.81]
        ])
        final_angles = np.zeros(N)
        final_side = calc_min_outer_side(final_positions, final_angles, 0.0)

    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0
