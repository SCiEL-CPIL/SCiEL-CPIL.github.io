import math
import time
import numpy as np

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit

    # Unit hexagon vertices (pointy-top, vertex at 0 deg)
    unit_verts = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts @ rot.T * side
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
        normals = np.array(normals)  # (6, 2)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T  # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        # For a regular hexagon with side length s, the apothem is s*sqrt(3)/2
        # The max projection onto a normal direction gives the apothem
        # So s = apothem / (sqrt(3)/2) = 2*apothem/sqrt(3)
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
            if norm < 1e-9:
                continue
            axis = axis / norm
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            if max1 < min2 or max2 < min1:
                return True
                
        return False

    def min_gap(params):
        """
        Returns the minimum gap between any pair of hexagons.
        Positive = disjoint, negative = overlapping.
        Used for SLSQP constraints.
        """
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        
        min_gap_val = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                p1 = polys[i]
                p2 = polys[j]
                
                axes = []
                for k in range(6):
                    e1 = p1[(k+1)%6] - p1[k]
                    axes.append(np.array([-e1[1], e1[0]]))
                    e2 = p2[(k+1)%6] - p2[k]
                    axes.append(np.array([-e2[1], e2[0]]))
                
                min_sep = float('inf')
                for axis in axes:
                    nrm = np.linalg.norm(axis)
                    if nrm < 1e-12:
                        continue
                    axis = axis / nrm
                    proj1 = p1 @ axis
                    proj2 = p2 @ axis
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    
                    if max1 < min2:
                        gap = min2 - max1
                    elif max2 < min1:
                        gap = min1 - max2
                    else:
                        gap = 0.0
                    if gap < min_sep:
                        min_sep = gap
                
                if min_sep < min_gap_val:
                    min_gap_val = min_sep
        return min_gap_val

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

    # Objective for SLSQP: just the outer side length
    def objective_slsqp(params):
        if time.time() > deadline - 10:
            return 1e6
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        return calc_min_outer_side(positions, angles, 0.0)

    # Constraint: minimum gap >= 0
    def gap_constraint(params):
        if time.time() > deadline - 10:
            return 0.0
        return min_gap(params)

    constraints = [
        {'type': 'ineq', 'fun': gap_constraint}
    ]

    # Phase 1: SLSQP with constraints
    from scipy.optimize import minimize
    
    try:
        res = minimize(
            objective_slsqp, 
            x0, 
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 2000, 'ftol': 1e-8, 'eps': 1e-5, 'disp': False}
        )
        
        if res.fun < best_score and res.success:
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
                if not valid:
                    break
            
            if valid:
                best_score = res.fun
                best_params = res.x.copy()
    except Exception:
        pass

    # Phase 2: Nelder-Mead refinement (no constraints, just minimize side length)
    # This can sometimes push to a better local minimum if SLSQP got stuck
    def objective_nm(params):
        if time.time() > deadline - 10:
            return 1e6
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        # Penalty for overlap to guide away from invalid regions
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if not check_disjoint_fast(polys[i], polys[j]):
                    penalty += 50.0
        
        return calc_min_outer_side(positions, angles, 0.0) + penalty

    if time.time() < deadline - 30:
        try:
            res_nm = minimize(
                objective_nm, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if res_nm.fun < best_score + 1e-6:  # Allow small penalty in comparison
                new_positions = res_nm.x[:2*N].reshape(N, 2)
                new_angles = res_nm.x[2*N:]
                polys = [get_hex_verts(new_positions[i][0], new_positions[i][1], 1.0, new_angles[i]) for i in range(N)]
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if not check_disjoint_fast(polys[i], polys[j]):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best_score = calc_min_outer_side(new_positions, new_angles, 0.0)
                    best_params = res_nm.x.copy()
        except Exception:
            pass

    # Phase 3: Random restarts / perturbations
    if time.time() < deadline - 20:
        current_best = best_score
        current_params = best_params
        
        np.random.seed(42)
        for attempt in range(5):
            if time.time() > deadline - 15:
                break
            
            # Perturb current best
            noise_pos = np.random.normal(0, 0.2, 2*N)
            noise_ang = np.random.normal(0, 10, N) * math.pi / 180
            x_perturbed = current_params.copy()
            x_perturbed[:2*N] += noise_pos
            x_perturbed[2*N:] += noise_ang
            
            try:
                res = minimize(
                    objective_slsqp, 
                    x_perturbed, 
                    method='SLSQP',
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-8, 'eps': 1e-5, 'disp': False}
                )
                
                if res.fun < current_best and res.success:
                    new_positions = res.x[:2*N].reshape(N, 2)
                    new_angles = res.x[2*N:]
                    polys = [get_hex_verts(new_positions[i][0], new_positions[i][1], 1.0, new_angles[i]) for i in range(N)]
                    valid = True
                    for i in range(N):
                        for j in range(i+1, N):
                            if not check_disjoint_fast(polys[i], polys[j]):
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        current_best = res.fun
                        current_params = res.x.copy()
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
