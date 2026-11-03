import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    # Precompute unit hexagon vertices (flat-top, vertex at 0 deg is not standard, 
    # let's use pointy-top: vertices at 30, 90, 150, 210, 270, 330? 
    # Actually, standard regular hexagon with side 1:
    # Vertices at angles 0, 60, 120, 180, 240, 300 relative to center.
    # If we align one vertex with x-axis, it's "pointy" in x direction.
    # Let's stick to the seed's definition for consistency in distance checks, 
    # but the symmetry strategy is independent of initial orientation as long as consistent.
    
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
        Returns side length.
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
        
        # For a regular hexagon of side s, the distance from center to side is s * sqrt(3)/2.
        # The projection of a vertex onto the normal of a side is the distance to that side.
        # So max_proj is the required inradius.
        # s = inradius / (sqrt(3)/2) = max_proj * 2 / sqrt(3)
        return max_proj * 2.0 / math.sqrt(3)

    def check_disjoint_fast(poly1, poly2):
        """
        Fast check for disjointness using SAT.
        Returns True if disjoint (or touching), False if overlapping.
        """
        def project_polygon(poly, axis):
            proj = poly @ axis
            return np.min(proj), np.max(proj)
            
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
            min1, max1 = project_polygon(poly1, axis)
            min2, max2 = project_polygon(poly2, axis)
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True
                
        return False

    def min_gap(params):
        """
        Returns the minimum gap between any pair of hexagons.
        Positive = disjoint, negative = overlapping.
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
                    if nrm < 1e-12: continue
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

    def expand_symmetric_params(params):
        """
        Expands a set of symmetric parameters into the full 12-hexagon configuration.
        Strategy: Use 6-fold symmetry.
        We define orbits.
        
        Topology 1: Ring of 6 + Center 1 + Ring of 5? No, 12 is divisible by 6.
        Let's try:
        - Orbit A: 6 hexagons in a ring at radius r1, angle offset a1, rotation rot1.
        - Orbit B: 6 hexagons in a ring at radius r2, angle offset a2, rotation rot2.
        
        Total 12.
        Parameters: [r1, a1, rot1, r2, a2, rot2]
        a1, a2 are angular offsets for the positions.
        rot1, rot2 are rotations of the hexagons themselves.
        
        The 6 hexagons in an orbit are at angles: a_offset + k * 60 deg.
        """
        r1, a1, rot1, r2, a2, rot2 = params
        positions = []
        angles = []
        
        for k in range(6):
            ang_pos = math.radians(a1 + k * 60)
            x = r1 * math.cos(ang_pos)
            y = r1 * math.sin(ang_pos)
            positions.append([x, y])
            angles.append(rot1)
            
            ang_pos2 = math.radians(a2 + k * 60)
            x2 = r2 * math.cos(ang_pos2)
            y2 = r2 * math.sin(ang_pos2)
            positions.append([x2, y2])
            angles.append(rot2)
            
        return np.array(positions).flatten(), np.array(angles)

    def objective_symmetric(params):
        if time.time() > deadline - 10:
            return 1e6
        positions_flat, angles = expand_symmetric_params(params)
        positions = positions_flat.reshape(N, 2)
        
        # Check validity (disjointness)
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if not check_disjoint_fast(polys[i], polys[j]):
                    penalty += 100.0
                    
        side = calc_min_outer_side(positions, angles, 0.0)
        return side + penalty

    # Initial guesses for symmetric parameters
    # r1, a1, rot1, r2, a2, rot2
    # Typical radii for unit hexagons: distance between centers ~ 2*sqrt(3) ~ 3.46 for touching flat sides?
    # Side length 1. Width across flats = sqrt(3). 
    # If placed in a ring, center-to-center distance needs to be > sqrt(3) if aligned properly, or larger.
    # Let's start with r1=2.0, r2=4.0
    
    initial_guesses = [
        [2.0, 0.0, 0.0, 4.0, 0.0, 0.0],
        [1.8, 30.0, 0.0, 3.8, 30.0, 0.0],
        [2.2, 0.0, 30.0, 4.2, 0.0, 30.0],
        [1.5, 0.0, 0.0, 3.5, 0.0, 0.0],
        [2.5, 0.0, 0.0, 4.5, 0.0, 0.0]
    ]
    
    best_score = float('inf')
    best_params_sym = None
    
    # Optimize each initial guess
    for guess in initial_guesses:
        if time.time() > deadline - 20:
            break
            
        try:
            res = minimize(
                objective_symmetric, 
                guess, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if res.fun < best_score and res.fun < 10.0: # 10.0 is a sanity check for penalty
                # Verify validity
                positions_flat, angles = expand_symmetric_params(res.x)
                positions = positions_flat.reshape(N, 2)
                polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if not check_disjoint_fast(polys[i], polys[j]):
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    best_score = calc_min_outer_side(positions, angles, 0.0)
                    best_params_sym = res.x
        except Exception:
            continue

    # If symmetric optimization failed or didn't find anything good, fall back to free optimization
    if best_params_sym is None or best_score > 9.5:
        # Use the original free optimization strategy as fallback
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
        best_params_free = x0.copy()
        
        def objective_slsqp(params):
            if time.time() > deadline - 10:
                return 1e6
            positions = params[:2*N].reshape(N, 2)
            angles = params[2*N:]
            return calc_min_outer_side(positions, angles, 0.0)

        def gap_constraint(params):
            if time.time() > deadline - 10:
                return 0.0
            return min_gap(params)

        constraints = [
            {'type': 'ineq', 'fun': gap_constraint}
        ]

        try:
            res = minimize(
                objective_slsqp, 
                x0, 
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-8, 'eps': 1e-5, 'disp': False}
            )
            
            if res.fun < best_score and res.success:
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
                    best_params_free = res.x
        except Exception:
            pass
            
        # Use free params
        final_positions = best_params_free[:2*N].reshape(N, 2)
        final_angles = best_params_free[2*N:]
        final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
        
        inner = []
        for i in range(N):
            inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
        
        return inner, (0.0, 0.0), final_side, 0.0
    else:
        # Use symmetric params
        positions_flat, angles = expand_symmetric_params(best_params_sym)
        final_positions = positions_flat.reshape(N, 2)
        final_angles = angles
        final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
        
        inner = []
        for i in range(N):
            inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
        
        return inner, (0.0, 0.0), final_side, 0.0
