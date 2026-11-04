import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s

    # Precompute unit hexagon vertices (pointy-top: vertices at 30, 90, 150, 210, 270, 330 degrees)
    # Actually, let's use flat-top: vertices at 0, 60, 120, 180, 240, 300 degrees
    unit_verts = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        """Get vertices of a regular hexagon with given center, side length, and rotation."""
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle_deg=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at origin
        with rotation outer_angle_deg that contains all inner hexagons.
        
        For a regular hexagon with side s, the inradius (center to flat side) is s*sqrt(3)/2.
        The 6 outward normals are at angles: outer_angle + 30, outer_angle + 90, ..., outer_angle + 330 degrees.
        If max projection onto any normal is p, then s = 2p/sqrt(3).
        """
        theta_rad = math.radians(outer_angle_deg)
        # Normal directions: 30, 90, 150, 210, 270, 330 degrees offset by outer_angle
        normal_angles = theta_rad + math.pi/6 + np.arange(6) * math.pi/3
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj_all = 0.0
        for i in range(N):
            cx, cy = positions[i]
            ang_deg = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang_deg)
            projs = verts @ normals.T  # (6, 6)
            max_proj_i = np.max(projs, axis=0)  # (6,)
            m = np.max(max_proj_i)
            if m > max_proj_all:
                max_proj_all = m
        
        return 2.0 * max_proj_all / math.sqrt(3)

    def sat_gap(poly1, poly2):
        """
        Returns the minimum separation distance between two convex hexagons using SAT.
        If they overlap, returns a negative value (penetration depth).
        If they are separated, returns the positive distance.
        """
        axes = []
        for poly in [poly1, poly2]:
            for i in range(6):
                edge = poly[(i+1)%6] - poly[i]
                n = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(n)
                if norm > 1e-9:
                    axes.append(n / norm)
        
        min_gap = float('inf')
        for axis in axes:
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            gap = max(np.min(proj1) - np.max(proj2), np.min(proj2) - np.max(proj1))
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
        
        return min_gap

    def is_valid(positions, angles, tol=1e-5):
        """Check if all hexagons are non-overlapping."""
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True

    # --- Generate initial configuration from hexagonal lattice ---
    sqrt3 = math.sqrt(3)
    
    # Hexagonal lattice basis vectors
    base_v1 = np.array([1.5, sqrt3/2])
    base_v2 = np.array([0, sqrt3])
    
    # Generate candidate points from hexagonal lattice
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            p = i * base_v1 + j * base_v2
            lattice_points.append(p)
    
    # Sort by distance from origin and take 12 closest
    lattice_points.sort(key=lambda p: np.dot(p, p))
    init_positions = np.array(lattice_points[:12])
    init_positions -= np.mean(init_positions, axis=0)  # Center at origin
    init_angles = np.zeros(N)
    
    # Try a few rotations of the initial configuration
    best_init = None
    best_init_score = float('inf')
    
    for rot_deg in [0, 15, 30, 45]:
        rot_rad = math.radians(rot_deg)
        rot_mat = np.array([[math.cos(rot_rad), -math.sin(rot_rad)], 
                            [math.sin(rot_rad), math.cos(rot_rad)]])
        rotated = init_positions @ rot_mat.T
        angles = np.full(N, rot_deg)
        
        if is_valid(rotated, angles, tol=1e-6):
            score = calc_min_outer_side(rotated, angles, 0.0)
            if score < best_init_score:
                best_init_score = score
                best_init = (rotated.copy(), angles.copy())
    
    if best_init is None:
        best_init = (init_positions, init_angles)
        best_init_score = calc_min_outer_side(init_positions, init_angles, 0.0)
    
    init_positions, init_angles = best_init
    x0 = np.concatenate([init_positions.flatten(), init_angles, [0.0]])
    
    best_score = best_init_score
    best_result = x0.copy()

    # --- Phase 1: SLSQP with hard constraints ---
    def objective(params):
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:3*N]
        outer_angle = params[3*N]
        return calc_min_outer_side(positions, angles, outer_angle)
    
    def make_constraint(i, j):
        def constraint(params):
            pos_i = params[2*i:2*i+2]
            pos_j = params[2*j:2*j+2]
            ang_i = params[2*N + i]
            ang_j = params[2*N + j]
            
            poly_i = get_hex_verts(pos_i[0], pos_i[1], 1.0, ang_i)
            poly_j = get_hex_verts(pos_j[0], pos_j[1], 1.0, ang_j)
            
            return sat_gap(poly_i, poly_j)
        return constraint

    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
    
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
    for i in range(N):
        bounds.append((0, 60))  # Angles modulo 60
    bounds.append((0, 60))  # Outer angle modulo 60

    try:
        res = minimize(
            objective, 
            x0, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-10, 'disp': False}
        )
        
        if res.fun < best_score:
            pos = res.x[:2*N].reshape(N, 2)
            ang = res.x[2*N:3*N]
            outer_ang = res.x[3*N]
            
            if is_valid(pos, ang, tol=1e-4):
                score = calc_min_outer_side(pos, ang, outer_ang)
                if score < best_score:
                    best_score = score
                    best_result = res.x.copy()
    except Exception as e:
        pass

    # --- Phase 2: Nelder-Mead polishing with smooth penalty ---
    if time.time() < deadline - 10:
        def soft_overlap_penalty(polys):
            """Smooth penalty for overlapping hexagons using min gap."""
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    gap = sat_gap(polys[i], polys[j])
                    if gap < 0:
                        penalty += 1000.0 * gap * gap
            return penalty

        def obj_with_penalty(params):
            positions = params[:2*N].reshape(N, 2)
            angles = params[2*N:3*N]
            outer_angle = params[3*N]
            side_len = calc_min_outer_side(positions, angles, outer_angle)
            polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
            penalty = soft_overlap_penalty(polys)
            return side_len + penalty

        try:
            res2 = minimize(
                obj_with_penalty, 
                best_result, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True, 'disp': False}
            )
            
            pos = res2.x[:2*N].reshape(N, 2)
            ang = res2.x[2*N:3*N]
            outer_ang = res.x[3*N] if hasattr(res, 'x') else res2.x[3*N]
            
            if is_valid(pos, ang, tol=1e-5):
                score2 = calc_min_outer_side(pos, ang, outer_ang)
                if score2 < best_score:
                    best_score = score2
                    best_result = res2.x.copy()
        except Exception:
            pass

    # --- Phase 3: Fine-tune outer angle with fixed positions ---
    if time.time() < deadline - 5:
        final_positions = best_result[:2*N].reshape(N, 2)
        final_angles = best_result[2*N:3*N]
        
        # Grid search for optimal outer angle
        best_outer_angle = final_angles[0]  # placeholder
        best_outer_score = best_score
        
        for outer_angle in np.linspace(0, 60, 121):
            score = calc_min_outer_side(final_positions, final_angles, outer_angle)
            if score < best_outer_score:
                best_outer_score = score
                best_outer_angle = outer_angle
        
        if best_outer_score < best_score:
            best_score = best_outer_score
            best_result[3*N] = best_outer_angle

    # Finalize
    final_positions = best_result[:2*N].reshape(N, 2)
    final_angles = best_result[2*N:3*N]
    final_outer_angle = best_result[3*N]
    final_side = calc_min_outer_side(final_positions, final_angles, final_outer_angle)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, final_outer_angle

if __name__ == "__main__":
    construct()
