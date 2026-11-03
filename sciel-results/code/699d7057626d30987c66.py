import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Unit hexagon vertices (pointy-top, side 1)
    # Angles: 0, 60, 120, 180, 240, 300
    unit_verts_0 = np.array([
        [1.0, 0.0],
        [0.5, math.sqrt(3)/2],
        [-0.5, math.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -math.sqrt(3)/2],
        [0.5, -math.sqrt(3)/2]
    ])

    def get_hex_verts(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle_deg=0.0):
        """
        Calculate minimum side length S of regular hexagon (centered at 0,0, rotated by outer_angle_deg)
        that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle_deg)
        # Normals to the edges of the outer hexagon
        # Edge normals are at 30, 90, 150, 210, 270, 330 for flat-top?
        # For pointy-top outer hexagon (vertices at 0, 60...), normals are at 30, 90...
        # Let's assume standard orientation: vertices at 0, 60...
        # Then edge normals are at 30 + 60k.
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)
        
        max_proj = 0.0
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, ang)
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        # Distance from center to edge is R = S * sqrt(3) / 2
        # So S = 2 * R / sqrt(3)
        return max_proj * 2.0 / math.sqrt(3)

    def smooth_sat_penalty(polys, beta=50.0):
        """
        Calculate smooth penalty for overlaps between all pairs of polygons.
        """
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                p1 = polys[i]
                p2 = polys[j]
                
                axes = []
                for k in range(6):
                    edge1 = p1[(k+1)%6] - p1[k]
                    n1 = np.array([-edge1[1], edge1[0]])
                    axes.append(n1)
                    edge2 = p2[(k+1)%6] - p2[k]
                    n2 = np.array([-edge2[1], edge2[0]])
                    axes.append(n2)
                
                min_gap = float('inf')
                for axis in axes:
                    norm = np.linalg.norm(axis)
                    if norm < 1e-9: continue
                    axis = axis / norm
                    proj1 = p1 @ axis
                    proj2 = p2 @ axis
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    # Gap s: positive if separated
                    s = min(min2 - max1, min1 - max2)
                    if s < min_gap:
                        min_gap = s
                        if min_gap > 0.1: # Early exit if clearly separated
                            break
                
                if min_gap < 0:
                    # Softplus approximation of max(0, -min_gap)^2
                    # softplus(x) = log(1 + exp(x))
                    # We want penalty for overlap depth d = -min_gap > 0
                    # penalty = (softplus(d*beta)/beta)^2 ?
                    # Simpler: penalty = (max(0, -min_gap))^2 is non-smooth.
                    # Use: penalty = (1/beta * log(1 + exp(-beta * min_gap)))^2
                    # If min_gap < 0, exp(-beta * min_gap) = exp(positive) -> large.
                    # log(1+large) ~ -beta * min_gap.
                    # Result ~ min_gap.
                    # Squared ~ min_gap^2.
                    val = -min_gap
                    # Avoid overflow
                    if val * beta > 500:
                        pen += val * val
                    else:
                        soft_val = (1.0/beta) * math.log(1.0 + math.exp(val * beta))
                        pen += soft_val * soft_val
        return pen

    def is_valid(positions, angles, tol=1e-5):
        polys = [get_hex_verts(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                p1 = polys[i]
                p2 = polys[j]
                axes = []
                for k in range(6):
                    edge1 = p1[(k+1)%6] - p1[k]
                    n1 = np.array([-edge1[1], edge1[0]])
                    axes.append(n1)
                    edge2 = p2[(k+1)%6] - p2[k]
                    n2 = np.array([-edge2[1], edge2[0]])
                    axes.append(n2)
                
                min_gap = float('inf')
                for axis in axes:
                    norm = np.linalg.norm(axis)
                    if norm < 1e-9: continue
                    axis = axis / norm
                    proj1 = p1 @ axis
                    proj2 = p2 @ axis
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    s = min(min2 - max1, min1 - max2)
                    if s < min_gap:
                        min_gap = s
                    if min_gap > tol:
                        break
                if min_gap <= tol:
                    return False
        return True

    def objective_smooth(params):
        if time.time() > deadline - 5:
            return 1000.0
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:2*N+N]
        outer_angle = params[2*N+N]
        
        side = calc_min_outer_side(positions, angles, outer_angle)
        
        polys = [get_hex_verts(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
        pen = smooth_sat_penalty(polys)
        
        # We want side to be small, penalty to be 0.
        # If penalty > 0, it should dominate.
        return side + 100.0 * pen

    def generate_initial_guesses():
        guesses = []
        sqrt3 = math.sqrt(3)
        
        # 1. Honeycomb Lattice Subsets
        # Generate a large grid of hex lattice points
        lattice_points = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * 1.5
                y = i * sqrt3 / 2.0 + j * sqrt3
                lattice_points.append((x, y))
        
        # Sort by distance to origin
        lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
        
        # Take first 12
        init_pos = np.array(lattice_points[:12])
        init_pos -= np.mean(init_pos, axis=0)
        init_ang = np.zeros(N)
        init_outer = 0.0
        guesses.append(np.concatenate([init_pos.flatten(), init_ang, [init_outer]]))
        
        # 2. Rectangular Grid 3x4
        pos = []
        for r in range(3):
            for c in range(4):
                x = c * 1.5
                y = r * sqrt3
                if r % 2 == 1:
                    x += 0.75
                pos.append([x, y])
        pos = np.array(pos)
        pos -= np.mean(pos, axis=0)
        ang = np.zeros(N)
        guesses.append(np.concatenate([pos.flatten(), ang, [0.0]]))
        
        # 3. Rectangular Grid 4x3
        pos = []
        for r in range(4):
            for c in range(3):
                x = c * 1.5
                y = r * sqrt3
                if r % 2 == 1:
                    x += 0.75
                pos.append([x, y])
        pos = np.array(pos)
        pos -= np.mean(pos, axis=0)
        ang = np.zeros(N)
        guesses.append(np.concatenate([pos.flatten(), ang, [0.0]]))
        
        # 4. Rotated Honeycomb
        init_pos = np.array(lattice_points[:12])
        init_pos -= np.mean(init_pos, axis=0)
        # Rotate positions by 15 deg
        rot_angle = math.radians(15)
        rot_mat = np.array([[math.cos(rot_angle), -math.sin(rot_angle)], [math.sin(rot_angle), math.cos(rot_angle)]])
        init_pos = init_pos @ rot_mat.T
        init_ang = np.full(N, 15.0)
        guesses.append(np.concatenate([init_pos.flatten(), init_ang, [15.0]]))
        
        # 5. Random Scatter (few)
        for _ in range(5):
            pos = np.random.uniform(-3, 3, (N, 2))
            pos -= np.mean(pos, axis=0)
            ang = np.random.uniform(0, 60, N)
            outer = np.random.uniform(0, 60)
            guesses.append(np.concatenate([pos.flatten(), ang, [outer]]))
            
        return guesses

    best_score = float('inf')
    best_params = None

    guesses = generate_initial_guesses()
    
    # Optimize each guess
    for i, x0 in enumerate(guesses):
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                objective_smooth, 
                x0, 
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-7, 'maxls': 50}
            )
            
            # Check if result is promising
            if res.fun < best_score and res.fun < 50.0:
                final_positions = res.x[:2*N].reshape(N, 2)
                final_angles = res.x[2*N:2*N+N]
                final_outer_angle = res.x[2*N+N]
                
                # Validate
                if is_valid(final_positions, final_angles):
                    true_score = calc_min_outer_side(final_positions, final_angles, final_outer_angle)
                    if true_score < best_score:
                        best_score = true_score
                        best_params = res.x.copy()
                        
        except Exception:
            pass

    # Fallback
    if best_params is None:
        # Use the best guess's initial state if optimization failed
        # Or a very large safe box
        init_pos = np.array([[0,0]] * N)
        init_ang = np.zeros(N)
        final_positions = init_pos
        final_angles = init_ang
        final_outer_angle = 0.0
        final_side = calc_min_outer_side(final_positions, final_angles, final_outer_angle)
    else:
        final_positions = best_params[:2*N].reshape(N, 2)
        final_angles = best_params[2*N:2*N+N]
        final_outer_angle = best_params[2*N+N]
        final_side = calc_min_outer_side(final_positions, final_angles, final_outer_angle)

    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    # The interface requires: inner, center, side, angle
    # center is (0,0)
    # side is final_side
    # angle is final_outer_angle
    return inner, (0.0, 0.0), final_side, final_outer_angle

