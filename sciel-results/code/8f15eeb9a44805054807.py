import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute unit hexagon vertices for angle 0 (pointy-top)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
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
        The side length s is related to the max projection R on normals by R = s * sqrt(3)/2.
        So s = 2 * R / sqrt(3).
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

    def sat_gap(poly1, poly2):
        """
        Returns the minimum separation distance between two convex polygons.
        If they overlap, returns a negative value (approximate penetration depth).
        If they are separated, returns the positive distance.
        """
        min_gap = float('inf')
        
        # Axes from poly1 edges
        for i in range(6):
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            norm = np.linalg.norm(normal1)
            if norm < 1e-9: continue
            axis = normal1 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Gap on this axis
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                # Overlapping on this axis
                min_gap = min(min_gap, gap)
            else:
                # Separated on this axis -> disjoint
                return gap
                
        # Axes from poly2 edges
        for i in range(6):
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            norm = np.linalg.norm(normal2)
            if norm < 1e-9: continue
            axis = normal2 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                # Overlapping on this axis
                min_gap = min(min_gap, gap)
            else:
                return gap
                
        return min_gap

    def is_valid(positions, angles, tol=1e-5):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True

    def objective_smooth(params, penalty_weight=10.0):
        if time.time() > deadline - 5:
            return 1000.0
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        # Calculate outer side length
        side = calc_min_outer_side(positions, angles, 0.0)
        
        # Calculate overlap penalty
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                gap = sat_gap(polys[i], polys[j])
                if gap < 0:
                    pen += gap * gap
        return side + penalty_weight * pen

    def generate_initial_guesses():
        guesses = []
        sqrt3 = math.sqrt(3)
        
        # 1. Honeycomb subset (compact)
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
        guesses.append(np.concatenate([init_positions.flatten(), init_angles]))
        
        # 2. 2x6 Grid (rectangular block)
        pos = []
        for row in range(2):
            for col in range(6):
                x = col * 1.5
                y = row * sqrt3
                if row % 2 == 1:
                    x += 0.75
                pos.append([x, y])
        pos = np.array(pos)
        pos -= np.mean(pos, axis=0)
        ang = np.zeros(N)
        guesses.append(np.concatenate([pos.flatten(), ang]))
        
        # 3. Linear Chain
        pos = []
        for i in range(12):
            pos.append([i * 1.5, 0])
        pos = np.array(pos)
        pos -= np.mean(pos, axis=0)
        ang = np.zeros(N)
        guesses.append(np.concatenate([pos.flatten(), ang]))
        
        # 4. Random Scatter
        for _ in range(5):
            pos = np.random.uniform(-3, 3, (N, 2))
            ang = np.random.uniform(0, 360, N)
            guesses.append(np.concatenate([pos.flatten(), ang]))
            
        # 5. Discrete Lattice Search (Inspiration 1)
        # Search a small grid of lattice scales and offsets to find better topologies
        base_v1 = np.array([1.5, sqrt3/2])
        base_v2 = np.array([0, sqrt3])
        
        for scale_x in [0.95, 1.0, 1.05]:
            for scale_y in [0.95, 1.0, 1.05]:
                v1 = base_v1 * np.array([scale_x, 1.0])
                v2 = base_v2 * np.array([1.0, scale_y])
                
                points = []
                for i in range(-4, 5):
                    for j in range(-4, 5):
                        p = i * v1 + j * v2
                        points.append(p)
                
                points.sort(key=lambda p: np.dot(p, p))
                selected = np.array(points[:12])
                centroid = np.mean(selected, axis=0)
                selected -= centroid
                
                ang = np.zeros(N)
                guesses.append(np.concatenate([selected.flatten(), ang]))
        
        return guesses

    best_score = float('inf')
    best_params = None

    guesses = generate_initial_guesses()
    
    # Optimize each guess
    for i, x0 in enumerate(guesses):
        if time.time() > deadline - 10:
            break
            
        try:
            # Use L-BFGS-B with the new objective
            res = minimize(
                objective_smooth, 
                x0, 
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if res.fun < best_score and res.fun < 100.0:
                # Check validity of the result
                final_positions = res.x[:2*N].reshape(N, 2)
                final_angles = res.x[2*N:]
                
                # Re-calculate true score (without penalty) if valid
                if is_valid(final_positions, final_angles):
                    true_score = calc_min_outer_side(final_positions, final_angles, 0.0)
                    if true_score < best_score:
                        best_score = true_score
                        best_params = res.x
        except Exception:
            pass

    # If no good result found, use a safe fallback
    if best_params is None:
        # Fallback to a known valid configuration (even if not optimal)
        # A large hexagon containing 12 small ones
        final_positions = np.array([
            [0, 0], [2.1, 0], [-2.1, 0],
            [1.05, 1.81], [-1.05, 1.81], [0, 3.62],
            [1.05, -1.81], [-1.05, -1.81], [0, -3.62],
            [3.15, 1.81], [-3.15, 1.81], [3.15, -1.81]
        ])
        final_angles = np.zeros(N)
        final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    else:
        final_positions = best_params[:2*N].reshape(N, 2)
        final_angles = best_params[2*N:]
        final_side = calc_min_outer_side(final_positions, final_angles, 0.0)

    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0
