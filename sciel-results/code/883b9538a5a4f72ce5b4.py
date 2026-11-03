import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0 (pointy-top)
    # Vertices at angles 0, 60, ..., 300 degrees
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

    def compute_sat_constraints(params):
        """
        Computes the separation values for all pairs of hexagons.
        Positive value means separated (constraint satisfied).
        Negative value means overlapping (constraint violated).
        Returns a list of floats, one for each pair.
        """
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        
        constraints = []
        for i in range(N):
            for j in range(i+1, N):
                p1 = polys[i]
                p2 = polys[j]
                
                axes = []
                for k in range(6):
                    e1 = p1[(k+1)%6] - p1[k]
                    n1 = np.array([-e1[1], e1[0]])
                    axes.append(n1)
                    e2 = p2[(k+1)%6] - p2[k]
                    n2 = np.array([-e2[1], e2[0]])
                    axes.append(n2)
                    
                min_separation = float('inf')
                for axis in axes:
                    norm = np.linalg.norm(axis)
                    if norm < 1e-9: continue
                    axis = axis / norm
                    proj1 = p1 @ axis
                    proj2 = p2 @ axis
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    
                    # Separation = min(max1, max2) - max(min1, min2)
                    # If > 0, they are separated.
                    sep = min(max1, max2) - max(min1, min2)
                    if sep < min_separation:
                        min_separation = sep
                        
                constraints.append(min_separation)
                
        return constraints

    def objective(params):
        """
        Objective function: Minimize the outer hexagon side length.
        """
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        outer_angle = params[-1]
        
        side = calc_min_outer_side(positions, angles, outer_angle)
        return side

    def constraint_func(params):
        """
        Constraint function for SLSQP.
        Returns array of separation values. All must be >= 0.
        """
        return np.array(compute_sat_constraints(params))

    # Multi-start optimization
    best_score = float('inf')
    best_params = None

    sqrt3 = math.sqrt(3)
    
    # Start 1: Honeycomb subset
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
    
    x0_list = [np.concatenate([init_positions.flatten(), init_angles, [0.0]])]
    
    # Start 2: Random
    np.random.seed(42)
    for _ in range(5):
        pos = np.random.uniform(-5, 5, (N, 2))
        ang = np.random.uniform(0, 360, N)
        outer_ang = np.random.uniform(0, 60)
        x0_list.append(np.concatenate([pos.flatten(), ang, [outer_ang]]))
        
    # Start 3: Two rings
    r1, r2 = 2.0, 4.0
    pos_rings = []
    ang_rings = []
    for k in range(6):
        theta = k * math.pi / 3
        pos_rings.append([r1 * math.cos(theta), r1 * math.sin(theta)])
        ang_rings.append(math.degrees(theta))
    for k in range(6):
        theta = k * math.pi / 3 + math.pi/6
        pos_rings.append([r2 * math.cos(theta), r2 * math.sin(theta)])
        ang_rings.append(math.degrees(theta))
    x0_list.append(np.concatenate([np.array(pos_rings).flatten(), np.array(ang_rings), [0.0]]))

    # Bounds: positions (-10, 10), angles (0, 360), outer angle (0, 60)
    bounds = [(-10, 10)] * (2*N) + [(0, 360)] * N + [(0, 60)]

    # Constraints for SLSQP: sep >= 0
    cons = {'type': 'ineq', 'fun': constraint_func}

    for idx, x0 in enumerate(x0_list):
        if time.time() > deadline - 30:
            break
            
        try:
            # SLSQP is a constrained solver. It uses finite difference gradients if not provided.
            # Providing analytical gradients for SAT is complex and error-prone in a short script.
            # Given N=12, the constraint function is called frequently. 
            # To speed it up, we could vectorize, but for now, we rely on SLSQP's robustness.
            # We limit iterations to ensure we don't hit the time limit.
            
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 100, 'ftol': 1e-6, 'disp': False}
            )
            
            # Check validity and score
            if res.success or res.fun < best_score:
                # Verify validity strictly
                final_positions = res.x[:2*N].reshape(N, 2)
                final_angles = res.x[2*N:-1]
                final_outer_angle = res.x[-1]
                
                if is_valid(final_positions, final_angles):
                    score = calc_min_outer_side(final_positions, final_angles, final_outer_angle)
                    if score < best_score:
                        best_score = score
                        best_params = res.x
        except Exception as e:
            pass

    # If SLSQP failed to find a valid tight packing, try Nelder-Mead on best found or fallback
    if best_params is None:
        # Fallback to a safe, valid configuration
        # Use the honeycomb subset
        final_positions = init_positions
        final_angles = init_angles
        final_outer_angle = 0.0
        # Verify
        if not is_valid(final_positions, final_angles):
            # Force separation
            # This is a rough fallback
            final_positions = np.array([
                [0, 0], [2.1, 0], [-2.1, 0],
                [1.05, 1.81], [-1.05, 1.81], [0, 3.62],
                [1.05, -1.81], [-1.05, -1.81], [0, -3.62],
                [3.15, 1.81], [-3.15, 1.81], [3.15, -1.81]
            ])
            final_angles = np.zeros(N)
            final_outer_angle = 0.0
        best_score = calc_min_outer_side(final_positions, final_angles, final_outer_angle)
    else:
        final_positions = best_params[:2*N].reshape(N, 2)
        final_angles = best_params[2*N:-1]
        final_outer_angle = best_params[-1]
        
        # Final polish with Nelder-Mead if time permits
        if time.time() < deadline - 30:
            try:
                def obj_nm(params):
                    positions = params[:2*N].reshape(N, 2)
                    angles = params[2*N:-1]
                    outer_angle = params[-1]
                    if not is_valid(positions, angles):
                        return 100.0
                    return calc_min_outer_side(positions, angles, outer_angle)
                
                res_nm = minimize(
                    obj_nm,
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                
                if res_nm.fun < best_score:
                    nm_positions = res_nm.x[:2*N].reshape(N, 2)
                    nm_angles = res_nm.x[2*N:-1]
                    nm_outer = res_nm.x[-1]
                    if is_valid(nm_positions, nm_angles):
                        best_score = res_nm.fun
                        final_positions = nm_positions
                        final_angles = nm_angles
                        final_outer_angle = nm_outer
            except Exception:
                pass

    final_side = calc_min_outer_side(final_positions, final_angles, final_outer_angle)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, final_outer_angle

if __name__ == "__main__":
    construct()
