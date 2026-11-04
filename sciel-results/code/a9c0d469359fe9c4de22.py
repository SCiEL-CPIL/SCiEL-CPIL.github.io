import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    sqrt3 = math.sqrt(3)
    
    # Precompute angles for hexagon vertices relative to center
    # Unit hexagon vertices are at angles 0, 60, 120, 180, 240, 300 degrees
    hex_vertex_angles = np.array([k * math.pi / 3.0 for k in range(6)])

    def support_hex(cx, cy, rot_rad, dir_x, dir_y):
        """
        Calculate the maximum projection of a unit hexagon centered at (cx, cy) 
        rotated by rot_rad onto the direction (dir_x, dir_y).
        
        The support function of a regular polygon with vertices at angles 
        v_k + rot is max_k (cos(v_k + rot - dir_angle)).
        Since dir = (cos(dir_angle), sin(dir_angle)), 
        dot product is cos(v_k + rot - dir_angle).
        """
        # Vectorized calculation of cos(v_k + rot - dir_angle)
        # Note: cos(A - B) = cos(A)cos(B) + sin(A)sin(B)
        # But direct cos is fine.
        # dir_angle = atan2(dir_y, dir_x)
        # We can compute cos(v_k + rot - dir_angle) efficiently.
        
        # Let's just use the property:
        # proj_k = cos(v_k + rot) * dir_x + sin(v_k + rot) * dir_y
        # This avoids atan2.
        
        # Pre-compute cos/sin of vertex angles
        # cos(v_k + rot) = cos(v_k)cos(rot) - sin(v_k)sin(rot)
        # sin(v_k + rot) = sin(v_k)cos(rot) + cos(v_k)sin(rot)
        
        # To keep it simple and fast enough for N=12, we do this per call.
        # However, since this is called inside optimization loops, we want it tight.
        
        # Optimization: 
        # The support function of a regular hexagon is a periodic function.
        # max_k cos(v_k + rot - theta_dir)
        
        # Let's implement the direct dot product of rotated vertices with direction.
        # Vertices are R(rot) * V_k.
        # Dot = (R(rot) V_k) . d = V_k . R(-rot) d
        # Let d' = R(-rot) d = (d_x cos(rot) + d_y sin(rot), -d_x sin(rot) + d_y cos(rot))
        # Then max_k (V_k . d')
        
        c_rot = math.cos(rot_rad)
        s_rot = math.sin(rot_rad)
        
        dx_prime = dir_x * c_rot + dir_y * s_rot
        dy_prime = -dir_x * s_rot + dir_y * c_rot
        
        # V_k are fixed: (1,0), (0.5, sqrt3/2), (-0.5, sqrt3/2), (-1,0), (-0.5, -sqrt3/2), (0.5, -sqrt3/2)
        # Projections:
        # k=0: dx_prime
        # k=1: 0.5*dx_prime + sqrt3/2*dy_prime
        # k=2: -0.5*dx_prime + sqrt3/2*dy_prime
        # k=3: -dx_prime
        # k=4: -0.5*dx_prime - sqrt3/2*dy_prime
        # k=5: 0.5*dx_prime - sqrt3/2*dy_prime
        
        p0 = dx_prime
        p1 = 0.5 * dx_prime + 0.8660254037844386 * dy_prime # sqrt3/2
        p2 = -0.5 * dx_prime + 0.8660254037844386 * dy_prime
        p3 = -dx_prime
        p4 = -0.5 * dx_prime - 0.8660254037844386 * dy_prime
        p5 = 0.5 * dx_prime - 0.8660254037844386 * dy_prime
        
        max_proj = max(p0, p1, p2, p3, p4, p5)
        
        return cx * dir_x + cy * dir_y + max_proj

    def calc_side_length(params):
        """
        Calculate the side length of the outer hexagon containing all inner hexagons.
        params: [x1, y1, a1, ..., x12, y12, a12, phi_outer]
        """
        # Unpack
        # 0..23: positions and angles of 12 hexagons
        # 24: outer angle phi
        phi = params[24]
        
        # Outer hexagon normals
        # A regular hexagon centered at origin with rotation phi has edge normals at:
        # phi + pi/6 + k*pi/3 for k=0..5
        # The side length S is related to the max projection M by:
        # M = S * sqrt(3) / 2  (inradius)
        # So S = 2 * M / sqrt(3)
        
        max_proj = 0.0
        
        for k in range(6):
            normal_angle = phi + math.pi/6 + k * math.pi/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            
            for i in range(N):
                cx = params[3*i]
                cy = params[3*i+1]
                ai = params[3*i+2]
                
                proj = support_hex(cx, cy, ai, nx, ny)
                if proj > max_proj:
                    max_proj = proj
                    
        return 2.0 * max_proj / sqrt3

    def poly_distance(params, i, j):
        """
        Calculate the minimum distance between hex i and hex j.
        Returns positive if separated, negative if overlapping.
        Uses SAT (Separating Axis Theorem) on the 12 edge normals (6 from each).
        """
        cx1, cy1, a1 = params[3*i], params[3*i+1], params[3*i+2]
        cx2, cy2, a2 = params[3*j], params[3*j+1], params[3*j+2]
        
        min_gap = float('inf')
        
        # Check axes from Hex 1
        for k in range(6):
            # Normal of edge k of hex 1
            # Edge k connects vertex k and k+1.
            # Normal direction is angle a1 + k*pi/3 + pi/6
            normal_angle = a1 + k * math.pi/3 + math.pi/6
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            
            # Project both hexagons onto this axis
            # We need min and max projections.
            # support_hex gives max.
            # min_proj = -support_hex(-cx, -cy, a+pi, -nx, -ny) ?
            # Actually, min projection of P onto n is - max projection of -P onto n.
            # Or simply, we can compute min by checking all vertices.
            # For efficiency, let's compute min and max for a hex on an axis.
            
            def min_max_proj(cx, cy, rot, nx, ny):
                c_rot = math.cos(rot)
                s_rot = math.sin(rot)
                dx_prime = nx * c_rot + ny * s_rot
                dy_prime = -nx * s_rot + ny * c_rot
                
                p0 = dx_prime
                p1 = 0.5 * dx_prime + 0.8660254037844386 * dy_prime
                p2 = -0.5 * dx_prime + 0.8660254037844386 * dy_prime
                p3 = -dx_prime
                p4 = -0.5 * dx_prime - 0.8660254037844386 * dy_prime
                p5 = 0.5 * dx_prime - 0.8660254037844386 * dy_prime
                
                vals = np.array([p0, p1, p2, p3, p4, p5])
                min_v = cx * nx + cy * ny + np.min(vals)
                max_v = cx * nx + cy * ny + np.max(vals)
                return min_v, max_v

            min1, max1 = min_max_proj(cx1, cy1, a1, nx, ny)
            min2, max2 = min_max_proj(cx2, cy2, a2, nx, ny)
            
            gap = max(min1 - max2, min2 - max1)
            
            if gap < 0:
                if gap < min_gap:
                    min_gap = gap
            else:
                # Separating axis found
                return gap

        # Check axes from Hex 2
        for k in range(6):
            normal_angle = a2 + k * math.pi/3 + math.pi/6
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            
            min1, max1 = min_max_proj(cx1, cy1, a1, nx, ny)
            min2, max2 = min_max_proj(cx2, cy2, a2, nx, ny)
            
            gap = max(min1 - max2, min2 - max1)
            
            if gap < 0:
                if gap < min_gap:
                    min_gap = gap
            else:
                return gap

        return min_gap

    def is_valid(params, tol=1e-6):
        for i in range(N):
            for j in range(i+1, N):
                if poly_distance(params, i, j) < -tol:
                    return False
        return True

    # --- Phase 1: Generate Initial Configurations ---
    # Honeycomb lattice
    a1_vec = np.array([1.5, sqrt3/2])
    a2_vec = np.array([0.0, sqrt3])
    
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            p = i * a1_vec + j * a2_vec
            lattice_points.append(p)
            
    # Sort by distance from origin
    lattice_points.sort(key=lambda p: float(np.dot(p, p)))
    init_points = np.array(lattice_points[:12])
    init_points -= np.mean(init_points, axis=0)
    
    best_params = None
    best_score = float('inf')
    
    # Try a few global rotations for the initial lattice
    for rot_deg in [0, 30, 60]:
        rot_rad = math.radians(rot_deg)
        rot_mat = np.array([[math.cos(rot_rad), -math.sin(rot_rad)],
                            [math.sin(rot_rad), math.cos(rot_rad)]])
        rotated_points = (init_points @ rot_mat.T)
        
        # Scale slightly to ensure validity if they are touching
        for scale in [1.0, 1.01, 1.05]:
            scaled_points = rotated_points * scale
            params = np.zeros(3*N + 1)
            for i in range(N):
                params[3*i] = scaled_points[i][0]
                params[3*i+1] = scaled_points[i][1]
                params[3*i+2] = rot_deg # All aligned initially
            params[24] = 0.0 # Outer angle 0
            
            if is_valid(params, tol=1e-5):
                score = calc_side_length(params)
                if score < best_score:
                    best_score = score
                    best_params = params.copy()
                    break
        if best_params is not None:
            # Check if we found a good one, maybe break early if very tight
            if best_score < 5.0: # Arbitrary threshold, just to speed up init
                break

    # Fallback if no valid config found (should not happen)
    if best_params is None:
        scaled_points = init_points * 2.0
        params = np.zeros(3*N + 1)
        for i in range(N):
            params[3*i] = scaled_points[i][0]
            params[3*i+1] = scaled_points[i][1]
            params[3*i+2] = 0.0
        params[24] = 0.0
        best_params = params
        best_score = calc_side_length(params)

    x0 = best_params.copy()
    
    # --- Phase 2: Optimization ---
    # We will run SLSQP with the outer angle as a variable.
    # Constraints: pairwise non-interpenetration.
    
    def objective(params):
        return calc_side_length(params)

    def make_constraint(i, j):
        def constraint(params):
            # We want distance >= 0
            return poly_distance(params, i, j)
        return constraint

    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})

    # Multi-start optimization to avoid local minima
    # We will restart from slightly perturbed versions of the best solution found so far.
    
    current_best_params = x0.copy()
    current_best_score = best_score
    
    np.random.seed(42)
    
    num_restarts = 15
    # Adjust number of restarts based on time remaining
    time_per_restart = 8.0 # seconds
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Perturb the initial guess
        x_start = current_best_params.copy()
        if r > 0:
            # Perturb positions and angles
            for i in range(N):
                x_start[3*i] += np.random.randn() * 0.5
                x_start[3*i+1] += np.random.randn() * 0.5
                x_start[3*i+2] += np.random.randn() * 10
            x_start[24] += np.random.randn() * 10 # Perturb outer angle
        
        try:
            result = minimize(
                objective, 
                x_start, 
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
            )
            
            if result.success or result.status == 0:
                # Verify validity with tight tolerance
                if is_valid(result.x, tol=1e-6):
                    score = calc_side_length(result.x)
                    if score < current_best_score:
                        current_best_score = score
                        current_best_params = result.x.copy()
                        # Update best_params for next restart to be centered around this
                        # This creates a local search around the best found so far
                        # But we also want to explore, so maybe keep x0 fixed for some restarts?
                        # Strategy: First few restarts explore, later ones exploit.
                        # If r < num_restarts/2, use x0 (original best init) + perturbation
                        # Else use current_best + perturbation
                        pass 
                        
        except Exception as e:
            pass

    # Final Polish: Run one more SLSQP from the very best solution found with tighter tolerance
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                objective, 
                current_best_params, 
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-12, 'disp': False}
            )
            if is_valid(result.x, tol=1e-7):
                score = calc_side_length(result.x)
                if score < current_best_score:
                    current_best_score = score
                    current_best_params = result.x.copy()
        except Exception:
            pass

    # Extract final result
    final_params = current_best_params
    inner = []
    for i in range(N):
        inner.append([final_params[3*i], final_params[3*i+1], final_params[3*i+2]])
    
    outer_center = [0.0, 0.0]
    outer_side = current_best_score
    outer_angle = final_params[24]
    
    # The evaluator expects: inner, center, side, angle
    # inner: list of (x, y, angle_degrees)
    # center: (x, y)
    # side: length
    # angle: rotation of outer hexagon
    
    return inner, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-END
