import numpy as np
import time
from scipy.optimize import minimize

N = 12
START_TIME = time.time()
TIME_LIMIT = 148.0  # Safety margin for 150s limit

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _compute_side_length(centers, angles, outer_angle):
    """
    Compute the minimum side length of the outer hexagon (centered at 0,0) 
    to contain all inner hexagons, given the outer angle.
    """
    max_proj = 0.0
    for k in range(6):
        angle = np.deg2rad(outer_angle + k * 60)
        normal = np.array([np.cos(angle), np.sin(angle)])
        
        for i in range(N):
            theta_i = np.deg2rad(angles[i])
            verts_i = np.array([
                [centers[i][0] + np.cos(theta_i + m * np.pi/3), 
                 centers[i][1] + np.sin(theta_i + m * np.pi/3)]
                for m in range(6)
            ])
            projs = np.dot(verts_i, normal)
            max_proj = max(max_proj, np.max(projs))
    
    return max_proj * 2.0 / np.sqrt(3.0)

def _check_validity(inner_hexes, outer_side, outer_center, outer_angle, eps=1e-7):
    """Check if inner hexagons are disjoint and inside outer hexagon."""
    outer_verts = _hex_vertices(outer_center, outer_side, outer_angle)
    inner_verts = [_hex_vertices(h[0:2], 1.0, h[2]) for h in inner_hexes]
    
    # Check containment
    for iv in inner_verts:
        for v in iv:
            inside = True
            for k in range(6):
                a = outer_verts[k]
                b = outer_verts[(k+1)%6]
                cross = (b[0]-a[0])*(v[1]-a[1]) - (b[1]-a[1])*(v[0]-a[0])
                if cross < -eps:
                    inside = False
                    break
            if not inside:
                return False, "outside"
                
    # Check disjointness using SAT
    def _sat_overlap(v1, v2):
        axes = []
        for verts in [v1, v2]:
            for i in range(6):
                p1 = verts[i]
                p2 = verts[(i + 1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-9:
                    normal = normal / norm
                axes.append(normal)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            if max1 < min2 or max2 < min1:
                return False
        return True
    
    for i in range(len(inner_hexes)):
        for j in range(i + 1, len(inner_hexes)):
            if _sat_overlap(inner_verts[i], inner_verts[j]):
                return False, "overlap"
                
    return True, "valid"

def _min_dist_polygons(v1, v2):
    """
    Compute the minimum distance between two convex polygons.
    Returns negative if overlapping.
    """
    # Check if any vertex of v1 is inside v2 or vice versa
    # If so, they overlap. We approximate overlap depth by -min_dist_to_edge.
    
    def point_in_poly(pt, poly):
        # Ray casting or cross product method for convex poly
        n = len(poly)
        sign = 0
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i+1)%n]
            cross = (p2[0]-p1[0])*(pt[1]-p1[1]) - (p2[1]-p1[1])*(pt[0]-p1[0])
            if cross > 1e-9:
                if sign == 0: sign = 1
                elif sign == -1: return False
            elif cross < -1e-9:
                if sign == 0: sign = -1
                elif sign == 1: return False
        return True

    if point_in_poly(v1[0], v2) or point_in_poly(v2[0], v1):
        # Overlapping. Estimate depth.
        # Find min distance from vertices to edges of the other polygon
        min_depth = 0.0
        for pt in v1:
            for i in range(len(v2)):
                p1 = v2[i]
                p2 = v2[(i+1)%len(v2)]
                edge = p2 - p1
                length = np.linalg.norm(edge)
                if length < 1e-9: continue
                normal = np.array([-edge[1], edge[0]]) / length
                # Distance from pt to line defined by p1, p2
                d = np.dot(pt - p1, normal)
                # Check if projection is on segment
                t = np.dot(pt - p1, edge) / (length * length)
                if 0 <= t <= 1:
                    if d < min_depth:
                        min_depth = d
                else:
                    # Check endpoints
                    d1 = np.linalg.norm(pt - p1)
                    d2 = np.linalg.norm(pt - p2)
                    # If point is inside, these distances are not the depth, 
                    # but for overlap penalty we just want a negative number.
                    # Using min_depth logic here is tricky for deep overlap.
                    # Let's just return a small negative number if we detect overlap via point-in-poly
                    return -1.0 
        return min_depth # Should be negative if vertex is inside
        
    # Not overlapping. Compute min distance between edges.
    min_d = np.inf
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i+1)%len(v1)]
        for j in range(len(v2)):
            q1 = v2[j]
            q2 = v2[(j+1)%len(v2)]
            # Segment-segment distance
            # Simplified: check distance from p1 to seg q1q2, p2 to q1q2, q1 to p1p2, q2 to p1p2
            def pt_seg_dist(pt, a, b):
                ab = b - a
                t = np.dot(pt - a, ab) / (np.dot(ab, ab) + 1e-9)
                t = np.clip(t, 0, 1)
                proj = a + t * ab
                return np.linalg.norm(pt - proj)
            
            d = min(pt_seg_dist(p1, q1, q2), pt_seg_dist(p2, q1, q2),
                    pt_seg_dist(q1, p1, p2), pt_seg_dist(q2, p1, p2))
            if d < min_d:
                min_d = d
    return min_d

def _simulate(initial_params, outer_angle, steps=2000, dt=0.01):
    """
    Simulate rigid bodies with repulsion and containment.
    """
    params = initial_params.copy()
    centers = params[:, :2].copy()
    angles = params[:, 2].copy()
    velocities = np.zeros_like(centers)
    angular_velocities = np.zeros_like(angles)
    
    # Initial side length estimate to start shrinking
    current_side = _compute_side_length(centers, angles, outer_angle)
    target_side = current_side * 0.9 # Start slightly tighter
    
    # We will iteratively reduce the target side
    # Actually, it's better to simulate with a fixed shrinking container
    
    # Let's use a force-based approach with a "soft" container that shrinks over time
    # Or simpler: simulate with repulsion only, then compute side length.
    # But repulsion alone doesn't minimize the bounding box.
    
    # Strategy:
    # 1. Simulate with repulsion to resolve overlaps.
    # 2. Apply a weak attractive force towards the center to compact them.
    # 3. Apply a containment force if they exceed a shrinking radius.
    
    # Let's try: Repulsion between hexes + Attraction to center + Hard constraint via penalty?
    # No, dynamics with hard constraints is hard.
    
    # Better dynamics:
    # Force = Repulsion (if dist < min_dist) + Centering (weak)
    # This will pack them into a ball-like shape.
    # Then we compute the hexagonal bounding box.
    
    # Let's implement a simple force simulation.
    
    for step in range(steps):
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
            
        forces = np.zeros_like(centers)
        torques = np.zeros_like(angles)
        
        # Compute vertices for all hexes
        verts = []
        for i in range(N):
            verts.append(_hex_vertices(centers[i], 1.0, angles[i]))
            
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                d = _min_dist_polygons(verts[i], verts[j])
                if d < 0.1: # Interaction range
                    # Force magnitude proportional to overlap or close proximity
                    # We want to push them apart.
                    # Direction: vector from center j to center i
                    vec = centers[i] - centers[j]
                    norm = np.linalg.norm(vec)
                    if norm < 1e-6:
                        vec = np.array([1.0, 0.0])
                        norm = 1.0
                    else:
                        vec = vec / norm
                    
                    # Force is stronger when closer
                    # F = k * (0.1 - d)
                    F = 10.0 * (0.1 - d)
                    if F > 0:
                        forces[i] += F * vec
                        forces[j] -= F * vec
                        
                        # Torque: try to rotate to increase distance
                        # This is complex to compute exactly. 
                        # Simple heuristic: add small random torque or 
                        # torque based on cross product of vec and edge normals?
                        # Let's skip torque for now or add small noise.
                        # torques[i] += 0.1 * np.sin(step * 0.1 + i)
                        # torques[j] -= 0.1 * np.sin(step * 0.1 + j)
                        
        # Centering force (to keep them compact)
        for i in range(N):
            forces[i] -= 0.5 * centers[i]
            
        # Integration (Symplectic Euler)
        # v += F * dt
        velocities += forces * dt
        # Damping
        velocities *= 0.95
        # x += v * dt
        centers += velocities * dt
        
        # Update angles with noise to help escape local minima
        # Or integrate angular velocity
        # Let's add small random perturbations to angles occasionally
        if step % 50 == 0:
            angles += np.random.uniform(-0.01, 0.01, N)
            
        params[:, :2] = centers
        params[:, 2] = angles
        
        # Update vertices for next iteration (expensive, but okay for 12 hexes)
        # We can optimize this later if needed
        
    return params

def _objective(x, outer_angle):
    """
    Objective function for L-BFGS-B.
    x: 3N vector of inner hexagon parameters (x1, y1, a1, x2, y2, a2, ...)
    Returns: (value, grad)
    """
    params = x.reshape(N, 3)
    centers = params[:, :2]
    angles = params[:, 2]
    
    # Compute side length
    side_length = _compute_side_length(centers, angles, outer_angle)
    
    # Compute penalty for overlaps
    penalty = 0.0
    grad = np.zeros_like(x)
    
    # We need gradients for the penalty.
    # For simplicity in the polish phase, we assume the simulation already resolved overlaps.
    # So we just minimize the side length.
    # However, if there are slight overlaps, we need to push them apart.
    
    # Let's compute a soft penalty for overlaps to ensure validity.
    verts = []
    for i in range(N):
        verts.append(_hex_vertices(centers[i], 1.0, angles[i]))
        
    for i in range(N):
        for j in range(i + 1, N):
            d = _min_dist_polygons(verts[i], verts[j])
            if d < 0:
                # Overlap penalty
                penalty += d * d
                # Gradient approximation: push apart
                vec = centers[i] - centers[j]
                norm = np.linalg.norm(vec)
                if norm > 1e-6:
                    vec = vec / norm
                    grad[i*3: i*3+2] += 2 * d * vec
                    grad[j*3: j*3+2] -= 2 * d * vec
                else:
                    grad[i*3] += 2 * d
                    grad[j*3] -= 2 * d

    # Gradient of side length
    # side = (2/sqrt(3)) * max_k (max_i (v_i . n_k))
    # d(side)/d(x_i) = (2/sqrt(3)) * n_k_active * d(v_i)/d(x_i)
    # v_i = c_i + R(a_i) * u_m
    # d(v_i)/d(c_i) = I
    # d(v_i)/d(a_i) = cross(u_m, v_i - c_i) ... complex
    
    # Approximation: The dominant term is the center.
    # Find the active constraint (max projection)
    max_proj_val = -np.inf
    active_k = 0
    active_i = 0
    active_m = 0
    
    for k in range(6):
        angle = np.deg2rad(outer_angle + k * 60)
        normal = np.array([np.cos(angle), np.sin(angle)])
        
        for i in range(N):
            theta_i = np.deg2rad(angles[i])
            verts_i = verts[i]
            projs = np.dot(verts_i, normal)
            if np.max(projs) > max_proj_val:
                max_proj_val = np.max(projs)
                active_k = k
                active_i = i
                active_m = np.argmax(projs)
                
    if max_proj_val > 0:
        angle_k = np.deg2rad(outer_angle + active_k * 60)
        normal = np.array([np.cos(angle_k), np.sin(angle_k)])
        
        # Gradient w.r.t. center of active hex
        grad[active_i*3: active_i*3+2] += (2.0 / np.sqrt(3.0)) * normal
        
        # Gradient w.r.t. angle is smaller, ignore for now or approximate
        # The vertex moves perpendicular to the radius vector
        # This is second order for the projection usually
        
    value = side_length + penalty
    
    return value, grad

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_params = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    outer_angles = [0.0, 30.0]
    
    # Generate initial guesses
    initial_guesses = []
    
    # Guess 1: Hexagonal ring
    pts = [[0,0,0]]
    for k in range(6):
        angle = np.pi/3 * k
        pts.append([np.sqrt(3)*np.cos(angle), np.sqrt(3)*np.sin(angle), 0.0])
    for k in range(6):
        angle = np.pi/3 * k
        pts.append([2*np.sqrt(3)*np.cos(angle), 2*np.sqrt(3)*np.sin(angle), 0.0])
        if len(pts) == 12:
            break
    initial_guesses.append(np.array(pts).flatten())
    
    # Guess 2: Random
    np.random.seed(42)
    for _ in range(3):
        pts = np.random.uniform(-3, 3, (N, 3))
        pts[:, 2] = np.random.choice([0, 30, 60, 90, 120, 150], N)
        initial_guesses.append(pts.flatten())
        
    for outer_angle in outer_angles:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
            
        for guess in initial_guesses:
            if time.time() - start_time > TIME_LIMIT - 20:
                break
                
            try:
                # Phase 1: Simulation
                params_sim = _simulate(guess.reshape(N, 3), outer_angle, steps=1000, dt=0.02)
                
                # Phase 2: Gradient Polish
                # Use the result of simulation as initial guess for L-BFGS-B
                res = minimize(
                    _objective,
                    params_sim.flatten(),
                    args=(outer_angle,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-10, 'maxfun': 1000}
                )
                
                x_opt = res.x
                params_opt = x_opt.reshape(N, 3)
                
                inner_hexes = [[params_opt[i][0], params_opt[i][1], params_opt[i][2]] for i in range(N)]
                
                centers = np.array([h[:2] for h in inner_hexes])
                angles = np.array([h[2] for h in inner_hexes])
                
                side = _compute_side_length(centers, angles, outer_angle)
                
                is_valid, reason = _check_validity(inner_hexes, side, (0.0, 0.0), outer_angle)
                
                if is_valid and side < best_side:
                    best_side = side
                    best_params = inner_hexes
                    best_outer_center = (0.0, 0.0)
                    best_outer_angle = outer_angle
                    
            except Exception as e:
                pass
                
    if best_params is not None:
        return (best_params, best_outer_center, best_side, best_outer_angle)
    
    # Fallback
    return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
