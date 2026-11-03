import numpy as np
import time
import math

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    angle_rad = math.radians(angle_deg)
    r = side
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        x = cx + r * math.cos(theta)
        y = cy + r * math.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def polygons_overlap_sat(verts1, verts2):
    min_overlap = float('inf')
    min_axis = None
    
    def get_axes(verts):
        axes = []
        for i in range(len(verts)):
            p1 = verts[i]
            p2 = verts[(i + 1) % len(verts)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                normal /= norm
                axes.append(normal)
        return axes
    
    axes1 = get_axes(verts1)
    axes2 = get_axes(verts2)
    all_axes = axes1 + axes2
    
    for axis in all_axes:
        proj1 = np.dot(verts1, axis)
        proj2 = np.dot(verts2, axis)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            return 0, None
        
        center1 = (min1 + max1) / 2.0
        center2 = (min2 + max2) / 2.0
        
        if center1 < center2:
            depth = min(max1, max2) - max(min1, min2)
            if depth < min_overlap:
                min_overlap = depth
                min_axis = -axis
        else:
            depth = min(max1, max2) - max(min1, min2)
            if depth < min_overlap:
                min_overlap = depth
                min_axis = axis

    if min_overlap < float('inf'):
        return min_overlap, min_axis
    return 0, None

def point_in_hexagon(pt, center, side, angle_deg):
    dx = pt[0] - center[0]
    dy = pt[1] - center[1]
    
    angle_rad = -math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    local_x = dx * cos_a - dy * sin_a
    local_y = dx * sin_a + dy * cos_a
    
    s = side
    lx = local_x
    ly = local_y
    
    if lx > s: return False
    if lx < -s: return False
    if math.sqrt(3)*lx + ly > s * math.sqrt(3): return False
    if ly > s * math.sqrt(3)/2: return False
    if -math.sqrt(3)*lx - ly > s * math.sqrt(3): return False
    if math.sqrt(3)*lx - ly > s * math.sqrt(3): return False
    if ly < -s * math.sqrt(3)/2: return False
    
    return True

def construct():
    start_time = time.time()
    deadline = 115.0 # Leave 5s margin
    
    best_S = float('inf')
    best_xs = None
    best_ys = None
    best_angles = None
    
    dt = 0.01
    damping = 0.9
    force_k = 15.0
    boundary_force_k = 15.0
    
    num_restarts = 100
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Determine S to try
        # If we have a best_S, try slightly smaller. Else, random.
        if best_S == float('inf'):
            S_current = np.random.uniform(4.0, 6.0)
        else:
            # Try to improve
            S_current = best_S - np.random.uniform(0.0, 0.5)
            if S_current < 3.5:
                S_current = 3.5
        
        # Initialize
        xs = np.zeros(N)
        ys = np.zeros(N)
        angles = np.zeros(N)
        vx = np.zeros(N)
        vy = np.zeros(N)
        vangle = np.zeros(N)
        
        for i in range(N):
            r = np.random.uniform(0, S_current/2)
            theta = np.random.uniform(0, 2*np.pi)
            xs[i] = r * np.cos(theta)
            ys[i] = r * np.sin(theta)
            angles[i] = np.random.uniform(0, 360)
        
        max_steps = 1500
        stable_count = 0
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            forces_x = np.zeros(N)
            forces_y = np.zeros(N)
            forces_angle = np.zeros(N)
            
            verts_list = []
            for i in range(N):
                verts = hex_vertices(xs[i], ys[i], 1.0, angles[i])
                verts_list.append(verts)
            
            # Inner-Inner
            for i in range(N):
                for j in range(i+1, N):
                    overlap, axis = polygons_overlap_sat(verts_list[i], verts_list[j])
                    if overlap > 0:
                        f = force_k * overlap
                        forces_x[i] -= axis[0] * f
                        forces_y[i] -= axis[1] * f
                        forces_x[j] += axis[0] * f
                        forces_y[j] += axis[1] * f
                        forces_angle[i] += np.random.uniform(-0.2, 0.2)
                        forces_angle[j] += np.random.uniform(-0.2, 0.2)
            
            # Boundary
            outer_center = np.array([0.0, 0.0])
            for i in range(N):
                verts = verts_list[i]
                inside = True
                for v in verts:
                    if not point_in_hexagon(v, outer_center, S_current, 0.0):
                        inside = False
                        break
                
                if not inside:
                    dist = np.sqrt(xs[i]**2 + ys[i]**2)
                    if dist > 1e-6:
                        f = boundary_force_k
                        forces_x[i] -= (xs[i] / dist) * f
                        forces_y[i] -= (ys[i] / dist) * f
                    else:
                        forces_x[i] += np.random.uniform(-1, 1)
                        forces_y[i] += np.random.uniform(-1, 1)
                    forces_angle[i] += np.random.uniform(-0.5, 0.5)
            
            # Update
            vx += forces_x * dt
            vy += forces_y * dt
            vangle += forces_angle * dt
            
            vx *= damping
            vy *= damping
            vangle *= damping
            
            xs += vx * dt
            ys += vy * dt
            angles += vangle * dt
            
            total_force = np.sum(np.abs(forces_x)) + np.sum(np.abs(forces_y)) + np.sum(np.abs(forces_angle))
            if total_force < 1e-3:
                stable_count += 1
            else:
                stable_count = 0
                
            if stable_count > 50:
                break
        
        # Validate
        valid = True
        verts_list = []
        for i in range(N):
            verts = hex_vertices(xs[i], ys[i], 1.0, angles[i])
            verts_list.append(verts)
            for v in verts:
                if not point_in_hexagon(v, np.array([0.0, 0.0]), S_current, 0.0):
                    valid = False
                    break
            if not valid: break
        
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    overlap, _ = polygons_overlap_sat(verts_list[i], verts_list[j])
                    if overlap > 1e-6:
                        valid = False
                        break
                if not valid: break
        
        if valid:
            if S_current < best_S:
                best_S = S_current
                best_xs = xs.copy()
                best_ys = ys.copy()
                best_angles = angles.copy()
                
                # Try to shrink further immediately
                for _ in range(20):
                    if time.time() > deadline: break
                    S_try = best_S - 0.02
                    if S_try < 3.5: break
                    
                    fits = True
                    for v in verts_list:
                        for pt in v:
                            if not point_in_hexagon(pt, np.array([0.0, 0.0]), S_try, 0.0):
                                fits = False
                                break
                        if not fits: break
                    if fits:
                        best_S = S_try
                    else:
                        break
    
    if best_xs is None:
        # Fallback
        best_S = 10.0
        best_xs = np.array([4.0 * k for k in range(-(N // 2), N - N // 2)])
        best_ys = np.zeros(N)
        best_angles = np.zeros(N)
        
    inner = []
    for i in range(N):
        inner.append([float(best_xs[i]), float(best_ys[i]), float(best_angles[i])])
        
    return inner, (0.0, 0.0), float(best_S), 0.0
