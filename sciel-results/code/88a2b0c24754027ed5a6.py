import numpy as np
import time

N = 12
DEADLINE = 140.0

def hex_vertices(cx, cy, angle_deg, scale=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    angles = np.radians(angle_deg) + np.arange(6) * np.pi / 3
    r = scale
    return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])

def polygons_overlap_sat(verts_a, verts_b):
    """SAT test for two convex polygons. Returns (overlap, min_axis, min_dist)."""
    n_a, n_b = len(verts_a), len(verts_b)
    min_overlap = float('inf')
    best_axis = None
    
    for verts in [verts_a, verts_b]:
        for i in range(len(verts)):
            j = (i + 1) % len(verts)
            edge = verts[j] - verts[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            
            proj_a = np.dot(verts_a, normal)
            proj_b = np.dot(verts_b, normal)
            
            min_a, max_a = proj_a.min(), proj_a.max()
            min_b, max_b = proj_b.min(), proj_b.max()
            
            if max_a <= min_b or max_b <= min_a:
                return (False, normal, max_a - min_b if max_a <= min_b else max_b - min_a)
            
            overlap = min(max_a, max_b) - max(min_a, min_b)
            if overlap < min_overlap:
                min_overlap = overlap
                best_axis = normal
    
    if min_overlap < 0:
        return (False, best_axis, 0)
    
    center_a = verts_a.mean(axis=0)
    center_b = verts_b.mean(axis=0)
    diff = center_b - center_a
    if np.dot(diff, best_axis) < 0:
        best_axis = -best_axis
        
    return (True, best_axis, -min_overlap)

def compute_min_outer_side(positions, angles, outer_center, outer_angle):
    """
    positions: np.array (N, 2)
    angles: np.array (N,)
    outer_center: (x, y)
    outer_angle: degrees
    """
    sqrt3_2 = np.sqrt(3) / 2
    outer_center = np.array(outer_center)
    
    max_required = 0.0
    # Precompute outer normals
    outer_normals = []
    for k in range(6):
        theta_k = np.radians(outer_angle) + k * np.pi / 3
        n_k = np.array([np.cos(theta_k), np.sin(theta_k)])
        outer_normals.append(n_k)
    
    for i in range(N):
        cx, cy = positions[i]
        ang = angles[i]
        verts = hex_vertices(cx, cy, ang)
        verts_rel = verts - outer_center
        
        for n_k in outer_normals:
            dots = np.dot(verts_rel, n_k)
            max_dot = dots.max()
            required_s = max_dot / sqrt3_2
            if required_s > max_required:
                max_required = required_s
                
    return max_required

def check_validity(positions, angles):
    """Check if all inner hexagons are disjoint."""
    verts_list = []
    for i in range(N):
        verts_list.append(hex_vertices(positions[i][0], positions[i][1], angles[i]))
        
    for i in range(N):
        for j in range(i + 1, N):
            overlap, _, dist = polygons_overlap_sat(verts_list[i], verts_list[j])
            if overlap:
                return False, dist
    return True, 0.0

def construct():
    start_time = time.time()
    
    # Initial configuration: Honeycomb cluster
    # 1 center, 6 around, 5 more? 
    # Let's use a 4x3 grid initially as it's simple, then optimize.
    # Or better: a compact cluster.
    # 12 hexagons can fit in a roughly circular shape.
    
    # Let's try a specific known good layout:
    # 3 rows of 4? No, offset.
    # Row 0: 4 hexes
    # Row 1: 4 hexes (offset)
    # Row 2: 4 hexes
    # This is 12.
    
    positions = []
    angles = []
    
    # Generate grid
    for row in range(3):
        for col in range(4):
            x = col * 1.0 + (0.5 if row % 2 == 1 else 0.0) * 2.0 # Wait, hex width is 2.
            # For unit side, distance between centers in same row is 2? No, 2*sqrt(3)/2?
            # Side 1. Width (flat-to-flat) is sqrt(3). Height (point-to-point) is 2.
            # If flat top:
            # x dist = 1.5? No.
            # Let's use standard honeycomb coordinates.
            # Axial coords to Cartesian.
            # q, r
            # x = 1.5 * q
            # y = sqrt(3) * (r + q/2)
            
            pass
            
    # Let's just use a simple grid of centers with spacing 2.0 (touching)
    # and let the optimizer fix the rotations and slight overlaps if any.
    # Actually, for hexagons, if aligned, center distance 2.0 means they touch.
    
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    
    # 4x3 grid
    for i in range(N):
        row = i // 4
        col = i % 4
        pos[i, 0] = col * 2.0
        pos[i, 1] = row * np.sqrt(3)
        if row % 2 == 1:
            pos[i, 0] += 1.0 # Offset row
            
    # Center the grid
    pos -= pos.mean(axis=0)
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Optimization
    best_side = float('inf')
    best_pos = pos.copy()
    best_ang = ang.copy()
    best_outr = outer_angle
    
    # Local search with random restarts
    np.random.seed(42)
    
    max_iter = 5000
    step_size = 0.5
    decay = 0.99
    
    for restart in range(20):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # If not first restart, perturb from best
        if restart > 0:
            pos = best_pos + np.random.normal(0, 0.5, pos.shape)
            ang = best_ang + np.random.normal(0, 0.2, ang.shape)
            outer_angle = best_outr + np.random.normal(0, 0.1)
        else:
            outer_angle = 0.0
            step_size = 0.5
            
        current_side = compute_min_outer_side(pos, ang, outer_center, outer_angle)
        
        for it in range(max_iter):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            # Randomly pick a hexagon to move/rotate
            i = np.random.randint(N)
            
            old_pos = pos[i].copy()
            old_ang = ang[i]
            
            # Random perturbation
            pos[i] += np.random.normal(0, step_size, 2)
            ang[i] += np.random.normal(0, step_size * 0.5)
            
            # Also perturb outer angle slightly
            old_outr = outer_angle
            outer_angle += np.random.normal(0, step_size * 0.1)
            
            # Check validity (disjointness)
            valid, overlap_dist = check_validity(pos, ang)
            
            if valid:
                new_side = compute_min_outer_side(pos, ang, outer_center, outer_angle)
                # We want to minimize side length
                if new_side < current_side:
                    current_side = new_side
                    if new_side < best_side:
                        best_side = new_side
                        best_pos = pos.copy()
                        best_ang = ang.copy()
                        best_outr = outer_angle
                else:
                    # Reject move if it increases side length (greedy)
                    # Or accept with probability if it's close? Keep simple: greedy.
                    pos[i] = old_pos
                    ang[i] = old_ang
                    outer_angle = old_outr
            else:
                # Reject if overlap
                pos[i] = old_pos
                ang[i] = old_ang
                outer_angle = old_outr
                
            # Decay step size
            step_size *= decay
            
    # Final check
    if best_side == float('inf'):
        # Fallback to initial
        best_side = compute_min_outer_side(best_pos, best_ang, outer_center, best_outr)
        
    # Format output
    inner = []
    for i in range(N):
        inner.append([float(best_pos[i][0]), float(best_pos[i][1]), float(best_ang[i])])
        
    return inner, (float(outer_center[0]), float(outer_center[1])), float(best_side), float(best_outr)
