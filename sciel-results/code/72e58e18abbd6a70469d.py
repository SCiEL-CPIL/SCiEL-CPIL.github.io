import numpy as np
import time

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

def _compute_required_side(centers, angles, outer_angle=0.0):
    """
    Compute the minimum side length of the outer hexagon (centered at 0,0) 
    to contain all inner hexagons, given the outer angle.
    """
    max_proj = 0.0
    # Outer hexagon normals
    theta_o = np.deg2rad(outer_angle)
    normals = []
    for k in range(6):
        angle = theta_o + k * np.pi / 3
        normals.append(np.array([np.cos(angle), np.sin(angle)]))
    
    # Precompute inner vertices
    all_verts = []
    for i in range(N):
        theta_i = np.deg2rad(angles[i])
        verts_i = np.array([
            [centers[i][0] + np.cos(theta_i + m * np.pi/3), 
             centers[i][1] + np.sin(theta_i + m * np.pi/3)]
            for m in range(6)
        ])
        all_verts.append(verts_i)
        
    # Project all vertices onto all outer normals
    for normal in normals:
        for verts in all_verts:
            projs = np.dot(verts, normal)
            max_proj = max(max_proj, np.max(projs))
            
    # Apothem = max_proj
    # Side = Apothem / (sqrt(3)/2)
    return max_proj * 2.0 / np.sqrt(3.0)

def _simulate(initial_positions, initial_angles, target_side, outer_angle, steps, dt):
    """
    Simulate the hexagons using symplectic Euler integration.
    Forces:
    1. Repulsion between inner hexagons (based on min distance between edges).
    2. Confinement to a hexagon of side `target_side`.
    3. Damping.
    """
    # State: positions (N, 2), velocities (N, 2), angles (N,), angular_vels (N,)
    pos = initial_positions.copy()
    vel = np.zeros_like(pos)
    ang = initial_angles.copy()
    ang_vel = np.zeros_like(ang)
    
    # Precompute edge normals for local frame (angle=0)
    # Edge k is perpendicular to direction (k * 60 degrees)
    # Normal points outward
    local_normals = np.array([
        [np.cos(k * np.pi / 3), np.sin(k * np.pi / 3)] for k in range(6)
    ])
    
    # Force parameters
    k_repel = 50.0
    k_contain = 100.0
    damping = 0.8
    
    for step in range(steps):
        # Check time limit
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        # Compute forces
        force = np.zeros_like(pos)
        torque = np.zeros_like(ang)
        
        # 1. Repulsion between inner hexagons
        for i in range(N):
            for j in range(i + 1, N):
                # Get vertices
                theta_i = np.deg2rad(ang[i])
                theta_j = np.deg2rad(ang[j])
                
                verts_i = pos[i] + np.array([
                    [np.cos(theta_i + k * np.pi/3), np.sin(theta_i + k * np.pi/3)] for k in range(6)
                ])
                verts_j = pos[j] + np.array([
                    [np.cos(theta_j + k * np.pi/3), np.sin(theta_j + k * np.pi/3)] for k in range(6)
                ])
                
                # Compute min distance between edges
                # Approximation: check all pairs of edges? Too slow.
                # Better: Check vertices of i against edges of j and vice versa.
                min_dist = np.inf
                min_vec = np.zeros(2)
                
                # Vertices of i to edges of j
                for v_i in verts_i:
                    for k in range(6):
                        p1 = verts_j[k]
                        p2 = verts_j[(k+1)%6]
                        # Distance from point v_i to segment p1-p2
                        d_vec = p2 - p1
                        d_len_sq = np.dot(d_vec, d_vec)
                        if d_len_sq < 1e-9: continue
                        t = np.dot(v_i - p1, d_vec) / d_len_sq
                        t = np.clip(t, 0.0, 1.0)
                        closest = p1 + t * d_vec
                        dist = np.linalg.norm(v_i - closest)
                        if dist < min_dist:
                            min_dist = dist
                            min_vec = v_i - closest
                            
                # Vertices of j to edges of i
                for v_j in verts_j:
                    for k in range(6):
                        p1 = verts_i[k]
                        p2 = verts_i[(k+1)%6]
                        d_vec = p2 - p1
                        d_len_sq = np.dot(d_vec, d_vec)
                        if d_len_sq < 1e-9: continue
                        t = np.dot(v_j - p1, d_vec) / d_len_sq
                        t = np.clip(t, 0.0, 1.0)
                        closest = p1 + t * d_vec
                        dist = np.linalg.norm(v_j - closest)
                        if dist < min_dist:
                            min_dist = dist
                            min_vec = v_j - closest
                            
                # If overlapping (min_dist < 0 is not possible with this metric, but if very small)
                # We want to push them apart if min_dist < 0 (overlap) or even if < epsilon
                # Actually, for convex polygons, min_dist is 0 if touching, >0 if separate.
                # If they overlap, this metric fails to give negative distance.
                # We need a better overlap measure or just use center distance if close?
                # Let's use a potential that activates when min_dist < 0.1
                # But min_dist is always >= 0.
                # We need to detect overlap.
                # SAT is robust but expensive.
                # Let's assume if min_dist < 0.01, they are overlapping or touching.
                # To handle overlap, we can use the center-to-center vector if min_dist is very small.
                # Or better: use the SDF approach for force calculation?
                # Let's stick to min_dist for now, but add a center repulsion if very close.
                
                # Force is proportional to overlap depth (negative min_dist if we had it)
                # Since min_dist >= 0, we use a spring that pushes apart if min_dist < threshold
                threshold = 0.01
                if min_dist < threshold:
                    # Push apart
                    # Direction: min_vec is from closest point on j to v_i (or similar)
                    # We need direction from j to i
                    # If min_vec was from v_i to closest on j, it points i->j? No, v_i - closest.
                    # So v_i - closest points from j to i.
                    # So force on i is + min_vec, force on j is - min_vec.
                    # But if min_dist is 0 (touching), min_vec might be 0 or ambiguous.
                    # If centers are closer than sqrt(3), they might overlap.
                    
                    # Let's use center vector if min_dist is very small
                    if min_dist < 1e-3:
                        c_vec = pos[i] - pos[j]
                        c_norm = np.linalg.norm(c_vec)
                        if c_norm < 1e-6:
                            c_vec = np.array([np.random.rand()-0.5, np.random.rand()-0.5])
                            c_norm = np.linalg.norm(c_vec)
                        dir_vec = c_vec / c_norm
                    else:
                        dir_vec = min_vec / np.linalg.norm(min_vec)
                        
                    overlap_depth = threshold - min_dist
                    f_mag = k_repel * overlap_depth
                    force[i] += f_mag * dir_vec
                    force[j] -= f_mag * dir_vec
                    
                    # Torque? For now, ignore torque to keep it simple.
                    # Rotation might help but adds complexity.
        
        # 2. Confinement
        # Outer hexagon centered at 0, side target_side
        apothem = target_side * np.sqrt(3) / 2
        theta_o = np.deg2rad(outer_angle)
        
        for i in range(N):
            # Get vertices of i
            theta_i = np.deg2rad(ang[i])
            verts_i = pos[i] + np.array([
                [np.cos(theta_i + k * np.pi/3), np.sin(theta_i + k * np.pi/3)] for k in range(6)
            ])
            
            # Check if any vertex is outside
            # SDF of vertex to outer hexagon
            # Outer hexagon normals in world frame
            for k in range(6):
                angle_k = theta_o + k * np.pi / 3
                n = np.array([np.cos(angle_k), np.sin(angle_k)])
                
                # Project all vertices onto this normal
                projs = np.dot(verts_i, n)
                max_proj = np.max(projs)
                
                # Distance to boundary
                dist = max_proj - apothem
                if dist > 0:
                    # Push back in
                    # Force direction is -n
                    f_mag = k_contain * dist
                    force[i] -= f_mag * n
    
        # Integrate
        vel += force * dt
        vel *= damping
        pos += vel * dt
        
        ang_vel += torque * dt
        ang_vel *= damping
        ang += ang_vel * dt
        
        # Normalize angles
        ang = np.mod(ang, 360.0)
        
    return pos, ang

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

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_params = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Try a few outer angles
    outer_angles = [0.0, 30.0]
    
    # Initial guess: Hexagonal packing
    # Center at 0,0
    # Ring 1: 6 hexagons at distance sqrt(3)
    # Ring 2: 6 hexagons at distance 2*sqrt(3) (but only 5 needed? No, 1+6+5=12)
    # Let's use a known good packing: 1 in center, 6 around it, 5 in the gaps of the next ring?
    # Actually, 1+6+12 is too many.
    # 12 hexagons: Center (1), Ring 1 (6), Ring 2 (5).
    
    initial_positions = np.zeros((N, 2))
    initial_angles = np.zeros(N)
    
    # Center
    initial_positions[0] = [0, 0]
    
    # Ring 1
    for k in range(1, 7):
        angle = np.pi/3 * (k-1)
        initial_positions[k] = [np.sqrt(3)*np.cos(angle), np.sqrt(3)*np.sin(angle)]
        
    # Ring 2 (5 hexagons)
    # Place them at distance 2*sqrt(3) + small offset?
    # Or just start with a random valid-ish configuration and let simulation fix it.
    # Let's start with a larger ring and shrink.
    for k in range(7, 12):
        angle = np.pi/3 * (k-7) + np.pi/6 # Offset
        initial_positions[k] = [3.5*np.cos(angle), 3.5*np.sin(angle)]
        
    for outer_angle in outer_angles:
        if time.time() - start_time > TIME_LIMIT - 20:
            break
            
        # Start with a large target side and shrink
        current_side = 10.0
        pos = initial_positions.copy()
        ang = initial_angles.copy()
        
        # Shrink phase
        while current_side > 3.0 and time.time() - start_time < TIME_LIMIT - 10:
            # Simulate for a few steps
            pos, ang = _simulate(pos, ang, current_side, outer_angle, steps=200, dt=0.01)
            
            # Check validity
            inner_hexes = [[pos[i][0], pos[i][1], ang[i]] for i in range(N)]
            side_needed = _compute_required_side(pos, ang, outer_angle)
            
            is_valid, reason = _check_validity(inner_hexes, side_needed, (0.0, 0.0), outer_angle)
            
            if is_valid:
                if side_needed < best_side:
                    best_side = side_needed
                    best_params = inner_hexes
                    best_outer_center = (0.0, 0.0)
                    best_outer_angle = outer_angle
                    
            # Shrink target side
            # If we are valid, we can shrink more aggressively
            if is_valid:
                current_side = max(3.0, side_needed * 0.95)
            else:
                # If invalid, keep target side or increase slightly
                current_side = max(3.0, side_needed * 1.05)
                
    # Polish phase
    if best_params is not None:
        # Use the best found configuration as start for a final simulation with very small target
        # Or just return the best found.
        # Let's try a few more simulations with the best config as start
        for _ in range(5):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            pos = np.array([h[:2] for h in best_params])
            ang = np.array([h[2] for h in best_params])
            
            # Try to tighten further
            target = best_side * 0.98
            pos, ang = _simulate(pos, ang, target, best_outer_angle, steps=500, dt=0.005)
            
            inner_hexes = [[pos[i][0], pos[i][1], ang[i]] for i in range(N)]
            side_needed = _compute_required_side(pos, ang, best_outer_angle)
            
            is_valid, reason = _check_validity(inner_hexes, side_needed, (0.0, 0.0), best_outer_angle)
            
            if is_valid and side_needed < best_side:
                best_side = side_needed
                best_params = inner_hexes
                best_outer_center = (0.0, 0.0)
                
    if best_params is not None:
        return (best_params, best_outer_center, best_side, best_outer_angle)
    
    # Fallback
    return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

