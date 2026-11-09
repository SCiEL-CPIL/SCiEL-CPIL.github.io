# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    ang = a + np.deg2rad(60 * np.arange(6))
    return np.stack([cx + s * np.cos(ang), cy + s * np.sin(ang)], axis=1)


def _is_point_in_hexagon(p, ox, oy, side, oa_deg):
    a = np.deg2rad(oa_deg)
    cos_a = np.cos(a)
    sin_a = np.sin(a)
    dx = p[0] - ox
    dy = p[1] - oy
    x_local = dx * cos_a + dy * sin_a
    y_local = -dx * sin_a + dy * cos_a
    
    verts = _hex_vertices(0, 0, side, 0)
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i + 1) % 6]
        v = p2 - p1
        w = np.array([x_local, y_local]) - p1
        cross = v[0] * w[1] - v[1] * w[0]
        if cross < -1e-9:
            return False
    return True


def _valid(inner_data, outer_data, side):
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if side < np.sqrt(11) - 1e-12:
        return False
    
    ox, oy, oa = outer_data
    
    # Check separation (assuming angles are 0)
    for i in range(len(inner_data)):
        for j in range(i + 1, len(inner_data)):
            dist = np.linalg.norm(inner_data[i][:2] - inner_data[j][:2])
            if dist < np.sqrt(3) - 1e-6:
                return False
            
    # Check containment
    for i in range(len(inner_data)):
        x, y, a = inner_data[i]
        v = _hex_vertices(x, y, 1.0, a)
        for p in v:
            if not _is_point_in_hexagon(p, ox, oy, side, oa):
                return False
                
    return True


def _simulate(inner_init, outer_side, outer_angle, steps=300, dt=0.005, damping=0.85):
    n = len(inner_init)
    pos = inner_init[:, :2].copy()
    vel = np.zeros((n, 2))
    
    inner_angle = 0.0
    min_dist = np.sqrt(3)
    k_rep = 50.0
    k_cont = 100.0
    ox, oy = 0.0, 0.0
    
    for step in range(steps):
        acc = np.zeros((n, 2))
        
        for i in range(n):
            for j in range(i + 1, n):
                d_vec = pos[i] - pos[j]
                d = np.linalg.norm(d_vec)
                if d < min_dist:
                    if d < 1e-6:
                        d_vec = np.array([1e-6, 0])
                        d = 1e-6
                    force = k_rep * (min_dist - d)
                    acc[i] += force * d_vec / d
                    acc[j] -= force * d_vec / d
        
        for i in range(n):
            x, y = pos[i]
            v = _hex_vertices(x, y, 1.0, inner_angle)
            for p in v:
                if not _is_point_in_hexagon(p, ox, oy, outer_side, outer_angle):
                    vec_to_center = np.array([ox - p[0], oy - p[1]])
                    dist_to_center = np.linalg.norm(vec_to_center)
                    if dist_to_center > 1e-6:
                        force = k_cont * vec_to_center / dist_to_center
                        acc[i] += force
        
        vel = vel * damping + acc * dt
        pos = pos + vel * dt
        
        # Hard constraint projection
        inradius = outer_side * np.sqrt(3) / 2
        max_r = inradius - np.sqrt(3)/2
        for i in range(n):
            r = np.linalg.norm(pos[i])
            if r > max_r:
                pos[i] = pos[i] * (max_r / r)
                vel[i] = 0.0
                
    inner_data = inner_init.copy()
    inner_data[:, :2] = pos
    return inner_data


def hexagon_packing_11():
    t0 = time.time()
    deadline = t0 + 620.0
    
    best_side = float('inf')
    best_inner = None
    best_outer = None
    
    def get_initial_config():
        init = []
        init.append([0.0, 0.0, 0.0])
        for k in range(6):
            ang = np.deg2rad(60 * k)
            x = np.sqrt(3) * np.cos(ang)
            y = np.sqrt(3) * np.sin(ang)
            init.append([x, y, 0.0])
        for k in [0, 2, 3, 5]:
            ang = np.deg2rad(60 * k + 30)
            x = 2 * np.sqrt(3) * np.cos(ang)
            y = 2 * np.sqrt(3) * np.sin(ang)
            init.append([x, y, 0.0])
        return np.array(init)

    init_config = get_initial_config()
    
    current_side = 6.0
    last_valid_side = 6.0
    
    while time.time() < deadline - 10:
        inner_sim = _simulate(init_config, current_side, 0.0)
        
        outer = np.array([0.0, 0.0, 0.0])
        if _valid(inner_sim, outer, current_side):
            if current_side < best_side:
                best_side = current_side
                best_inner = inner_sim.copy()
                best_outer = outer.copy()
            last_valid_side = current_side
            current_side -= 0.02
            if current_side < np.sqrt(11) + 0.2:
                current_side -= 0.005
        else:
            # If invalid, go back to last valid side and try to recover
            # or just increase side
            current_side = last_valid_side + 0.05
            # To avoid infinite loop, if we can't shrink, we stop
            # Actually, if we can't shrink from last_valid_side, we should stop.
            # Let's add a counter for failures
            # For simplicity, just break if we fail to shrink from a known good side
            # But we might just need more simulation steps.
            # Let's just increase side to ensure we have a valid state to start from
            # and try again with slightly larger side.
            # If current_side was already last_valid_side + 0.05, we are stuck.
            # Let's just return best found.
            break

    if best_inner is not None:
        return best_inner, best_outer, best_side
    
    # Fallback
    inner = init_config
    outer = np.array([0.0, 0.0, 0.0])
    side = 6.0
    return inner, outer, side

# EVOLVE-BLOCK-END
