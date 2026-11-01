import numpy as np
import time
from scipy.optimize import minimize

N = 11
DEADLINE = 145.0
START_TIME = time.time()

def hexagon_vertices(cx, cy, angle_deg, side=1.0):
    angle_rad = np.radians(angle_deg)
    vertices = []
    for k in range(6):
        a = angle_rad + k * np.pi / 3.0
        vertices.append((cx + side * np.cos(a), cy + side * np.sin(a)))
    return np.array(vertices)

def all_inner_vertices(positions, angles):
    vertices_list = []
    for i in range(len(positions)):
        cx, cy = positions[i]
        ang = angles[i]
        v = hexagon_vertices(cx, cy, ang, side=1.0)
        vertices_list.append(v)
    return np.vstack(vertices_list)

def enclosing_hexagon_side(verts, outer_angle_deg=0.0):
    theta = np.radians(outer_angle_deg)
    u = np.array([
        [np.cos(theta), np.sin(theta)],
        [np.cos(theta + np.pi/3), np.sin(theta + np.pi/3)],
        [np.cos(theta + 2*np.pi/3), np.sin(theta + 2*np.pi/3)]
    ])
    proj = verts @ u.T
    min_proj = proj.min(axis=0)
    max_proj = proj.max(axis=0)
    ideal_b = (min_proj + max_proj) / 2.0
    S_ideal = (max_proj - min_proj) / 2.0
    S_base = S_ideal.max()
    
    if abs(ideal_b.sum()) > 1e-8:
        def obj(b_vec):
            b0, b1 = b_vec
            b2 = -b0 - b1
            b = np.array([b0, b1, b2])
            return np.abs(proj - b[None, :]).max()
        
        res = minimize(obj, x0=[ideal_b[0], ideal_b[1]], method='Nelder-Mead',
                       options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
        S_val = res.fun
    else:
        S_val = S_base
    
    return S_val

def overlap_penalty(positions, angles, weight=1000.0):
    """Penalty for overlapping hexagons."""
    penalty = 0.0
    min_dist = np.sqrt(3)  # minimum center-to-center distance for non-overlap
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < min_dist:
                penalty += weight * (min_dist - dist)**2
    return penalty

def objective(params, outer_angle_deg=0.0):
    N_local = len(params) // 3
    positions = params.reshape(N_local, 3)[:, :2]
    angles = params.reshape(N_local, 3)[:, 2]
    verts = all_inner_vertices(positions, angles)
    S = enclosing_hexagon_side(verts, outer_angle_deg)
    pen = overlap_penalty(positions, angles)
    return S + pen

def construct():
    t0 = time.time()
    hex_dist = np.sqrt(3)
    rng = np.random.RandomState(42)
    
    configs = []
    
    # Config 1: hexagonal cluster (center + 6 ring + 4 more)
    pos1 = [(0, 0)]
    for k in range(6):
        a = k * np.pi / 3
        pos1.append((hex_dist * np.cos(a), hex_dist * np.sin(a)))
    for k in range(4):
        a = k * np.pi / 2
        pos1.append((1.8*hex_dist * np.cos(a), 1.8*hex_dist * np.sin(a)))
    pos1 = np.array(pos1)
    ang1 = np.zeros(N)
    configs.append(np.concatenate([pos1.flatten(), ang1]))
    
    # Config 2: two rows
    pos2 = []
    for i in range(6):
        pos2.append((i * hex_dist, 0))
    for i in range(5):
        pos2.append((i * hex_dist + hex_dist/2, hex_dist * np.sqrt(3)/2))
    pos2 = np.array(pos2[:N])
    ang2 = np.zeros(N)
    configs.append(np.concatenate([pos2.flatten(), ang2]))
    
    # Config 3: random
    pos3 = rng.uniform(-4, 4, size=(N, 2))
    ang3 = rng.uniform(0, 60, size=N)
    configs.append(np.concatenate([pos3.flatten(), ang3]))
    
    # Config 4: tight hex
    pos4 = [(0, 0)]
    for k in range(6):
        a = k * np.pi / 3
        pos4.append((hex_dist * np.cos(a), hex_dist * np.sin(a)))
    for k in range(4):
        a = (k + 0.5) * np.pi / 3
        pos4.append((1.5*hex_dist * np.cos(a), 1.5*hex_dist * np.sin(a)))
    pos4 = np.array(pos4)
    ang4 = np.zeros(N)
    configs.append(np.concatenate([pos4.flatten(), ang4]))
    
    best_params = None
    best_val = np.inf
    best_outer_angle = 0.0
    
    for cfg in configs:
        if time.time() - t0 > DEADLINE - 15:
            break
        
        for outer_ang in [0.0, 30.0, 15.0, 45.0]:
            if time.time() - t0 > DEADLINE - 15:
                break
            
            try:
                res = minimize(objective, cfg, args=(outer_ang,), method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8,
                                      'adaptive': True})
                if res.fun < best_val and res.fun < 100:  # sanity check
                    best_val = res.fun
                    best_params = res.x.copy()
                    best_outer_angle = outer_ang
            except Exception:
                pass
    
    # Polish
    if best_params is not None and time.time() - t0 < DEADLINE - 5:
        try:
            res = minimize(objective, best_params, args=(best_outer_angle,), method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-12,
                                  'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    params = best_params.reshape(N, 3)
    inner = [[float(params[i, 0]), float(params[i, 1]), float(params[i, 2])] for i in range(N)]
    
    theta = np.radians(best_outer_angle)
    u = np.array([
        [np.cos(theta), np.sin(theta)],
        [np.cos(theta + np.pi/3), np.sin(theta + np.pi/3)],
        [np.cos(theta + 2*np.pi/3), np.sin(theta + 2*np.pi/3)]
    ])
    
    positions = params[:, :2]
    angles = params[:, 2]
    verts = all_inner_vertices(positions, angles)
    proj = verts @ u.T
    min_proj = proj.min(axis=0)
    max_proj = proj.max(axis=0)
    ideal_b = (min_proj + max_proj) / 2.0
    
    c, _, _, _ = np.linalg.lstsq(u.T, ideal_b, rcond=None)
    cx, cy = float(c[0]), float(c[1])
    
    # Compute actual S (without penalty)
    S_actual = enclosing_hexagon_side(verts, best_outer_angle)
    
    return inner, (cx, cy), float(S_actual), float(best_outer_angle)

