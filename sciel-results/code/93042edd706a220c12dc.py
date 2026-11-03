import numpy as np
from scipy.optimize import minimize
import time

N = 12

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    a = np.radians(angle_deg)
    angles = a + np.pi/6 + np.arange(6) * np.pi/3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _required_outer_side(inner_positions, inner_angles, outer_center, outer_rotation_deg):
    cx, cy = outer_center
    rot = np.radians(outer_rotation_deg)
    normals = []
    for k in range(6):
        angle = rot + k * np.pi / 3
        normals.append(np.array([np.cos(angle), np.sin(angle)]))
    
    max_extent = 0.0
    for pos, ang in zip(inner_positions, inner_angles):
        verts = _hex_vertices(pos[0], pos[1], ang)
        for n in normals:
            s1 = np.max(verts @ n) - (cx * n[0] + cy * n[1])
            s2 = np.max(verts @ (-n)) - (-cx * n[0] - cy * n[1])
            extent = max(s1, s2)
            if extent > max_extent:
                max_extent = extent
    return 2.0 * max_extent / np.sqrt(3)

def construct():
    deadline = time.time() + 140.0
    
    # Try a hexagonal grid arrangement: 1 center + 6 ring + 5 outer
    # Positions on triangular lattice with spacing 2 (touching hexagons)
    # Hex grid basis: e1 = (2, 0), e2 = (1, sqrt(3))
    e1 = np.array([2.0, 0.0])
    e2 = np.array([1.0, np.sqrt(3)])
    
    # Generate candidate positions on hex grid
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            pos = i * e1 + j * e2
            candidates.append(pos)
    candidates = np.array(candidates)
    
    # Initial arrangement: pick 12 positions close to center
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    init_idx = order[:12]
    init_pos = candidates[init_idx]
    init_angles = np.zeros(12)
    
    def objective(x):
        if time.time() > deadline:
            return 1e10
        pos = x[:24].reshape(12, 2)
        angles = x[24:36]
        ocx, ocy, orot = x[36], x[37], x[38]
        
        # Overlap penalty: use center distance < sqrt(3) as indicator
        penalty = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < 1.73:
                    penalty += (1.73 - d) ** 2 * 50
        side = _required_outer_side(pos, angles, (ocx, ocy), np.degrees(orot))
        return side + penalty
    
    x0 = np.concatenate([init_pos.flatten(), init_angles, [0.0, 0.0, 0.0]])
    
    best_x = x0.copy()
    best_val = objective(x0)
    
    rng = np.random.RandomState(123)
    for trial in range(30):
        if time.time() > deadline - 15:
            break
        start = x0 + rng.randn(*x0.shape) * 0.3
        try:
            res = minimize(objective, start, method='L-BFGS-B', options={'maxiter': 150, 'ftol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except:
            pass
    
    pos = best_x[:24].reshape(12, 2)
    angles_deg = np.degrees(best_x[24:36])
    ocx, ocy = best_x[36], best_x[37]
    orot_deg = np.degrees(best_x[38])
    
    side = _required_outer_side(pos, angles_deg, (ocx, ocy), orot_deg)
    
    # If side is unreasonably large, fallback
    if side > 15:
        pos = np.array([[4.0 * k, 0.0] for k in range(-6, 6)])
        angles_deg = np.zeros(12)
        ocx, ocy, orot_deg = 0.0, 0.0, 0.0
        side = _required_outer_side(pos, angles_deg, (ocx, ocy), orot_deg)
    
    inner = [[pos[i][0], pos[i][1], angles_deg[i]] for i in range(12)]
    return inner, (ocx, ocy), side, orot_deg

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
