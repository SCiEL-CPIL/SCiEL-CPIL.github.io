import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initial positions: Rectangular grid of hexagons
    # dx = sqrt(3), dy = 1.5
    pos = []
    for i in range(4):
        for j in range(3):
            if len(pos) < N:
                x = i * np.sqrt(3)
                y = j * 1.5
                # Shift to center roughly
                pos.append([x - 1.5*np.sqrt(3), y - 1.5])
                
    # Ensure exactly N points
    while len(pos) < N:
        pos.append([0.0, 0.0])
    pos = np.array(pos[:N])
    
    # Normals and constants for outer hexagon (flat-top, S=1)
    normals = np.array([
        [0.0, 1.0],
        [0.0, -1.0],
        [np.sqrt(3), 1.0],
        [np.sqrt(3), -1.0],
        [-np.sqrt(3), 1.0],
        [-np.sqrt(3), -1.0]
    ])
    consts = np.array([
        np.sqrt(3)/2,
        np.sqrt(3)/2,
        np.sqrt(3),
        np.sqrt(3),
        np.sqrt(3),
        np.sqrt(3)
    ])
    
    # Vertices of a unit flat-top hexagon
    hex_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    def calc_S(pos):
        # pos: (N, 2)
        verts = pos[:, np.newaxis, :] + hex_verts[np.newaxis, :, :]
        max_s = 0.0
        for m in range(6):
            proj = verts @ normals[m] # (N, 6)
            s_vals = proj / consts[m]
            max_s = max(max_s, np.max(s_vals))
        return max_s

    def calc_overlap_penalty(pos):
        penalty = 0.0
        min_dist = np.sqrt(3)
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    penalty += (min_dist - d) ** 2
        return penalty

    def objective(pos_flat):
        pos = pos_flat.reshape(N, 2)
        S = calc_S(pos)
        pen = calc_overlap_penalty(pos)
        return S + 10.0 * pen

    best_pos = pos.copy()
    best_val = objective(best_pos.flatten())
    
    import random
    random.seed(42)
    np.random.seed(42)
    
    num_iter = 2000
    step_size = 0.2
    
    for it in range(num_iter):
        if time.time() > deadline:
            break
        
        # Random perturbation
        new_pos = best_pos + np.random.normal(0, step_size, (N, 2))
        
        val = objective(new_pos.flatten())
        
        if val < best_val:
            best_val = val
            best_pos = new_pos
        else:
            T = 1.0 / (1.0 + it/50.0)
            if random.random() < np.exp(-(val - best_val) / T):
                best_pos = new_pos
                best_val = val
        
        step_size *= 0.999

    final_S = calc_S(best_pos)
    
    inner = []
    for i in range(N):
        inner.append([best_pos[i][0], best_pos[i][1], 0.0])
        
    return inner, (0.0, 0.0), final_S, 0.0
# EVOLVE-BLOCK-END
