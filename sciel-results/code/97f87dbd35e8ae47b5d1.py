import numpy as np
import time

N = 11
MAX_TIME = 140.0

def get_hex_verts(cx, cy, angle_rad):
    angles = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack((cx + np.cos(angles), cy + np.sin(angles)))

def point_in_hexagon_violation(x, y, S):
    apothem = S * np.sqrt(3) / 2.0
    alphas = np.arange(6) * np.pi / 3.0
    projs = x * np.cos(alphas) + y * np.sin(alphas)
    return np.max(projs) - apothem

def sat_overlap_depth(v1, v2):
    axes = []
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i+1)%6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            axes.append(normal / norm)
    for i in range(6):
        p1 = v2[i]
        p2 = v2[(i+1)%6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            axes.append(normal / norm)
    
    min_overlap = float('inf')
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            return 0.0
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
            
    return min_overlap

def calculate_penalty(state, S):
    verts = []
    penalty = 0.0
    for i in range(N):
        x = state[i*3]
        y = state[i*3+1]
        a = state[i*3+2]
        v = get_hex_verts(x, y, a)
        verts.append(v)
        
        for p in v:
            d = point_in_hexagon_violation(p[0], p[1], S)
            if d > 0:
                penalty += d * d
    
    for i in range(N):
        for j in range(i+1, N):
            pen = sat_overlap_depth(verts[i], verts[j])
            if pen > 0:
                penalty += pen * pen
                
    return penalty

def construct():
    start_time = time.time()
    np.random.seed(42)
    
    # Initial State
    current_S = 4.5
    state = np.zeros(N * 3)
    for i in range(N):
        angle = i * 2 * np.pi / N
        r = 1.0
        state[i*3] = r * np.cos(angle)
        state[i*3+1] = r * np.sin(angle)
        state[i*3+2] = 0.0
        
    # SA to find feasible config
    T = 1.0
    T_min = 0.01
    alpha = 0.99
    step_size = 0.5
    
    best_state = state.copy()
    best_pen = calculate_penalty(best_state, current_S)
    
    while time.time() - start_time < MAX_TIME - 20.0:
        new_state = best_state.copy()
        idx = np.random.randint(0, N*3)
        if idx % 3 == 2:
            new_state[idx] += np.random.normal(0, step_size * 0.5)
        else:
            new_state[idx] += np.random.normal(0, step_size)
            
        new_pen = calculate_penalty(new_state, current_S)
        
        if new_pen < best_pen or np.random.random() < np.exp(-(new_pen - best_pen) / T):
            best_state = new_state
            best_pen = new_pen
            
        T *= alpha
        if T < T_min:
            T = T_min
            
    min_S = current_S
    min_state = best_state.copy()
    
    # Shrink Loop
    for _ in range(30):
        if time.time() - start_time > MAX_TIME - 5.0:
            break
            
        new_S = min_S - 0.05
        if new_S < 3.0: break
        
        temp_state = min_state.copy()
        T = 0.5
        # Short SA to adapt to smaller S
        for _ in range(1000):
            if time.time() - start_time > MAX_TIME - 5.0: break
            new_state = temp_state.copy()
            idx = np.random.randint(0, N*3)
            if idx % 3 == 2:
                new_state[idx] += np.random.normal(0, 0.2)
            else:
                new_state[idx] += np.random.normal(0, 0.2)
                
            new_pen = calculate_penalty(new_state, new_S)
            old_pen = calculate_penalty(temp_state, new_S)
            
            if new_pen < old_pen or np.random.random() < np.exp(-(new_pen - old_pen) / T):
                temp_state = new_state
                 
            T *= 0.95
             
        final_pen = calculate_penalty(temp_state, new_S)
        if final_pen < 1e-6:
            min_S = new_S
            min_state = temp_state
        else:
            break
            
    return [[min_state[i*3], min_state[i*3+1], np.degrees(min_state[i*3+2])] for i in range(N)], (0.0, 0.0), min_S, 0.0
