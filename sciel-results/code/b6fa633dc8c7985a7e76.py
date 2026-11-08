import numpy as np
import time
import math
import random

N = 11

# EVOLVE-BLOCK-START

def hex_vertices(angle=0.0):
    """Return vertices of a unit regular hexagon centered at origin, rotated by angle."""
    return np.array([
        [math.cos(angle + k * math.pi / 3), math.sin(angle + k * math.pi / 3)]
        for k in range(6)
    ])

def minkowski_sum_inradius(v1, v2):
    """
    Compute the inradius of the Minkowski sum of two convex polygons v1 and v2.
    """
    sums = []
    for a in v1:
        for b in v2:
            sums.append(a + b)
    sums = np.array(sums)
    
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(sums)
        vertices = sums[hull.vertices]
    except Exception:
        return 0.0

    min_dist = float('inf')
    n = len(vertices)
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        cross = abs(p1[0] * p2[1] - p1[1] * p2[0])
        length = math.sqrt(dx*dx + dy*dy)
        if length > 1e-9:
            dist = cross / length
            if dist < min_dist:
                min_dist = dist
    return min_dist

# Precompute R_min table
ALPHA_STEPS = 360
R_MIN_TABLE = []
v1_base = hex_vertices(0.0)
for i in range(ALPHA_STEPS + 1):
    alpha = (i / ALPHA_STEPS) * (math.pi / 3)
    v2_rot = hex_vertices(-alpha)
    r_min = minkowski_sum_inradius(v1_base, v2_rot)
    R_MIN_TABLE.append(r_min)

def get_r_min(alpha):
    """Interpolate R_min for a given relative angle alpha."""
    alpha = alpha % (math.pi / 3)
    if alpha < 0:
        alpha += math.pi / 3
    idx = alpha / (math.pi / 3) * ALPHA_STEPS
    i0 = int(idx)
    i1 = min(i0 + 1, ALPHA_STEPS)
    frac = idx - i0
    return R_MIN_TABLE[i0] * (1 - frac) + R_MIN_TABLE[i1] * frac

def calculate_outer_side_length(positions, angles):
    """
    Calculate the minimum side length of a regular hexagon (centered at origin, 
    vertices at 0, 60, ... degrees) that contains all hexagons.
    """
    max_s = 0.0
    for pos, ang in zip(positions, angles):
        verts = pos + hex_vertices(ang)
        for v in verts:
            r = math.sqrt(v[0]**2 + v[1]**2)
            if r < 1e-9:
                continue
            phi = math.atan2(v[1], v[0])
            if phi < 0:
                phi += 2 * math.pi
            
            delta = (phi - math.pi / 6) % (math.pi / 3)
            if delta > math.pi / 6:
                delta = math.pi / 3 - delta
            
            s_val = r * math.cos(delta) / (math.sqrt(3) / 2)
            if s_val > max_s:
                max_s = s_val
    return max_s

def calculate_penalty(positions, angles):
    """
    Calculate penalty for overlapping inner hexagons.
    """
    penalty = 0.0
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            p1 = positions[i]
            p2 = positions[j]
            dist = np.linalg.norm(p1 - p2)
            rel_angle = angles[i] - angles[j]
            r_min = get_r_min(rel_angle)
            if dist < r_min:
                overlap = r_min - dist
                penalty += overlap * overlap
    return penalty

def objective(state):
    """
    state: array of size 3*N: [x1, y1, theta1, x2, y2, theta2, ...]
    """
    positions = []
    angles = []
    for i in range(N):
        x = state[3*i]
        y = state[3*i+1]
        th = state[3*i+2]
        positions.append(np.array([x, y]))
        angles.append(th)
    
    s_outer = calculate_outer_side_length(positions, angles)
    penalty = calculate_penalty(positions, angles)
    
    return s_outer + 100.0 * penalty

def simulated_annealing(initial_state, time_limit=140.0):
    """
    Perform Simulated Annealing.
    """
    current_state = np.array(initial_state)
    current_obj = objective(current_state)
    best_state = current_state.copy()
    best_obj = current_obj
    best_valid_state = None
    best_valid_obj = float('inf')
    
    # Check if initial is valid
    if calculate_penalty([np.array(state[3*i:3*i+2]) for i in range(N)], [state[3*i+2] for i in range(N)]) < 1e-6:
        best_valid_state = current_state.copy()
        best_valid_obj = current_obj

    start_time = time.time()
    t_end = start_time + time_limit
    
    T = 1.0
    T_min = 1e-6
    cooling_rate = 0.99
    
    n_moves = 0
    while time.time() < t_end:
        neighbor = current_state.copy()
        idx = random.randint(0, len(neighbor) - 1)
        if idx % 3 == 2: 
            step = random.uniform(-0.1, 0.1)
        else: 
            step = random.uniform(-0.5, 0.5)
        neighbor[idx] += step
        
        neighbor_obj = objective(neighbor)
        
        delta = neighbor_obj - current_obj
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_state = neighbor
            current_obj = neighbor_obj
            
            if current_obj < best_obj:
                best_state = current_state.copy()
                best_obj = current_obj
                
            # Check validity
            pos_list = [np.array(current_state[3*i:3*i+2]) for i in range(N)]
            ang_list = [current_state[3*i+2] for i in range(N)]
            pen = calculate_penalty(pos_list, ang_list)
            if pen < 1e-6:
                if current_obj < best_valid_obj:
                    best_valid_state = current_state.copy()
                    best_valid_obj = current_obj
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        n_moves += 1
        if n_moves % 1000 == 0:
            if time.time() > t_end:
                break
                
    # Return best valid state if found, else best state
    if best_valid_state is not None:
        return best_valid_state, best_valid_obj
    else:
        return best_state, best_obj

def get_initial_hex_lattice():
    positions = []
    angles = [0.0] * N
    
    positions.append([0.0, 0.0])
    
    for k in range(6):
        angle = k * math.pi / 3
        x = math.sqrt(3) * math.cos(angle)
        y = math.sqrt(3) * math.sin(angle)
        positions.append([x, y])
        
    for k in range(4):
        angle = k * math.pi / 3
        x = 2 * math.sqrt(3) * math.cos(angle)
        y = 2 * math.sqrt(3) * math.sin(angle)
        positions.append([x, y])
        
    state = []
    for p, a in zip(positions, angles):
        state.extend([p[0], p[1], a])
    return state

def get_initial_random():
    positions = []
    angles = []
    for _ in range(N):
        x = random.uniform(-5, 5)
        y = random.uniform(-5, 5)
        a = random.uniform(0, 2 * math.pi)
        positions.append([x, y])
        angles.append(a)
    state = []
    for p, a in zip(positions, angles):
        state.extend([p[0], p[1], a])
    return state

def construct():
    start_time = time.time()
    deadline = 140.0
    
    best_state = None
    best_score = float('inf')
    
    # 1. Hex lattice
    init1 = get_initial_hex_lattice()
    state1, score1 = simulated_annealing(init1, time_limit=deadline - (time.time() - start_time))
    if score1 < best_score:
        best_score = score1
        best_state = state1
        
    # 2. Random
    init2 = get_initial_random()
    state2, score2 = simulated_annealing(init2, time_limit=deadline - (time.time() - start_time))
    if score2 < best_score:
        best_score = score2
        best_state = state2
        
    # 3. Another random
    init3 = get_initial_random()
    state3, score3 = simulated_annealing(init3, time_limit=deadline - (time.time() - start_time))
    if score3 < best_score:
        best_score = score3
        best_state = state3

    positions = []
    angles = []
    for i in range(N):
        x = best_state[3*i]
        y = best_state[3*i+1]
        a = best_state[3*i+2]
        positions.append([x, y, a * 180.0 / math.pi])
        angles.append(a)
        
    final_s = calculate_outer_side_length(
        [np.array(p[:2]) for p in positions], 
        angles
    )
    
    return positions, (0.0, 0.0), final_s, 0.0

# EVOLVE-BLOCK-END
