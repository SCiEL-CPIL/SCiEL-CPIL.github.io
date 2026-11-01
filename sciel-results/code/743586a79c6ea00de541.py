import numpy as np
import time

N = 50

def calculate_radius(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_inter = np.min(dists) / 2.0
    bound_dists = 0.5 - np.abs(points)
    min_bound = np.min(bound_dists)
    return min(min_inter, min_bound)

def hex_init():
    d = 0.15
    x_range = np.arange(-0.6, 0.6, d)
    y_range = np.arange(-0.6, 0.6, d * np.sqrt(3))
    points = []
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            y_offset = (d * np.sqrt(3) / 2) if (i % 2 == 1) else 0
            px, py = x, y + y_offset
            if -0.5 <= px <= 0.5 and -0.5 <= py <= 0.5:
                points.append([px, py])
    points = np.array(points)
    if len(points) == 0:
        return np.random.uniform(-0.5, 0.5, (N, 2))
    if len(points) < N:
        random_pts = np.random.uniform(-0.5, 0.5, (N - len(points), 2))
        points = np.vstack([points, random_pts])
    elif len(points) > N:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    return points

def sa_optimize(init_points, time_limit):
    points = init_points.copy()
    current_R = calculate_radius(points)
    best_points = points.copy()
    best_R = current_R
    T = 0.01
    T_min = 1e-6
    cooling_rate = 0.9995
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        i = np.random.randint(N)
        old_p = points[i].copy()
        step_size = 0.005 * (T / 0.01 + 0.5)
        points[i] += np.random.normal(0, step_size, 2)
        new_R = calculate_radius(points)
        
        if new_R > current_R:
            current_R = new_R
            if new_R > best_R:
                best_R = new_R
                best_points = points.copy()
        else:
            delta = new_R - current_R
            if T > 0:
                prob = np.exp(delta / T)
            else:
                prob = 0.0
            if np.random.rand() < prob:
                current_R = new_R
            else:
                points[i] = old_p
                
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_points, best_R

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    init1 = hex_init()
    pts1, R1 = sa_optimize(init1, time_limit=60)
    
    remaining = deadline - time.time()
    if remaining > 30:
        init2 = np.random.uniform(-0.5, 0.5, (N, 2))
        pts2, R2 = sa_optimize(init2, time_limit=remaining - 5)
        if R2 > R1:
            return pts2
    return pts1
