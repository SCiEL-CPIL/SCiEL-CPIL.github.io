import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def generate_start():
        spacing = 0.25
        k = 10
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0 - 0.01
        points = points[mask]
        
        if len(points) < N:
             spacing = 0.2
             k = 12
             rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                     for i in range(-k, k + 1) for j in range(-k, k + 1)]
             points = np.array(rows)
             radius = np.hypot(points[:, 0], points[:, 1])
             mask = radius <= 1.0 - spacing / 2.0 - 0.01
             points = points[mask]
             
        if len(points) < N:
             return np.random.uniform(-0.8, 0.8, (N, 2))
             
        idx = np.argsort(radius, kind="stable")[:N]
        return points[idx]

    C = generate_start()
    
    # Initialize Distance Matrix
    def update_dist_matrix(C):
        diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
        D = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(D, np.inf)
        return D

    D = update_dist_matrix(C)
    
    def calc_score_fast(C, D):
        # D is already updated for C
        min_dist = np.min(D)
        r_pair = min_dist / 2.0
        dists_center = np.hypot(C[:, 0], C[:, 1])
        if np.any(dists_center > 1.0):
            return 0.0
        r_bound = np.min(1.0 - dists_center)
        return min(r_bound, r_pair)
        
    current_score = calc_score_fast(C, D)
    best_score = current_score
    best_C = C.copy()
    best_D = D.copy()
    
    T = 0.002
    T_min = 1e-7
    cooling = 0.9995
    step_size = 0.005
    
    accept_count = 0
    total_count = 0
    window_size = 100
    
    while time.time() < deadline:
        if total_count > window_size:
            rate = accept_count / total_count
            if rate > 0.6:
                step_size *= 1.05
            elif rate < 0.2:
                step_size *= 0.95
            accept_count = 0
            total_count = 0
            
        i = np.random.randint(N)
        
        C_new = C.copy()
        C_new[i] += np.random.normal(0, step_size, 2)
        
        if np.hypot(C_new[i, 0], C_new[i, 1]) > 1.0:
            total_count += 1
            continue
            
        # Update D for the moved point
        D_new = D.copy()
        # Distances from i to all j
        diffs = C_new[i] - C_new
        dists_i = np.sqrt(np.sum(diffs**2, axis=1))
        D_new[i, :] = dists_i
        D_new[:, i] = dists_i
        D_new[i, i] = np.inf
        
        new_score = calc_score_fast(C_new, D_new)
        total_count += 1
        
        if new_score > current_score:
            C = C_new
            D = D_new
            current_score = new_score
            accept_count += 1
            if current_score > best_score:
                best_score = current_score
                best_C = C.copy()
                best_D = D.copy()
        else:
            delta = new_score - current_score
            if delta < 0:
                if T > 0:
                    prob = np.exp(delta / T)
                    if np.random.rand() < prob:
                        C = C_new
                        D = D_new
                        current_score = new_score
                        accept_count += 1
        
        T *= cooling
        if T < T_min:
            T = T_min

    return best_C
