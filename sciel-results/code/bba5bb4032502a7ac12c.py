import numpy as np
import time

N = 21
DEADLINE = time.time() + 110

def get_radii(centers, w):
    h = 2.0 - w
    rx = np.minimum(centers[:, 0], w - centers[:, 0])
    ry = np.minimum(centers[:, 1], h - centers[:, 1])
    r_wall = np.minimum(rx, ry)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r_pair = np.min(dist, axis=1) / 2.0
    
    r = np.minimum(r_wall, r_pair)
    return np.maximum(r, 0.0)

def score(centers, w):
    return np.sum(get_radii(centers, w))

def construct():
    start_time = time.time()
    best_score = -1
    best_centers = None
    best_w = 1.0
    
    grid_shapes = [(5, 5), (4, 6), (6, 4), (3, 7), (7, 3), (2, 11), (11, 2)]
    
    for rows, cols in grid_shapes:
        if time.time() > DEADLINE:
            break
        if rows * cols < N:
            continue
            
        x = np.linspace(0.5/cols, 1 - 0.5/cols, cols)
        y = np.linspace(0.5/rows, 1 - 0.5/rows, rows)
        centers_grid = np.array([(xi, yj) for yj in y for xi in x])
        centers_init = centers_grid[:N].copy()
        
        c = centers_init.copy()
        w = 1.0
        current_score = score(c, w)
        
        T = 0.05
        T_min = 0.001
        alpha = 0.999
        max_iter = 5000
        
        for i in range(max_iter):
            if time.time() > DEADLINE:
                break
            
            c_new = c.copy()
            w_new = w
            
            idx = np.random.randint(N)
            step_size = 0.1 * (T / 0.05)
            c_new[idx, 0] += np.random.uniform(-step_size, step_size)
            c_new[idx, 1] += np.random.uniform(-step_size, step_size)
            
            if np.random.random() < 0.1:
                w_new += np.random.uniform(-0.05, 0.05)
                w_new = np.clip(w_new, 0.2, 1.8)
            
            h_new = 2.0 - w_new
            c_new[:, 0] = np.clip(c_new[:, 0], 0, w_new)
            c_new[:, 1] = np.clip(c_new[:, 1], 0, h_new)
            
            new_score = score(c_new, w_new)
            delta = new_score - current_score
            if delta > 0 or (T > T_min and np.random.random() < np.exp(delta / T)):
                c = c_new
                w = w_new
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_centers = c.copy()
                    best_w = w
            
            T *= alpha
            if T < T_min:
                break
                
    if best_centers is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    r = get_radii(best_centers, best_w)
    return np.column_stack((best_centers, r))
