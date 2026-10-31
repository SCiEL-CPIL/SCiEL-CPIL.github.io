import numpy as np
import time

N = 11
H = np.sqrt(3) / 2

# Precompute triangle indices for vectorization
idx_i = []
idx_j = []
idx_k = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            idx_i.append(i)
            idx_j.append(j)
            idx_k.append(k)
idx_i = np.array(idx_i)
idx_j = np.array(idx_j)
idx_k = np.array(idx_k)

def _bary_to_cart(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    x = u + 0.5 * v
    y = H * v
    return np.column_stack((x, y))

def _min_triangle_area_vec(points):
    p1 = points[idx_i]
    p2 = points[idx_j]
    p3 = points[idx_k]
    dx1 = p2[:, 0] - p1[:, 0]
    dy1 = p2[:, 1] - p1[:, 1]
    dx2 = p3[:, 0] - p1[:, 0]
    dy2 = p3[:, 1] - p1[:, 1]
    areas = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
    return np.min(areas)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_bary = None
    best_score = 0.0
    
    rng = np.random.default_rng(42)
    
    def gen_valid_bary(n_points):
        b = rng.random((n_points, 2))
        mask = b[:, 0] + b[:, 1] > 1
        b[mask, 0] = 1 - b[mask, 0]
        b[mask, 1] = 1 - b[mask, 1]
        return b

    # Initial structured guess
    init_bary = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
        [1/3, 1/3]
    ])
    rand_pts = gen_valid_bary(4)
    init_bary = np.vstack([init_bary, rand_pts])
    
    def run_sa(initial_bary, time_limit):
        current_bary = initial_bary.copy()
        current_score = _min_triangle_area_vec(_bary_to_cart(current_bary))
        best_local_bary = current_bary.copy()
        best_local_score = current_score
        
        temp = 0.005
        cooling = 0.995
        noise_std = 0.02
        
        local_start = time.time()
        
        while time.time() - local_start < time_limit:
            noise = rng.normal(0, noise_std, size=current_bary.shape)
            new_bary = current_bary + noise
            new_bary = np.clip(new_bary, 0, 1)
            sum_uv = new_bary[:, 0] + new_bary[:, 1]
            mask = sum_uv > 1
            if np.any(mask):
                scale = np.where(sum_uv > 1, 1 / sum_uv, 1)
                new_bary *= scale[:, np.newaxis]
            
            new_score = _min_triangle_area_vec(_bary_to_cart(new_bary))
            
            if new_score > current_score:
                current_bary = new_bary
                current_score = new_score
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_bary = current_bary.copy()
            else:
                delta = current_score - new_score
                if delta < 0:
                    accept = True
                else:
                    if temp > 1e-6:
                        prob = np.exp(-delta / temp)
                    else:
                        prob = 0
                    accept = rng.random() < prob
                    
                if accept:
                    current_bary = new_bary
                    current_score = new_score
            
            temp *= cooling
            
        return best_local_bary, best_local_score

    # Run 1
    b, s = run_sa(init_bary, time_limit=30)
    if s > best_score:
        best_score = s
        best_bary = b
        
    # Run 2-5
    for i in range(4):
        if time.time() > deadline - 10:
            break
        if best_bary is None:
            break
        perturbed = best_bary + rng.normal(0, 0.1, size=(N, 2))
        perturbed = np.clip(perturbed, 0, 1)
        sum_uv = perturbed[:, 0] + perturbed[:, 1]
        mask = sum_uv > 1
        if np.any(mask):
            scale = np.where(sum_uv > 1, 1 / sum_uv, 1)
            perturbed *= scale[:, np.newaxis]
            
        b, s = run_sa(perturbed, time_limit=15)
        if s > best_score:
            best_score = s
            best_bary = b

    final_points = _bary_to_cart(best_bary)
    return final_points
