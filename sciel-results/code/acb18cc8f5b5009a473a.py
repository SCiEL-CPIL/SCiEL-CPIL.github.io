import numpy as np
import time
import random

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    sqrt3 = np.sqrt(3)
    min_dist = sqrt3  # Minimum distance between centers of aligned unit hexagons
    
    def calculate_penalty(positions, S):
        A = (S - 1.0) * sqrt3 / 2.0
        
        x = positions[:, 0]
        y = positions[:, 1]
        
        # Boundary Penalty
        v1 = np.maximum(0, y - A)
        v2 = np.maximum(0, -y - A)
        v3 = np.maximum(0, -sqrt3/2.0 * x + 0.5 * y - A)
        v4 = np.maximum(0, -sqrt3/2.0 * x - 0.5 * y - A)
        v5 = np.maximum(0, sqrt3/2.0 * x + 0.5 * y - A)
        v6 = np.maximum(0, sqrt3/2.0 * x - 0.5 * y - A)
        
        boundary_pen = np.sum(v1**2) + np.sum(v2**2) + np.sum(v3**2) + np.sum(v4**2) + np.sum(v5**2) + np.sum(v6**2)
        
        # Overlap Penalty
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        violations = np.maximum(0, min_dist - dists)
        overlap_pen = np.sum(violations**2)
        
        return boundary_pen + overlap_pen

    def random_positions(S_init):
        A_init = (S_init - 1.0) * sqrt3 / 2.0
        while True:
            x = random.uniform(-A_init, A_init)
            y = random.uniform(-A_init, A_init)
            if (abs(y) <= A_init and 
                abs(-sqrt3/2.0 * x + 0.5 * y) <= A_init and
                abs(-sqrt3/2.0 * x - 0.5 * y) <= A_init):
                return np.array([x, y])

    def sa_optimize(S, max_iters=2000):
        positions = np.array([random_positions(S + 0.5) for _ in range(N)])
        current_pen = calculate_penalty(positions, S)
        best_positions = positions.copy()
        best_pen = current_pen
        
        T = 1.0
        T_min = 1e-4
        cooling = 0.995
        
        for i in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            idx = random.randint(0, N-1)
            coord = random.randint(0, 1)
            delta = random.gauss(0, 0.1)
            
            new_positions = positions.copy()
            new_positions[idx, coord] += delta
            
            new_pen = calculate_penalty(new_positions, S)
            
            if new_pen < current_pen:
                positions = new_positions
                current_pen = new_pen
                if current_pen < best_pen:
                    best_pen = current_pen
                    best_positions = positions.copy()
            else:
                prob = np.exp(-(new_pen - current_pen) / T)
                if random.random() < prob:
                    positions = new_positions
                    current_pen = new_pen
            
            T *= cooling
            if T < T_min:
                break
        
        return best_positions, best_pen

    low_S = 3.5
    high_S = 4.5
    best_S = high_S
    best_positions = None
    best_pen = float('inf')
    
    # Binary search
    for step in range(20):
        if time.time() - start_time > time_limit - 10:
            break
            
        mid_S = (low_S + high_S) / 2.0
        
        pos, pen = sa_optimize(mid_S, max_iters=1500)
        
        if pen < best_pen:
            best_pen = pen
            best_S = mid_S
            best_positions = pos
        
        if pen < 1e-4:
            high_S = mid_S
        else:
            low_S = mid_S
            
        if time.time() - start_time > time_limit - 5:
            break

    if best_positions is None:
        S_fallback = 4.0
        pos, pen = sa_optimize(S_fallback, max_iters=5000)
        best_positions = pos
        best_S = S_fallback

    inner = []
    for i in range(N):
        inner.append([best_positions[i, 0], best_positions[i, 1], 0.0])
    
    return inner, (0.0, 0.0), best_S, 0.0
# EVOLVE-BLOCK-END
