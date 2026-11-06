# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 16
DEADLINE = 325.0

# Precompute points and distance matrix
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = N * N

# Distance squared matrix
D = np.zeros((NUM_POINTS, NUM_POINTS), dtype=np.int32)
for i in range(NUM_POINTS):
    xi, yi = POINTS[i]
    for j in range(i, NUM_POINTS):
        xj, yj = POINTS[j]
        dx = xi - xj
        dy = yi - yj
        d2 = dx * dx + dy * dy
        D[i, j] = d2
        D[j, i] = d2

def _can_add(p_idx, chosen_indices, D):
    n = len(chosen_indices)
    D_local = D
    p = p_idx
    for k in range(n):
        i = chosen_indices[k]
        d_pi = D_local[p, i]
        for m in range(k + 1, n):
            j = chosen_indices[m]
            d_pj = D_local[p, j]
            d_ij = D_local[i, j]
            if d_pi == d_pj or d_pi == d_ij or d_pj == d_ij:
                return False
    return True

def _greedy(order_indices, D):
    chosen_indices = []
    D_local = D
    for p_idx in order_indices:
        if _can_add(p_idx, chosen_indices, D_local):
            chosen_indices.append(p_idx)
    return chosen_indices

def construct():
    start_time = time.time()
    
    # 1. Deterministic row-major greedy
    base_order = list(range(NUM_POINTS))
    best_indices = _greedy(base_order, D)
    best_size = len(best_indices)
    
    # 2. Randomized restarts
    while time.time() - start_time < DEADLINE:
        order = base_order.copy()
        random.shuffle(order)
        
        current_indices = _greedy(order, D)
        current_size = len(current_indices)
        
        if current_size < best_size:
            best_size = current_size
            best_indices = current_indices
            
    return [POINTS[i] for i in best_indices]
# EVOLVE-BLOCK-END
