import numpy as np
import time
import random

def construct():
    start_time = time.time()
    limit = 145.0  # Safety margin before 150s kill
    
    n = 256
    
    # Precompute all 4-term arithmetic progressions
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
            
    # Precompute neighbors for fast delta calculation
    neighbors = [[] for _ in range(n)]
    for ap in aps:
        i1, i2, i3, i4 = ap
        neighbors[i1].append((i2, i3, i4))
        neighbors[i2].append((i1, i3, i4))
        neighbors[i3].append((i1, i2, i4))
        neighbors[i4].append((i1, i2, i3))
        
    # Convert neighbors to numpy arrays
    neighbors_arr = []
    for i in range(n):
        if neighbors[i]:
            neighbors_arr.append(np.array(neighbors[i], dtype=np.int32))
        else:
            neighbors_arr.append(np.array([], dtype=np.int32).reshape(0, 3))

    def count_mono(coloring):
        count = 0
        for i1, i2, i3, i4 in aps:
            if coloring[i1] == coloring[i2] == coloring[i3] == coloring[i4]:
                count += 1
        return count

    def get_delta_single(p, coloring):
        if len(neighbors[p]) == 0:
            return 0
        current_bit = coloring[p]
        delta = 0
        for o1, o2, o3 in neighbors[p]:
            c1 = coloring[o1]
            c2 = coloring[o2]
            c3 = coloring[o3]
            if c1 == c2 == c3:
                if c1 == current_bit:
                    delta -= 1
                else:
                    delta += 1
        return delta

    def get_all_deltas(coloring):
        deltas = np.zeros(n, dtype=np.int32)
        for p in range(n):
            if len(neighbors[p]) == 0:
                continue
            others = neighbors_arr[p]
            c_others = coloring[others]
            others_equal = (c_others[:, 0] == c_others[:, 1]) & (c_others[:, 1] == c_others[:, 2])
            if not others_equal.any():
                continue
            c_p = coloring[p]
            contrib = np.where(others_equal, np.where(c_others[:, 0] == c_p, -1, 1), 0)
            deltas[p] = contrib.sum()
        return deltas

    best_coloring = None
    best_score = float('inf')
    
    # Restart 1: Random
    coloring = np.random.randint(0, 2, n).astype(np.int32)
    
    # Greedy Phase
    current_score = count_mono(coloring)
    improved = True
    while improved and time.time() < limit - 5:
        improved = False
        deltas = get_all_deltas(coloring)
        min_delta = np.min(deltas)
        if min_delta < 0:
            idx = np.argmin(deltas)
            coloring[idx] = 1 - coloring[idx]
            current_score += min_delta
            improved = True
            
    if current_score < best_score:
        best_score = current_score
        best_coloring = coloring.copy()
        
    # SA Phase
    T = 0.5
    T_min = 0.01
    cooling_rate = 0.9999
    sa_steps = 10000 # Increased steps since it's fast
    
    for _ in range(sa_steps):
        if time.time() > limit - 5:
            break
        p = random.randint(0, n - 1)
        delta = get_delta_single(p, coloring)
        if delta < 0 or random.random() < np.exp(-delta / T):
            coloring[p] = 1 - coloring[p]
            current_score += delta
            if current_score < best_score:
                best_score = current_score
                best_coloring = coloring.copy()
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Subsequent Restarts: Perturb Best
    while time.time() < limit - 5:
        # Perturb best coloring
        coloring = best_coloring.copy()
        # Flip 10-30 random bits
        num_flips = random.randint(10, 30)
        indices = np.random.choice(n, num_flips, replace=False)
        coloring[indices] = 1 - coloring[indices]
        
        current_score = count_mono(coloring)
        
        # Greedy Phase
        improved = True
        while improved and time.time() < limit - 5:
            improved = False
            deltas = get_all_deltas(coloring)
            min_delta = np.min(deltas)
            if min_delta < 0:
                idx = np.argmin(deltas)
                coloring[idx] = 1 - coloring[idx]
                current_score += min_delta
                improved = True
                
        if current_score < best_score:
            best_score = current_score
            best_coloring = coloring.copy()
            
        # SA Phase (Shorter, to allow more restarts)
        T = 0.2
        T_min = 0.01
        cooling_rate = 0.9999
        sa_steps = 2000
        
        for _ in range(sa_steps):
            if time.time() > limit - 5:
                break
            p = random.randint(0, n - 1)
            delta = get_delta_single(p, coloring)
            if delta < 0 or random.random() < np.exp(-delta / T):
                coloring[p] = 1 - coloring[p]
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_coloring = coloring.copy()
            T *= cooling_rate
            if T < T_min:
                T = T_min

    if best_coloring is None:
        return [int(x) for x in np.random.randint(0, 2, n)]
    return [int(x) for x in best_coloring]
