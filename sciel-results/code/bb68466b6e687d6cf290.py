import numpy as np
import time


def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    deadline = time.time() + 140.0  # 10s safety margin
    
    # Helper: compute cost of permutation
    def compute_cost(pos):
        return int(np.sum(a * b[np.ix_(pos, pos)]))
    
    # Helper: compute delta for swapping positions i and j
    def swap_delta(pos, i, j):
        x = pos[i]
        y = pos[j]
        # b[y, pos] - b[x, pos]: vector of length n
        diff1 = b[y, pos] - b[x, pos]
        diff2 = b[x, pos] - b[y, pos]
        diff3 = b[pos, y] - b[pos, x]
        diff4 = b[pos, x] - b[pos, y]
        return (a[i, :] @ diff1 + a[j, :] @ diff2 + a[:, i] @ diff3 + a[:, j] @ diff4)
    
    # Initialization: greedy rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind='stable')
    central = np.argsort(b.sum(axis=1), kind='stable')
    pos = np.empty(n, dtype=np.int64)
    pos[heavy] = central
    pos = pos.astype(np.int64)
    
    best_pos = pos.copy()
    best_cost = compute_cost(pos)
    current_cost = best_cost
    
    # Simulated Annealing with tabu-like diversification
    rng = np.random.default_rng(42)
    
    # Precompute for speed: b_rows = b, b_cols = b
    # We'll do SA with restarts
    
    t_start = time.time()
    max_temp = 1e7
    min_temp = 1e-2
    cooling = 0.9999
    
    # We'll do multiple SA runs
    for restart in range(20):
        if time.time() > deadline - 5.0:
            break
        
        if restart == 0:
            pos = best_pos.copy()
            current_cost = best_cost
        else:
            # Perturb best solution
            pos = best_pos.copy()
            for _ in range(n // 4):
                i, j = rng.choice(n, size=2, replace=False)
                pos[i], pos[j] = pos[j], pos[i]
            current_cost = compute_cost(pos)
        
        temp = max_temp
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < deadline - 2.0 and temp > min_temp:
            # Pick random swap
            i = rng.integers(n)
            j = rng.integers(n)
            if i == j:
                continue
            
            delta = swap_delta(pos, i, j)
            
            if delta < 0 or (temp > 0 and rng.random() < np.exp(-delta / temp)):
                pos[i], pos[j] = pos[j], pos[i]
                current_cost += delta
                no_improve = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_pos = pos.copy()
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
            
            temp *= cooling
        
        # Local search polish at end of each SA run
        if time.time() > deadline - 5.0:
            break
        
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            # Evaluate all pairs is O(n^2) = 65536, each O(n) = 256, total ~16M ops
            # That's too slow per iteration. Let's do a subset.
            # Try random subset of pairs
            for _ in range(2000):
                if time.time() > deadline - 3.0:
                    break
                i = rng.integers(n)
                j = rng.integers(n)
                if i == j:
                    continue
                delta = swap_delta(pos, i, j)
                if delta < 0:
                    pos[i], pos[j] = pos[j], pos[i]
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_pos = pos.copy()
    
    return best_pos.tolist()
