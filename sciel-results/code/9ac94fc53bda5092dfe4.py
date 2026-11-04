import numpy as np
import time
import itertools
from scipy.optimize import linear_sum_assignment # Not needed, just for import check

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    def solve_mis(w, max_time):
        # Generate all w-subsets
        supports = list(itertools.combinations(range(D), w))
        N_S = len(supports)
        
        # Precompute masks
        masks = []
        for s in supports:
            m = 0
            for i in s:
                m |= (1 << i)
            masks.append(m)
            
        # Conflict: intersection >= w-1 (for all signs to be valid, need k <= w/2? No, k < w/2? )
        # Condition for all signs: 2(w-k) >= w => k <= w/2.
        # So conflict if k > w/2.
        # For w=4, k > 2 => k >= 3.
        # For w=5, k > 2.5 => k >= 3.
        # For w=6, k > 3 => k >= 4.
        
        threshold = w // 2
        # Conflict if inter > threshold
        
        adj = [set() for _ in range(N_S)]
        for i in range(N_S):
            for j in range(i+1, N_S):
                inter = masks[i] & masks[j]
                if inter.bit_count() > threshold:
                    adj[i].add(j)
                    adj[j].add(i)
                    
        # Simple Greedy + Local Search
        best_size = 0
        best_set = set()
        
        # Heuristic: Sort by degree
        degrees = [len(adj[i]) for i in range(N_S)]
        order = np.argsort(degrees)
        
        def greedy(order):
            x = np.zeros(N_S, dtype=bool)
            blocked = np.zeros(N_S, dtype=bool)
            for v in order:
                if not x[v] and not blocked[v]:
                    x[v] = True
                    for u in adj[v]:
                        blocked[u] = True
            return x
            
        x0 = greedy(order)
        best_size = int(np.sum(x0))
        best_set = set(np.where(x0)[0].tolist())
        
        # Local Search
        rng = np.random.RandomState(42)
        for _ in range(1000):
            if time.time() > max_time:
                break
            # Random restart or kick
            # ... (similar to before)
            pass
            
        return best_size, best_set

    # Try w=4
    size4, set4 = solve_mis(4, deadline - 10)
    score4 = 22 + 16 * size4
    
    # Try w=6
    size6, set6 = solve_mis(6, deadline - 10)
    score6 = 22 + 64 * size6
    
    # Pick best
    if score4 > score6:
        w = 4
        best_set = set4
    else:
        w = 6
        best_set = set6
        
    # Build vectors
    vectors = []
    for i in range(D):
        v1 = [0]*D; v1[i] = 2; vectors.append(v1)
        v2 = [0]*D; v2[i] = -2; vectors.append(v2)
        
    supports = list(itertools.combinations(range(D), w))
    for s_idx in best_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=w):
            v = [0]*D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int8)
