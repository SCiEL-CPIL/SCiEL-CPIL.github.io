import time
import random
import math

def construct():
    N = 128
    
    # Precompute 4-term arithmetic progressions
    aps = []
    ap_indices = [[] for _ in range(N)]
    
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            ap_id = len(aps)
            indices = [a - 1 + j * d for j in range(4)]
            aps.append(indices)
            for j, idx in enumerate(indices):
                ap_indices[idx].append((ap_id, j))
                
    def total_violations(colors):
        count = 0
        for ap_id, _ in enumerate(aps):
            c = {colors[i] for i in aps[ap_id]}
            if len(c) <= 2:
                count += 1
        return count

    def calc_delta(colors, idx, new_c, old_c):
        if new_c == old_c:
            return 0
        delta = 0
        for ap_id, pos in ap_indices[idx]:
            # Fetch colors for this AP
            # Optimization: copy is fast for small lists
            c_list = [colors[i] for i in aps[ap_id]]
            
            # Old state
            c_list[pos] = old_c
            d_old = len(set(c_list))
            
            # New state
            c_list[pos] = new_c
            d_new = len(set(c_list))
            
            if d_old <= 2 and d_new > 2:
                delta -= 1
            elif d_old > 2 and d_new <= 2:
                delta += 1
        return delta

    def sa_solve(K, time_limit):
        start_time = time.time()
        best_v = float('inf')
        best_c = None
        
        # Allow multiple restarts if time permits
        while time.time() - start_time < time_limit:
            cur_c = [random.randint(0, K - 1) for _ in range(N)]
            cur_v = total_violations(cur_c)
            
            if cur_v < best_v:
                best_v = cur_v
                best_c = cur_c[:]
                
            if cur_v == 0:
                return cur_c, 0

            T = 5.0
            min_T = 0.05
            cooling = 0.9995
            
            while time.time() - start_time < time_limit:
                if cur_v == 0:
                    if cur_v < best_v:
                        best_v = cur_v
                        best_c = cur_c[:]
                    break

                i = random.randint(0, N - 1)
                old_c = cur_c[i]
                new_c = random.randint(0, K - 1)
                
                if new_c == old_c:
                    continue
                
                d_v = calc_delta(cur_c, i, new_c, old_c)
                
                # Acceptance criterion
                if d_v < 0:
                    accept = True
                else:
                    # Avoid log of 0 or negative
                    if T > 0:
                        prob = math.exp(-d_v / T)
                    else:
                        prob = 0.0
                    accept = random.random() < prob

                if accept:
                    cur_c[i] = new_c
                    cur_v += d_v
                    if cur_v < best_v:
                        best_v = cur_v
                        best_c = cur_c[:]
                
                T *= cooling
                if T < min_T:
                    break
        
        return best_c, best_v

    # Global deadline guard
    global_start = time.time()
    deadline = 60.0 # Safety margin
    
    # Try increasing K
    # Start with small K to minimize colors
    for K in range(3, 11):
        remaining = deadline - (time.time() - global_start)
        if remaining <= 5:
            break
            
        # Allocate time: smaller K get slightly less time if we are optimistic, 
        # but 3 and 4 are hard. 5+ are easier.
        alloc = min(5.0, remaining - 1.0)
        
        c, v = sa_solve(K, alloc)
        
        if v == 0:
            # Found valid coloring with K colors (or fewer)
            # Return immediately as we are iterating K upwards
            return c

    # If we reached here, no valid coloring found for K in [3, 10]
    # Use remaining time to find a valid one with larger K
    remaining = deadline - (time.time() - global_start)
    if remaining > 1:
        c, v = sa_solve(15, remaining)
        if v == 0:
            return c
            
    # Fallback: Random assignment is likely invalid.
    # We need a valid solution.
    # Run a greedy CSP like the seed if time permits?
    # Or just return the best invalid one? No, score 0.
    # Let's try one last desperate SA with high K.
    remaining = deadline - (time.time() - global_start)
    if remaining > 0:
        c, v = sa_solve(20, remaining)
        return c
        
    # Absolute fallback: return an array (might be invalid, but better than exception)
    return [0] * N

