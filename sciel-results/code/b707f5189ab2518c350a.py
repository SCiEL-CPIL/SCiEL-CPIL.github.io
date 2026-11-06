import time
import sys
import random
from math import gcd

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    N = 256
    vertices = list(range(2, N + 1))
    n = len(vertices)
    
    # Precompute compatibility
    # compat[i] is a bitmask of vertices j such that lcm(vertices[i], vertices[j]) > N
    compat = [0] * n
    for i in range(n):
        a = vertices[i]
        for j in range(i + 1, n):
            b = vertices[j]
            g = gcd(a, b)
            if a * b // g > N:
                compat[i] |= (1 << j)
                compat[j] |= (1 << i)
                
    weights = [1.0 / v for v in vertices]
    
    # Helper to get value of a mask
    # We can precompute weight sums for chunks or just iterate bits. 
    # Iterating bits is fast enough for 255 bits.
    def get_value(mask):
        s = 0.0
        m = mask
        while m:
            lsb = m & -m
            idx = (lsb.bit_length() - 1)
            s += weights[idx]
            m ^= lsb
        return s

    # Helper to check if a vertex can be added to a mask
    # Returns True if v is not in mask and mask is compatible with v
    def can_add(mask, v):
        if mask & (1 << v):
            return False
        return (mask & ~compat[v]) == 0

    # 1. Greedy Initialization
    S = 0
    # Sort indices by weight descending (i.e., value ascending)
    # Smaller values have larger weights, so we prioritize them.
    # vertices are 2..256, so index 0 is 2 (weight 0.5), index 1 is 3 (weight 0.33)...
    # This is already sorted by weight descending.
    for v in range(n):
        if can_add(S, v):
            S |= (1 << v)
            
    best_mask = S
    best_val = get_value(S)
    
    # 2. Simulated Annealing
    # Temperature schedule
    T_start = 0.1
    T_end = 1e-6
    alpha = 0.999  # Cooling rate
    # We run SA until time limit or T is low
    
    T = T_start
    current_mask = best_mask
    current_val = best_val
    
    # Precompute list of indices for random selection
    indices = list(range(n))
    
    while time.time() < deadline:
        if T < T_end:
            break
            
        # Generate a neighbor state
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add a random vertex
            v = random.choice(indices)
            if can_add(current_mask, v):
                new_mask = current_mask | (1 << v)
                delta_val = weights[v]
                new_val = current_val + delta_val
            else:
                continue
        elif move_type < 0.8:
            # Try to remove a random vertex from current set
            m = current_mask
            if m == 0:
                continue
            # Pick a random set bit
            # To pick a random set bit efficiently, we can pick a random index and check if it's set,
            # or iterate. For 255 bits, picking random index is fine.
            v = random.choice(indices)
            if current_mask & (1 << v):
                new_mask = current_mask & ~(1 << v)
                delta_val = -weights[v]
                new_val = current_val + delta_val
            else:
                continue
        else:
            # Swap: remove one, add one
            m = current_mask
            if m == 0:
                continue
            v_remove = random.choice(indices)
            if not (current_mask & (1 << v_remove)):
                continue
            
            # Find a random v_add to add
            # We need v_add not in mask, and compatible with (mask without v_remove)
            # Try a few random candidates for v_add
            found = False
            for _ in range(10): # Try up to 10 random candidates
                v_add = random.choice(indices)
                if current_mask & (1 << v_add):
                    continue
                temp_mask = current_mask & ~(1 << v_remove)
                if can_add(temp_mask, v_add):
                    new_mask = temp_mask | (1 << v_add)
                    delta_val = weights[v_add] - weights[v_remove]
                    new_val = current_val + delta_val
                    found = True
                    break
            if not found:
                continue
        
        # Metropolis criterion
        if delta_val > 0:
            accept = True
        else:
            # Accept with probability exp(delta / T)
            # delta_val is negative
            prob = 2.718281828 ** (delta_val / T)
            accept = random.random() < prob
            
        if accept:
            current_mask = new_mask
            current_val = new_val
            
            if current_val > best_val:
                best_val = current_val
                best_mask = current_mask
                
        T *= alpha

    # 3. Final Local Search (Greedy Refill) to polish
    # Try to add any compatible vertex
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for v in range(n):
            if can_add(best_mask, v):
                best_mask |= (1 << v)
                best_val += weights[v]
                changed = True
                
    # Try to remove vertices that allow better additions? 
    # Simple removal-refill again
    improved = True
    while improved and time.time() < deadline:
        improved = False
        m = best_mask
        while m:
            lsb = m & -m
            u_idx = (lsb.bit_length() - 1)
            m ^= lsb
            
            S_candidate = best_mask & ~lsb
            val_candidate = best_val - weights[u_idx]
            
            # Greedy Refill
            new_mask = S_candidate
            new_val = val_candidate
            
            for v in range(n):
                if new_mask & (1 << v):
                    continue
                if (new_mask & ~compat[v]) == 0:
                    new_mask |= (1 << v)
                    new_val += weights[v]
                    
            if new_val > best_val:
                best_val = new_val
                best_mask = new_mask
                improved = True
                break

    # Extract result
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = (lsb.bit_length() - 1)
        result.append(vertices[idx])
        m ^= lsb
        
    if not result:
        return [2]
        
    return result
