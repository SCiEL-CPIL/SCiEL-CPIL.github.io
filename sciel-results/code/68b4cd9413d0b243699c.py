import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 512
    # Precompute compatibility
    # compatible[i] is a set of j such that lcm(i+1, j+1) <= 512
    # Using bitsets for speed
    compat = [0] * (n + 1)
    for i in range(1, n + 1):
        bits = 0
        for j in range(1, n + 1):
            g = gcd(i, j)
            if (i * j) // g <= n:
                bits |= (1 << j)
        compat[i] = bits
        
    # Initial candidate: divisors of 504
    # 504 = 2^3 * 3^2 * 7
    init_set = set()
    for d in range(1, n + 1):
        if 504 % d == 0:
            init_set.add(d)
            
    best_set = set(init_set)
    best_size = len(best_set)
    
    # Local Search
    # Represent set as a bitset for fast intersection checks
    # current_mask is a big int where bit i is 1 if i in S
    def set_to_mask(S):
        m = 0
        for x in S:
            m |= (1 << x)
        return m
        
    def mask_to_set(m):
        S = set()
        for x in range(1, n + 1):
            if m & (1 << x):
                S.add(x)
        return S

    current_mask = set_to_mask(best_set)
    
    def is_compatible_mask(mask, v):
        # Check if v is compatible with all elements in mask
        # compat[v] has bits for compatible numbers
        # We need (mask & compat[v]) == mask
        return (mask & compat[v]) == mask

    def try_add(mask, v):
        if not (mask & (1 << v)):
            if is_compatible_mask(mask, v):
                return mask | (1 << v)
        return mask

    def try_swap(mask, v, u):
        # Remove u, add v
        new_mask = mask & ~(1 << u)
        if is_compatible_mask(new_mask, v):
            return new_mask | (1 << v)
        return mask

    random.seed(42)
    
    while time.time() < deadline:
        # Try to add a random vertex
        improved = False
        for _ in range(1000):
            v = random.randint(1, n)
            if current_mask & (1 << v):
                continue
            if is_compatible_mask(current_mask, v):
                current_mask |= (1 << v)
                improved = True
                break
                
        if not improved:
            # Try swaps
            # Pick a random u to remove, random v to add
            for _ in range(1000):
                # Get list of elements in current set
                # Optimization: maintain list
                elems = [x for x in range(1, n + 1) if current_mask & (1 << x)]
                if not elems: break
                u = random.choice(elems)
                v = random.randint(1, n)
                if v == u: continue
                if current_mask & (1 << v): continue
                
                new_mask = try_swap(current_mask, v, u)
                if new_mask != current_mask:
                    current_mask = new_mask
                    improved = True
                    break
        
        if not improved:
            # Try multi-swaps or restart with a different seed?
            # For now, just keep trying. 
            # To avoid infinite loop in dead end, we can perturb.
            # Remove a random element to escape local optimum?
            # But we want to maximize, so removing is bad unless it allows adding more.
            # We'll stick to single swaps for simplicity.
            pass

        # Update best
        curr_size = bin(current_mask).count('1')
        if curr_size > best_size:
            best_size = curr_size
            best_set = mask_to_set(current_mask)
            
    return list(best_set)
