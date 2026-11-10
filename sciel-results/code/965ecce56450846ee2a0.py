import numpy as np
import random
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Finds a large square-difference-free set modulo 205 using Simulated Annealing.
    """
    m = 205
    k = 2
    
    # 1. Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)
    
    # 2. Precompute conflict masks
    # conflicts[u] is a bitmask where bit v is 1 if (u - v) % m is in forbidden
    # Note: The condition is symmetric for the set, but the difference direction matters for calculation.
    # (a-b) in F OR (b-a) in F. Since F contains squares, and -1 is a square mod 205?
    # 205 = 5 * 41.
    # (-1)^2 = 1. Is -1 a square mod 5? No (1, 4). Is -1 a square mod 41? Yes (9^2=81=-1).
    # By CRT, -1 is a square mod 205 if it is a square mod 5 AND mod 41. It is not mod 5.
    # So if d is a square, -d is NOT necessarily a square.
    # However, the problem states: "neither (a - b) mod 205 nor (b - a) mod 205 may be in F".
    # This means the graph is undirected if we define edge if EITHER difference is in F.
    # So conflicts[u] should have bit v set if ((u-v)%m in forbidden) or ((v-u)%m in forbidden).
    
    conflicts = [0] * m
    for u in range(m):
        mask = 0
        for v in range(m):
            if u == v:
                continue
            d1 = (u - v) % m
            d2 = (v - u) % m
            if d1 in forbidden or d2 in forbidden:
                mask |= (1 << v)
        conflicts[u] = mask

    # 3. Simulated Annealing
    # State: current_mask (int), current_size (int)
    
    best_mask = 0
    best_size = 0
    
    # Initialize with a greedy set to have a decent starting point
    # Randomize order for different runs? No, deterministic is safer for reproducibility unless we seed random.
    # We will run multiple restarts with different random seeds to find the global max.
    
    start_time = time.time()
    time_limit = 500  # Stay well under 600s
    
    for restart in range(50):  # Multiple restarts
        random.seed(42 + restart)
        
        # Greedy initialization
        cur_mask = 0
        cur_size = 0
        available = list(range(m))
        random.shuffle(available)
        
        for v in available:
            if (cur_mask & conflicts[v]) == 0:
                cur_mask |= (1 << v)
                cur_size += 1
        
        # Local Search / Simulated Annealing
        T = 1.0
        cooling = 0.9999
        min_T = 0.001
        
        # If greedy already found a good size, we can start with higher T or just run.
        # We want to maximize size.
        
        while T > min_T:
            # Choose move type
            # Probability of Add: proportional to (m - cur_size)
            # Probability of Swap: proportional to cur_size * (m - cur_size)
            # We can just pick randomly.
            
            if cur_size == 0:
                # Must add
                v = random.randint(0, m - 1)
                if (cur_mask & conflicts[v]) == 0:
                    cur_mask |= (1 << v)
                    cur_size += 1
                    if cur_size > best_size:
                        best_size = cur_size
                        best_mask = cur_mask
                T *= cooling
                continue
            
            if cur_size == m:
                break
            
            move_type = random.random()
            
            if move_type < 0.4:
                # Try Add
                # Pick random v not in cur_mask
                # Efficient way: pick random index, if in mask, retry or pick another
                # Since m is small, we can just pick random and check
                attempts = 0
                added = False
                while attempts < 10:
                    v = random.randint(0, m - 1)
                    if not (cur_mask & (1 << v)):
                        if (cur_mask & conflicts[v]) == 0:
                            cur_mask |= (1 << v)
                            cur_size += 1
                            added = True
                            break
                    attempts += 1
                
                if added:
                    if cur_size > best_size:
                        best_size = cur_size
                        best_mask = cur_mask
                    # Reset T? No, just continue.
                
            elif move_type < 0.9:
                # Try Swap
                # Pick u in cur_mask, v not in cur_mask
                # Get list of elements in cur_mask
                # For efficiency, we can maintain a list, but bitmask is faster for checks.
                # Let's just pick random u and v.
                attempts = 0
                swapped = False
                while attempts < 10:
                    u = random.randint(0, m - 1)
                    if cur_mask & (1 << u):
                        v = random.randint(0, m - 1)
                        if not (cur_mask & (1 << v)):
                            # Check if v is compatible with cur_mask without u
                            mask_without_u = cur_mask ^ (1 << u)
                            if (mask_without_u & conflicts[v]) == 0:
                                cur_mask = mask_without_u | (1 << v)
                                # Size unchanged
                                swapped = True
                                break
                    attempts += 1
                
                if swapped:
                    # Accept neutral move to explore
                    # We don't update best_size here as it's same, but we might find a path to larger later.
                    # To avoid infinite loops on same-size states, we rely on T cooling.
                    pass
            
            else:
                # Try Remove (to escape local minima)
                # Only do this if we are stuck or with low probability
                if random.random() < 0.1: # Low prob remove
                    u = random.randint(0, m - 1)
                    if cur_mask & (1 << u):
                        cur_mask ^= (1 << u)
                        cur_size -= 1
            
            T *= cooling
            
            if time.time() - start_time > time_limit:
                break
        
        if time.time() - start_time > time_limit:
            break
    
    # Convert best_mask to array
    result = []
    for i in range(m):
        if best_mask & (1 << i):
            result.append(i)
    
    return np.array(result, dtype=np.int64)
