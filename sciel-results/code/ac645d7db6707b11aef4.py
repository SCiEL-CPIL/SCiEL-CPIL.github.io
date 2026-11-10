import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin before 60s kill
    
    n = 512
    k = 4
    
    # Precompute all 4-APs
    ap_list = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
            
    num_aps = len(ap_list)
    
    # Map element -> list of AP indices it belongs to
    # Elements are 1-indexed
    elem_to_aps = [[] for _ in range(n + 1)]
    for idx, ap in enumerate(ap_list):
        for elem in ap:
            elem_to_aps[elem].append(idx)
            
    # State
    in_set = np.zeros(n + 1, dtype=bool)
    ap_counts = np.zeros(num_aps, dtype=np.int8)
    
    def add_elem(x):
        in_set[x] = True
        for ap_idx in elem_to_aps[x]:
            ap_counts[ap_idx] += 1
            
    def remove_elem(x):
        in_set[x] = False
        for ap_idx in elem_to_aps[x]:
            ap_counts[ap_idx] -= 1
            
    def can_add(x):
        for ap_idx in elem_to_aps[x]:
            if ap_counts[ap_idx] >= 3:
                return False
        return True

    # Initial Solution: Base-4 construction
    initial_set = []
    for x in range(1, n + 1):
        valid = True
        t = x
        while t > 0:
            if t % 4 == 3:
                valid = False
                break
            t //= 4
        if valid:
            initial_set.append(x)
            
    for x in initial_set:
        add_elem(x)
        
    current_size = len(initial_set)
    best_size = current_size
    best_set = set(initial_set)
    
    random.seed(42)
    
    # Local Search
    while time.time() < deadline:
        improved = False
        
        # 1. Try to add any element
        candidates = [x for x in range(1, n + 1) if not in_set[x]]
        if not candidates:
            break
            
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x):
                add_elem(x)
                current_size += 1
                improved = True
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(np.nonzero(in_set)[0])
                    
        if not improved:
            # Stuck. Try perturbation.
            elems_in_set = [x for x in range(1, n + 1) if in_set[x]]
            if not elems_in_set:
                break
                
            # Try a few kicks
            for _ in range(20):
                if time.time() > deadline - 2:
                    break
                
                rem = random.choice(elems_in_set)
                remove_elem(rem)
                current_size -= 1
                
                # Greedy add
                candidates = [x for x in range(1, n + 1) if not in_set[x]]
                random.shuffle(candidates)
                
                for x in candidates:
                    if can_add(x):
                        add_elem(x)
                        current_size += 1
                        
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(np.nonzero(in_set)[0])
                    
                # Check if we can add more (i.e., not stuck)
                # If we can, break out of kick loop and let main loop handle it
                stuck = True
                for x in candidates:
                    if can_add(x):
                        stuck = False
                        break
                if not stuck:
                    break
                
    # Final Verification
    bs = set(best_set)
    valid = True
    for ap in ap_list:
        if all(e in bs for e in ap):
            valid = False
            break
            
    if not valid:
        return initial_set
        
    return list(best_set)
