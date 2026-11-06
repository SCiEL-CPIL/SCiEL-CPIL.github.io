import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 100
    TARGET = 200
    MASK = (1 << (TARGET + 1)) - 1
    
    def get_bitset(S):
        r = 1
        for x in S:
            r |= (r << x)
        return r & MASK

    def is_valid_bitset(r):
        return (r >> TARGET) & 1 == 0

    def local_search(S):
        S = set(S)
        
        # Helper to check if adding x to set S with bitset r is valid
        # Returns new bitset if valid, else None
        def try_add(r, S, x):
            if x in S:
                return None
            new_r = r | (r << x)
            new_r &= MASK
            if is_valid_bitset(new_r):
                return new_r
            return None

        # Phase 1: Fill with any valid additions
        r = get_bitset(S)
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                new_r = try_add(r, S, x)
                if new_r is not None:
                    S.add(x)
                    r = new_r
                    changed = True
                    break
        
        # Phase 2: Try swaps to enable further additions
        improved = True
        while improved:
            improved = False
            # We look for a swap (u out, v in) that allows an addition (w in)
            for u in list(S):
                # Remove u
                # Recalculating bitset after removal is hard. 
                # So we compute bitset for S - {u} + {v} from scratch? 
                # Or we can just iterate v, compute S_swap, get bitset, then try adds.
                
                for v in range(1, N + 1):
                    if v in S:
                        continue
                    S_swap = (S - {u}) | {v}
                    r_swap = get_bitset(S_swap)
                    if not is_valid_bitset(r_swap):
                        continue
                    
                    # Try to add w to S_swap
                    for w in range(1, N + 1):
                        if w in S_swap:
                            continue
                        new_r = try_add(r_swap, S_swap, w)
                        if new_r is not None:
                            S = S_swap | {w}
                            r = new_r
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return S

    best_size = 0
    best_set = set()
    
    # 1. Try standard greedy (descending)
    S_desc = set()
    r_desc = 1
    for x in range(N, 0, -1):
        new_r = r_desc | (r_desc << x)
        new_r &= MASK
        if is_valid_bitset(new_r):
            S_desc.add(x)
            r_desc = new_r
    S_desc = local_search(S_desc)
    if len(S_desc) > best_size:
        best_size = len(S_desc)
        best_set = S_desc

    # 2. Try standard greedy (ascending)
    S_asc = set()
    r_asc = 1
    for x in range(1, N + 1):
        new_r = r_asc | (r_asc << x)
        new_r &= MASK
        if is_valid_bitset(new_r):
            S_asc.add(x)
            r_asc = new_r
    S_asc = local_search(S_asc)
    if len(S_asc) > best_size:
        best_size = len(S_asc)
        best_set = S_asc

    # 3. Random restarts
    while time.time() < deadline:
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        S_rand = set()
        r_rand = 1
        for x in order:
            new_r = r_rand | (r_rand << x)
            new_r &= MASK
            if is_valid_bitset(new_r):
                S_rand.add(x)
                r_rand = new_r
        
        S_rand = local_search(S_rand)
        if len(S_rand) > best_size:
            best_size = len(S_rand)
            best_set = S_rand
            
    return list(best_set)
