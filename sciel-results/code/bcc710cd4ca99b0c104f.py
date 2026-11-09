import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 580.0
    
    # Precompute all 4-sets and their bitmasks
    all_4sets = list(combinations(range(d), 4))
    n_sets = len(all_4sets)
    
    # Map each 4-set to a bitmask of its elements (0-10)
    set_masks = []
    for s in all_4sets:
        mask = 0
        for idx in s:
            mask |= (1 << idx)
        set_masks.append(mask)
        
    # Precompute conflict bitmasks: conflict_mask[i] is a bitmask of sets j that conflict with i
    # Two sets conflict if their intersection size >= 3.
    # Intersection of masks m1, m2: m1 & m2.
    # Popcount of intersection >= 3.
    conflict_masks = [0] * n_sets
    
    # Optimization: Precompute popcount for all possible 11-bit masks?
    # There are 2^11 = 2048 masks. We can precompute popcount for all.
    popcount = [0] * (1 << d)
    for m in range(1, 1 << d):
        popcount[m] = popcount[m >> 1] + (m & 1)
        
    for i in range(n_sets):
        mi = set_masks[i]
        for j in range(i + 1, n_sets):
            mj = set_masks[j]
            inter = mi & mj
            if popcount[inter] >= 3:
                conflict_masks[i] |= (1 << j)
                conflict_masks[j] |= (1 << i)
                
    # Search for Maximum Independent Set
    # State: bitmask of selected sets
    # We want to maximize the number of bits in the selection mask.
    
    import random
    rng = random.Random(42)
    
    best_selection_mask = 0
    best_count = 0
    
    # Greedy initialization with random order
    def greedy_random():
        order = list(range(n_sets))
        rng.shuffle(order)
        sel = 0
        for idx in order:
            # Check if idx conflicts with any selected set
            # sel & conflict_masks[idx] == 0 means no conflict
            if (sel & conflict_masks[idx]) == 0:
                sel |= (1 << idx)
        return sel

    # Local Search: Swap, Add, Remove
    # We represent selection as an integer mask.
    # To check validity of a new mask, we can't just check pairwise easily without iterating bits.
    # However, we can maintain a list of selected indices for faster checks if needed, 
    # but bitset intersection is very fast.
    # Let's stick to bitmask for state, but iterate bits for local moves.
    
    def popcount_mask(m):
        return popcount[m]

    def is_valid_mask(sel):
        # Check if sel is an independent set.
        # For each set i in sel, check if (sel & conflict_masks[i]) has any bits other than i itself?
        # Actually, conflict_masks[i] does NOT include i.
        # So if (sel & conflict_masks[i]) != 0, there is a conflict.
        # We need to check for all i in sel.
        # This is O(k) per check.
        # Since k is small (< 100), this is fast.
        temp = sel
        while temp:
            # Get lowest set bit
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            if (sel & conflict_masks[i]) != 0:
                return False
            temp ^= lsb
        return True

    # Better: Just use the greedy result as a valid start, and ensure moves preserve validity or check explicitly.
    # Since we start from a valid set, we can do:
    # 1. Try adding a set: check if it conflicts with current sel.
    # 2. Try removing a set: always valid.
    # 3. Try swapping: remove i, add j. Check if j conflicts with (sel without i).
    
    def local_search(sel, time_limit):
        current_count = popcount_mask(sel)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # 1. Try to add
            # Find all j not in sel that are compatible with sel
            # compatible_j: (sel & conflict_masks[j]) == 0
            # We can iterate j from 0 to n_sets-1
            for j in range(n_sets):
                if not (sel >> j) & 1:
                    if (sel & conflict_masks[j]) == 0:
                        new_sel = sel | (1 << j)
                        current_count += 1
                        sel = new_sel
                        improved = True
                        break # Restart loop to try adding more
            
            if time.time() > time_limit:
                break
                
            # 2. Try to swap
            # For each i in sel, for each j not in sel, check if j is compatible with sel \ {i}
            temp = sel
            while temp:
                lsb_i = temp & (-temp)
                i = lsb_i.bit_length() - 1
                temp ^= lsb_i
                
                sel_without_i = sel ^ lsb_i
                
                for j in range(n_sets):
                    if not (sel >> j) & 1:
                        # Check if j conflicts with sel_without_i
                        if (sel_without_i & conflict_masks[j]) == 0:
                            # Swap is valid
                            new_sel = sel_without_i | (1 << j)
                            # Count stays same, but maybe allows future additions?
                            # We don't gain count immediately, but might unblock.
                            # To avoid cycling, we only accept if it leads to a better state later?
                            # Simple heuristic: accept swap if j has higher "degree" or random.
                            # Or just accept and see if we can add.
                            # Let's try the swap, then immediately try to add.
                            # This is a bit expensive. 
                            # Alternative: Only swap if it reduces "potential conflicts" or similar.
                            # Given time constraints, let's just do random swaps if stuck?
                            # Or simply: if we can't add, try a random valid swap.
                            pass
            
            # If we couldn't add, try a random valid swap to escape local optimum
            if not improved:
                # Pick random i in sel
                temp = sel
                indices_in_sel = []
                while temp:
                    lsb = temp & (-temp)
                    indices_in_sel.append(lsb.bit_length() - 1)
                    temp ^= lsb
                    
                if indices_in_sel:
                    i = rng.choice(indices_in_sel)
                    lsb_i = 1 << i
                    sel_without_i = sel ^ lsb_i
                    
                    # Find valid j to swap in
                    valid_js = []
                    for j in range(n_sets):
                        if not (sel >> j) & 1:
                            if (sel_without_i & conflict_masks[j]) == 0:
                                valid_js.append(j)
                    
                    if valid_js:
                        j = rng.choice(valid_js)
                        sel = sel_without_i | (1 << j)
                        improved = True # Force continue loop to try adding again
                    
        return sel, popcount_mask(sel)

    # Main Loop: Multiple restarts
    n_restarts = 1000
    for r in range(n_restarts):
        if time.time() > deadline - 30:
            break
            
        sel = greedy_random()
        count = popcount_mask(sel)
        
        if count > best_count:
            best_count = count
            best_selection_mask = sel
        
        # Local search from this start
        # Limit time per local search to ensure we do many restarts
        # Or just run until no improvement? 
        # Since adding is fast, it will stop quickly when stuck.
        # Swaps are expensive. Let's limit swap attempts.
        
        # Simplified local search: Just try to add. If stuck, stop.
        # This is very fast.
        while True:
            added = False
            for j in range(n_sets):
                if not (sel >> j) & 1:
                    if (sel & conflict_masks[j]) == 0:
                        sel |= (1 << j)
                        count += 1
                        added = True
                        break
            if not added:
                break
                
        if count > best_count:
            best_count = count
            best_selection_mask = sel
            
        # Try some random swaps if we have time
        if time.time() < deadline - 10:
            for _ in range(10): # 10 random swap attempts
                if time.time() > deadline - 10:
                    break
                temp = sel
                indices_in_sel = []
                while temp:
                    lsb = temp & (-temp)
                    indices_in_sel.append(lsb.bit_length() - 1)
                    temp ^= lsb
                    
                if not indices_in_sel:
                    break
                    
                i = rng.choice(indices_in_sel)
                lsb_i = 1 << i
                sel_without_i = sel ^ lsb_i
                
                valid_js = []
                for j in range(n_sets):
                    if not (sel >> j) & 1:
                        if (sel_without_i & conflict_masks[j]) == 0:
                            valid_js.append(j)
                
                if valid_js:
                    j = rng.choice(valid_js)
                    new_sel = sel_without_i | (1 << j)
                    # Try to add from new_sel
                    new_count = count
                    added = True
                    while added:
                        added = False
                        for k in range(n_sets):
                            if not (new_sel >> k) & 1:
                                if (new_sel & conflict_masks[k]) == 0:
                                    new_sel |= (1 << k)
                                    new_count += 1
                                    added = True
                                    break
                    if new_count > count:
                        sel = new_sel
                        count = new_count
                        if count > best_count:
                            best_count = count
                            best_selection_mask = sel

    # Generate points from best_selection_mask
    points = []
    temp = best_selection_mask
    while temp:
        lsb = temp & (-temp)
        idx = lsb.bit_length() - 1
        temp ^= lsb
        
        s = all_4sets[idx]
        i, j, k, l = s
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        points.append(v)
    
    if not points:
        # Fallback: 220 norm-2 vectors
        for i, j in combinations(range(d), 2):
            for si, sj in [(1,1), (1,-1), (-1,1), (-1,-1)]:
                v = np.zeros(d, dtype=np.int64)
                v[i] = si
                v[j] = sj
                points.append(v)
    
    points_array = np.array(points, dtype=np.int64)
    return np.around(points_array).astype(np.int64)
