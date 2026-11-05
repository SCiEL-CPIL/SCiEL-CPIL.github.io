import time
import random

def construct():
    n = 512
    start_time = time.time()
    deadline = start_time + 82.0  # 8s safety margin before 90s kill
    
    # --- Fast 4-AP check using bitmask representation ---
    # We use a Python int as a bitset: bit i set means (i+1) is in S
    # So element x (1-indexed) corresponds to bit (x-1)
    
    def make_bitset(S):
        b = 0
        for x in S:
            b |= (1 << (x - 1))
        return b
    
    def has_4ap_in_bitset(bs, x):
        """Check if x (1-indexed) creates a 4-AP with elements already in bitset bs."""
        # x is 1st: x, x+d, x+2d, x+3d
        for d in range(1, (n - x) // 3 + 1):
            if (bs >> (x + d - 1) & 1) and (bs >> (x + 2*d - 1) & 1) and (bs >> (x + 3*d - 1) & 1):
                return True
        # x is 2nd: x-d, x, x+d, x+2d
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            if (bs >> (x - d - 1) & 1) and (bs >> (x + d - 1) & 1) and (bs >> (x + 2*d - 1) & 1):
                return True
        # x is 3rd: x-2d, x-d, x, x+d
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            if (bs >> (x - 2*d - 1) & 1) and (bs >> (x - d - 1) & 1) and (bs >> (x + d - 1) & 1):
                return True
        # x is 4th: x-3d, x-2d, x-d, x
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (bs >> (x - 3*d - 1) & 1) and (bs >> (x - 2*d - 1) & 1) and (bs >> (x - d - 1) & 1):
                return True
        return False
    
    def get_conflict_count(bs, x):
        """Count how many distinct elements in S are in a 4-AP with x."""
        conflicts = set()
        for d in range(1, (n - x) // 3 + 1):
            if (bs >> (x + d - 1) & 1) and (bs >> (x + 2*d - 1) & 1) and (bs >> (x + 3*d - 1) & 1):
                conflicts.add(x + d)
                conflicts.add(x + 2*d)
                conflicts.add(x + 3*d)
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            if (bs >> (x - d - 1) & 1) and (bs >> (x + d - 1) & 1) and (bs >> (x + 2*d - 1) & 1):
                conflicts.add(x - d)
                conflicts.add(x + d)
                conflicts.add(x + 2*d)
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            if (bs >> (x - 2*d - 1) & 1) and (bs >> (x - d - 1) & 1) and (bs >> (x + d - 1) & 1):
                conflicts.add(x - 2*d)
                conflicts.add(x - d)
                conflicts.add(x + d)
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (bs >> (x - 3*d - 1) & 1) and (bs >> (x - 2*d - 1) & 1) and (bs >> (x - d - 1) & 1):
                conflicts.add(x - 3*d)
                conflicts.add(x - 2*d)
                conflicts.add(x - d)
        return len(conflicts), conflicts
    
    # --- Base-3 (Salem-Spencer) construction ---
    # Numbers in [1, n] whose base-3 representation uses only digits 0 and 1
    def base3_construction():
        S = set()
        x = 1
        while x <= n:
            # Convert x to base 3, check if all digits are 0 or 1
            temp = x
            valid = True
            while temp > 0:
                if temp % 3 == 2:
                    valid = False
                    break
                temp //= 3
            if valid:
                S.add(x)
            x += 1
        return S
    
    # --- Alternative: greedy with randomized order ---
    def greedy_random(seed_val):
        rng = random.Random(seed_val)
        S = set()
        order = list(range(1, n + 1))
        rng.shuffle(order)
        for x in order:
            if not has_4ap_in_bitset(make_bitset(S), x):
                S.add(x)
        return S
    
    # --- Local search: try to add elements, then do swaps ---
    def local_search(S, seed_val):
        rng = random.Random(seed_val)
        bs = make_bitset(S)
        size = len(S)
        
        # Phase 1: Try to add any element
        while True:
            if time.time() > deadline:
                break
            added = False
            candidates = [y for y in range(1, n + 1) if y not in S]
            rng.shuffle(candidates)
            for y in candidates:
                if time.time() > deadline:
                    break
                if not has_4ap_in_bitset(bs, y):
                    S.add(y)
                    bs |= (1 << (y - 1))
                    size += 1
                    added = True
                    break
            if not added:
                break
        
        # Phase 2: Try swaps to enable further additions
        for _ in range(50):
            if time.time() > deadline:
                break
            # Find candidate y not in S with minimal conflict count
            candidates = [y for y in range(1, n + 1) if y not in S]
            rng.shuffle(candidates)
            
            best_y = -1
            best_conflict_count = n + 1
            best_conflicts = set()
            
            for y in candidates:
                if time.time() > deadline:
                    break
                cnt, confs = get_conflict_count(bs, y)
                if cnt < best_conflict_count:
                    best_conflict_count = cnt
                    best_y = y
                    best_conflicts = confs
                    if cnt == 0:
                        break
            
            if best_y == -1:
                break
            
            if best_conflict_count == 0:
                S.add(best_y)
                bs |= (1 << (best_y - 1))
                size += 1
            elif best_conflict_count == 1:
                # Remove the one conflicting element, add best_y
                elem_to_remove = list(best_conflicts)[0]
                if elem_to_remove in S:
                    S.remove(elem_to_remove)
                    bs &= ~(1 << (elem_to_remove - 1))
                    S.add(best_y)
                    bs |= (1 << (best_y - 1))
                    # Size unchanged, but structure changed - try adding again
                    # Quick add check
                    added_more = False
                    for y2 in [y for y in range(1, n + 1) if y not in S]:
                        if time.time() > deadline:
                            break
                        if not has_4ap_in_bitset(bs, y2):
                            S.add(y2)
                            bs |= (1 << (y2 - 1))
                            size += 1
                            added_more = True
                            break
                    if not added_more:
                        break
                else:
                    break
            else:
                # Try 2-for-2 swap: remove 2 conflicting elements, add 2 new ones
                if best_conflict_count == 2:
                    elems_to_remove = list(best_conflicts)
                    # Remove them temporarily
                    temp_S = set(S)
                    temp_bs = bs
                    for e in elems_to_remove:
                        temp_S.remove(e)
                        temp_bs &= ~(1 << (e - 1))
                    
                    # Now try to add best_y and see if we can also add another
                    if not has_4ap_in_bitset(temp_bs, best_y):
                        temp_S.add(best_y)
                        temp_bs |= (1 << (best_y - 1))
                        # Try to add one more
                        for y2 in [y for y in range(1, n + 1) if y not in temp_S and y != best_y]:
                            if time.time() > deadline:
                                break
                            if not has_4ap_in_bitset(temp_bs, y2):
                                temp_S.add(y2)
                                temp_bs |= (1 << (y2 - 1))
                                S = temp_S
                                bs = temp_bs
                                size = len(S)
                                break
                        else:
                            # Only added 1, net -1, skip
                            break
                    else:
                        break
                else:
                    break
        
        return S, size
    
    # --- Main: try multiple strategies and keep the best ---
    best_S = set()
    best_size = 0
    
    # Strategy 1: Base-3 construction
    S1 = base3_construction()
    S1, sz1 = local_search(S1, seed_val=42)
    if sz1 > best_size:
        best_size = sz1
        best_S = S1
    
    # Strategy 2: Greedy sequential
    S2 = set()
    bs2 = 0
    for x in range(1, n + 1):
        if not has_4ap_in_bitset(bs2, x):
            S2.add(x)
            bs2 |= (1 << (x - 1))
    S2, sz2 = local_search(S2, seed_val=123)
    if sz2 > best_size:
        best_size = sz2
        best_S = S2
    
    # Strategy 3: Multiple random greedy + local search
    for seed_val in range(10):
        if time.time() > deadline:
            break
        S3 = greedy_random(seed_val * 7 + 1)
        S3, sz3 = local_search(S3, seed_val=seed_val * 13 + 5)
        if sz3 > best_size:
            best_size = sz3
            best_S = S3
    
    # Strategy 4: More aggressive local search with multiple restarts
    for restart in range(20):
        if time.time() > deadline:
            break
        # Start from best_S with perturbation
        S4 = set(best_S)
        rng = random.Random(restart * 31 + 7)
        # Remove k random elements
        k = 3 + (restart % 5)
        elems = list(S4)
        rng.shuffle(elems)
        for e in elems[:k]:
            S4.discard(e)
        # Greedy fill
        bs4 = make_bitset(S4)
        for x in range(1, n + 1):
            if x not in S4 and not has_4ap_in_bitset(bs4, x):
                S4.add(x)
                bs4 |= (1 << (x - 1))
        
        S4, sz4 = local_search(S4, seed_val=restart * 17 + 3)
        if sz4 > best_size:
            best_size = sz4
            best_S = S4
    
    return sorted(best_S)

# EVOLVE-BLOCK-START
# The construct function is defined above. This block is for structural clarity.
# The entry point is construct() which returns a sorted list of integers.
# EVOLVE-BLOCK-END
