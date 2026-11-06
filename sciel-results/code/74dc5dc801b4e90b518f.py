import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit

    # 1. Precompute 4-AP conflicts using bitmasks
    # forbidden_masks[v] = list of masks. If S contains any mask, v is forbidden.
    forbidden_masks = [[] for _ in range(n + 1)]
    
    def val_to_bit(x):
        return 1 << (x - 1)

    for v in range(1, n + 1):
        masks = []
        # v is 1st: v, v+d, v+2d, v+3d. Need {v+d, v+2d, v+3d} in S.
        for d in range(1, n // 3 + 1):
            if v + 3 * d <= n:
                m = val_to_bit(v + d) | val_to_bit(v + 2 * d) | val_to_bit(v + 3 * d)
                masks.append(m)
            # v is 2nd: v-d, v, v+d, v+2d. Need {v-d, v+d, v+2d} in S.
            if v - d >= 1:
                m = val_to_bit(v - d) | val_to_bit(v + d) | val_to_bit(v + 2 * d)
                masks.append(m)
            # v is 3rd: v-2d, v-d, v, v+d. Need {v-2d, v-d, v+d} in S.
            if v - 2 * d >= 1:
                m = val_to_bit(v - 2 * d) | val_to_bit(v - d) | val_to_bit(v + d)
                masks.append(m)
            # v is 4th: v-3d, v-2d, v-d, v. Need {v-3d, v-2d, v-d} in S.
            if v - 3 * d >= 1:
                m = val_to_bit(v - 3 * d) | val_to_bit(v - 2 * d) | val_to_bit(v - d)
                masks.append(m)
        forbidden_masks[v] = masks

    # 2. Precompute 3-AP contributions for delta
    # ap3_masks[u] = list of masks for pairs (j, k) such that {u, j, k} is a 3-AP.
    ap3_masks = [[] for _ in range(n + 1)]
    for u in range(1, n + 1):
        masks = []
        # u is 1st: u, u+d, u+2d. Other: u+d, u+2d.
        for d in range(1, (n - u) // 2 + 1):
            masks.append(val_to_bit(u + d) | val_to_bit(u + 2 * d))
        # u is 2nd: u-d, u, u+d. Other: u-d, u+d.
        for d in range(1, min(u - 1, n - u) + 1):
            masks.append(val_to_bit(u - d) | val_to_bit(u + d))
        # u is 3rd: u-2d, u-d, u. Other: u-2d, u-d.
        for d in range(1, (u - 1) // 2 + 1):
            masks.append(val_to_bit(u - 2 * d) | val_to_bit(u - d))
        ap3_masks[u] = masks

    # Helper: Check if v can be added to S (S_mask)
    def can_add(S_mask, v):
        masks = forbidden_masks[v]
        for m in masks:
            if (S_mask & m) == m:
                return False
        return True

    # Helper: Calculate delta 3-APs for swap u -> v
    def calc_delta_swap(S_mask, u, v):
        S_mask_without_u = S_mask & ~val_to_bit(u)
        
        lost = 0
        for m in ap3_masks[u]:
            if (S_mask & m) == m:
                lost += 1
                
        gained = 0
        for m in ap3_masks[v]:
            if (S_mask_without_u & m) == m:
                gained += 1
                
        return gained - lost

    # Helper: Calculate delta 3-APs for remove u, add v1, add v2
    def calc_delta_2swap(S_mask, u, v1, v2):
        S_mask_without_u = S_mask & ~val_to_bit(u)
        
        lost = 0
        for m in ap3_masks[u]:
            if (S_mask & m) == m:
                lost += 1
                
        gained = 0
        for m in ap3_masks[v1]:
            if (S_mask_without_u & m) == m:
                gained += 1
        for m in ap3_masks[v2]:
            if (S_mask_without_u & m) == m:
                gained += 1
                
        # Check if v1 and v2 form a 3-AP with u? No, u is removed.
        # Check if v1 and v2 form a 3-AP with each other and some element in S?
        # Wait, the delta calculation above only counts 3-APs containing v1 or v2.
        # It misses 3-APs formed by {v1, v2, w} where w is in S.
        # Actually, if {v1, v2, w} is a 3-AP, then v1 is in a 3-AP with (v2, w) and v2 is in a 3-AP with (v1, w).
        # The mask for v1 would be bit(v2)|bit(w). If w is in S, then (S_without_u & mask) == mask is true.
        # So it is counted once for v1 and once for v2?
        # No, the 3-AP {v1, v2, w} is a single progression.
        # My counting method counts "3-APs containing v".
        # If I sum gained for v1 and gained for v2, I might double count the AP {v1, v2, w}.
        # Let's check:
        # AP {v1, v2, w}.
        # For v1, the mask is bit(v2)|bit(w). If w in S, this mask is satisfied. Count +1.
        # For v2, the mask is bit(v1)|bit(w). If w in S, this mask is satisfied. Count +1.
        # Total +2. But there is only 1 new AP.
        # So I need to subtract the overlap.
        
        # Overlap: 3-APs containing both v1 and v2.
        # This happens if {v1, v2, w} is a 3-AP for some w in S.
        # There are at most 2 such w's (if v1, v2 are ends, w is middle; if v1, v2 are middle/end, etc).
        # Actually, for any pair (v1, v2), there are at most 2 values of w such that {v1, v2, w} is a 3-AP.
        # Let's find them.
        overlap = 0
        # Case 1: v1, v2 are ends. w = (v1+v2)/2. If integer and in S.
        if (v1 + v2) % 2 == 0:
            w = (v1 + v2) // 2
            if 1 <= w <= n and (S_mask_without_u >> (w-1)) & 1:
                overlap += 1
        # Case 2: v1 is middle, v2 is end. w = 2*v1 - v2.
        w = 2*v1 - v2
        if 1 <= w <= n and (S_mask_without_u >> (w-1)) & 1:
            overlap += 1
        # Case 3: v2 is middle, v1 is end. w = 2*v2 - v1.
        w = 2*v2 - v1
        if 1 <= w <= n and (S_mask_without_u >> (w-1)) & 1:
            overlap += 1
            
        return gained - lost - overlap

    # 1. Initial Greedy Construction
    # Sort elements by a heuristic? Or just 1..n?
    # Let's try a few different greedy orders.
    
    best_mask = 0
    best_count = -1
    best_set = set()
    
    def count_3aps_from_bitmask(S_mask):
        S_set = {i+1 for i in range(n) if (S_mask >> i) & 1}
        count = 0
        for a in S_set:
            for d in range(1, (n - a) // 2 + 1):
                if (a + d in S_set) and (a + 2 * d in S_set):
                    count += 1
        return count

    # Multi-start
    num_starts = 10
    for start_idx in range(num_starts):
        if time.time() - start_time > time_limit:
            break
            
        # Different random permutations for greedy
        elements = list(range(1, n + 1))
        random.shuffle(elements)
        
        S_mask = 0
        S_set = set()
        
        for x in elements:
            if can_add(S_mask, x):
                S_mask |= val_to_bit(x)
                S_set.add(x)
                
        current_count = count_3aps_from_bitmask(S_mask)
        
        if current_count > best_count:
            best_count = current_count
            best_mask = S_mask
            best_set = set(S_set)
            
        # Local Search on this start
        current_mask = S_mask
        current_count = current_count
        improved = True
        max_iters = 500
        iters = 0
        
        while improved and iters < max_iters and time.time() - start_time < time_limit:
            improved = False
            iters += 1
            
            # Try 1-swap
            S_list = [i+1 for i in range(n) if (current_mask >> i) & 1]
            random.shuffle(S_list)
            
            for u in S_list:
                if time.time() - start_time > time_limit:
                    break
                # Try to find a better v
                # Instead of all v, try random v's?
                # Or deterministic?
                # Let's try a limited set of v's.
                
                # Generate candidate v's: random
                candidates = set()
                for _ in range(20):
                    v = random.randint(1, n)
                    if not ((current_mask >> (v-1)) & 1):
                        candidates.add(v)
                
                for v in candidates:
                    S_mask_without_u = current_mask & ~val_to_bit(u)
                    if not can_add(S_mask_without_u, v):
                        continue
                    delta = calc_delta_swap(current_mask, u, v)
                    if delta > 0:
                        current_mask = S_mask_without_u | val_to_bit(v)
                        current_count += delta
                        improved = True
                        if current_count > best_count:
                            best_count = current_count
                            best_mask = current_mask
                            best_set = {i+1 for i in range(n) if (best_mask >> i) & 1}
                        break # Restart loop for this u? Or continue?
                        # Breaking u loop to restart search from new state is safer.
                        # But we are in a for u loop. Let's just break out and restart while loop.
                        # Actually, just setting improved=True and breaking the v loop is enough.
                        # But we should probably break the u loop too to re-evaluate.
                        break
            
            # Try 2-swap (remove 1, add 2)
            if not improved:
                S_list = [i+1 for i in range(n) if (current_mask >> i) & 1]
                random.shuffle(S_list)
                
                for u in S_list:
                    if time.time() - start_time > time_limit:
                        break
                    S_mask_without_u = current_mask & ~val_to_bit(u)
                    
                    # Pick 2 random v's
                    v1 = 0
                    v2 = 0
                    while True:
                        v1 = random.randint(1, n)
                        if not ((S_mask_without_u >> (v1-1)) & 1):
                            break
                    while True:
                        v2 = random.randint(1, n)
                        if not ((S_mask_without_u >> (v2-1)) & 1) and v2 != v1:
                            break
                    
                    # Check validity
                    if not can_add(S_mask_without_u, v1):
                        continue
                    if not can_add(S_mask_without_u | val_to_bit(v1), v2):
                        continue
                        
                    delta = calc_delta_2swap(current_mask, u, v1, v2)
                    if delta > 0:
                        current_mask = S_mask_without_u | val_to_bit(v1) | val_to_bit(v2)
                        current_count += delta
                        improved = True
                        if current_count > best_count:
                            best_count = current_count
                            best_mask = current_mask
                            best_set = {i+1 for i in range(n) if (best_mask >> i) & 1}
                        break

    return sorted(best_set)

if __name__ == "__main__":
    res = construct()
    print(len(res), res[:10])
