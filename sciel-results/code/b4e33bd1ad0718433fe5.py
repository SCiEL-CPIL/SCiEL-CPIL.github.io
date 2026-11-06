import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    time_limit = 58.0  # Safety margin for 60s limit
    
    # Precompute bit values
    bit_val = [1 << i for i in range(n)]  # bit_val[x-1] = 1 << (x-1)
    
    # Precompute forbidden 4-AP masks
    # forbidden_masks[v] contains masks M such that if (S & M) == M, v cannot be added
    forbidden_masks = [[] for _ in range(n + 1)]
    for v in range(1, n + 1):
        masks = []
        # v is 1st: v, v+d, v+2d, v+3d -> need {v+d, v+2d, v+3d}
        for d in range(1, n // 3 + 1):
            if v + 3 * d <= n:
                m = bit_val[v + d - 1] | bit_val[v + 2 * d - 1] | bit_val[v + 3 * d - 1]
                masks.append(m)
        # v is 2nd: v-d, v, v+d, v+2d -> need {v-d, v+d, v+2d}
        for d in range(1, n // 3 + 1):
            if v - d >= 1 and v + 2 * d <= n:
                m = bit_val[v - d - 1] | bit_val[v + d - 1] | bit_val[v + 2 * d - 1]
                masks.append(m)
        # v is 3rd: v-2d, v-d, v, v+d -> need {v-2d, v-d, v+d}
        for d in range(1, n // 3 + 1):
            if v - 2 * d >= 1 and v + d <= n:
                m = bit_val[v - 2 * d - 1] | bit_val[v - d - 1] | bit_val[v + d - 1]
                masks.append(m)
        # v is 4th: v-3d, v-2d, v-d, v -> need {v-3d, v-2d, v-d}
        for d in range(1, n // 3 + 1):
            if v - 3 * d >= 1:
                m = bit_val[v - 3 * d - 1] | bit_val[v - 2 * d - 1] | bit_val[v - d - 1]
                masks.append(m)
        forbidden_masks[v] = masks
        
    # Precompute 3-AP "witness" masks for each element
    # ap3_masks[u] contains masks M such that if (S & M) == M, u completes a 3-AP
    ap3_masks = [[] for _ in range(n + 1)]
    for u in range(1, n + 1):
        masks = []
        # u is 1st: u, u+d, u+2d -> need {u+d, u+2d}
        for d in range(1, (n - u) // 2 + 1):
            masks.append(bit_val[u + d - 1] | bit_val[u + 2 * d - 1])
        # u is 2nd: u-d, u, u+d -> need {u-d, u+d}
        for d in range(1, min(u - 1, n - u) + 1):
            masks.append(bit_val[u - d - 1] | bit_val[u + d - 1])
        # u is 3rd: u-2d, u-d, u -> need {u-2d, u-d}
        for d in range(1, (u - 1) // 2 + 1):
            masks.append(bit_val[u - 2 * d - 1] | bit_val[u - d - 1])
        ap3_masks[u] = masks
        
    # Helper: Check if v can be added to S (S_mask)
    def can_add(S_mask, v):
        for m in forbidden_masks[v]:
            if (S_mask & m) == m:
                return False
        return True

    # Helper: Count 3-APs in S
    def count_3aps(S_mask):
        # Convert to set for fast lookup? No, use bitmask logic or set.
        # For n=256, set is fast enough for initial count.
        S_set = {i+1 for i in range(n) if (S_mask >> i) & 1}
        count = 0
        for a in S_set:
            for d in range(1, (n - a) // 2 + 1):
                if (a + d in S_set) and (a + 2 * d in S_set):
                    count += 1
        return count

    # Helper: Calculate delta 3-APs for swap u -> v
    # Optimization: Instead of counting from scratch, use the precomputed masks.
    def calc_delta(S_mask, u, v):
        S_mask_without_u = S_mask & ~bit_val[u - 1]
        
        # Lost: count masks in ap3_masks[u] that are subset of S_mask
        # Note: S_mask contains u. The masks in ap3_masks[u] do NOT contain u's bit.
        # So we check against S_mask (or S_mask_without_u, same result for these bits).
        lost = 0
        for m in ap3_masks[u]:
            if (S_mask & m) == m:
                lost += 1
                
        # Gained: count masks in ap3_masks[v] that are subset of S_mask_without_u
        gained = 0
        for m in ap3_masks[v]:
            if (S_mask_without_u & m) == m:
                gained += 1
                
        return gained - lost

    best_count = 0
    best_mask = 0
    
    # Multi-Start Local Search
    # We use several random seeds for the initial greedy construction
    # and then perform aggressive hill climbing.
    
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # 1. Initial Greedy Construction (with randomization)
        S_mask = 0
        S_set = set()
        
        # Randomize the order of elements to get diverse starting points
        candidates = list(range(1, n + 1))
        if restart == 0:
            pass # Deterministic first
        else:
            random.shuffle(candidates)
            
        for x in candidates:
            if can_add(S_mask, x):
                S_mask |= bit_val[x - 1]
                S_set.add(x)
                
        current_count = count_3aps(S_mask)
        
        # 2. Local Search (Hill Climbing)
        # We will iterate until no improvement is found or time runs out.
        # To speed up, we maintain a list of valid moves? 
        # For n=256, iterating all u in S and v not in S is |S|*(n-|S|) ~ 128*128 = 16k checks per pass.
        # Each check is O(1) validity + O(k) delta. k is small.
        # This is very fast. We can do many passes.
        
        improved = True
        while improved and (time.time() - start_time < time_limit):
            improved = False
            # Shuffle to avoid bias? Or just iterate.
            # Iterating is fine.
            S_list = sorted(S_set)
            N_list = [x for x in range(1, n + 1) if x not in S_set]
            
            # To make it faster, we can pick random swaps if the full scan is too slow,
            # but 16k checks * ~10 operations is ~160k ops, which is < 1ms in Python if optimized?
            # Actually Python loops are slow. 16k iterations * 10us = 160ms per pass.
            # 58s / 0.16s = ~360 passes per restart. 20 restarts = 7200 passes.
            # This is feasible.
            
            # Let's try to optimize the inner loop.
            # Pre-compute S_mask_without_u for each u? No, u changes.
            
            for u in S_list:
                if time.time() - start_time > time_limit:
                    break
                S_mask_without_u = S_mask & ~bit_val[u - 1]
                
                for v in N_list:
                    # Check validity of v in S \ {u}
                    valid = True
                    for m in forbidden_masks[v]:
                        if (S_mask_without_u & m) == m:
                            valid = False
                            break
                    if not valid:
                        continue
                        
                    delta = calc_delta(S_mask, u, v)
                    
                    if delta > 0:
                        # Accept
                        S_mask = S_mask_without_u | bit_val[v - 1]
                        S_set.remove(u)
                        S_set.add(v)
                        current_count += delta
                        improved = True
                        
                        if current_count > best_count:
                            best_count = current_count
                            best_mask = S_mask
                            
                        # Break out of v loop? 
                        # If we accept a move, S changes, so the remaining v's in N_list
                        # are no longer valid candidates against the OLD S.
                        # We should restart the inner loop or just break and let the outer loop
                        # re-evaluate. Breaking is safer and simpler.
                        break
                if improved:
                    # S changed, so S_list and N_list are stale.
                    # We break the u loop to re-generate them.
                    break
                    
    # Final verification and return
    if best_mask == 0:
        # Fallback: just return a small valid set
        return [1]
        
    best_set = {i+1 for i in range(n) if (best_mask >> i) & 1}
    return sorted(best_set)

if __name__ == "__main__":
    res = construct()
    print(len(res), res[:10])
