import time
import random
import math
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    # Evaluator kills at 90s, task says 60s. 
    # Use 58s to be safe.
    time_limit = 58.0
    
    # 1. Precomputation of 4-AP conflicts and 3-AP contributions
    
    # conflict_mask[v] is a bitmask (int) where bit u is 1 if {u, v} forms part of a 4-AP.
    # Actually, we need to know if adding v to S creates a 4-AP.
    # A 4-AP is {a, a+d, a+2d, a+3d}.
    # If we add v, we check if there exist d such that v is one of the terms and the other 3 are in S.
    # It's faster to precompute for each v, the set of "witnesses" W_v.
    # If (S & W_v) != 0, adding v is forbidden? No, we need ALL 3 others.
    # So for each v, we have a list of triples T_v = { (u1, u2, u3) } such that {v, u1, u2, u3} is a 4-AP.
    # Adding v is valid iff for all (u1, u2, u3) in T_v, NOT (u1 in S AND u2 in S AND u3 in S).
    
    # Alternative: Precompute a "forbidden pair" graph?
    # If u in S, which v's are forbidden?
    # Let's stick to the bitmask approach but optimized.
    
    # Bitmask representation for S: S is an integer < 2^256.
    # Python handles big ints natively, bitwise ops are fast.
    
    # Precompute for each v (1..n), the list of masks M such that if (S & M) == M, v cannot be added.
    # M is a 3-bit mask representing the other 3 elements of a 4-AP containing v.
    
    forbidden_masks = [[] for _ in range(n + 1)]
    for v in range(1, n + 1):
        masks = []
        # v is 1st: v, v+d, v+2d, v+3d
        for d in range(1, (n - v) // 3 + 1):
            u1, u2, u3 = v + d, v + 2 * d, v + 3 * d
            if u3 <= n:
                m = (1 << (u1 - 1)) | (1 << (u2 - 1)) | (1 << (u3 - 1))
                masks.append(m)
        # v is 2nd: v-d, v, v+d, v+2d
        for d in range(1, min(v - 1, (n - v) // 2) + 1):
            u1, u2, u3 = v - d, v + d, v + 2 * d
            if u1 >= 1 and u3 <= n:
                m = (1 << (u1 - 1)) | (1 << (u2 - 1)) | (1 << (u3 - 1))
                masks.append(m)
        # v is 3rd: v-2d, v-d, v, v+d
        for d in range(1, (v - 1) // 2 + 1):
            u1, u2, u3 = v - 2 * d, v - d, v + d
            if u1 >= 1 and u3 <= n:
                m = (1 << (u1 - 1)) | (1 << (u2 - 1)) | (1 << (u3 - 1))
                masks.append(m)
        # v is 4th: v-3d, v-2d, v-d, v
        for d in range(1, (v - 1) // 3 + 1):
            u1, u2, u3 = v - 3 * d, v - 2 * d, v - d
            if u1 >= 1:
                m = (1 << (u1 - 1)) | (1 << (u2 - 1)) | (1 << (u3 - 1))
                masks.append(m)
        forbidden_masks[v] = masks

    # Precompute 3-AP contributions
    # For each v, ap3_masks[v] contains masks M such that if (S & M) == M, v completes a 3-AP.
    # Specifically, M is a 2-bit mask of the other two elements.
    ap3_masks = [[] for _ in range(n + 1)]
    for v in range(1, n + 1):
        masks = []
        # v is 1st: v, v+d, v+2d
        for d in range(1, (n - v) // 2 + 1):
            u1, u2 = v + d, v + 2 * d
            if u2 <= n:
                m = (1 << (u1 - 1)) | (1 << (u2 - 1))
                masks.append(m)
        # v is 2nd: v-d, v, v+d
        for d in range(1, min(v - 1, n - v) + 1):
            u1, u2 = v - d, v + d
            if u1 >= 1 and u2 <= n:
                m = (1 << (u1 - 1)) | (1 << (u2 - 1))
                masks.append(m)
        # v is 3rd: v-2d, v-d, v
        for d in range(1, (v - 1) // 2 + 1):
            u1, u2 = v - 2 * d, v - d
            if u1 >= 1:
                m = (1 << (u1 - 1)) | (1 << (u2 - 1))
                masks.append(m)
        ap3_masks[v] = masks

    def count_3aps(S_mask):
        count = 0
        # Iterate over all v in S
        # To speed this up, we can just iterate 1..n and check bit
        for v in range(1, n + 1):
            if (S_mask >> (v - 1)) & 1:
                for m in ap3_masks[v]:
                    if (S_mask & m) == m:
                        count += 1
        return count

    def can_add(S_mask, v):
        for m in forbidden_masks[v]:
            if (S_mask & m) == m:
                return False
        return True

    # Strategy: Multi-Start Local Search
    # 1. Generate several random valid initial sets (or greedy).
    # 2. For each, perform local search: 
    #    - Try adding a random v not in S. If valid, accept if it increases 3-AP count (or always, since we want max 3-APs? No, we want max 3-APs subject to no 4-APs. Adding an element never decreases 3-APs. So always add if valid? 
    #      Wait, adding an element might block future additions? 
    #      Actually, the problem is to MAXIMIZE 3-APs. 
    #      Adding an element always adds >= 0 to the 3-AP count. 
    #      However, adding an element might prevent adding OTHER elements later that would have added MORE 3-APs.
    #      So simple greedy "add if valid" might get stuck in a local optimum where we can't add more, but a different set of choices would allow more total 3-APs.
    #      
    #      Local search moves:
    #      1. Add v (if valid and v not in S). Delta = gained 3-APs. Always >= 0.
    #      2. Remove v (if v in S). Delta = -lost 3-APs. Always <= 0.
    #      3. Swap u in S with v not in S. Delta = gained(v) - lost(u).
    #      
    #      Since we want to maximize, we should primarily look for Add moves. 
    #      If no Add moves are valid, we are in a maximal 4-AP-free set. 
    #      Then we must look for Swap moves to improve the count.
    #      Or Remove moves to unblock better Adds?
    #      
    #      Let's focus on:
    #      - Phase 1: Greedy Add.
    #      - Phase 2: Local Search with Swaps.
    #        - For each u in S, for each v not in S, check if swap is valid and improves count.
    #        - This is O(|S| * (n-|S|) * Cost_Check). 
    #        - Cost_Check is O(1) if we precompute deltas? 
    #        - Delta(u, v) = count_3aps(S - u + v) - count_3aps(S).
    #        - count_3aps(S - u + v) = count_3aps(S) - lost(u) + gained(v, S-u).
    #        - lost(u) is number of 3-APs containing u in S.
    #        - gained(v, S-u) is number of 3-APs containing v in S-u.
    #        - We can precompute lost(u) for current S? It changes when S changes.
    #        - But we can compute it on the fly: 
    #          lost(u) = sum(1 for m in ap3_masks[u] if (S & m) == m)
    #          gained(v) = sum(1 for m in ap3_masks[v] if ((S & ~bit_u) & m) == m)
    #        - This is fast enough for a few thousand swaps.
    
    best_set = set()
    best_count = 0
    best_mask = 0
    
    # Run multiple restarts
    num_restarts = 500 # We will run until time limit
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 2.0:
            break
            
        # Generate initial set
        # Random subset, then filter to be 4-AP free? 
        # Or Greedy Add from random order.
        S_mask = 0
        S_set = set()
        
        # Greedy Add with random tie-breaking
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for v in candidates:
            if can_add(S_mask, v):
                S_mask |= (1 << (v - 1))
                S_set.add(v)
                
        current_count = count_3aps(S_mask)
        
        # Local Search
        improved = True
        # Limit steps per restart to avoid getting stuck in one bad restart
        max_steps_per_restart = 5000 
        
        while improved and max_steps_per_restart > 0:
            improved = False
            max_steps_per_restart -= 1
            
            # 1. Try to add any valid element
            # Iterate through candidates not in S
            # To speed up, we can just pick a random few
            # Or check all if n is small? n=256 is small.
            
            added = False
            # Check all v not in S
            for v in range(1, n + 1):
                if v not in S_set:
                    if can_add(S_mask, v):
                        # Calculate gain
                        gain = 0
                        for m in ap3_masks[v]:
                            if (S_mask & m) == m:
                                gain += 1
                        # Since gain >= 0, and we want max, we should add.
                        # But does adding v block future adds that yield more?
                        # Yes. So greedy add might be suboptimal.
                        # However, if we are in a local search, we can also remove.
                        # Let's just add if gain > 0. If gain == 0, it might still be worth it to unblock? No, adding never unblocks, it only blocks.
                        # So only add if gain > 0? 
                        # Actually, adding an element with gain 0 doesn't help immediately, but might it help? No, it only adds constraints.
                        # So we should only add if gain > 0.
                        if gain > 0:
                            S_mask |= (1 << (v - 1))
                            S_set.add(v)
                            current_count += gain
                            improved = True
                            added = True
                            break # Restart inner loop? Or continue? 
                            # Breaking allows us to re-evaluate other adds.
            
            if added:
                continue # Restart the while loop to check for more adds
                
            # 2. If no adds, try swaps
            # Find best swap u->v that increases count
            best_swap = None
            best_delta = 0
            
            # Sample some u's and v's to save time? 
            # Or check all? 
            # |S| is around 100-150? 
            # n=256. 
            # 150 * 100 = 15000 pairs. 
            # Each pair check is ~100 ops. 1.5M ops. 
            # Python might do 1M ops/sec. So 1.5 sec per full pass. 
            # Too slow for many passes.
            
            # Optimization: 
            # Precompute "potential" for each v not in S?
            # potential(v) = number of 3-APs v would complete if added to S.
            # But validity depends on S.
            
            # Let's just sample random swaps.
            # Pick random u in S, random v not in S.
            # If valid and delta > 0, accept.
            
            # To be faster, let's just do a few random swaps per iteration.
            for _ in range(20): # 20 random swap attempts
                if not S_set: break
                u = random.choice(list(S_set))
                v = random.randint(1, n)
                if v in S_set:
                    continue
                if v == u:
                    continue
                
                # Check validity of swap
                S_mask_without_u = S_mask & ~(1 << (u - 1))
                
                valid = True
                for m in forbidden_masks[v]:
                    if (S_mask_without_u & m) == m:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Calculate delta
                # lost(u) in S
                lost = 0
                for m in ap3_masks[u]:
                    if (S_mask & m) == m:
                        lost += 1
                        
                # gained(v) in S - u
                gained = 0
                for m in ap3_masks[v]:
                    if (S_mask_without_u & m) == m:
                        gained += 1
                        
                delta = gained - lost
                
                if delta > 0:
                    # Accept swap
                    S_mask = S_mask_without_u | (1 << (v - 1))
                    S_set.remove(u)
                    S_set.add(v)
                    current_count += delta
                    improved = True
                    break # Restart while loop
                    
            # If no swap improved, we are in a local optimum for this restart.
            
        if current_count > best_count:
            best_count = current_count
            best_set = S_set.copy()
            best_mask = S_mask

    # Final verification and return
    if not best_set:
        return [1]
        
    return sorted(best_set)

if __name__ == "__main__":
    res = construct()
    print(len(res), res[:10])
