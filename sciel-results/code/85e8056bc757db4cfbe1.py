import time
from math import gcd

def construct():
    start = time.time()
    deadline = 80.0  # leave 10s safety margin before 90s kill
    
    N = 256
    # All numbers from 2 to 256
    # Split into "small" (2..128) and "large" (129..256)
    # Key insight: lcm(a,b) > 256 for all pairs in the set.
    
    # If a is "small" (<= 128) and b is "large" (> 128):
    #   lcm(a,b) = a*b/gcd(a,b)
    #   For lcm(a,b) <= 256, we need a*b/gcd(a,b) <= 256
    #   Since b > 128, a must be small enough.
    #   Specifically, if a divides b, then lcm(a,b) = b > 128, but could be <= 256.
    #   If a does not divide b, lcm(a,b) >= b * a/gcd(a,b) >= b/1 * (a/gcd) >= b/2 * ... 
    #   Actually, let's just precompute incompatibility.
    
    # Two numbers a, b in [2,256] are INCOMPATIBLE if lcm(a,b) <= 256.
    # We need to find a subset S such that no two elements are incompatible.
    # This is a Maximum Weight Independent Set problem on a graph where:
    #   - Vertices are 2..256
    #   - Edge between a,b if lcm(a,b) <= 256
    #   - Weight of vertex a is 1/a
    #   - We want max weight independent set.
    
    # The graph has 255 vertices. MIS is NP-hard, but this graph has special structure.
    # 
    # Structural observation:
    # - For any b in [129, 256], the only small a that can be incompatible with b 
    #   are those where lcm(a,b) <= 256.
    # - If a <= 2 and b >= 129: lcm(a,b) is either b or 2b. 
    #   If a=2: lcm(2,b) = b if b odd, 2b if b even. 2b > 256 when b > 128. 
    #   So lcm(2,b) > 256 for b in [129,256]. Wait: if b=130, lcm(2,130)=130. 130 <= 256! Incompatible!
    #   So a=2 is incompatible with even b in [130, 256].
    # - For large b, the set of incompatible small a is limited.
    
    # Strategy:
    # 1. Precompute the incompatibility graph.
    # 2. Use branch-and-bound on the independent set.
    # 3. Key: vertices with higher weight (smaller numbers) should be explored first.
    # 4. Use bitset-based representation for speed.
    
    # Let's precompute incompatibility pairs
    # For each a in [2, 256], find all b > a such that lcm(a,b) <= 256
    # lcm(a,b) = a*b // gcd(a,b)
    
    # Build adjacency: adj[a] = set of b where a < b and lcm(a,b) <= 256
    # Represent vertices as 0..254 (for values 2..256)
    # idx = value - 2
    
    num_vertices = 255  # values 2..256
    # weight[i] = 1/(i+2)
    weights = [1.0 / (i + 2) for i in range(num_vertices)]
    
    # Precompute gcd table for speed
    # gcd_table[a][b] for a,b in 2..256
    # But 255x255 is fine to compute on the fly with math.gcd
    
    # Build incompatibility as bitmasks
    # conflict[i] is a bitmask of vertices j > i such that lcm(i+2, j+2) <= 256
    # Also need conflict in both directions for B&B
    
    # For B&B, we use the standard approach:
    # Process vertices in order of descending weight (ascending value).
    # At each step, decide include/exclude.
    # If we include vertex i, we must exclude all its neighbors.
    
    # Let's precompute for each vertex i, the set of later vertices (j > i) that conflict with i.
    # And for each vertex i, the set of earlier vertices that conflict with i (for pruning).
    
    # Actually, for the standard B&B MIS:
    # Order vertices by weight descending.
    # Maintain a set of available vertices.
    # When we include vertex v, remove v and all its neighbors from available.
    # Upper bound: sum of weights of available vertices (simple) or tighter.
    
    # Let's use a more efficient approach:
    # Since the graph is sparse (each vertex has limited conflicts), 
    # and we want to include small numbers (high weight), let's try:
    
    # Phase 1: Greedy initial solution
    # Phase 2: Local search / B&B to improve
    
    # Precompute conflict masks
    # conflict[i] = bitmask of all j where lcm(i+2, j+2) <= 256 and j != i
    conflict = [0] * num_vertices
    
    for i in range(num_vertices):
        a = i + 2
        for j in range(i + 1, num_vertices):
            b = j + 2
            g = gcd(a, b)
            l = a * b // g
            if l <= 256:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # Sort vertices by weight descending (i.e., by value ascending)
    # Order: vertex 0 (value 2), vertex 1 (value 3), ..., vertex 254 (value 256)
    # This is already in descending weight order!
    
    # B&B with bitmasks
    # State: (start_idx, available_mask, current_weight, current_set)
    # But available_mask can be up to 255 bits. Python ints handle this.
    
    best_weight = 0.0
    best_set = []
    
    # Precompute: for each vertex i, the mask of vertices j > i that conflict with i
    # This helps when we include i: we need to remove those from future consideration.
    
    # Upper bound: sum of weights of all available vertices
    # But this is loose. Better: for each available vertex, its weight.
    # Since weights decrease with index, a tighter bound is possible.
    
    # Let's use a recursive B&B.
    # To avoid deep recursion, use iterative or limit depth.
    # With 255 vertices, depth 255 is too deep for Python recursion.
    
    # Alternative: use a greedy + local search approach.
    # Or: use the fact that the graph has special structure.
    
    # Let's try a different approach:
    # Key observation: 
    # - All numbers > 128 are pairwise compatible? 
    #   lcm(a,b) for a,b in [129,256]: min lcm is when a and b share large factors.
    #   lcm(a,b) >= max(a,b) >= 129. But can it be <= 256?
    #   lcm(a,b) = a*b/gcd(a,b). For this to be <= 256, we need a*b <= 256*gcd(a,b).
    #   Since a,b >= 129, a*b >= 129*129 = 16641. 256*gcd(a,b) <= 256*256 = 65536.
    #   So it's possible! E.g., a=128, b=128 (but they're not distinct in range).
    #   a=129, b=258? No, b <= 256.
    #   a=129, b=130: gcd=1, lcm=129*130=16770 > 256. Compatible.
    #   a=129, b=193: gcd=1, lcm=129*193 > 256. Compatible.
    #   a=128, b=192: gcd=64, lcm=128*192/64=384 > 256. Compatible.
    #   a=128, b=256: gcd=128, lcm=256. <= 256! Incompatible!
    #   So 128 and 256 are incompatible.
    #   a=130, b=256: gcd=2, lcm=130*256/2=16640 > 256. Compatible.
    #   a=129, b=258? Out of range.
    #   a=144, b=192: gcd=48, lcm=144*192/48=576 > 256. Compatible.
    #   a=128, b=128: same number.
    #   a=136, b=256: gcd=8, lcm=136*256/8=4352 > 256. Compatible.
    #   a=160, b=256: gcd=32, lcm=160*256/32=1280 > 256. Compatible.
    #   a=192, b=256: gcd=64, lcm=192*256/64=768 > 256. Compatible.
    #   a=224, b=256: gcd=32, lcm=224*256/32=1792 > 256. Compatible.
    #   a=256, b=128: same as above.
    #   
    # So among [129,256], only some pairs are incompatible.
    # Let's check: which pairs (a,b) with 129 <= a < b <= 256 have lcm(a,b) <= 256?
    # lcm(a,b) <= 256 implies a*b/gcd(a,b) <= 256.
    # Since a >= 129, b >= a+1 >= 130: a*b >= 129*130 = 16770.
    # So gcd(a,b) >= 16770/256 ≈ 65.5. So gcd(a,b) >= 66.
    # For a,b in [129,256], when is gcd(a,b) >= 66?
    # a and b must share a large factor.
    # E.g., a=129, b=193: gcd=1. No.
    # a=129, b=258: out of range.
    # a=132, b=198: gcd=6, lcm=132*198/6=4356. No.
    # a=128, b=192: a is not >= 129.
    # a=135, b=180: gcd=45, lcm=135*180/45=540. No.
    # a=129, b=256: gcd=1, lcm=129*256. No.
    # It seems like for a,b both >= 129, a*b is large enough that lcm > 256 almost always.
    # Let's verify: min a*b for a<b in [129,256] is 129*130=16770.
    # Max gcd is min(a,b) = a <= 128 (since a < b <= 256, a can be up to 255, but a < b).
    # Wait, a can be 129, b=258? No, b <= 256. So a < b <= 256, a >= 129.
    # Max gcd(a,b) for a < b: when b = 2a, gcd = a. But 2a <= 256 means a <= 128.
    # So for a >= 129, b cannot be 2a (since 2a >= 258 > 256).
    # Next: b = 3a/2? gcd might be a/2.
    # In general, for a >= 129, b <= 256, the ratio b/a < 2.
    # gcd(a,b) <= a/2 (since b is not a multiple of a).
    # Actually, gcd(a,b) <= a/2 is not always true. E.g., a=129, b=193, gcd=1.
    # But max gcd for a < b is at most a/2 (when b=2a, but that's not in range).
    # More precisely, if d = gcd(a,b), then a = d*a', b = d*b', gcd(a',b')=1, a' < b'.
    # lcm = d*a'*b'. For this to be <= 256, and a = d*a' >= 129:
    # d*a'*b' <= 256 and d*a' >= 129.
    # So b' <= 256/(d*a') <= 256/129 ≈ 1.98. Since b' > a' >= 1 and integer, b' <= 1.
    # But b' > a' >= 1, so b' >= 2. Contradiction!
    # Therefore, for a,b in [129, 256] with a < b, lcm(a,b) > 256 ALWAYS.
    # So all numbers in [129, 256] are pairwise compatible!
    
    # This is a huge simplification. The "large" set [129,256] is a clique in the compatibility graph.
    # So we can include all of [129,256] freely.
    # The question is: which small numbers [2,128] can we add without creating conflicts?
    # A small number a is compatible with all large numbers b in [129,256] iff lcm(a,b) > 256 for all b.
    # lcm(a,b) <= 256 for some b in [129,256] means a is "blocked" by that b.
    # If we include a, we must exclude all b in [129,256] where lcm(a,b) <= 256.
    
    # So the problem reduces to:
    # - Start with all large numbers [129,256].
    # - Consider adding small numbers [2,128]. Each small number a, if added, 
    #   forces us to remove certain large numbers (those b where lcm(a,b) <= 256).
    #   Also, small numbers may conflict with each other.
    # - We want to maximize sum of reciprocals.
    
    # For each small a, define:
    #   cost(a) = sum of 1/b for b in [129,256] where lcm(a,b) <= 256
    #   gain(a) = 1/a
    #   net_gain(a) = gain(a) - cost(a)
    # But costs overlap if multiple small numbers are chosen.
    
    # Also, small numbers conflict with each other if lcm(a1, a2) <= 256.
    
    # This is still a weighted independent set on the small numbers, 
    # with modified weights that account for the large number costs.
    
    # But the overlap of costs makes it complex. Let's use B&B on the small numbers.
    
    # Small numbers: 2..128 (127 numbers)
    # For each small a, precompute:
    #   - Conflicts with other small numbers: lcm(a, a') <= 256
    #   - Large numbers that must be removed: L(a) = {b in [129,256] : lcm(a,b) <= 256}
    
    smalls = list(range(2, 129))
    num_small = len(smalls)
    small_idx = {x: i for i, x in enumerate(smalls)}
    
    # Precompute small-small conflicts
    small_conflict = [0] * num_small
    for i in range(num_small):
        a = smalls[i]
        for j in range(i + 1, num_small):
            b = smalls[j]
            if lcm(a, b) <= 256:
                small_conflict[i] |= (1 << j)
                small_conflict[j] |= (1 << i)
    
    # Precompute large removals for each small
    # large_removal[i] = bitmask of large indices to remove
    # Large numbers: 129..256, indexed 0..127
    larges = list(range(129, 257))
    num_large = len(larges)
    large_idx = {x: i for i, x in enumerate(larges)}
    
    large_removal = [0] * num_small
    large_weights = [1.0 / x for x in larges]
    sum_large = sum(large_weights)
    
    for i, a in enumerate(smalls):
        for j, b in enumerate(larges):
            if lcm(a, b) <= 256:
                large_removal[i] |= (1 << j)
    
    # Now B&B on small numbers
    # State: (start_idx, available_small_mask, removed_large_mask, current_weight_small)
    # Total weight = current_weight_small + (sum of 1/b for b not removed)
    #            = current_weight_small + sum_large - sum(1/b for b removed)
    
    # Precompute weight of each large removal mask? Too many masks.
    # Instead, track removed_large_mask and compute penalty on the fly.
    # But that's slow. Alternative: track penalty_sum.
    
    best_total = 0.0
    best_small_mask = 0
    
    # Upper bound: current_weight_small + sum of weights of available smalls + sum_large
    # (ignoring penalties, so it's an upper bound)
    
    def popcount_weight(mask, weights_list):
        """Sum of weights for bits set in mask."""
        s = 0.0
        m = mask
        while m:
            lb = m & (-m)
            idx = lb.bit_length() - 1
            s += weights_list[idx]
            m ^= lb
        return s
    
    # Precompute prefix sums of small weights for bounding? 
    # Actually, let's just use the simple bound.
    
    # To speed up, precompute for each small i, the weight contribution and removal cost.
    small_weights = [1.0 / x for x in smalls]
    
    # Precompute removal cost for each small
    removal_costs = []
    for i in range(num_small):
        cost = 0.0
        m = large_removal[i]
        while m:
            lb = m & (-m)
            idx = lb.bit_length() - 1
            cost += large_weights[idx]
            m ^= lb
        removal_costs.append(cost)
    
    # B&B
    def bb(start, avail_mask, removed_mask, cur_small_weight, cur_removal_weight):
        nonlocal best_total, best_small_mask
        
        if time.time() > deadline:
            return
        
        # Current total
        total = cur_small_weight + sum_large - cur_removal_weight
        if total > best_total:
            best_total = total
            # Reconstruct small mask from avail and decisions... 
            # We need to track which smalls are selected.
            # Let's pass the selected mask instead.
        
        # Upper bound
        if not avail_mask:
            return
        # Bound: cur_small_weight + sum of weights of available smalls + sum_large
        # (worst case: no additional removals)
        bound = cur_small_weight + popcount_weight(avail_mask, small_weights) + sum_large
        if bound <= best_total + 1e-12:
            return
        
        # Pick next small to branch on
        # Choose the one with highest weight (smallest value)
        # Or the one with best net gain?
        # Let's pick the smallest available small (highest weight).
        m = avail_mask
        best_idx = -1
        best_w = -1.0
        while m:
            lb = m & (-m)
            idx = lb.bit_length() - 1
            if small_weights[idx] > best_w:
                best_w = small_weights[idx]
                best_idx = idx
            m ^= lb
        
        if best_idx == -1:
            return
        
        bit = 1 << best_idx
        x = smalls[best_idx]
        
        # Branch 1: Include x
        # Remove x and its conflicts from avail
        new_avail = avail_mask & ~bit
        new_avail &= ~small_conflict[best_idx]
        
        new_removed = removed_mask | large_removal[best_idx]
        # Additional removal weight
        add_removed = new_removed & ~removed_mask
        add_weight = 0.0
        m2 = add_removed
        while m2:
            lb = m2 & (-m2)
            idx = lb.bit_length() - 1
            add_weight += large_weights[idx]
            m2 ^= lb
        
        new_small_weight = cur_small_weight + small_weights[best_idx]
        new_removal_weight = cur_removal_weight + add_weight
        
        bb(best_idx + 1, new_avail, new_removed, new_small_weight, new_removal_weight)
        
        if time.time() > deadline:
            return
        
        # Branch 2: Exclude x
        bb(best_idx + 1, avail_mask & ~bit, removed_mask, cur_small_weight, cur_removal_weight)
    
    # We need to track the actual set. Let's modify to pass selected mask.
    # But that makes the recursion signature longer. 
    # Alternative: after B&B finds best_total, do a second pass to reconstruct.
    # Or: store the best selection.
    
    # Let's redo with selected mask tracking.
    best_total = 0.0
    best_selected = 0  # bitmask of selected smalls
    
    def bb2(start, avail_mask, removed_mask, selected_mask, cur_small_weight, cur_removal_weight):
        nonlocal best_total, best_selected
        
        if time.time() > deadline:
            return
        
        total = cur_small_weight + sum_large - cur_removal_weight
        if total > best_total:
            best_total = total
            best_selected = selected_mask
        
        if not avail_mask:
            return
        
        bound = cur_small_weight + popcount_weight(avail_mask, small_weights) + sum_large
        if bound <= best_total + 1e-12:
            return
        
        m = avail_mask
        best_idx = -1
        best_w = -1.0
        while m:
            lb = m & (-m)
            idx = lb.bit_length() - 1
            if small_weights[idx] > best_w:
                best_w = small_weights[idx]
                best_idx = idx
            m ^= lb
        
        if best_idx == -1:
            return
        
        bit = 1 << best_idx
        
        # Branch 1: Include
        new_avail = avail_mask & ~bit
        new_avail &= ~small_conflict[best_idx]
        new_removed = removed_mask | large_removal[best_idx]
        add_removed = new_removed & ~removed_mask
        add_weight = 0.0
        m2 = add_removed
        while m2:
            lb = m2 & (-m2)
            idx = lb.bit_length() - 1
            add_weight += large_weights[idx]
            m2 ^= lb
        
        bb2(best_idx + 1, new_avail, new_removed, selected_mask | bit, 
            cur_small_weight + small_weights[best_idx], cur_removal_weight + add_weight)
        
        if time.time() > deadline:
            return
        
        # Branch 2: Exclude
        bb2(best_idx + 1, avail_mask & ~bit, removed_mask, selected_mask, 
            cur_small_weight, cur_removal_weight)
    
    initial_avail = (1 << num_small) - 1
    bb2(0, initial_avail, 0, 0, 0.0, 0.0)
    
    # Construct final set
    final_set = set()
    
    # Add all large numbers not removed
    removed_mask_final = 0
    for i in range(num_small):
        if best_selected & (1 << i):
            removed_mask_final |= large_removal[i]
    
    for j, b in enumerate(larges):
        if not (removed_mask_final & (1 << j)):
            final_set.add(b)
    
    # Add selected smalls
    for i in range(num_small):
        if best_selected & (1 << i):
            final_set.add(smalls[i])
    
    if not final_set:
        final_set = {2}
    
    return list(final_set)
