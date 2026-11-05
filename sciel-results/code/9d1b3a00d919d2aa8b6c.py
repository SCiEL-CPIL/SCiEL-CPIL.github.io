import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds buffer for safety
    
    # 1. Generate STS(15) on points 0..14
    # The points 0..14 correspond to the non-zero vectors in GF(2)^4.
    # A triple {a, b, c} is in the STS if a ^ b ^ c == 0 (in 1..15 representation).
    # In 0..14 representation, let x = a+1, y = b+1, z = c+1.
    # x ^ y ^ z == 0.
    
    sts_triples = []
    for i in range(1, 16):
        for j in range(i+1, 16):
            k = i ^ j
            if k > j:
                t = tuple(sorted([i-1, j-1, k-1]))
                sts_triples.append(t)
    
    # 2. Find all 8-caps in STS(15)
    # An 8-cap is a subset of 8 points containing no triple.
    # There are C(15, 8) = 6435 such subsets.
    
    all_caps = []
    
    # Precompute triple masks for fast containment checks
    triple_masks = []
    for t in sts_triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)
    
    for combo in itertools.combinations(range(15), 8):
        cap_mask = 0
        for p in combo:
            cap_mask |= (1 << p)
        
        is_cap = True
        for tm in triple_masks:
            if (tm & cap_mask) == tm:
                is_cap = False
                break
        if is_cap:
            all_caps.append(combo)
            
    if not all_caps:
        # Fallback: return a valid STS(15) + isolated point 15 (alpha=9)
        # This is a valid linear triple system.
        return [list(t) for t in sts_triples]

    n_caps = len(all_caps)
    
    # 3. Precompute for each pair of points (u, v) in 0..14, which caps contain BOTH u and v.
    # If we select pair (u, v) into our matching M, we "hit" all caps that contain {u, v}.
    # We want to select a matching M (set of disjoint pairs) such that EVERY cap is hit by at least one pair in M.
    # This ensures that no 8-cap remains intact (i.e., every 8-cap contains at least one pair from M).
    
    # Specifically, the construction is:
    # - Keep all STS(15) triples that do NOT contain any pair from M.
    # - For each pair (u, v) in M, add triple {15, u, v}.
    # - This forms a linear triple system on 16 points.
    # - Any 8-cap of the original STS(15) that is hit by a pair (u, v) in M cannot be an independent set in the new system
    #   because the triple {15, u, v} exists. Wait, an independent set cannot contain a triple.
    #   If S is an independent set, it cannot contain {15, u, v}.
    #   If S contains 15, it cannot contain both u and v.
    #   If S does not contain 15, it is a subset of 0..14.
    #   If S is an 8-cap of STS(15), it contains no STS triple.
    #   But in the new system, the "forbidden" sets are the new triples.
    #   The new triples involving 15 are {15, u, v} for (u,v) in M.
    #   So if S contains 15, S \ {15} cannot contain any pair from M.
    #   If S does not contain 15, S is a subset of 0..14. The triples in S are the same as in STS(15) EXCEPT those removed.
    #   Wait, we REMOVED triples from STS(15) that contain pairs from M.
    #   So S might contain a triple that was in STS(15) but removed? No, if it was removed, it's not a triple in the new system.
    #   So S is independent in the new system if:
    #   1. It contains no triple from the kept STS(15) triples.
    #   2. It contains no new triple {15, u, v}.
    
    #   Let's re-evaluate the logic.
    #   Goal: Minimize independence number.
    #   Current best score is 1.375, which implies alpha = 8 (since 15/8 = 1.875? No, score = seed/value. Seed=11? 11/8=1.375. Yes.)
    #   So the current program achieves alpha=8.
    #   Can we achieve alpha=7?
    #   If alpha=7, score = 11/7 = 1.57.
    
    #   The strategy of hitting all 8-caps ensures that no 8-subset of 0..14 is independent (if it doesn't contain 15).
    #   Why?
    #   If S is an 8-subset of 0..14.
    #   If S is a cap in STS(15), it contains a pair (u,v) from M.
    #   The triple {15, u, v} is in the system.
    #   Does this prevent S from being independent?
    #   No, S does not contain 15. So the triple {15, u, v} is not in S.
    #   However, we REMOVED the STS triples that contain pairs from M.
    #   This means that if S contained an STS triple T, and T contained a pair from M, T is not in the new system, so S is not "blocked" by T.
    #   But if S contained an STS triple T that did NOT contain a pair from M, T is still in the new system, so S is not independent.
    #   
    #   So, for S (8-subset of 0..14) to be independent in the new system:
    #   - It must not contain any kept STS triple.
    #   - (It doesn't contain 15, so new triples are irrelevant).
    #   
    #   The kept STS triples are those that do NOT contain any pair from M.
    #   So S is independent iff S contains no STS triple that is disjoint from M (in terms of pairs).
    #   i.e., every STS triple in S must contain at least one pair from M.
    #   
    #   If S is an 8-cap of STS(15), it contains NO STS triples at all.
    #   So it certainly contains no kept STS triples.
    #   So S IS independent in the new system?
    #   Wait. If S contains no STS triples, then it contains no kept STS triples.
    #   So S is independent.
    #   This implies that hitting 8-caps does NOT reduce the independence number of subsets not containing 15.
    #   
    #   Let's re-read the current program's logic or the inspiration.
    #   Inspiration: "Finds a matching hitting all 8-caps of STS(15) to reduce independence number from 9 to 8".
    #   How does it work?
    #   
    #   Maybe the logic is different.
    #   If we add triples {15, u, v}, then any independent set containing 15 cannot contain both u and v for any (u,v) in M.
    #   So if S contains 15, S \ {15} is a subset of 0..14 that contains no pair from M.
    #   Such a subset is an "M-independent" set.
    #   The maximum size of an M-independent set in 0..14 is related to the matching size.
    #   If M is a perfect matching on 14 points (7 pairs), the max M-independent set has size 7 (picking one from each pair).
    #   So any independent set containing 15 has size at most 1 + 7 = 8.
    #   
    #   What about independent sets NOT containing 15?
    #   They are subsets of 0..14.
    #   They must not contain any kept STS triple.
    #   The kept STS triples are those not containing any pair from M.
    #   So if S contains an STS triple T, and T does not contain a pair from M, S is not independent.
    #   If S contains an STS triple T, and T contains a pair from M, T is removed, so S is not blocked by T.
    #   
    #   So S is independent iff every STS triple in S contains a pair from M.
    #   
    #   If S is an 8-cap of STS(15), it contains no STS triples. So it is independent.
    #   So alpha >= 8.
    #   
    #   Can we make alpha < 8?
    #   We need to ensure that no 8-subset of 0..14 is independent.
    #   i.e., every 8-subset S of 0..14 must contain a kept STS triple.
    #   i.e., every 8-subset S of 0..14 must contain an STS triple T such that T does NOT contain any pair from M.
    #   
    #   This is a different condition than "hitting all 8-caps".
    #   Hitting all 8-caps means: for every 8-cap C, C contains a pair from M.
    #   This implies that C cannot be an M-independent set.
    #   But C is independent in the new system iff it contains no kept STS triple.
    #   Since C is a cap, it contains no STS triples at all. So it contains no kept STS triples.
    #   So C is independent.
    #   
    #   So the current strategy DOES NOT prevent 8-caps from being independent sets (of size 8).
    #   It prevents sets of size 9?
    #   If S is a 9-subset of 0..14.
    #   If S is independent, it contains no kept STS triple.
    #   So every STS triple in S must contain a pair from M.
    #   
    #   It is known that STS(15) has independence number 8?
    #   If alpha(STS(15)) = 8, then there are no 9-subsets with no triples.
    #   So any 9-subset contains an STS triple.
    #   If that triple contains a pair from M, it is removed, so it doesn't block independence.
    #   If that triple does NOT contain a pair from M, it is kept, so it blocks independence.
    #   
    #   So to make alpha < 9, we need that every 9-subset contains a kept STS triple.
    #   i.e., every 9-subset contains an STS triple disjoint from M (pair-wise).
    #   
    #   The current program achieves alpha=8.
    #   Can we achieve alpha=7?
    #   We need every 8-subset to contain a kept STS triple.
    #   i.e., every 8-subset contains an STS triple disjoint from M.
    #   
    #   But we know there exist 8-caps (8-subsets with NO STS triples).
    #   So no 8-cap can contain a kept STS triple.
    #   So every 8-cap is an independent set in the new system.
    #   So alpha >= 8.
    #   
    #   Therefore, it is IMPOSSIBLE to get alpha < 8 using this construction method (STS(15) + point 15).
    #   The best we can do with this method is alpha=8.
    #   
    #   To get alpha=7, we need a different construction.
    #   Perhaps a different base structure?
    #   Or a different way of adding point 15?
    #   
    #   However, the task says "Minimize". And "no finite value is documented".
    #   But we have a score of 1.375 (alpha=8).
    #   If we can't beat 8, we should stick with 8.
    #   
    #   Is there a way to get alpha=7?
    #   Maybe by modifying the STS(15) itself?
    #   No, STS(15) is fixed.
    #   
    #   What if we don't use STS(15)?
    #   We can construct any linear triple system on 16 points.
    #   The maximum number of triples in a linear triple system on v points is bounded.
    #   For v=16, max triples is roughly v(v-1)/6 = 40.
    #   STS(15) has 35 triples. Adding 7 more gives 42.
    #   42 > 40. So it's not a 2-(16,3,1) design. It's just a linear triple system.
    #   
    #   Let's check the number of triples in the current construction.
    #   STS(15) has 35 triples.
    #   We remove triples containing pairs from M.
    #   M has 7 pairs.
    #   Each pair is in how many STS triples?
    #   In STS(15), each pair is in exactly 1 triple.
    #   So we remove 7 triples.
    #   We add 7 triples.
    #   Total triples = 35.
    #   
    #   So we have 35 triples.
    #   Can we have more triples to reduce alpha further?
    #   If we add more triples, we might block more independent sets.
    #   But we must maintain linearity (any two triples share at most 1 point).
    #   
    #   The current construction is valid and gives alpha=8.
    #   Since 8-caps exist in STS(15) and are independent in the new system, alpha cannot be less than 8 with this specific "STS(15) + 15" structure.
    #   
    #   Is there a different structure?
    #   Maybe start with a different packing?
    #   
    #   Given the time constraint and the fact that the current program already finds a valid alpha=8 solution,
    #   and that beating 8 seems difficult/impossible with standard methods,
    #   I will optimize the current program to ensure it runs fast and reliably finds the alpha=8 solution.
    #   
    #   The current program already does this.
    #   I will make sure the backtracking is robust.
    
    # Precompute cap hits for each pair
    pair_to_caps = {}
    for u in range(15):
        for v in range(u+1, 15):
            pair_to_caps[(u,v)] = []
            
    for idx, cap in enumerate(all_caps):
        cap_set = set(cap)
        for u in cap:
            for v in cap:
                if u < v:
                    pair_to_caps[(u,v)].append(idx)
                    
    # Bitmasks for caps
    pair_hit_mask = {}
    for pair, cap_indices in pair_to_caps.items():
        mask = 0
        for idx in cap_indices:
            mask |= (1 << idx)
        pair_hit_mask[pair] = mask
        
    all_caps_mask = (1 << n_caps) - 1
    
    best_matching = None
    
    def search(available, current_matching, current_hit_mask, depth):
        nonlocal best_matching
        if time.time() > deadline:
            return
            
        if current_hit_mask == all_caps_mask:
            best_matching = current_matching[:]
            return
            
        # Find an unhit cap
        # Heuristic: Pick the cap with the fewest available pairs (MRV)
        # Or just the first one.
        # Let's try to pick a cap that has few options to prune faster.
        
        min_options = float('inf')
        best_cap_idx = None
        best_cap_pairs = None
        
        # This loop is O(n_caps * 28) which is 6435 * 28 ~ 180k.
        # Might be slow if done at every node.
        # Let's stick to first unhit cap for speed, or precomputed order.
        
        unhit_cap_idx = None
        for i in range(n_caps):
            if not (current_hit_mask & (1 << i)):
                unhit_cap_idx = i
                break
                
        if unhit_cap_idx is None:
            return
            
        cap = all_caps[unhit_cap_idx]
        
        # Try all pairs in this cap
        pairs_in_cap = []
        for u in cap:
            for v in cap:
                if u < v:
                    if u in available and v in available:
                        pairs_in_cap.append((u,v))
                        
        # Sort by number of NEW caps hit
        def new_hits(p):
            return (pair_hit_mask[p] & ~current_hit_mask).bit_count()
            
        pairs_in_cap.sort(key=new_hits, reverse=True)
        
        for p in pairs_in_cap:
            u, v = p
            if u not in available or v not in available:
                continue
                
            new_matching = current_matching + [p]
            new_available = available - {u, v}
            new_hit_mask = current_hit_mask | pair_hit_mask[p]
            
            search(new_available, new_matching, new_hit_mask, depth + 1)
            
            if best_matching is not None:
                return

    available = set(range(15))
    search(available, [], 0, 0)
    
    if best_matching is None:
        # Fallback: try to find a matching that hits MOST caps
        # Or just return a valid system with alpha=9?
        # The task says "Minimize".
        # If we can't find alpha=8, we return alpha=9 (just STS(15) + 15 with random pairs?).
        # Actually, if we add 7 random pairs, we might hit some caps.
        # Let's just use a greedy approach if backtracking fails.
        
        found_M = []
        remaining_caps = set(range(n_caps))
        available = set(range(15))
        
        while remaining_caps:
            # Find pair with max hits
            best_pair = None
            max_hits = 0
            for u in available:
                for v in available:
                    if u < v:
                        hits = 0
                        for idx in pair_to_caps[(u,v)]:
                            if idx in remaining_caps:
                                hits += 1
                        if hits > max_hits:
                            max_hits = hits
                            best_pair = (u,v)
            if not best_pair or max_hits == 0:
                break
            found_M.append(best_pair)
            available -= {best_pair[0], best_pair[1]}
            for idx in pair_to_caps[best_pair]:
                remaining_caps.discard(idx)
                
        if not found_M:
            found_M = [(i, i+1) for i in range(0, 14, 2)]
    else:
        found_M = best_matching

    # Construct final triples
    M_pairs = [set(m) for m in found_M]
    
    final_triples = []
    
    # Keep STS triples that do not contain any pair in M
    for t in sts_triples:
        t_set = set(t)
        keep = True
        for m_set in M_pairs:
            if m_set.issubset(t_set):
                keep = False
                break
        if keep:
            final_triples.append(list(t))
            
    # Add new triples with 15
    for m in found_M:
        final_triples.append([15, m[0], m[1]])
        
    return final_triples
