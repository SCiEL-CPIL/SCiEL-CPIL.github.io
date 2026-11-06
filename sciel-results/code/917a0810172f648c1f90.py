import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    n = 16
    k = 4
    all_sets = list(combinations(range(16), 4))
    total = len(all_sets)  # 1820
    
    # Bitmask representation
    all_masks = []
    for combo in all_sets:
        m = 0
        for x in combo:
            m |= (1 << x)
        all_masks.append(m)
    
    # Precompute popcount for 16-bit values
    popcount = [0] * (1 << 16)
    for i in range(1, 1 << 16):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # For each pair (i, j), precompute:
    #   S = all_masks[i] & all_masks[j]
    #   union = all_masks[i] | all_masks[j]
    #   available = full_mask ^ union
    #   need = k - popcount[S]
    # A mask c conflicts with pair (i,j) iff:
    #   (c & S) == S  AND  (c & union) == S  AND  popcount[c] == k
    # i.e., c contains S, c has no elements from union\S, and c is size k
    
    # Precompute conflict info for each pair
    full_mask = (1 << 16) - 1
    pair_info = []  # (S, union, need)
    for i in range(total):
        for j in range(i + 1, total):
            S = all_masks[i] & all_masks[j]
            union = all_masks[i] | all_masks[j]
            need = k - popcount[S]
            pair_info.append((S, union, need))
    
    # For a candidate c, check against family F:
    # For each pair (a,b) in F, check if c conflicts
    # We can speed this up by precomputing for each c, which pairs it conflicts with
    
    # Actually, let's precompute for each mask c, the set of pair-indices it conflicts with
    # conflict_mask[c] = bitmask of pair indices where c forms sunflower with that pair
    # But total pairs = C(1820,2) ~ 1.6M, so this is large.
    
    # Alternative: for each c, precompute a "signature" and use it for fast checking.
    # For c to conflict with pair (i,j): c must contain S_ij and be disjoint from (union_ij \ S_ij)
    # i.e., c & S_ij == S_ij and c & (union_ij ^ S_ij) == 0
    
    # Let me just use a different strategy: maintain family as list, and for checking
    # a new candidate c against family F, iterate over pairs in F.
    # To make this fast, use numpy or just accept O(|F|^2) per check.
    
    # With |F| up to ~200, that's 40K pair checks per candidate, 1820 candidates = 73M ops.
    # In Python this might be slow. Let me optimize.
    
    # Precompute for each c: for each pair (i,j), does c conflict?
    # Store as a list of bitsets (one bit per pair)
    # 1.6M pairs * 1820 candidates = 3B bits = 375MB. Too much.
    
    # Let me use a smarter approach: for each c, the set of pairs it conflicts with
    # can be computed on the fly. For pair (i,j) with (S, union, need):
    # c conflicts iff (c & S) == S and (c & (union ^ S)) == 0
    # Note: union ^ S = (i|j) & ~(i&j) = symmetric difference
    
    # Let me precompute for each pair: S and diff = union ^ S
    pair_S = []
    pair_diff = []
    for i in range(total):
        for j in range(i + 1, total):
            S = all_masks[i] & all_masks[j]
            diff = all_masks[i] ^ all_masks[j]
            pair_S.append(S)
            pair_diff.append(diff)
    
    num_pairs = len(pair_S)
    
    # For fast checking: given c and family (list of indices), check all pairs in family
    def check_candidate(c, family_indices):
        """Check if c can be added. family_indices is list of indices into all_masks."""
        for ai in range(len(family_indices)):
            a = all_masks[family_indices[ai]]
            for bi in range(ai + 1, len(family_indices)):
                b = all_masks[family_indices[bi]]
                # Find pair index for (a,b) - need mapping
                # This is expensive. Let me use a different approach.
                S = a & b
                diff = a ^ b
                if (c & S) == S and (c & diff) == 0:
                    return False
        return True
    
    # Even better: precompute a dictionary mapping (mask_a, mask_b) -> pair_index
    # But that's 1820^2 entries.
    
    # Let me just do the direct check. With optimization, it should be fast enough.
    
    def greedy_randomized(seed):
        rng = random.Random(seed)
        order = list(range(total))
        rng.shuffle(order)
        family = []
        for idx in order:
            c = all_masks[idx]
            # Check against all pairs in family
            ok = True
            for ai in range(len(family)):
                a = family[ai]
                for bi in range(ai + 1, len(family)):
                    b = family[bi]
                    S = a & b
                    diff = a ^ b
                    if (c & S) == S and (c & diff) == 0:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                family.append(c)
        return family
    
    # Run multiple greedy starts
    best_family = []
    num_starts = 30
    for s in range(num_starts):
        if time.time() > DEADLINE - 40:
            break
        fam = greedy_randomized(s * 7919 + 13)
        if len(fam) > len(best_family):
            best_family = fam
    
    # Local search: try to improve by swaps
    if time.time() < DEADLINE - 5:
        family = list(best_family)
        best = family
        rng = random.Random(999)
        
        while time.time() < DEADLINE - 5:
            improved = False
            # Try adding one more
            for idx in range(total):
                c = all_masks[idx]
                if c in set(family):
                    continue
                ok = True
                for ai in range(len(family)):
                    a = family[ai]
                    for bi in range(ai + 1, len(family)):
                        b = family[bi]
                        S = a & b
                        diff = a ^ b
                        if (c & S) == S and (c & diff) == 0:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    family.append(c)
                    if len(family) > len(best):
                        best = list(family)
                    improved = True
                    break
            
            if not improved:
                # Try 1-out, 2-in
                if len(family) >= 1:
                    rem_idx = rng.randint(0, len(family) - 1)
                    rem = family[rem_idx]
                    temp = [x for x in family if x != rem]
                    temp_set = set(temp)
                    
                    added = []
                    for idx in range(total):
                        if len(added) == 2:
                            break
                        c = all_masks[idx]
                        if c in temp_set or c in added:
                            continue
                        ok = True
                        for ai in range(len(temp) + len(added)):
                            a = (temp + added)[ai]
                            for bi in range(ai + 1, len(temp) + len(added)):
                                b = (temp + added)[bi]
                                S = a & b
                                diff = a ^ b
                                if (c & S) == S and (c & diff) == 0:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            added.append(c)
                    
                    if len(added) == 2:
                        new_fam = temp + added
                        if len(new_fam) > len(best):
                            best = new_fam
                            family = new_fam
                            improved = True
            
            if not improved:
                # Try 2-out, 3-in
                if len(family) >= 2:
                    rem1_idx = rng.randint(0, len(family) - 1)
                    rem2_idx = rng.randint(0, len(family) - 1)
                    if rem1_idx == rem2_idx:
                        rem2_idx = (rem2_idx + 1) % len(family)
                    rem1 = family[rem1_idx]
                    rem2 = family[rem2_idx]
                    temp = [x for i, x in enumerate(family) if i != rem1_idx and i != rem2_idx]
                    temp_set = set(temp)
                    
                    added = []
                    for idx in range(total):
                        if len(added) == 3:
                            break
                        c = all_masks[idx]
                        if c in temp_set or c in added:
                            continue
                        ok = True
                        for ai in range(len(temp) + len(added)):
                            a = (temp + added)[ai]
                            for bi in range(ai + 1, len(temp) + len(added)):
                                b = (temp + added)[bi]
                                S = a & b
                                diff = a ^ b
                                if (c & S) == S and (c & diff) == 0:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            added.append(c)
                    
                    if len(added) == 3:
                        new_fam = temp + added
                        if len(new_fam) > len(best):
                            best = new_fam
                            family = new_fam
                            improved = True
            
            if not improved:
                break
        
        best_family = best
    
    # Convert to sorted lists
    result = []
    seen = set()
    for m in best_family:
        if m in seen:
            continue
        seen.add(m)
        elems = [i + 1 for i in range(16) if (m >> i) & 1]
        result.append(sorted(elems))
    
    return result
