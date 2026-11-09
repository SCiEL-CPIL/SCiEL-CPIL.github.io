import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 16
    k = 4
    
    # Generate all 4-subsets as bitmasks
    all_masks = []
    mask_to_idx = {}
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)
        mask_to_idx[mask] = len(all_masks) - 1
        
    total = len(all_masks)
    
    # Precompute conflictors for each pair of masks
    # conflictors[i][j] is a list of indices w such that {all_masks[i], all_masks[j], all_masks[w]} is a sunflower
    # This is memory intensive if stored as lists of lists.
    # Let's estimate: 2184 * 2183 / 2 pairs.
    # Each list has size <= 70.
    # Total elements ~ 2.3M * 20 = 46M.
    # Python list overhead is high.
    # Maybe compute on the fly in the update step?
    # Update step: For a pair (u, v), find all w conflicting.
    # u, v are masks.
    # I = u & v
    # Forbid = (u | v) & ~I
    # w must contain I and be disjoint from Forbid.
    # w is a 4-subset.
    
    def get_conflictors(u, v):
        """Returns list of indices w that form a sunflower with u and v."""
        I = u & v
        Forbid = (u | v) & ~I
        # w must contain I.
        # w \ I must be a subset of [n] \ (I | Forbid)
        # Available elements for w \ I: All elements not in I and not in Forbid.
        # Note: I and Forbid are disjoint.
        avail_mask = ((1 << n) - 1) & ~I & ~Forbid
        need = k - bin(I).count('1')
        
        res = []
        # Iterate all combinations of 'need' elements from avail_mask
        avail_elems = [i for i in range(n) if avail_mask & (1 << i)]
        if len(avail_elems) < need:
            return res
        
        for combo in combinations(avail_elems, need):
            w_mask = I
            for x in combo:
                w_mask |= (1 << x)
            res.append(mask_to_idx[w_mask])
        return res

    # We will maintain conflict_count for each set index
    conflict_count = [0] * total
    
    def add_to_family(idx):
        """Add set idx to family, updating conflict counts."""
        u = all_masks[idx]
        # For each existing set v in family, the pair (u, v) is new.
        # We need to increment conflict_count[w] for all w conflicting with (u, v).
        for v_idx in family:
            if v_idx == idx: continue
            v = all_masks[v_idx]
            conflictors = get_conflictors(u, v)
            for w_idx in conflictors:
                conflict_count[w_idx] += 1

    def remove_from_family(idx):
        """Remove set idx from family, updating conflict counts."""
        u = all_masks[idx]
        for v_idx in family:
            if v_idx == idx: continue
            v = all_masks[v_idx]
            conflictors = get_conflictors(u, v)
            for w_idx in conflictors:
                conflict_count[w_idx] -= 1

    family = []
    family_set = set()

    def try_add(idx):
        if conflict_count[idx] == 0 and idx not in family_set:
            family.append(idx)
            family_set.add(idx)
            add_to_family(idx)
            return True
        return False

    def run_greedy(order_indices):
        nonlocal family, family_set, conflict_count
        family = []
        family_set = set()
        conflict_count = [0] * total
        
        for idx in order_indices:
            if time.time() > deadline - 2:
                break
            try_add(idx)
        return family[:]

    def local_search(family_init, max_rounds=50):
        nonlocal family, family_set, conflict_count
        family = list(family_init)
        family_set = set(family)
        conflict_count = [0] * total
        
        # Rebuild conflict counts
        for i in range(len(family)):
            for j in range(i+1, len(family)):
                u = all_masks[family[i]]
                v = all_masks[family[j]]
                conflictors = get_conflictors(u, v)
                for w_idx in conflictors:
                    conflict_count[w_idx] += 1
        
        for _ in range(max_rounds):
            if time.time() > deadline - 3:
                break
            
            added = False
            # Try to add any compatible set
            # Optimization: Only check sets not in family
            candidates = [idx for idx in range(total) if idx not in family_set]
            random.shuffle(candidates)
            
            for cand in candidates:
                if time.time() > deadline - 3:
                    break
                if conflict_count[cand] == 0:
                    try_add(cand)
                    added = True
                    break
            
            if not added:
                # Try swap: remove 1, add 2
                if len(family) >= 1:
                    swap_done = False
                    # Shuffle family indices to try different removals
                    rem_indices = list(range(len(family)))
                    random.shuffle(rem_indices)
                    
                    for ri in rem_indices:
                        if time.time() > deadline - 3:
                            break
                        rem_idx = family[ri]
                        
                        # Remove rem_idx
                        remove_from_family(rem_idx)
                        family.pop(ri)
                        family_set.remove(rem_idx)
                        
                        # Try to add 2 sets
                        found = []
                        candidates2 = [idx for idx in range(total) if idx not in family_set and idx != rem_idx]
                        random.shuffle(candidates2)
                        
                        for cand in candidates2:
                            if len(found) >= 2:
                                break
                            if time.time() > deadline - 3:
                                break
                            if conflict_count[cand] == 0:
                                found.append(cand)
                                # Tentatively add to update counts for the second one?
                                # No, we need to check if the second one is compatible with the first found one too.
                                # So we must simulate adding the first one.
                                # This is getting complex. 
                                # Let's just check if conflict_count is 0. 
                                # If we add found[0], conflict_count for found[1] might change.
                                # So we must update conflict_count after adding found[0].
                                
                                # Let's do this properly:
                                # Save state? Or just check compatibility of found[1] against family + found[0].
                                # Since we have conflict_count for the *current* family (without found[0]),
                                # we can't just check conflict_count[found[1]] == 0.
                                # We need to check if found[1] conflicts with any pair in (family + found[0]).
                                # Pairs in family: already counted.
                                # Pairs involving found[0]: (found[0], v) for v in family.
                                # So we need to check if found[1] is in conflictors(found[0], v) for any v in family.
                                
                                # This is O(N) check.
                                # Let's implement a helper: is_compatible_with_set(cand, other_idx)
                                # which checks if cand conflicts with (other_idx, v) for all v in family?
                                # No, it checks if cand forms sunflower with any pair in family U {other_idx}.
                                
                                # Simpler: Just use the old is_compatible for the second candidate?
                                # It's O(N^2). N is ~200. 40k ops.
                                # If we do this for many candidates, it might be slow.
                                
                                # Let's just try to add the first one, then check the second with the updated counts.
                                # But we are in a "what if" scenario.
                                
                                # Alternative:
                                # Just pick one candidate at a time?
                                # "Remove 1, Add 1" doesn't increase size.
                                # "Remove 1, Add 2" increases size by 1.
                                
                                # Let's implement a helper to check compatibility of cand with family + extra_sets
                                def check_compat(cand, extra):
                                    # Check against pairs in family
                                    # conflict_count[cand] is for family.
                                    # If conflict_count[cand] > 0, return False.
                                    if conflict_count[cand] > 0:
                                        return False
                                    
                                    # Check against pairs (extra, v) for v in family
                                    # And pairs (extra1, extra2) if len(extra) > 1
                                    
                                    u = all_masks[cand]
                                    for e in extra:
                                        ve = all_masks[e]
                                        # Check if cand, e, v form sunflower for any v in family
                                        # This is equivalent to:
                                        # Does cand conflict with (e, v)?
                                        # We can check by seeing if cand is in get_conflictors(e, v)
                                        # Or just compute intersection logic.
                                        I_ev = ve & all_masks[v] # No, v is in family
                                        
                                        # Let's just use the bitmask logic directly for speed in this specific check
                                        # Sunflower: A, B, C. Intersections equal.
                                        # A=cand, B=e.
                                        # Inter = cand & e.
                                        # We need C (from family) such that C & cand = Inter AND C & e = Inter.
                                        # This is expensive.
                                        
                                        # Let's stick to the conflict_count approach for the main loop,
                                        # and for the swap, we'll just try adding one set at a time 
                                        # and see if we can add a second one later?
                                        # No, that's not a swap.
                                        
                                        # Let's simplify: 
                                        # In the swap loop, just try to add ONE set.
                                        # If we remove 1 and add 1, size is same.
                                        # If we remove 1 and add 2, size +1.
                                        
                                        # To avoid complexity, let's just do "Add if possible".
                                        # If we can't add, we try to remove a "bad" set and see if 
                                        # removing it allows adding 2 others.
                                        
                                        # For now, let's just break out of the inner loop 
                                        # and rely on the next round of "Try to add" 
                                        # after a removal?
                                        
                                        # Actually, if we remove a set, conflict_count decreases.
                                        # So sets that were blocked might now be unblocked.
                                        # So we can just remove a set, then go back to "Try to add" loop.
                                        # If we add 1, size is same. 
                                        # If we add 2, size +1.
                                        
                                        # Let's modify the structure:
                                        # 1. Try Add.
                                        # 2. If fail, Try Remove 1, then Try Add (multiple times).
                                        
                                        return False
                                    
                                    return True
                                
                                # This is getting messy. Let's revert to a simpler local search 
                                # that uses the fast conflict_count for Add, 
                                # and for Swap, we just remove one set, 
                                # then repeatedly try to add any available set.
                                # If we add 2, we gain.
                                
                                # Let's implement that.
                                pass

                        # Re-add rem_idx if we didn't find improvement?
                        # Or just keep it removed if we added >= 1?
                        # If we added 1, size is same. 
                        # If we added 2, size +1.
                        
                        # Let's just do this:
                        # Remove rem_idx.
                        # Loop:
                        #   Find any idx with conflict_count == 0.
                        #   If found, add it.
                        #   If not, break.
                        # If added_count >= 2:
                        #   Keep new family.
                        # Else:
                        #   Re-add rem_idx.
                        
                        added_count = 0
                        while True:
                            if time.time() > deadline - 3:
                                break
                            # Find candidate
                            # Optimization: Check a random subset of candidates
                            # Or just iterate all? 2184 is small.
                            cand_found = -1
                            for c in range(total):
                                if c not in family_set and conflict_count[c] == 0:
                                    cand_found = c
                                    break
                            
                            if cand_found == -1:
                                break
                            
                            try_add(cand_found)
                            added_count += 1
                            
                        if added_count >= 2:
                            swap_done = True
                            break
                        else:
                            # Revert
                            family.append(rem_idx)
                            family_set.add(rem_idx)
                            add_to_family(rem_idx)
                            
                    if not swap_done:
                        break

        return family

    best_family = []
    
    # 1. Standard Lexicographic Order
    order = list(range(total))
    fam = run_greedy(order)
    if len(fam) > len(best_family):
        best_family = fam[:]
        
    # 2. Reverse Lexicographic Order
    if time.time() < deadline - 5:
        order_rev = list(reversed(range(total)))
        fam = run_greedy(order_rev)
        if len(fam) > len(best_family):
            best_family = fam[:]

    # 3. Random Permutations
    for attempt in range(25):
        if time.time() > deadline - 8:
            break
        
        random.seed(42 + attempt)
        perm = list(range(n))
        random.shuffle(perm)
        
        keys = []
        for idx in range(total):
            mask = all_masks[idx]
            elems = []
            for i in range(n):
                if mask & (1 << i):
                    elems.append(perm[i])
            elems_sorted_by_perm = sorted(elems)
            keys.append((tuple(elems_sorted_by_perm), idx))
            
        keys.sort()
        order_perm = [idx for _, idx in keys]
        
        fam = run_greedy(order_perm)
        if len(fam) > len(best_family):
            best_family = fam[:]
            
        # Local search on this family
        if len(fam) >= len(best_family) - 3:
            improved = local_search(fam, max_rounds=20)
            if len(improved) > len(best_family):
                best_family = improved[:]

    # 4. Pure random orderings
    for attempt in range(15):
        if time.time() > deadline - 5:
            break
        
        random.seed(100 + attempt)
        order_rand = list(range(total))
        random.shuffle(order_rand)
        
        fam = run_greedy(order_rand)
        if len(fam) > len(best_family):
            best_family = fam[:]
        
        if len(fam) >= len(best_family) - 3:
            improved = local_search(fam, max_rounds=15)
            if len(improved) > len(best_family):
                best_family = improved[:]

    # Final local search on best family
    if time.time() < deadline - 3 and len(best_family) > 0:
        improved = local_search(best_family, max_rounds=50)
        if len(improved) > len(best_family):
            best_family = improved[:]

    # Convert best_family indices back to lists of 1-based integers
    result = []
    for idx in best_family:
        elements = []
        for i in range(n):
            if all_masks[idx] & (1 << i):
                elements.append(i + 1)
        result.append(sorted(elements))
        
    return result
