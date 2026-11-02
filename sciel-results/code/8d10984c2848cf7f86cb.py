import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit

    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(start_time, deadline)
    
    return _direct_sum_construct(q, n, radius)

def _direct_sum_construct(q, n, radius):
    if n < 4:
        if n == 0:
            return [[]]
        if n == 1:
            return [[i] for i in range(q)]
        if n == 2:
            return [[a, b] for a in range(q) for b in range(q)]
        if n == 3:
            return [[a, b, (a + b) % q] for a in range(q) for b in range(q)]
    
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = []
        for h in tetracode:
            for tail in product(range(3), repeat=n - 4):
                result.append(h + list(tail))
        return result
    else:
        return [list(w) for w in product(range(q), repeat=n)]

def _solve_ternary_7_1(start_time, deadline):
    q, n, radius = 3, 7, 1
    N = q ** n  # 2187
    
    # Precompute all words
    all_words = [list(w) for w in product(range(q), repeat=n)]
    
    # Precompute neighbors (ball contents)
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    neighbors = []
    for w in all_words:
        covered = [word_to_idx[tuple(w)]]
        w_list = list(w)
        for i in range(n):
            orig_val = w_list[i]
            for val in range(q):
                if val != orig_val:
                    w_list[i] = val
                    covered.append(word_to_idx[tuple(w_list)])
            w_list[i] = orig_val
        neighbors.append(covered)
    
    # Precompute neighbors as numpy arrays for faster indexing
    neighbors_np = [np.array(nb, dtype=np.int32) for nb in neighbors]
    
    # Precompute coverers: coverers[point] = list of codewords that cover point
    coverers = [[] for _ in range(N)]
    for cw_idx, ball in enumerate(neighbors):
        for pt in ball:
            coverers[pt].append(cw_idx)
    
    # Convert coverers to numpy arrays for set intersection speed if needed, 
    # but lists are often faster for small intersections in Python.
    # We'll stick to lists for coverers but use sets for candidate generation.

    # 1. Initial Construction: Greedy Set Cover using Bitsets
    ball_bitsets = [0] * N
    for i in range(N):
        mask = 0
        for j in neighbors[i]:
            mask |= (1 << j)
        ball_bitsets[i] = mask
        
    all_mask = (1 << N) - 1
    current_cover_mask = 0
    codewords = []
    is_chosen = np.zeros(N, dtype=bool)
    
    # Greedy Set Cover
    while current_cover_mask != all_mask:
        if time.time() > deadline - 80:
            break
            
        best_i = -1
        best_count = 0
        
        for i in range(N):
            if is_chosen[i]:
                continue
            new_bits = ball_bitsets[i] & ~current_cover_mask
            if new_bits:
                count = new_bits.bit_count()
                if count > best_count:
                    best_count = count
                    best_i = i
                    
        if best_i == -1:
            break # Should not happen
            
        codewords.append(best_i)
        is_chosen[best_i] = True
        current_cover_mask |= ball_bitsets[best_i]
        
    # Ensure validity
    if current_cover_mask != all_mask:
        while current_cover_mask != all_mask:
            i = random.randint(0, N-1)
            if not is_chosen[i]:
                codewords.append(i)
                is_chosen[i] = True
            current_cover_mask |= ball_bitsets[i]
            if time.time() > deadline - 10:
                break

    # 2. Local Search Setup
    # cover_count[pt] = number of codewords covering pt
    cover_count = np.zeros(N, dtype=np.int32)
    for cw in codewords:
        cover_count[neighbors_np[cw]] += 1
        
    # critical_count[cw] = number of points in ball(cw) that are covered ONLY by cw
    # A codeword is removable if critical_count[cw] == 0
    critical_count = np.zeros(N, dtype=np.int32)
    for cw in codewords:
        b = neighbors_np[cw]
        critical_count[cw] = np.sum(cover_count[b] == 1)

    best_codewords = list(codewords)
    best_size = len(codewords)
    
    # Helper to update critical counts efficiently? 
    # When we add/remove a codeword, we update cover_count.
    # Then we need to recalculate critical_count for affected codewords.
    # Affected codewords are those sharing neighbors with the changed codeword.
    # Let's define a function to recalculate critical counts for a set of codewords.
    
    def recalc_critical(cw_list):
        for cw in cw_list:
            if cw in is_chosen: # Only care about chosen ones? 
                # Actually critical_count is only valid for chosen codewords.
                # But we might check candidates.
                pass
            b = neighbors_np[cw]
            critical_count[cw] = np.sum(cover_count[b] == 1)

    # We need a mapping from point to codewords that cover it (coverers)
    # To find which codewords are affected by a change in cover_count at a point,
    # we look at coverers[point].
    
    def try_remove_redundant(codewords, cover_count, critical_count):
        removed = 0
        i = 0
        while i < len(codewords):
            if time.time() > deadline - 15:
                break
            cw = codewords[i]
            if critical_count[cw] == 0:
                # Remove cw
                b = neighbors_np[cw]
                cover_count[b] -= 1
                is_chosen[cw] = False
                codewords.pop(i)
                removed += 1
                
                # Update critical counts for all codewords that cover any point in b
                # These are the codewords in coverers[pt] for pt in b
                # We need to recalculate critical_count for them.
                affected_cws = set()
                for pt in b:
                    for other_cw in coverers[pt]:
                        if other_cw != cw:
                            affected_cws.add(other_cw)
                
                # Recalculate critical counts for affected codewords
                for acw in affected_cws:
                    if is_chosen[acw]: # Only if still in solution
                        b_acw = neighbors_np[acw]
                        critical_count[acw] = np.sum(cover_count[b_acw] == 1)
            else:
                i += 1
        return removed

    # Initial redundancy removal
    try_remove_redundant(codewords, cover_count, critical_count)
    if len(codewords) < best_size:
        best_size = len(codewords)
        best_codewords = list(codewords)
        
    # Local Search Loop with Iterated Local Search (Perturbation)
    random.seed(42)
    no_improvement_count = 0
    max_no_improvement = 20
    perturbation_strength = 5 # Number of random swaps/removals in perturbation
    
    while time.time() < deadline - 5:
        improved = False
        
        # 1. Try Removals
        removed = try_remove_redundant(codewords, cover_count, critical_count)
        if removed > 0:
            improved = True
            no_improvement_count = 0
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
            continue # Restart loop after modification
        
        # 2. Try Swaps
        # We look for a codeword 'old' and a candidate 'new' such that:
        # - new covers all critical points of old
        # - new is not already in codewords
        # - The "cost" (number of new critical points created) is minimized
        # For simplicity and speed, we pick a random codeword 'old' with critical points > 0
        # and find candidates covering its critical points.
        
        if codewords:
            # Filter codewords that have critical points
            candidates_for_swap = [cw for cw in codewords if critical_count[cw] > 0]
            
            if candidates_for_swap:
                # Pick one randomly to avoid bias
                idx = random.randint(0, len(candidates_for_swap) - 1)
                old_cw = candidates_for_swap[idx]
                
                # Find critical points of old_cw
                b_old = neighbors_np[old_cw]
                critical_points = []
                for pt in b_old:
                    if cover_count[pt] == 1:
                        critical_points.append(pt)
                
                if critical_points:
                    # Find candidates that cover ALL critical points
                    # Intersection of coverers[pt] for pt in critical_points
                    candidate_set = set(coverers[critical_points[0]])
                    for p in critical_points[1:]:
                        candidate_set.intersection_update(coverers[p])
                    
                    # Remove old_cw and existing codewords
                    candidate_set.discard(old_cw)
                    for cw in codewords:
                        candidate_set.discard(cw)
                        
                    if candidate_set:
                        # Evaluate candidates
                        # We want to minimize the number of "new" critical points created.
                        # A point pt in ball(new) becomes critical if cover_count[pt] was 1 
                        # and new is the only one covering it? No.
                        # If we swap old -> new:
                        # cover_count[pt] decreases by 1 for pt in b_old
                        # cover_count[pt] increases by 1 for pt in b_new
                        # A point pt in b_new becomes critical (cover_count==1) if:
                        # - It was not in b_old (so count unchanged) and count was 1? No, count increases to 2.
                        # - It was in b_old (count decreases by 1, then increases by 1 -> net 0).
                        # Wait, the critical count for *other* codewords might change.
                        
                        # Simpler heuristic: Minimize sum of cover_count in b_new (redundancy)
                        # Or minimize number of points in b_new that are covered by exactly 1 codeword (after swap).
                        
                        best_new_cw = -1
                        min_cost = float('inf')
                        
                        # Limit candidates to check
                        candidates_list = list(candidate_set)
                        if len(candidates_list) > 50:
                            random.shuffle(candidates_list)
                            candidates_list = candidates_list[:50]
                        
                        for new_cw in candidates_list:
                            b_new = neighbors_np[new_cw]
                            # Cost: How many points in b_new will have cover_count == 1 AFTER swap?
                            # cover_count_new[pt] = cover_count[pt] - (1 if pt in b_old else 0) + 1
                            # We want to minimize the number of pts where cover_count_new[pt] == 1
                            # AND also consider that we remove old_cw, so other codewords' critical counts change.
                            
                            # Fast approximation:
                            # 1. Points in b_new \ b_old: count increases. If count was 1, becomes 2 (good, less critical).
                            #    If count was 0 (impossible since covered), 
                            # 2. Points in b_new \cap b_old: count unchanged.
                            # 3. Points in b_old \ b_new: count decreases. If count was 1, becomes 0 (INVALID, but we ensured new covers critical pts of old, so this shouldn't happen for critical pts. But non-critical pts with count 1? If count was 1, it was critical. So all pts in b_old with count 1 are in critical_points, which are covered by new. So count won't drop to 0).
                            #    If count was >1, it drops by 1. If it becomes 1, it becomes critical for the *other* codeword covering it.
                            
                            # So the "cost" is the number of points in (b_old \ b_new) that will have count 1 after swap.
                            # i.e., points pt in b_old such that pt not in b_new and cover_count[pt] == 2.
                            
                            # Let's calculate this.
                            # b_old_set = set(b_old)
                            # b_new_set = set(b_new)
                            # diff = b_old_set - b_new_set
                            # cost = sum(1 for pt in diff if cover_count[pt] == 2)
                            
                            # This is O(ball_size).
                            # We can use numpy for speed.
                            # Create a boolean mask for b_new
                            # mask_new = np.zeros(N, dtype=bool)
                            # mask_new[b_new] = True
                            # diff_mask = (np.ones(N, dtype=bool) & ~mask_new) # Not efficient
                            
                            # Better:
                            # count_2_in_b_old = np.sum((cover_count[b_old] == 2) & ~np.isin(b_old, b_new))
                            # np.isin is slow.
                            
                            # Let's just use the set approach for the small number of candidates.
                            b_old_set = set(b_old)
                            b_new_set = set(b_new)
                            diff = b_old_set - b_new_set
                            cost = 0
                            for pt in diff:
                                if cover_count[pt] == 2:
                                    cost += 1
                            
                            # Also add a small penalty for size of b_new to prefer "tighter" fits?
                            # cost += len(b_new) * 0.01
                            
                            if cost < min_cost:
                                min_cost = cost
                                best_new_cw = new_cw
                            
                        if best_new_cw != -1:
                            # Perform swap
                            b_old = neighbors_np[old_cw]
                            b_new = neighbors_np[best_new_cw]
                            
                            # Update cover counts
                            cover_count[b_old] -= 1
                            cover_count[b_new] += 1
                            
                            # Update codewords list
                            # Find index of old_cw
                            for i, cw in enumerate(codewords):
                                if cw == old_cw:
                                    codewords[i] = best_new_cw
                                    break
                            
                            is_chosen[old_cw] = False
                            is_chosen[best_new_cw] = True
                            
                            # Update critical counts for affected codewords
                            # Affected: all codewords covering any pt in b_old or b_new
                            affected_cws = set()
                            for pt in b_old:
                                for other_cw in coverers[pt]:
                                    if other_cw != old_cw:
                                        affected_cws.add(other_cw)
                            for pt in b_new:
                                for other_cw in coverers[pt]:
                                    if other_cw != best_new_cw:
                                        affected_cws.add(other_cw)
                            
                            # Also the new codeword itself
                            affected_cws.add(best_new_cw)
                            
                            for acw in affected_cws:
                                if is_chosen[acw]:
                                    b_acw = neighbors_np[acw]
                                    critical_count[acw] = np.sum(cover_count[b_acw] == 1)
                            
                            improved = True
                            no_improvement_count = 0
                            
                            # Try to remove redundancies immediately after swap
                            try_remove_redundant(codewords, cover_count, critical_count)
                            
                            if len(codewords) < best_size:
                                best_size = len(codewords)
                                best_codewords = list(codewords)
                            
                            continue # Restart loop
        
        if not improved:
            no_improvement_count += 1
            
            if no_improvement_count >= max_no_improvement:
                # Perturbation
                # Remove K random codewords and re-add greedily
                K = min(perturbation_strength, len(codewords))
                if K > 0:
                    # Pick K random codewords to remove
                    to_remove = random.sample(codewords, K)
                    for cw in to_remove:
                        b = neighbors_np[cw]
                        cover_count[b] -= 1
                        is_chosen[cw] = False
                        # Remove from list
                        codewords.remove(cw)
                    
                    # Re-add greedily to cover any uncovered points
                    # Find uncovered points
                    uncovered_mask = (cover_count == 0)
                    uncovered_pts = np.where(uncovered_mask)[0]
                    
                    if len(uncovered_pts) > 0:
                        # Greedy add
                        while len(uncovered_pts) > 0:
                            if time.time() > deadline - 10:
                                break
                            best_i = -1
                            best_count = 0
                            # Sample candidates? Or scan all?
                            # Scan all N is 2187, fast enough.
                            for i in range(N):
                                if is_chosen[i]:
                                    continue
                                # Count how many uncovered pts this covers
                                b_i = neighbors_np[i]
                                count = np.sum(uncovered_mask[b_i])
                                if count > best_count:
                                    best_count = count
                                    best_i = i
                            
                            if best_i == -1:
                                break
                            
                            codewords.append(best_i)
                            is_chosen[best_i] = True
                            cover_count[neighbors_np[best_i]] += 1
                            
                            # Update uncovered_mask
                            uncovered_mask[neighbors_np[best_i]] = False
                            uncovered_pts = np.where(uncovered_mask)[0]
                    
                    # Recalculate critical counts for all codewords
                    for cw in codewords:
                        b_cw = neighbors_np[cw]
                        critical_count[cw] = np.sum(cover_count[b_cw] == 1)
                    
                    no_improvement_count = 0
                    # Reset perturbation strength? Or keep it.

    # Final Polish
    final_codewords = list(best_codewords)
    final_cover_count = np.zeros(N, dtype=np.int32)
    for cw in final_codewords:
        final_cover_count[neighbors_np[cw]] += 1
        
    # Run redundancy removal on the best solution found
    # Need to reconstruct critical counts for final_codewords
    final_critical_count = np.zeros(N, dtype=np.int32)
    for cw in final_codewords:
        b = neighbors_np[cw]
        final_critical_count[cw] = np.sum(final_cover_count[b] == 1)
    
    # We need a local copy of the state for the final polish
    # But try_remove_redundant uses global cover_count, critical_count, is_chosen
    # Let's just do a simple check: if any codeword has critical_count 0, remove it.
    # Repeat until no more removals.
    
    # Re-initialize local state for final polish to be safe
    temp_codewords = list(final_codewords)
    temp_cover_count = np.zeros(N, dtype=np.int32)
    for cw in temp_codewords:
        temp_cover_count[neighbors_np[cw]] += 1
    temp_critical_count = np.zeros(N, dtype=np.int32)
    for cw in temp_codewords:
        b = neighbors_np[cw]
        temp_critical_count[cw] = np.sum(temp_cover_count[b] == 1)
    
    # Simple loop
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(temp_codewords):
            cw = temp_codewords[i]
            if temp_critical_count[cw] == 0:
                b = neighbors_np[cw]
                temp_cover_count[b] -= 1
                temp_codewords.pop(i)
                changed = True
                # Update critical counts for affected
                affected = set()
                for pt in b:
                    for other in coverers[pt]:
                        if other != cw:
                            affected.add(other)
                for acw in affected:
                    if acw in temp_codewords:
                        b_acw = neighbors_np[acw]
                        temp_critical_count[acw] = np.sum(temp_cover_count[b_acw] == 1)
            else:
                i += 1
    
    if np.min(temp_cover_count) > 0:
        return [list(all_words[cw]) for cw in temp_codewords]
    else:
        return [list(all_words[cw]) for cw in final_codewords]

