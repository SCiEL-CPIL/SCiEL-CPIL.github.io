import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 90s kill
    
    n = 12
    m = 40
    total_masks = 1 << n  # 4096
    
    # Helper to convert mask to list
    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]
    
    # Precompute all masks
    all_masks = list(range(total_masks))
    
    # Precompute subset relationship: subset_of[a][b] = True if mask a is subset of mask b
    # Actually, we can just use bit operations: a is subset of b iff (a & b) == a
    # Let's precompute a fast lookup or just use bit ops directly.
    
    def is_comparable(a, b):
        if a == b:
            return False
        return (a & b) == a or (a & b) == b
    
    def count_comparable(family):
        count = 0
        for i in range(len(family)):
            mi = family[i]
            for j in range(i + 1, len(family)):
                mj = family[j]
                if (mi & mj) == mi or (mi & mj) == mj:
                    count += 1
        return count
    
    def score_candidate(cand, family):
        s = 0
        for fs in family:
            if (cand & fs) == cand or (cand & fs) == fs:
                s += 1
        return s

    # --- Phase 1: Generate multiple diverse initial solutions ---
    
    def initial_chain_greedy():
        """Start with a full chain, then greedily add."""
        family = []
        used = set()
        
        # Chain: 0, 1, 3, 7, ..., 4095
        chain_mask = 0
        for k in range(n + 1):
            family.append(chain_mask)
            used.add(chain_mask)
            if k < n:
                chain_mask |= (1 << k)
        
        remaining = [m for m in all_masks if m not in used]
        random.shuffle(remaining)
        
        for _ in range(m - len(family)):
            if time.time() > deadline:
                break
            best = None
            best_score = -1
            for cand in remaining:
                s = score_candidate(cand, family)
                if s > best_score:
                    best_score = s
                    best = cand
            if best is None:
                break
            family.append(best)
            used.add(best)
            remaining.remove(best)
        
        return family
    
    def initial_star_greedy():
        """Start with empty set and all singletons, then greedily add."""
        family = [0]
        used = {0}
        
        for i in range(n):
            mask = 1 << i
            family.append(mask)
            used.add(mask)
        
        remaining = [m for m in all_masks if m not in used]
        random.shuffle(remaining)
        
        for _ in range(m - len(family)):
            if time.time() > deadline:
                break
            best = None
            best_score = -1
            for cand in remaining:
                s = score_candidate(cand, family)
                if s > best_score:
                    best_score = s
                    best = cand
            if best is None:
                break
            family.append(best)
            used.add(best)
            remaining.remove(best)
        
        return family
    
    def initial_random_greedy():
        """Start with a random set of size 10, then greedily add."""
        family = []
        used = set()
        initial_size = 10
        initial = random.sample(all_masks, initial_size)
        for m in initial:
            family.append(m)
            used.add(m)
        
        remaining = [m for m in all_masks if m not in used]
        random.shuffle(remaining)
        
        for _ in range(m - len(family)):
            if time.time() > deadline:
                break
            best = None
            best_score = -1
            for cand in remaining:
                s = score_candidate(cand, family)
                if s > best_score:
                    best_score = s
                    best = cand
            if best is None:
                break
            family.append(best)
            used.add(best)
            remaining.remove(best)
        
        return family

    # Generate a few initial solutions
    solutions = []
    scores = []
    
    # Always try chain and star
    fam1 = initial_chain_greedy()
    if len(fam1) == m:
        s1 = count_comparable(fam1)
        solutions.append(fam1)
        scores.append(s1)
    
    fam2 = initial_star_greedy()
    if len(fam2) == m:
        s2 = count_comparable(fam2)
        solutions.append(fam2)
        scores.append(s2)
    
    # Try a few random starts if time permits
    time_check = time.time()
    if time_check < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            fam_r = initial_random_greedy()
            if len(fam_r) == m:
                s_r = count_comparable(fam_r)
                solutions.append(fam_r)
                scores.append(s_r)
    
    # Pick the best initial solution
    if not solutions:
        # Fallback
        solutions.append(initial_chain_greedy())
        scores.append(count_comparable(solutions[0]))
    
    best_idx = max(range(len(solutions)), key=lambda i: scores[i])
    current = solutions[best_idx][:]
    current_score = scores[best_idx]
    
    # --- Phase 2: Local Search ---
    # We will perform local search by trying to replace one element with another.
    # To speed this up, we maintain the current score and compute the delta.
    
    def local_search(family, time_limit):
        current_score = count_comparable(family)
        end_time = time.time() + time_limit
        used = set(family)
        
        # Precompute comparability with all masks? No, 40*4096 is fine, but let's be smart.
        # For each i, we want to find a mask not in family that maximizes gain.
        # Gain = (comparables with new mask) - (comparables with old mask)
        
        improved = True
        while improved and time.time() < end_time:
            improved = False
            used = set(family)
            
            # Shuffle indices to avoid bias
            indices = list(range(len(family)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > end_time:
                    break
                
                old_mask = family[i]
                # Count how many other masks are comparable with old_mask
                old_comparables = 0
                for j in range(len(family)):
                    if j == i:
                        continue
                    mj = family[j]
                    if (old_mask & mj) == old_mask or (old_mask & mj) == mj:
                        old_comparables += 1
                
                best_new_mask = None
                best_gain = 0
                
                # We can iterate through all unused masks
                # Optimization: only check masks that are likely to be comparable?
                # For now, check all unused. 4096 * 40 checks per iteration is ~160k ops.
                # 160k * 40 swaps = 6.4M ops per pass. Should be fast in Python if optimized.
                
                # Let's try to speed up the inner loop.
                # We can precompute a list of unused masks.
                unused_masks = [m for m in all_masks if m not in used]
                
                for new_mask in unused_masks:
                    new_comparables = 0
                    for j in range(len(family)):
                        if j == i:
                            continue
                        mj = family[j]
                        if (new_mask & mj) == new_mask or (new_mask & mj) == mj:
                            new_comparables += 1
                    
                    gain = new_comparables - old_comparables
                    if gain > best_gain:
                        best_gain = gain
                        best_new_mask = new_mask
                
                if best_new_mask is not None and best_gain > 0:
                    # Perform swap
                    family[i] = best_new_mask
                    current_score += best_gain
                    improved = True
                    # Update used set? We can just rebuild it next iteration or update it here.
                    # Rebuilding is O(N), updating is O(1) but set ops are slow.
                    # Let's just let the next loop iteration rebuild 'used' via set(family)
        
        return family, current_score

    # Run local search on the best initial solution
    time_remaining = deadline - 2 - time.time()
    if time_remaining > 1:
        current, current_score = local_search(current, time_remaining)
    
    # --- Phase 3: Final Polish (Optional) ---
    # If we have time left, try a few more random restarts of local search from the current best?
    # Or just return the current best.
    
    # Convert to list of lists
    result = [mask_to_list(mask) for mask in current]
    
    # Ensure types are correct
    for i in range(len(result)):
        for j in range(len(result[i])):
            result[i][j] = int(result[i][j])
    
    # Validation
    assert len(result) == m
    assert len(set(tuple(s) for s in result)) == m
    for s in result:
        assert isinstance(s, list)
        for x in s:
            assert isinstance(x, int)
            assert not isinstance(x, bool)
            assert 0 <= x <= 11
        assert len(s) == len(set(s))
    
    return result
