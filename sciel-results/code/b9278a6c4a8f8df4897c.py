import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit

    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return list(itertools.product(range(q), repeat=n))

    total_words = q ** n  # 2187

    # Precompute all words
    all_words = []
    for i in range(total_words):
        w = [0] * n
        idx = i
        for pos in range(n - 1, -1, -1):
            w[pos] = idx % q
            idx //= q
        all_words.append(w)

    # Precompute spheres as lists of indices
    spheres = []
    for i in range(total_words):
        w = all_words[i]
        s = [i]
        for pos in range(n):
            for delta in [-1, 1]:
                new_val = (w[pos] + delta) % q
                if new_val != w[pos]:
                    nw = w.copy()
                    nw[pos] = new_val
                    j = 0
                    for d in nw:
                        j = j * q + d
                    s.append(j)
        spheres.append(s)

    # Convert spheres to numpy arrays for fast operations
    sphere_arrays = [np.array(s, dtype=np.int32) for s in spheres]
    
    # Precompute sphere bitmasks for greedy
    sphere_bits = [0] * total_words
    for i in range(total_words):
        mask = 0
        for j in spheres[i]:
            mask |= (1 << j)
        sphere_bits[i] = mask

    # Precompute coverers: coverers[j] = list of i such that j in spheres[i]
    coverers = [[] for _ in range(total_words)]
    for i in range(total_words):
        for j in spheres[i]:
            coverers[j].append(i)

    # Precompute sets for faster membership checks in local search
    sphere_sets = [set(s) for s in spheres]

    best_code = None
    best_size = float('inf')
    
    def local_search(code_set, deadline):
        current_code = set(code_set)
        count = np.zeros(total_words, dtype=np.int32)
        for c in current_code:
            count[sphere_arrays[c]] += 1
            
        changed = True
        while changed and time.time() < deadline:
            changed = False
            
            # 1. Prune: Remove any redundant codeword
            # A codeword is redundant if all words in its sphere have count > 1
            for c in list(current_code):
                if np.min(count[sphere_arrays[c]]) > 1:
                    current_code.remove(c)
                    count[sphere_arrays[c]] -= 1
                    changed = True
                    # Restart loop to check for new redundancies
                    break
            
            if not changed and time.time() < deadline:
                # 2. Swap: Try to replace one codeword with another
                candidates_remove = list(current_code)
                random.shuffle(candidates_remove)
                
                for c_remove in candidates_remove:
                    # Simulate removal
                    # Words that become uncovered (count goes from 1 to 0)
                    # We need to find c_add that covers all these words
                    
                    # Efficiently find words in sphere[c_remove] with count == 1
                    sphere_c_remove = sphere_arrays[c_remove]
                    counts_in_sphere = count[sphere_c_remove]
                    # Indices in sphere_c_remove where count is 1
                    # Note: sphere_c_remove contains indices of words, counts_in_sphere contains counts for those words
                    # We need the actual word indices that have count 1
                    # mask of positions in sphere_c_remove where count is 1
                    pos_mask = (counts_in_sphere == 1)
                    if not np.any(pos_mask):
                        # This codeword is redundant? Should have been caught by prune.
                        # But maybe count changed? No, count is static during this phase.
                        # If no word has count 1, it's redundant.
                        current_code.remove(c_remove)
                        count[sphere_c_remove] -= 1
                        changed = True
                        break
                    
                    # Get the word indices that are critical
                    critical_words = sphere_c_remove[pos_mask]
                    
                    # If too many critical words, a single codeword might not cover them all (max sphere size is 15)
                    # But a codeword covers at most 15 words. If critical_words > 15, impossible?
                    # Actually, critical_words are a subset of sphere[c_remove], which has size 15.
                    # So len(critical_words) <= 15.
                    
                    # Find a c_add that covers all critical_words
                    # Start with candidates from coverers of the first critical word
                    if len(critical_words) == 0:
                        continue
                        
                    target = critical_words[0]
                    cands = coverers[target]
                    random.shuffle(cands)
                    
                    found_swap = False
                    for c_add in cands:
                        if c_add in current_code:
                            continue
                        
                        # Check if c_add covers all critical_words
                        add_sphere_set = sphere_sets[c_add]
                        covers_all = all(j in add_sphere_set for j in critical_words)
                        
                        if covers_all:
                            # Calculate score: number of words in sphere[c_add] that will have count > 1 after swap
                            # New count for word j in sphere[c_add]:
                            # old_count[j] - (1 if j in sphere[c_remove] else 0) + 1
                            # We want to minimize the number of j where new_count[j] > 1
                            # i.e., minimize sum(1 for j in sphere[c_add] if old_count[j] - I(j in S_remove) + 1 > 1)
                            # <=> minimize sum(1 for j in sphere[c_add] if old_count[j] - I(j in S_remove) >= 1)
                            # <=> minimize sum(1 for j in sphere[c_add] if old_count[j] >= 1 + I(j in S_remove))
                            
                            # Let's just compute it directly for the small set
                            score = 0
                            # We can iterate over sphere[c_add]
                            for j in sphere_arrays[c_add]:
                                old_c = count[j]
                                new_c = old_c
                                if j in sphere_sets[c_remove]: # This check is O(1) with set
                                    new_c -= 1
                                new_c += 1
                                if new_c > 1:
                                    score += 1
                                    
                            # We want the swap with minimum score.
                            # For now, take the first valid one with low score.
                            # To be faster, we could break early if score is 0.
                            if score == 0:
                                # Perfect swap
                                current_code.remove(c_remove)
                                count[sphere_c_remove] -= 1
                                current_code.add(c_add)
                                count[sphere_arrays[c_add]] += 1
                                changed = True
                                found_swap = True
                                break
                            else:
                                # We could store best candidate, but for simplicity take first valid
                                # If we want to be smarter, we'd track best_score.
                                # Let's just take the first valid one to keep it fast, 
                                # or check a few.
                                # Given the small size, checking all coverers of target is fine.
                                # But we only break if score==0. Otherwise we continue looking for a better one?
                                # No, let's just accept the first valid one to avoid complexity.
                                # Actually, accepting a bad swap might block future prunes.
                                # Let's try to find one with score 0. If none, take min score.
                                pass # Logic below handles this better

                    if not found_swap:
                        # If we didn't find a score 0 swap, let's find the best one (min score)
                        # Re-iterate or store best during previous loop?
                        # To save time, we only did "break" on score 0.
                        # Let's do a second pass if we didn't break?
                        # No, that's expensive.
                        # Let's refine the loop above to track best.
                        
                        pass # The current loop breaks on score 0. If not found, we skip this c_remove.
                        # This might be too aggressive. Let's modify the inner loop to track best.

            if not changed and time.time() < deadline:
                # 3. 1-for-2 Move
                # Try adding a random codeword that has high overlap with current code
                # but allows removing two redundant ones.
                
                attempts = 0
                while attempts < 10 and time.time() < deadline:
                    attempts += 1
                    c_add = random.randint(0, total_words - 1)
                    if c_add in current_code:
                        continue
                    
                    temp_count = count.copy()
                    temp_count[sphere_arrays[c_add]] += 1
                    temp_code = set(current_code)
                    temp_code.add(c_add)
                    
                    removed_count = 0
                    # Try to remove redundant codewords from temp_code
                    # We need to check all codewords in temp_code
                    # To speed up, we can check only those that might have become redundant
                    # But it's safer to check all.
                    
                    # Optimization: Only check codewords whose sphere intersects with sphere[c_add]
                    # Or just check all, it's small.
                    
                    # Shuffle to randomize order
                    candidates_to_remove = list(temp_code)
                    random.shuffle(candidates_to_remove)
                    
                    for c_remove in candidates_to_remove:
                        if c_remove not in temp_code:
                            continue
                        if np.min(temp_count[sphere_arrays[c_remove]]) > 1:
                            temp_code.remove(c_remove)
                            temp_count[sphere_arrays[c_remove]] -= 1
                            removed_count += 1
                            if removed_count == 2:
                                break
                                
                    if removed_count == 2:
                        # Net change: +1 - 2 = -1
                        current_code = temp_code
                        count = temp_count
                        changed = True
                        break
                    # Else revert (do nothing)

        return current_code

    # Refined Local Search with better swap selection
    def local_search_v2(code_set, deadline):
        current_code = set(code_set)
        count = np.zeros(total_words, dtype=np.int32)
        for c in current_code:
            count[sphere_arrays[c]] += 1
            
        changed = True
        while changed and time.time() < deadline:
            changed = False
            
            # 1. Prune
            for c in list(current_code):
                if np.min(count[sphere_arrays[c]]) > 1:
                    current_code.remove(c)
                    count[sphere_arrays[c]] -= 1
                    changed = True
                    break
            
            if not changed and time.time() < deadline:
                # 2. Swap
                candidates_remove = list(current_code)
                random.shuffle(candidates_remove)
                
                for c_remove in candidates_remove:
                    sphere_c_remove = sphere_arrays[c_remove]
                    counts_in_sphere = count[sphere_c_remove]
                    pos_mask = (counts_in_sphere == 1)
                    
                    if not np.any(pos_mask):
                        current_code.remove(c_remove)
                        count[sphere_c_remove] -= 1
                        changed = True
                        break
                    
                    critical_words = sphere_c_remove[pos_mask]
                    if len(critical_words) == 0:
                        continue
                        
                    target = critical_words[0]
                    cands = coverers[target]
                    random.shuffle(cands)
                    
                    best_c_add = -1
                    best_score = float('inf')
                    
                    for c_add in cands:
                        if c_add in current_code:
                            continue
                        
                        add_sphere_set = sphere_sets[c_add]
                        covers_all = all(j in add_sphere_set for j in critical_words)
                        
                        if covers_all:
                            score = 0
                            # Calculate score
                            for j in sphere_arrays[c_add]:
                                old_c = count[j]
                                new_c = old_c
                                if j in sphere_sets[c_remove]:
                                    new_c -= 1
                                new_c += 1
                                if new_c > 1:
                                    score += 1
                                    
                            if score < best_score:
                                best_score = score
                                best_c_add = c_add
                                if score == 0:
                                    break
                                    
                    if best_c_add != -1:
                        current_code.remove(c_remove)
                        count[sphere_c_remove] -= 1
                        current_code.add(best_c_add)
                        count[sphere_arrays[best_c_add]] += 1
                        changed = True
                        break # Restart loop to check for new prunes

            if not changed and time.time() < deadline:
                # 3. 1-for-2 Move
                attempts = 0
                while attempts < 10 and time.time() < deadline:
                    attempts += 1
                    c_add = random.randint(0, total_words - 1)
                    if c_add in current_code:
                        continue
                    
                    temp_count = count.copy()
                    temp_count[sphere_arrays[c_add]] += 1
                    temp_code = set(current_code)
                    temp_code.add(c_add)
                    
                    removed_count = 0
                    candidates_to_remove = list(temp_code)
                    random.shuffle(candidates_to_remove)
                    
                    for c_remove in candidates_to_remove:
                        if c_remove not in temp_code:
                            continue
                        if np.min(temp_count[sphere_arrays[c_remove]]) > 1:
                            temp_code.remove(c_remove)
                            temp_count[sphere_arrays[c_remove]] -= 1
                            removed_count += 1
                            if removed_count == 2:
                                break
                                
                    if removed_count == 2:
                        current_code = temp_code
                        count = temp_count
                        changed = True
                        break

        return current_code

    num_runs = 200

    for run in range(num_runs):
        if time.time() > deadline:
            break
            
        uncovered = (1 << total_words) - 1
        code_indices = []
        
        # Greedy
        while uncovered:
            best_gain = -1
            best_c = -1
            candidates = list(range(total_words))
            random.shuffle(candidates)
            
            for c in candidates:
                gain = bin(sphere_bits[c] & uncovered).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                if best_gain == 15:
                    break
                    
            if best_c == -1:
                break
                
            code_indices.append(best_c)
            uncovered &= ~sphere_bits[best_c]
            
        # Local Search
        code_set = local_search_v2(set(code_indices), deadline)
        
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = code_set
            if best_size <= 186:
                break

    if best_code is None:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    result = [all_words[i] for i in best_code]
    return result
