import random
import time
import sys
from collections import defaultdict

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses a combination of greedy initialization and aggressive local search 
    including redundancy removal, 1-for-1 swaps, and 2-for-1 replacements.
    """
    if n == 0:
        return [[]]
    
    total_words = q ** n
    
    # Convert between integer representation and list representation
    def int_to_word(val, n, q):
        word = []
        for _ in range(n):
            word.append(val % q)
            val //= q
        return word
        
    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val

    # Precompute neighbors (balls of radius 'radius')
    # For q=3, n=8, radius=1, each word has 1 + 8*(3-1) = 17 neighbors.
    # Total 6561 words. 6561 * 17 is small.
    neighbors = [set() for _ in range(total_words)]
    
    # Efficient neighbor generation
    for w in range(total_words):
        word = int_to_word(w, n, q)
        neigh = {w}
        if radius >= 1:
            for pos in range(n):
                orig_val = word[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    word[pos] = new_val
                    new_w = word_to_int(word, q)
                    neigh.add(new_w)
            # Restore original word for next position
            word = int_to_word(w, n, q)
        if radius >= 2:
             # For this specific task radius is 1, but keep generic just in case
             # Radius 2 is 1 + 8*2 + C(8,2)*4 = 1 + 16 + 112 = 129 neighbors.
             # Still manageable for 6561 words.
             for pos1 in range(n):
                 for pos2 in range(pos1 + 1, n):
                     v1_orig = word[pos1]
                     v2_orig = word[pos2]
                     for v1 in range(q):
                         if v1 == v1_orig: continue
                         for v2 in range(q):
                             if v2 == v2_orig: continue
                             word[pos1] = v1
                             word[pos2] = v2
                             new_w = word_to_int(word, q)
                             neigh.add(new_w)
                     word[pos1] = v1_orig
                     word[pos2] = v2_orig
        neighbors[w] = neigh

    def greedy_init():
        covered = [False] * total_words
        uncovered = set(range(total_words))
        codewords = []
        
        while uncovered:
            best_w = -1
            best_count = -1
            # Sample a subset if uncovered is large? No, 6561 is small.
            # Iterate through all potential candidates (all words)
            # Optimization: Only consider words that are in 'uncovered' or cover many uncovered?
            # Standard greedy: Pick w that maximizes |neighbors[w] \ covered|
            # We can iterate through all w in [0, total_words)
            for w in range(total_words):
                # Quick check: if w is already covered, it might still be a good center?
                # Yes, a center doesn't have to be uncovered itself.
                count = 0
                for u in neighbors[w]:
                    if u in uncovered:
                        count += 1
                        if count > best_count:
                            best_count = count
                            best_w = w
                # Early exit if we found a perfect cover (unlikely in one step)
                if best_count == len(uncovered):
                    break
            
            if best_w == -1:
                break
                
            codewords.append(best_w)
            for u in neighbors[best_w]:
                if u in uncovered:
                    uncovered.remove(u)
        
        return codewords

    def optimize(codewords, time_limit):
        # Build coverage counts
        coverage_count = [0] * total_words
        for cw in codewords:
            for u in neighbors[cw]:
                coverage_count[u] += 1
        
        cw_set = set(codewords)
        
        def try_remove_redundant():
            removed = False
            changed = True
            while changed:
                changed = False
                # Iterate in random order to avoid bias
                indices = list(range(len(codewords)))
                random.shuffle(indices)
                for i in indices:
                    cw = codewords[i]
                    is_redundant = True
                    for u in neighbors[cw]:
                        if coverage_count[u] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        for u in neighbors[cw]:
                            coverage_count[u] -= 1
                        codewords.pop(i)
                        cw_set.discard(cw)
                        removed = True
                        changed = True
                        break
            return removed

        def try_swaps():
            # 1-for-1 swaps
            made_swap = False
            indices = list(range(len(codewords)))
            random.shuffle(indices)
            for i in indices:
                old_cw = codewords[i]
                # Identify words uniquely covered by old_cw
                unique_words = [u for u in neighbors[old_cw] if coverage_count[u] == 1]
                
                if not unique_words:
                    # It's redundant, but try_remove_redundant should have caught it.
                    # Maybe it was just removed? Or logic error.
                    # Actually, if unique_words is empty, we can just remove it.
                    # But we handle that in try_remove_redundant.
                    # If we get here, it means we didn't remove it? 
                    # Let's just try to swap it with something else to break symmetry?
                    # Or just skip.
                    continue

                # Find a candidate new_cw that covers all unique_words
                # Candidates are neighbors of any word in unique_words
                candidates = set()
                for u in unique_words:
                    candidates.update(neighbors[u])
                
                # Remove old_cw and existing codewords from candidates
                candidates.discard(old_cw)
                candidates -= cw_set
                
                best_new_cw = -1
                best_score = -1 # Maximize redundancy reduction (or just find valid)
                
                for new_cw in candidates:
                    # Check if new_cw covers all unique_words
                    valid = True
                    for u in unique_words:
                        if u not in neighbors[new_cw]:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Score: How many redundant words does new_cw cover?
                    # We want to cover words with coverage_count > 1 to potentially enable further removals.
                    score = 0
                    for u in neighbors[new_cw]:
                        if coverage_count[u] > 1:
                            score += 1
                    
                    if score > best_score:
                        best_score = score
                        best_new_cw = new_cw
                
                if best_new_cw != -1:
                    # Perform swap
                    for u in neighbors[old_cw]:
                        coverage_count[u] -= 1
                    for u in neighbors[best_new_cw]:
                        coverage_count[u] += 1
                    codewords[i] = best_new_cw
                    cw_set.discard(old_cw)
                    cw_set.add(best_new_cw)
                    made_swap = True
                    # Don't break, try to find more swaps? 
                    # Usually we restart the loop after a swap to update coverage.
                    return True
            
            return made_swap

        def try_2_for_1():
            # Try to replace 2 codewords with 1
            # This is expensive O(N^2 * K)
            # Only try if N is small enough or we have time
            if len(codewords) > 100: # Heuristic
                return False
            
            made_swap = False
            indices = list(range(len(codewords)))
            random.shuffle(indices)
            
            # Pick a random pair
            if len(indices) < 2:
                return False
            
            i, j = indices[0], indices[1]
            if i == j: return False
            
            c1, c2 = codewords[i], codewords[j]
            
            # Words uniquely covered by c1 or c2
            unique_to_pair = set()
            for u in neighbors[c1]:
                if coverage_count[u] == 1:
                    unique_to_pair.add(u)
            for u in neighbors[c2]:
                if coverage_count[u] == 1:
                    unique_to_pair.add(u)
            
            if not unique_to_pair:
                # Both are redundant? try_remove_redundant should handle.
                return False

            # Find candidate new_cw that covers all unique_to_pair
            candidates = set()
            for u in unique_to_pair:
                candidates.update(neighbors[u])
            
            candidates.discard(c1)
            candidates.discard(c2)
            candidates -= cw_set
            
            best_new_cw = -1
            best_score = -1
            
            for new_cw in candidates:
                valid = True
                for u in unique_to_pair:
                    if u not in neighbors[new_cw]:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Score: Redundancy gain
                score = 0
                for u in neighbors[new_cw]:
                    if coverage_count[u] > 1:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_new_cw = new_cw
            
            if best_new_cw != -1:
                # Remove c1 and c2, add best_new_cw
                for u in neighbors[c1]:
                    coverage_count[u] -= 1
                for u in neighbors[c2]:
                    coverage_count[u] -= 1
                for u in neighbors[best_new_cw]:
                    coverage_count[u] += 1
                
                # Remove from list (careful with indices)
                # Remove larger index first to keep indices valid
                if i > j:
                    i, j = j, i
                codewords.pop(j)
                codewords.pop(i)
                
                cw_set.discard(c1)
                cw_set.discard(c2)
                
                codewords.append(best_new_cw)
                cw_set.add(best_new_cw)
                
                return True
            
            return False

        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            # Priority: Remove redundant, then 2-for-1, then swaps
            if try_remove_redundant():
                continue
            
            # Try 2-for-1 occasionally
            if random.random() < 0.2:
                if try_2_for_1():
                    continue
            
            if try_swaps():
                continue
            
            # If no moves found, we might be stuck.
            # Break if we haven't made progress for a while?
            # For now, just keep trying until time limit.
            pass
        
        try_remove_redundant()
        return codewords

    # Main Loop
    start_time = time.time()
    time_limit = 140 # Safety margin for 150s limit
    
    best_codewords = []
    
    # Run multiple restarts
    while time.time() - start_time < time_limit:
        # Allocate time for this iteration
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time <= 0:
            break
        
        # Greedy is fast (<1s usually for this size)
        # Give optimization most of the time
        opt_time = max(1, remaining_time - 5)
        
        cw = greedy_init()
        cw = optimize(cw, opt_time)
        
        if not best_codewords or len(cw) < len(best_codewords):
            best_codewords = cw[:]
        
        # If we found a very good solution, we can reduce restart time?
        # Or just let it run.
        
        # Check if we are close to time limit
        if time.time() - start_time > time_limit - 10:
             # Do one final fast pass on the best found
             # Actually, just break to ensure we return in time.
             break

    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
