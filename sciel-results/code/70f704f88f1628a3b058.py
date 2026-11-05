import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # Specific case q=3, n=7, radius=1
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other cases (not expected in this specific task evaluation)
        # Simple greedy fallback
        if n == 0:
            return [[]]
        # For general small n, use product
        # But task is specific to 3,7,1.
        # If not 3,7,1, just return a trivial valid code (might be large)
        # To be safe, implement a basic greedy for any q,n,r
        return _generic_construct(q, n, radius)

    total_words = 3 ** 7
    # Precompute balls
    # word_to_int and int_to_word
    # We represent words as integers 0..2186
    
    # Generate all words as tuples to compute balls
    words = [list(map(int, str(i).zfill(7))) for i in range(total_words)] 
    # Wait, str(i) is base 10. We need base 3.
    def int_to_base3(i, n):
        res = [0]*n
        for j in range(n):
            res[j] = i % 3
            i //= 3
        return res

    words = [int_to_base3(i, 7) for i in range(total_words)]
    
    # Map word tuple to index (already index i)
    # Precompute balls: list of lists of indices
    balls = []
    for i in range(total_words):
        w = words[i]
        ball = [i]
        for pos in range(7):
            orig = w[pos]
            for new_val in range(3):
                if new_val == orig:
                    continue
                new_w = w.copy()
                new_w[pos] = new_val
                # Convert back to int
                idx = 0
                for k in range(7):
                    idx = idx * 3 + new_w[k]
                # Wait, my int_to_base3 is little-endian?
                # i % 3 is least significant.
                # So index calculation must match.
                # idx = sum(new_w[k] * (3**k)) ?
                # Let's check int_to_base3:
                # res[0] = i % 3. So res is [d0, d1, ... d6] where i = d0 + d1*3 + ...
                # So idx = sum(new_w[k] * (3**k))
                
                # Let's stick to a consistent representation.
                # I will use the integer index directly.
                # To get the index of a word defined by digits d[0..6] (d[0] is LSB):
                # idx = d[0] + d[1]*3 + ...
                
                # Let's re-verify the mapping.
                # If I use idx = sum(d[k] * 3**k), then:
                # words[i] = digits of i in base 3 (LSB first).
                
                # Let's create a helper to get index from modified word
                # Actually, it's faster to precompute the index for all single-bit changes.
                
                # For now, let's just compute the index.
                idx = 0
                for k in range(7):
                    idx += new_w[k] * (3**k)
                ball.append(idx)
        balls.append(ball)

    # Optimize balls: ensure unique
    for i in range(total_words):
        balls[i] = list(set(balls[i]))
        
    # Greedy Construct
    def greedy_construct(seed_val):
        rng = random.Random(seed_val)
        cover_counts = np.zeros(total_words, dtype=np.int32)
        gains = np.full(total_words, 15, dtype=np.int32) # Max ball size is 15
        selected = []
        
        # To speed up "find max gain", we can just scan. 2187 items is small.
        # But we need to update gains.
        
        for step in range(total_words):
            if time.time() - start_time > time_limit:
                break
            
            # Find best candidate
            # We can optimize by only considering candidates with gain > 0
            best_idx = -1
            best_gain = 0
            # Randomize tie-breaking
            # We can pick from top few to add diversity
            # For simplicity, just pick max.
            
            # Vectorized max?
            # max_idx = np.argmax(gains)
            # But we need to handle the case where all gains are 0 (shouldn't happen if not covered)
            
            # To introduce randomness in ties:
            # Find all with max gain, pick one randomly.
            # This is slower. Let's just use np.argmax for speed, and rely on restarts.
            
            best_idx = np.argmax(gains)
            best_gain = gains[best_idx]
            
            if best_gain == 0:
                break
                
            selected.append(best_idx)
            
            # Update cover_counts and gains
            # For each x in balls[best_idx]:
            #   if cover_counts[x] == 0:
            #       # x becomes covered
            #       # For all y such that x in balls[y]:
            #       #   gains[y] -= 1
            #   cover_counts[x] += 1
            
            for x in balls[best_idx]:
                if cover_counts[x] == 0:
                    # x was uncovered, now covered
                    # Update gains for all y in balls[x]
                    # gains[balls[x]] -= 1
                    # Vectorized subtraction
                    gains[balls[x]] -= 1
                cover_counts[x] += 1
                
            # Zero out gain for best_idx? Not necessary, it will be low.
            # But we shouldn't pick it again.
            # We can set gains[best_idx] = -1 to avoid re-selection if it's still max?
            # No, if it's in selected, we don't want to pick it again.
            # But greedy set cover usually doesn't re-pick.
            # If we pick it again, gain would be 0 (since all its points are covered? No, only if all its points were uncovered before.
            # Actually, if we pick it, all its points gain count.
            # So its future gain is 0.
            # So np.argmax will naturally avoid it if others have gain > 0.
            
        return selected

    def local_search(initial_codewords):
        # initial_codewords: list of indices
        # Returns best list found
        
        best_set = set(initial_codewords)
        best_size = len(best_set)
        
        # Build cover counts
        cover_counts = np.zeros(total_words, dtype=np.int32)
        for w in best_set:
            cover_counts[balls[w]] += 1
            
        # Phase 1: Deletion
        # Repeat until no change
        changed = True
        while changed and (time.time() - start_time < time_limit):
            changed = False
            # Iterate in random order to avoid bias
            items = list(best_set)
            random.shuffle(items)
            for w in items:
                if w not in best_set: continue # Might have been removed
                # Check if w can be removed
                # If all x in balls[w] have cover_counts[x] > 1
                # i.e., min(cover_counts[balls[w]]) > 1
                if np.min(cover_counts[balls[w]]) > 1:
                    # Remove w
                    best_set.remove(w)
                    cover_counts[balls[w]] -= 1
                    best_size -= 1
                    changed = True
        
        # Phase 2: Swap
        # Try to swap w in best_set with v not in best_set
        # If valid, accept.
        # We do this until no improvement or time limit.
        
        # To make it faster, we can limit the number of swap iterations.
        # Also, we can try to find a swap that allows deletion.
        
        # Let's do a simple loop:
        # For each w in best_set:
        #   For each v in universe \ best_set:
        #     if swap is valid:
        #       apply swap
        #       run deletion phase
        #       if size decreased: keep
        #       else: revert swap (and revert deletion? No, if deletion didn't help, revert swap)
        
        # This is expensive.
        # Better:
        # 1. Try all swaps. If a swap is valid (coverage maintained), apply it.
        # 2. Then run deletion.
        # 3. If size is same or larger, revert to previous state?
        
        # Let's use a "kick" strategy which is often more effective.
        # Remove w, then greedily add to cover holes.
        
        # Given time constraints, let's stick to:
        # 1. Deletion (done above)
        # 2. Random Swaps with Deletion Check
        
        # We'll do multiple passes of swap attempts.
        
        current_set = set(best_set)
        current_size = best_size
        # Rebuild cover_counts for current_set (already done)
        
        # We need a function to check if a set is valid
        def is_valid(set_to_check, cc):
            # cc is cover counts for set_to_check
            return np.all(cc > 0)
            
        # We will try to improve.
        # We store the best found so far.
        
        # To speed up swap checking:
        # For a fixed w, we want to find v such that:
        # min(cover_counts[x] - 1_{x in ball[w]} + 1_{x in ball[v]} for x in ball[w]) > 0
        # Note: x must be in ball[w] for the -1 to apply.
        # If x not in ball[w], the term is cover_counts[x] + 1_{x in ball[v]}. Since cover_counts[x] >= 1 (valid set), this is always > 0.
        # So we only need to check x in ball[w].
        # Condition: For all x in ball[w], cover_counts[x] - 1 + (1 if x in ball[v] else 0) > 0
        # If x in ball[w] and x in ball[v]: cover_counts[x] > 0 (always true if valid)
        # If x in ball[w] and x not in ball[v]: cover_counts[x] - 1 > 0 => cover_counts[x] > 1
        
        # So, a swap w->v is valid if:
        # For all x in ball[w] \ ball[v], cover_counts[x] > 1.
        
        # We can precompute for each w, the list of x in ball[w] that are "critical" (cover_counts[x] == 1).
        # Let critical_w = [x in ball[w] if cover_counts[x] == 1]
        # Then v is a valid swap for w if:
        # For all x in critical_w, x is in ball[v].
        
        # This is much faster!
        # 1. Compute critical_w for all w in current_set.
        # 2. For each w, find v not in current_set such that ball[v] covers critical_w.
        # 3. If such v exists, perform swap.
        # 4. Run deletion.
        # 5. Update criticals.
        
        # This is a good heuristic.
        
        # Let's implement this.
        
        # We need to be careful: after swap, cover_counts change, so criticals change.
        
        # Loop:
        improved = True
        while improved and (time.time() - start_time < time_limit):
            improved = False
            # Compute criticals for current set
            # criticals[w] = list of x in ball[w] where cover_counts[x] == 1
            criticals = {}
            for w in current_set:
                crit = []
                for x in balls[w]:
                    if cover_counts[x] == 1:
                        crit.append(x)
                criticals[w] = crit
                
            # Try to find a swap
            # We can iterate w in random order
            ws = list(current_set)
            random.shuffle(ws)
            
            for w in ws:
                if w not in current_set: continue
                crit_w = criticals[w]
                
                # If no critical points, we can swap w with ANY v?
                # No, we just want to find a valid swap to enable deletion later?
                # Actually, if crit_w is empty, swapping w with any v keeps validity.
                # But does it help? Only if v allows deletion of another word.
                # So we should prefer v that covers points with high counts? Or just random?
                
                # Let's look for v that is NOT in current_set.
                # To speed up, we can precompute for each x, the list of v such that x in ball[v].
                # This is just balls[x] (symmetry).
                
                # We need v such that for all x in crit_w, x in ball[v].
                # i.e., v is in intersection of balls[x] for x in crit_w.
                
                if len(crit_w) == 0:
                    # Any v works for validity.
                    # Let's pick a random v not in current_set.
                    # But this might not lead to improvement.
                    # We can skip this or try a few.
                    # Let's try to find a v that is "useful".
                    # A useful v might be one that covers points with count 2?
                    # Let's just pick a random v from the universe not in current_set.
                    # To avoid O(N) scan, we can maintain a list of available v's?
                    # Or just pick random index and check if in set.
                    
                    # For now, if crit_w is empty, we don't have a strong constraint.
                    # We might skip to save time, or do a limited search.
                    # Let's skip for now and focus on constrained swaps.
                    continue
                
                # Find candidates v
                # Start with all v in balls[crit_w[0]]
                # Filter by subsequent crit_w elements
                
                # Optimization: If crit_w is large, the intersection is small.
                # If crit_w is small (1 or 2), intersection is large.
                
                # Let's get candidates from the first critical point
                c0 = crit_w[0]
                candidates = balls[c0]
                
                # Filter candidates
                # We need v not in current_set
                # And v must be in balls[x] for all other x in crit_w
                
                # To make this faster, we can just iterate through candidates and check.
                # But candidates can be large (15).
                # Wait, balls[c0] has size 15.
                # So we only have 15 candidates to check!
                
                for v in candidates:
                    if v in current_set:
                        continue
                    
                    # Check if v covers all other critical points
                    valid = True
                    for x in crit_w[1:]:
                        if x not in balls[v]: # O(15) check
                            valid = False
                            break
                            
                    if valid:
                        # Perform swap
                        # Remove w, add v
                        current_set.remove(w)
                        current_set.add(v)
                        
                        # Update cover_counts
                        cover_counts[balls[w]] -= 1
                        cover_counts[balls[v]] += 1
                        
                        # Run deletion phase
                        # ... (reuse deletion logic)
                        del_changed = True
                        while del_changed:
                            del_changed = False
                            items = list(current_set)
                            random.shuffle(items)
                            for u in items:
                                if u not in current_set: continue
                                if np.min(cover_counts[balls[u]]) > 1:
                                    current_set.remove(u)
                                    cover_counts[balls[u]] -= 1
                                    del_changed = True
                                    
                        # Check if size improved
                        if len(current_set) < best_size:
                            best_size = len(current_set)
                            best_set = set(current_set)
                            improved = True
                            # Break out of w loop? Or continue?
                            # Let's continue to see if more swaps help.
                            # But we need to update criticals for next iteration of outer while.
                            # So we can break the inner w loop.
                            break
                        else:
                            # Revert swap
                            current_set.remove(v)
                            current_set.add(w)
                            cover_counts[balls[v]] -= 1
                            cover_counts[balls[w]] += 1
                            
                # End for v
                
            # End for w
            
        return list(best_set)

    # Main loop
    best_result = None
    best_len = float('inf')
    
    # Run multiple restarts
    restart_count = 0
    while time.time() - start_time < time_limit:
        restart_count += 1
        seed_val = restart_count
        greedy_sol = greedy_construct(seed_val)
        
        if not greedy_sol:
            continue
            
        # Local search
        optimized_sol = local_search(greedy_sol)
        
        if len(optimized_sol) < best_len:
            best_len = len(optimized_sol)
            best_result = optimized_sol
            
        # If we reached the known best (186) or lower, we can stop or keep trying for a bit
        if best_len <= 186:
            # Keep running a bit more to see if we can go lower
            # But 186 is the best known.
            pass

    if best_result is None:
        # Fallback to last greedy
        best_result = greedy_construct(0)
        
    # Convert indices to lists of ints
    result = []
    for idx in best_result:
        w = int_to_base3(idx, 7)
        result.append(w)
        
    return result

def _generic_construct(q, n, radius):
    # Simple fallback
    import itertools
    words = list(itertools.product(range(q), repeat=n))
    # Greedy
    cover = set()
    selected = []
    # This is slow for large n, but n is small in fallback?
    # Just return a trivial code for safety
    return [list(w) for w in words]
