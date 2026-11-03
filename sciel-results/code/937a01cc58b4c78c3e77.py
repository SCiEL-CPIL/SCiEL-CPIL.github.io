import numpy as np
import itertools
import time
import random
import heapq

def construct(q, n, radius):
    # Fallback for non-specific parameters
    if q != 3 or n != 8 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 4:
                return tetracode
            else:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Specific optimization for K_3(8,1)
    total_points = q ** n  # 6561
    neighbors_per_word = 1 + n * (q - 1)  # 17

    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors for all 6561 words
    neighbors = np.zeros((total_points, neighbors_per_word), dtype=np.int32)
    for i in range(total_points):
        w = idx_to_word(i)
        neighbors[i, 0] = i
        count = 1
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                neighbors[i, count] = word_to_idx(w)
                count += 1
            w[pos] = orig_val

    all_words = [idx_to_word(i) for i in range(total_points)]

    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    # --- Lazy Deletion Max-Coverage Solver ---
    
    def solve_max_coverage():
        """
        Solves the covering problem using a lazy deletion priority queue.
        1. Greedy initialization to cover the space.
        2. Local search: Add best new word, Remove worst existing word.
        """
        cover_count = np.zeros(total_points, dtype=np.int32)
        in_code = np.zeros(total_points, dtype=bool)
        
        # Heap stores (-marginal_gain, word_index)
        # We use negative gain because heapq is a min-heap
        heap = []
        
        # Helper to push a word's gain
        def push_gain(word_idx):
            if in_code[word_idx]:
                return
            # Marginal gain = number of neighbors not currently covered
            gain = np.sum(cover_count[neighbors[word_idx]] == 0)
            if gain > 0:
                heapq.heappush(heap, (-gain, word_idx))

        # Initialize heap with all words
        for i in range(total_points):
            push_gain(i)

        # 1. Greedy Phase: Cover the space
        while not np.all(cover_count > 0):
            if time.time() - start_time > time_limit - 20.0:
                break
            # Get best candidate
            neg_gain, w_idx = heapq.heappop(heap)
            
            # Lazy deletion check:
            # 1. Is it already in the code?
            if in_code[w_idx]:
                continue
            # 2. Has its gain changed? (Recompute and re-push if needed)
            current_gain = np.sum(cover_count[neighbors[w_idx]] == 0)
            if -neg_gain != current_gain:
                if current_gain > 0:
                    heapq.heappush(heap, (-current_gain, w_idx))
                continue
            
            # Accept this word
            in_code[w_idx] = True
            cover_count[neighbors[w_idx]] += 1
            
            # Update gains for all words that share neighbors with this new word
            # A word v's gain changes if it covers any neighbor of w_idx
            # We just push them back; lazy deletion handles the rest
            # Optimization: Only push words that might have gained/lost value
            # Actually, any word covering a neighbor of w_idx is affected.
            # Since we don't have an inverted index (neighbor -> words), 
            # and total_points is 6561, iterating over all is 6561 ops.
            # 6561 * ~400 steps = ~2.6M ops. Acceptable.
            for v in range(total_points):
                if not in_code[v]:
                    # If v shares a neighbor with w_idx, its gain might change
                    # We can just push it. The check in pop will verify.
                    # To save heap size, only push if they share a neighbor?
                    # Intersection check is expensive. Just push all?
                    # No, that's 6561 pushes per step. 400*6561 = 2.6M pushes.
                    # Heap size grows. 
                    # Better: Only push if they are neighbors of neighbors?
                    # Let's just push all non-code words. It's O(N^2 log N) total roughly.
                    # 6561^2 is ~43M. With log factor, might be slow.
                    # Let's optimize: Only push v if v covers a point that w_idx covers?
                    # i.e. if neighbors[v] intersects neighbors[w_idx].
                    # Since we don't have inverted index, we check:
                    # if np.any(cover_count[neighbors[v]] == 0) changed?
                    # Simplest correct lazy deletion: Push ALL non-code words.
                    # But 6561 pushes per iteration is heavy.
                    # Alternative: Only push words that are neighbors of the covered points?
                    # Let's stick to pushing all, but optimize the inner loop.
                    pass
        
        # Wait, the "push all" strategy above is too slow if done naively in Python loop.
        # Let's refine the update strategy.
        # When we add w_idx, cover_count changes for neighbors[w_idx].
        # A word v's gain is sum(cover_count[neighbors[v]] == 0).
        # This changes only if neighbors[v] intersects neighbors[w_idx].
        # We can precompute an inverted index: for each point p, list of words covering p.
        # Then, when w_idx is added, for each p in neighbors[w_idx], for each v in inverted[p], push v.
        
        # Let's restart the solver with inverted index for efficiency.
        # Re-initialization
        cover_count = np.zeros(total_points, dtype=np.int32)
        in_code = np.zeros(total_points, dtype=bool)
        heap = []
        
        # Build inverted index: point -> list of words covering it
        # Each point is covered by 17 words.
        inverted = [[] for _ in range(total_points)]
        for i in range(total_points):
            for p in neighbors[i]:
                inverted[p].append(i)
        
        def push_gain(word_idx):
            if in_code[word_idx]:
                return
            gain = np.sum(cover_count[neighbors[word_idx]] == 0)
            if gain > 0:
                heapq.heappush(heap, (-gain, word_idx))

        # Initial push
        for i in range(total_points):
            push_gain(i)

        # 1. Greedy Phase
        while not np.all(cover_count > 0):
            if time.time() - start_time > time_limit - 30.0:
                break
            neg_gain, w_idx = heapq.heappop(heap)
            
            if in_code[w_idx]:
                continue
            
            current_gain = np.sum(cover_count[neighbors[w_idx]] == 0)
            if -neg_gain != current_gain:
                if current_gain > 0:
                    heapq.heappush(heap, (-current_gain, w_idx))
                continue
            
            # Accept
            in_code[w_idx] = True
            cover_count[neighbors[w_idx]] += 1
            
            # Update heap for affected words
            # Affected words are those covering any point in neighbors[w_idx]
            # We collect unique affected words to avoid duplicate pushes (though lazy handles dupes, 
            # duplicates bloat heap)
            affected = set()
            for p in neighbors[w_idx]:
                for v in inverted[p]:
                    if not in_code[v]:
                        affected.add(v)
            
            for v in affected:
                push_gain(v)

        # 2. Local Search Phase: Add-Best-Remove-Worst
        # Goal: Minimize size.
        # Strategy:
        # While possible:
        #   1. Find best word to ADD (max marginal gain among non-code words).
        #   2. Find worst word to REMOVE (word whose removal doesn't break coverage, 
        #      ideally one that frees up the most "slack" or is redundant).
        #      Actually, "worst" is usually the one with the most redundancy (cover_count >= 2 for all its neighbors).
        #      But we want to ADD a word that allows REMOVING a word.
        #      Standard 1-swap: Add u, Remove v.
        #      Condition: After adding u, v becomes redundant.
        
        # Let's implement a simpler 1-swap local search using the heap for "Add".
        # For "Remove", we maintain a list of "removable" words.
        
        removable = [] # List of indices in code that are currently redundant (cover_count[neighbors[i]] >= 2)
        
        def update_removable():
            removable.clear()
            for i in range(total_points):
                if in_code[i]:
                    if np.all(cover_count[neighbors[i]] >= 2):
                        removable.append(i)
        
        # Initial removable check
        update_removable()
        
        # Local Search Loop
        no_improvement_count = 0
        max_no_improvement = 1000
        
        while time.time() - start_time < time_limit - 5.0 and no_improvement_count < max_no_improvement:
            if not heap:
                break
            
            # Get best add candidate
            neg_gain, u = heapq.heappop(heap)
            if in_code[u]:
                continue
            current_gain = np.sum(cover_count[neighbors[u]] == 0)
            if -neg_gain != current_gain:
                if current_gain > 0:
                    heapq.heappush(heap, (-current_gain, u))
                continue
            
            # u is the best word to add.
            # If gain is 0, no word can improve coverage, but we might still swap?
            # If gain is 0, adding u doesn't cover new points. It might still allow removal if u overlaps heavily with v.
            # But typically we look for swaps that reduce size.
            
            # Try to find a v in removable that we can remove after adding u.
            # Actually, if u has gain 0, it covers only already covered points.
            # Adding u increases cover_count for its neighbors.
            # This might make MORE words removable.
            # Then we can remove one.
            # Net change: +1 (u) - 1 (v) = 0. No size reduction.
            # We need a swap where we add u and remove v, AND size decreases?
            # No, 1-swap keeps size same. We need to find a sequence or a 2-swap?
            # Or, does adding u make an existing word v removable, and then removing v makes another word w removable?
            # That's a chain.
            
            # Alternative: "Add-Best" then "Remove-Worst" repeatedly.
            # If we add u (size +1), then remove v (size -1), net 0.
            # To reduce size, we need to find a u that allows removing TWO words? Unlikely.
            # Or, we start with a non-minimal cover, and local search finds a smaller one?
            # Greedy usually gives near-minimal.
            
            # Let's try a different local move:
            # Replace v with u.
            # If v is removable WITHOUT u, we just remove v.
            # If v is NOT removable without u, but IS removable WITH u, we swap.
            # Size stays same.
            
            # To get size reduction, we need to find a set of words that can be removed.
            # The "Remove-Worst" phase after "Add-Best" might uncover redundancies.
            
            # Let's just do:
            # 1. Add u.
            # 2. Update cover counts.
            # 3. Update removable list.
            # 4. If any word in removable is NOT u, remove it.
            #    This reduces size by 0 (add 1, remove 1).
            #    Wait, if we add u, and remove v, size is constant.
            #    We only reduce size if we remove v WITHOUT adding u.
            
            # So the strategy is:
            # 1. Check if any word is removable. If yes, remove it. (Size -1)
            # 2. If no word is removable, try to find a swap (Add u, Remove v) that leads to a state where removal is possible?
            #    This is complex.
            
            # Let's stick to:
            # Phase A: Aggressive Removal.
            # While removable is not empty:
            #   Remove a random word from removable.
            #   Update cover counts.
            #   Update heap (gains of neighbors increase).
            #   Update removable list.
            
            # Phase B: If no removable words, try to find a "productive" swap.
            #   A swap (u, v) is productive if adding u makes v removable.
            #   Then we perform the swap.
            #   This keeps size same, but might enable future removals?
            
            # Let's implement Phase A first. It's the most direct way to reduce size.
            
            removed_any = False
            while removable:
                if time.time() - start_time > time_limit - 5.0:
                    break
                
                # Pick a removable word
                v = removable.pop()
                
                # Verify it's still removable (lazy check)
                if not in_code[v]:
                    continue
                if not np.all(cover_count[neighbors[v]] >= 2):
                    # It became non-removable due to other changes?
                    # If we are in a loop removing words, cover counts only decrease.
                    # So if it was removable, it stays removable?
                    # No, if we remove another word w that covered a neighbor of v,
                    # cover_count[neighbors[v]] might drop to 1.
                    # So v might no longer be removable.
                    # Re-add to removable? Or just skip.
                    # If we skip, we might miss a removal opportunity later.
                    # Let's just skip.
                    continue
                
                # Remove v
                in_code[v] = False
                cover_count[neighbors[v]] -= 1
                removed_any = True
                
                # Update heap: Gains of words covering neighbors of v increase
                # Words covering p (where p in neighbors[v]) might have higher gain now
                affected = set()
                for p in neighbors[v]:
                    for u in inverted[p]:
                        if not in_code[u]:
                            affected.add(u)
                
                for u in affected:
                    push_gain(u)
                
                # Update removable list
                # Words covering neighbors of v might become removable?
                # No, cover counts decreased, so fewer words are removable.
                # But words that were NOT removable might become removable? No.
                # Words that WERE removable might become NON-removable.
                # So we just need to check if any current code word is removable.
                # This is O(N * 17).
                # Let's do a full scan every few removals or if heap is empty.
                # For now, let's just do a full scan if we removed a word.
                # It's 6561 * 17 ~ 100k ops. If we remove 100 words, 10M ops. OK.
                
                removable.clear()
                for i in range(total_points):
                    if in_code[i]:
                        if np.all(cover_count[neighbors[i]] >= 2):
                            removable.append(i)
            
            if removed_any:
                no_improvement_count = 0
                continue
            
            # No removals possible. Try swaps?
            # If we don't do swaps, we are stuck at a local minimum.
            # Let's try a random swap to escape local minimum.
            # Pick random u (not in code), random v (in code).
            # If adding u makes v removable, do it.
            
            # Find best u from heap
            # We already popped u. If gain > 0, it covers new points.
            # If gain == 0, it covers old points.
            
            # Let's try a few random swaps
            found_swap = False
            for _ in range(10):
                if time.time() - start_time > time_limit - 5.0:
                    break
                
                # Pick random v from code
                code_list = np.where(in_code)[0]
                if len(code_list) == 0:
                    break
                v = code_list[random.randint(0, len(code_list) - 1)]
                
                # Pick random u from non-code
                non_code_list = np.where(~in_code)[0]
                if len(non_code_list) == 0:
                    break
                u = non_code_list[random.randint(0, len(non_code_list) - 1)]
                
                # Try swap
                # Temporarily add u, remove v
                in_code[u] = True
                in_code[v] = False
                cover_count[neighbors[u]] += 1
                cover_count[neighbors[v]] -= 1
                
                # Check if any word is removable
                # Specifically, is v removable? (It was removed, so N/A)
                # Is u removable? (It was just added, unlikely)
                # Is any other word w removable?
                
                removable_check = False
                for w in range(total_points):
                    if in_code[w]:
                        if np.all(cover_count[neighbors[w]] >= 2):
                            removable_check = True
                            break
                
                if removable_check:
                    # Accept swap
                    found_swap = True
                    # Update heap
                    affected = set()
                    for p in neighbors[u]:
                        for x in inverted[p]:
                            if not in_code[x]:
                                affected.add(x)
                    for p in neighbors[v]:
                        for x in inverted[p]:
                            if not in_code[x]:
                                affected.add(x)
                    for x in affected:
                        push_gain(x)
                    
                    # Update removable list
                    removable.clear()
                    for i in range(total_points):
                        if in_code[i]:
                            if np.all(cover_count[neighbors[i]] >= 2):
                                removable.append(i)
                    break
                else:
                    # Revert
                    in_code[u] = False
                    in_code[v] = True
                    cover_count[neighbors[u]] -= 1
                    cover_count[neighbors[v]] += 1
            
            if not found_swap:
                no_improvement_count += 1
                # To avoid getting stuck, we can add a random word and try to remove two?
                # Or just give up.
                # Let's try one more thing: Add best u, then try to remove ANY word.
                # If we can remove any word, size stays same.
                # But if we remove a word, we might enable another removal?
                # That's the "chain" idea.
                
                # Let's just break if no improvement for a while.
                if no_improvement_count > 500:
                    break

        return [i for i in range(total_points) if in_code[i]]

    # Run the solver
    result_indices = solve_max_coverage()
    
    # Verify validity
    cover_count = np.zeros(total_points, dtype=np.int32)
    for ci in result_indices:
        cover_count[neighbors[ci]] += 1
    if np.any(cover_count == 0):
        # Fallback to simple greedy if solver failed
        covered = np.zeros(total_points, dtype=bool)
        chosen = []
        for i in range(total_points):
            if not covered.all():
                if not np.any(covered[neighbors[i]]): # Only add if it covers something new? No, just add all if needed
                    pass
            # Simple greedy fallback
            newly = np.sum(~covered[neighbors[i]])
            if newly > 0:
                # Pick best
                best_i = i
                best_new = newly
                # This fallback is inefficient, just ensure validity
                pass
        
        # Actually, just use the standard greedy from the previous version if this fails
        # But let's trust the lazy solver. If it fails, it means a bug.
        # Let's do a quick repair:
        for i in range(total_points):
            if np.any(cover_count[neighbors[i]] == 0):
                # Add i to cover the gaps
                result_indices.append(i)
                cover_count[neighbors[i]] += 1
    
    return [all_words[i] for i in result_indices]
