import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases
    if q == 3 and n >= 4:
        # Direct sum with tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Generic fallback
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Precompute neighbors: neighbors[w] is an array of indices of words within distance 1 from w
    # There are 1 + 8*2 = 17 neighbors
    neighbors = np.zeros((total, 17), dtype=np.int32)
    for w in range(total):
        neighbors[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = 3 ** (n - 1 - pos)
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                neighbors[w, idx] = new_w
                idx += 1
    
    # 1. Greedy Set Cover to get an initial valid solution
    # cover_count[w] is the number of selected codewords within distance 1 of w
    cover_count = np.zeros(total, dtype=np.int32)
    selected = []
    uncovered_count = total
    
    # Greedy phase
    # We pick the codeword that covers the most currently uncovered words
    # To speed this up, we can use a priority queue or just iterate. 
    # Given the small size (6561), simple iteration with numpy vectorization is fast enough for the first few steps.
    # However, doing this for ~500 steps might be slow if done naively.
    # Let's use a slightly more efficient greedy:
    # Maintain a set of uncovered words.
    # For each candidate, count how many uncovered words it covers.
    
    # Optimization: Precompute the neighbor lists as a list of arrays for faster indexing? 
    # neighbors is already a 2D array. neighbors[cw] gives the ball.
    
    # Simple Greedy:
    # We need to find argmax of sum(cover_count[neighbors[cw]] == 0)
    # This is expensive to compute from scratch every time.
    # Alternative: Randomized Greedy or Local Search from a random start?
    # Let's try a faster greedy:
    # 1. Start with empty set.
    # 2. While uncovered > 0:
    #    a. Pick a random uncovered word w.
    #    b. Among all codewords c in Ball(w), pick the one that covers the most uncovered words.
    #    c. Add c to selected.
    
    # This "pick random uncovered, then best cover" heuristic is often faster and good enough for a starting point.
    uncovered_mask = np.ones(total, dtype=bool)
    
    while uncovered_count > 0 and time.time() < deadline - 60.0:
        # Pick a random uncovered word
        uncov_indices = np.where(uncovered_mask)[0]
        if len(uncov_indices) == 0:
            break
        target_w = uncov_indices[np.random.randint(len(uncov_indices))]
        
        # Candidates are all codewords within distance 1 of target_w
        # These are the neighbors of target_w
        candidates = neighbors[target_w]
        
        # For each candidate c, count how many uncovered words it covers
        # c covers word w if w is in neighbors[c]
        # We want to maximize sum(uncovered_mask[neighbors[c]])
        
        # Vectorized calculation of coverage for all candidates
        # candidates is shape (17,). neighbors[candidates] is shape (17, 17).
        # We need to sum uncovered_mask over the second axis for each candidate.
        # uncovered_mask is shape (6561,).
        # coverage_counts = uncovered_mask[neighbors[candidates]].sum(axis=1)
        
        # To avoid huge temporary arrays if possible, but 17x17 is tiny.
        coverage_counts = uncovered_mask[neighbors[candidates]].sum(axis=1)
        
        # Pick the best candidate
        best_local_idx = np.argmax(coverage_counts)
        best_cw = candidates[best_local_idx]
        
        # Add to selected
        selected.append(best_cw)
        
        # Update cover_count and uncovered_mask
        ball = neighbors[best_cw]
        cover_count[ball] += 1
        
        # Update uncovered_mask: words in ball that were uncovered become covered
        newly_covered = uncovered_mask[ball]
        uncovered_count -= np.sum(newly_covered)
        uncovered_mask[ball] = False
        
    # If greedy didn't finish, it's a problem, but unlikely for this size.
    if uncovered_count > 0:
        # Fallback: add random codewords until covered
        while uncovered_count > 0:
            cw = np.random.randint(total)
            if cw not in selected:
                selected.append(cw)
                ball = neighbors[cw]
                cover_count[ball] += 1
                newly_covered = uncovered_mask[ball]
                uncovered_count -= np.sum(newly_covered)
                uncovered_mask[ball] = False
    
    current_size = len(selected)
    best_size = current_size
    best_set = selected.copy()
    
    # 2. Local Search Phase
    # Objective: Minimize |S| subject to Cover(S) = All
    # Moves:
    # A. Remove a codeword c if it doesn't uncover any word (i.e., all words in Ball(c) are covered by other codewords).
    # B. Swap: Remove c1, add c2 if it reduces size or maintains size but allows future removals? 
    #    Actually, standard local search for set cover minimization:
    #    - Try removing any codeword. If valid, keep it.
    #    - Try swapping one codeword for another. This is complex.
    #    - Simpler: Just do removal passes. If stuck, try "perturbation" (random swap) and then removals again.
    
    # Efficient Removal Pass:
    # A codeword c is removable if min(cover_count[neighbors[c]]) > 1.
    # This means every word in its ball is covered by at least 2 codewords (including c).
    # So if we remove c, they are still covered by at least 1.
    
    def try_remove_pass(selected, cover_count, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Check all codewords in selected
            # We can vectorize this check for the whole selected set?
            # selected is a list of indices.
            # Let's iterate.
            for i in range(len(selected) - 1, -1, -1):
                cw = selected[i]
                # Check if removable
                if np.min(cover_count[neighbors[cw]]) > 1:
                    # Remove it
                    selected.pop(i)
                    cover_count[neighbors[cw]] -= 1
                    improved = True
                    # Break and restart the pass? Or continue? 
                    # Continuing is fine, but the condition for other codewords might change.
                    # Actually, removing a codeword only decreases cover counts, so it can't make another codeword *more* removable 
                    # (it makes it harder to satisfy min > 1). 
                    # Wait. If we remove c, cover_count drops. 
                    # If c was removable, min(cover_count[neighbors[c]]) > 1.
                    # This doesn't affect the removability of other codewords directly in a way that creates new removabilities.
                    # It only potentially destroys removability of others.
                    # So we can just continue the loop.
                    
            if improved:
                # We made changes, loop again to catch any that became removable? 
                # Actually, as noted, removals don't create new removabilities. 
                # So one pass is enough to find ALL removable codewords?
                # Let's verify. 
                # If c1 and c2 are both removable initially.
                # Remove c1. c2 is still removable? 
                # cover_count for words in Ball(c2) decreases by 1 if they are in Ball(c1).
                # If a word w in Ball(c2) was covered by c1 and c2 (count 2), now it's covered by c2 (count 1).
                # Then min(cover_count[neighbors[c2]]) might become 1, so c2 is NO LONGER removable.
                # So the order matters, and one pass might not find the maximal set of removable codewords.
                # However, any order of removals leads to a maximal independent set in the "redundancy" sense?
                # No, it's just a greedy removal. 
                # To maximize removals, we should perhaps remove the "most redundant" ones first? 
                # Or just iterate until no more can be removed.
                # Since removals can only hurt removability, we should do a single pass? 
                # No, if I remove c1, c2 might become non-removable. If I removed c2 first, c1 might become non-removable.
                # So the set of removed codewords depends on order.
                # To get the smallest set, we might want to be smart. 
                # But for now, let's just do repeated passes until stable.
                pass
        
        return len(selected)

    # We need to handle the case where we are stuck with a valid set that has no removable codewords.
    # This is a local minimum.
    # We can try a "swap" move: Replace c1 with c2.
    # Condition for valid swap: 
    # 1. c2 is not in selected.
    # 2. After removing c1 and adding c2, all words are still covered.
    # This is equivalent to: The words uniquely covered by c1 (words in Ball(c1) with cover_count == 1) 
    # must be covered by c2.
    # i.e., Ball(c1) \ Ball(c2) contains no word with cover_count == 1.
    # Actually, simpler: 
    # Let U_c1 = {w in Ball(c1) | cover_count[w] == 1}.
    # If we remove c1, these words become uncovered.
    # We need c2 to cover all of them.
    # So U_c1 must be a subset of Ball(c2).
    
    # Let's implement a local search loop with swaps.
    
    current_selected = selected
    current_cover_count = cover_count
    
    # Precompute a helper: for each codeword c, the list of words it uniquely covers?
    # This is dynamic.
    
    # To speed up swaps, we can precompute for each c, the set of words in Ball(c).
    # We already have neighbors[c].
    
    # Heuristic: 
    # While time remains:
    # 1. Try to remove any removable codeword.
    # 2. If no removal possible, try a random swap.
    #    - Pick random c1 in selected.
    #    - Pick random c2 not in selected.
    #    - Check if swap is valid.
    #    - If valid, perform swap.
    #    - Repeat removal pass.
    
    # To make swap checking fast:
    # We need to check if all words w in Ball(c1) with cover_count[w]==1 are in Ball(c2).
    # Let unique_words_c1 = neighbors[c1][cover_count[neighbors[c1]] == 1]
    # We need to check if all these words are in neighbors[c2].
    # This can be done with a set intersection or numpy in1d.
    
    # Let's optimize the removal pass first.
    
    def do_removal_pass(sel, cc, deadline):
        removed_any = False
        # Iterate backwards to allow safe popping
        for i in range(len(sel) - 1, -1, -1):
            if time.time() > deadline:
                break
            cw = sel[i]
            if np.min(cc[neighbors[cw]]) > 1:
                sel.pop(i)
                cc[neighbors[cw]] -= 1
                removed_any = True
        return removed_any

    # Main SA/Local Search Loop
    # We will use a simple random restart / perturbation strategy if stuck.
    
    # First, do a full removal pass on the greedy result.
    do_removal_pass(current_selected, current_cover_count, deadline)
    best_size = len(current_selected)
    best_set = current_selected.copy()
    
    # Now, try to improve via swaps.
    # We limit the number of swap attempts to avoid getting stuck in slow loops.
    
    swap_attempts = 0
    max_swap_attempts = 50000 # Adjust based on time
    
    while time.time() < deadline - 5.0:
        # Check for removable codewords
        if do_removal_pass(current_selected, current_cover_count, deadline):
            best_size = len(current_selected)
            best_set = current_selected.copy()
            if best_size <= 486:
                break
            continue # Restart swap search after removal
        
        # No removable codewords. Try a swap.
        if len(current_selected) <= 1:
            break
            
        # Pick a random codeword to remove
        idx_remove = np.random.randint(len(current_selected))
        c1 = current_selected[idx_remove]
        
        # Identify words uniquely covered by c1
        ball_c1 = neighbors[c1]
        unique_words = ball_c1[current_cover_count[ball_c1] == 1]
        
        if len(unique_words) == 0:
            # This shouldn't happen if removal pass was effective, but just in case
            current_selected.pop(idx_remove)
            current_cover_count[ball_c1] -= 1
            continue
            
        # We need to find a c2 such that unique_words is a subset of neighbors[c2]
        # And c2 is not in current_selected.
        
        # Strategy: Pick a random word w from unique_words.
        # c2 MUST be in neighbors[w] to cover w.
        # So candidates for c2 are in neighbors[w].
        
        w_target = unique_words[np.random.randint(len(unique_words))]
        candidates_c2 = neighbors[w_target]
        
        # Filter candidates: 
        # 1. Not in current_selected
        # 2. Must cover ALL unique_words.
        
        # To check "must cover all unique_words" efficiently:
        # For each candidate c2, check if unique_words is a subset of neighbors[c2].
        # This is expensive if done for all 17 candidates.
        # But 17 is small.
        
        valid_c2 = None
        # Shuffle candidates to randomize
        np.random.shuffle(candidates_c2)
        
        for c2 in candidates_c2:
            if c2 in current_selected:
                continue
            
            # Check if c2 covers all unique_words
            # unique_words is an array of indices.
            # neighbors[c2] is an array of indices.
            # We need to check if all elements of unique_words are in neighbors[c2].
            
            # Using numpy in1d is O(N*M). N=17, M=len(unique_words).
            # len(unique_words) can be up to 17.
            # So 17*17 = 289 ops. Very fast.
            
            if np.all(np.in1d(unique_words, neighbors[c2])):
                valid_c2 = c2
                break
        
        if valid_c2 is not None:
            # Perform swap
            # Remove c1
            current_selected[idx_remove] = valid_c2
            current_cover_count[neighbors[c1]] -= 1
            current_cover_count[neighbors[valid_c2]] += 1
            
            # After swap, check for removals again (loop will do it)
            swap_attempts += 1
            if swap_attempts % 100 == 0:
                # Optional: Check time
                if time.time() > deadline - 5.0:
                    break
        else:
            # No valid swap found for this c1.
            # Try another c1? Or just give up on this iteration?
            # To avoid getting stuck, we can just continue the loop.
            # But if we always pick the same c1 and fail, we waste time.
            # Let's just continue.
            pass

    # Final cleanup
    do_removal_pass(current_selected, current_cover_count, deadline)
    
    if len(current_selected) < best_size:
        best_size = len(current_selected)
        best_set = current_selected.copy()

    # Convert best_set to list of digit lists
    if best_set is None:
        # Fallback
        return [list(np.base_repr(w, 3).zfill(n).ljust(n, '0')) for w in range(total)]
        
    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
