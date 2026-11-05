import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    total_words = q ** n
    
    # Precompute word <-> index conversions
    def word_to_idx(word):
        idx = 0
        for v in word:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute balls (covered indices) for each codeword
    # ball[cw_idx] is a list of indices covered by cw_idx
    balls = []
    for cw_idx in range(total_words):
        cw = idx_to_word(cw_idx)
        indices = [cw_idx]
        if radius >= 1:
            for pos in range(n):
                orig = cw[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    lst = list(cw)
                    lst[pos] = new_val
                    indices.append(word_to_idx(lst))
        balls.append(np.array(indices, dtype=np.int32))

    # Convert to a 2D numpy array for faster slicing if needed, 
    # but list of arrays is fine for 729 size.
    # Let's use a flat array for balls to improve cache locality? 
    # No, list of int32 arrays is fast enough for 729*13.

    # Helper to update coverage counts
    # coverage: np.array of int32, size total_words
    def add_coverage(coverage, cw_idx):
        coverage[balls[cw_idx]] += 1

    def remove_coverage(coverage, cw_idx):
        coverage[balls[cw_idx]] -= 1

    def is_redundant(coverage, cw_idx):
        # A codeword is redundant if all words in its ball are covered by >1 codeword
        return np.all(coverage[balls[cw_idx]] > 1)

    def is_valid(coverage):
        return np.all(coverage > 0)

    # Phase 1: Initial Construction using Greedy Set Cover
    # This gives a valid starting point.
    def greedy_construct():
        remaining = set(range(total_words))
        selected = []
        coverage = np.zeros(total_words, dtype=np.int32)
        
        # To make it faster, we can just pick the best candidate each time
        # Candidates are all words. 
        # We need to count how many *new* words each candidate covers.
        
        # Optimization: 
        # Instead of iterating all 729 words and counting uncovered, 
        # we can maintain a list of uncovered words.
        
        uncovered = set(range(total_words))
        
        while uncovered:
            if time.time() > deadline:
                break
            best_cw = -1
            best_gain = -1
            # Sample a subset of candidates if the set is large? 
            # 729 is small, so check all.
            for cw in range(total_words):
                # Calculate gain: number of words in ball[cw] that are in uncovered
                # Using set intersection is slow. 
                # Better: use boolean mask.
                # But we have `uncovered` as a set.
                # Let's convert uncovered to a boolean array for speed?
                # That's 729 operations. 729 * 729 = 530k ops per iteration.
                # Iterations ~ 80. 42M ops. Acceptable in Python/NumPy mix?
                
                # Alternative: 
                # Precompute a matrix M where M[cw, w] = 1 if w in ball[cw].
                # Then gain = sum(M[cw, :][uncovered_mask]).
                
                # Let's stick to a simpler heuristic for the first pass to ensure we have a valid code.
                # Random greedy is often good enough to start.
                pass
            
            # Simpler Greedy: Pick a random word from remaining, if it covers many uncovered.
            # Let's just do a standard greedy with a boolean mask for uncovered.
            
            # This is getting complex to optimize perfectly inside the loop.
            # Let's use a different initialization: 
            # Start with all words, then remove redundancies? 
            # 729 words. 
            # coverage = np.ones(729) * 729? No.
            
            # Let's use the "Direct Sum" idea from the prompt as a seed, 
            # but ensure it's valid.
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n >= 4:
                initial_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
            else:
                initial_codewords = [list(word) for word in itertools.product(range(q), repeat=n)]
                
            initial_idx = [word_to_idx(cw) for cw in initial_codewords]
            
            # Verify validity
            cov = np.zeros(total_words, dtype=np.int32)
            for idx in initial_idx:
                add_coverage(cov, idx)
            
            if not is_valid(cov):
                 # Fallback to random greedy if seed is invalid (shouldn't happen)
                 # But let's just return this for now and let local search fix it? 
                 # No, local search assumes valid start.
                 # Let's add missing words.
                 while not is_valid(cov):
                     missing = np.where(cov == 0)[0]
                     # Pick a word that covers the most missing
                     best_cw = -1
                     best_cov = -1
                     for cw in range(total_words):
                         # Count how many missing words are in ball[cw]
                         # Use numpy
                         count = np.sum(cov[balls[cw]] == 0)
                         if count > best_cov:
                             best_cov = count
                             best_cw = cw
                     if best_cw != -1:
                         initial_idx.append(best_cw)
                         add_coverage(cov, best_cw)
                     else:
                         break
                         
            return initial_idx, cov

    initial_cw_idx_list, current_coverage = greedy_construct()
    best_cw_idx_list = initial_cw_idx_list.copy()
    best_len = len(initial_cw_idx_list)
    
    # Phase 2: Local Search
    # We maintain a list of codewords and a coverage array.
    
    def local_search(cw_list, coverage, max_iters=100000):
        # cw_list: list of int indices
        # coverage: np array
        # Returns improved list and coverage
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters:
            if time.time() > deadline:
                break
            improved = False
            iter_count += 1
            
            # 1. Try to remove redundant words
            # Shuffle to avoid bias
            indices = list(range(len(cw_list)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                cw_idx = cw_list[i]
                if is_redundant(coverage, cw_idx):
                    # Remove it
                    remove_coverage(coverage, cw_idx)
                    cw_list.pop(i)
                    improved = True
                    best_len_holder[0] = len(cw_list)
                    best_cw_holder[0] = cw_list.copy()
                    # Restart loop to check for new redundancies
                    break
            
            if improved:
                continue
                
            # 2. Try Swaps (Add-Remove)
            # Find a word to remove and a word to add such that we can remove more later.
            # Strategy: 
            # Pick a random word `rem_idx` from cw_list.
            # Remove it.
            # Find uncovered words.
            # Find a word `add_idx` that covers the most uncovered words.
            # Add it.
            # Try to remove redundancies again.
            # If the final size is smaller, accept.
            
            if len(cw_list) < 10:
                break
                
            rem_pos = random.randint(0, len(cw_list) - 1)
            rem_idx = cw_list[rem_pos]
            
            # Temporarily remove
            remove_coverage(coverage, rem_idx)
            
            # Find uncovered
            uncovered = np.where(coverage == 0)[0]
            
            if len(uncovered) == 0:
                # We just removed a redundant word? 
                # Should have been caught in step 1. 
                # But maybe it became redundant due to previous changes in this iteration? 
                # No, step 1 breaks on first removal.
                # So if uncovered is empty, we made an improvement.
                cw_list.pop(rem_pos)
                improved = True
                best_len_holder[0] = len(cw_list)
                best_cw_holder[0] = cw_list.copy()
                break
                
            # Find best replacement
            best_add_idx = -1
            best_gain = -1
            
            # Sample candidates to save time? 
            # With 729 words, checking all is ~729 * 13 ops = 9k ops. Very fast.
            for add_idx in range(total_words):
                if add_idx in cw_list:
                    continue
                # How many uncovered words does this cover?
                # We can use a precomputed mask or just sum
                # gain = sum(1 for u in uncovered if u in balls[add_idx])
                # Using numpy:
                # We need to check intersection of balls[add_idx] and uncovered.
                # Since uncovered is small usually, maybe loop is faster?
                # Or use a boolean mask for uncovered.
                
                # Let's use a set for uncovered for O(1) lookup?
                # No, numpy sum is fast.
                # Let's create a boolean mask for uncovered once per swap attempt?
                # That's 729 ops.
                
                pass
            
            # To speed up, let's precompute a matrix `covers` where covers[cw, w] = 1
            # Then gain = covers[add_idx][uncovered].sum()
            
            # I will implement the gain calculation using a precomputed 2D boolean array.
            
            if best_add_idx != -1:
                # Add the best candidate
                add_coverage(coverage, best_add_idx)
                cw_list.append(best_add_idx)
                
                # Now try to remove redundancies
                # We removed 1, added 1. Size is same.
                # But the new word might make others redundant.
                sub_improved = True
                while sub_improved:
                    if time.time() > deadline:
                        break
                    sub_improved = False
                    # Shuffle and try remove
                    sub_indices = list(range(len(cw_list)))
                    random.shuffle(sub_indices)
                    for i in sub_indices:
                        if time.time() > deadline:
                            break
                        cw_idx = cw_list[i]
                        if is_redundant(coverage, cw_idx):
                            remove_coverage(coverage, cw_idx)
                            cw_list.pop(i)
                            sub_improved = True
                            best_len_holder[0] = len(cw_list)
                            best_cw_holder[0] = cw_list.copy()
                            break
                            
                if len(cw_list) < best_len_holder[0]:
                    improved = True
                    # Accept the swap and removals
                else:
                    # Rollback
                    # This is tricky because we modified coverage and cw_list in place.
                    # We need to restore state.
                    # Easier: Copy state before swap.
                    pass
            else:
                # Rollback removal
                add_coverage(coverage, rem_idx)

    # To make the rollback safe, I'll refactor local_search to use copies or careful state management.
    
    # Let's refine the local search to be safer.
    
    best_len_holder = [best_len]
    best_cw_holder = [best_cw_idx_list.copy()]
    
    # Precompute covers matrix for fast gain calculation
    # covers[cw_idx, word_idx] = 1 if word_idx in ball[cw_idx]
    covers = np.zeros((total_words, total_words), dtype=np.uint8)
    for cw_idx in range(total_words):
        covers[cw_idx, balls[cw_idx]] = 1

    def perform_local_search(cw_list, coverage):
        # Returns best list and coverage found
        local_best_list = cw_list.copy()
        local_best_cov = coverage.copy()
        local_best_len = len(local_best_list)
        
        for step in range(5000): # Max steps per restart
            if time.time() > deadline:
                break
                
            # 1. Greedy Removal
            removed_any = False
            indices = list(range(len(local_best_list)))
            random.shuffle(indices)
            for i in indices:
                if time.time() > deadline:
                    break
                cw_idx = local_best_list[i]
                if is_redundant(local_best_cov, cw_idx):
                    remove_coverage(local_best_cov, cw_idx)
                    local_best_list.pop(i)
                    local_best_len -= 1
                    removed_any = True
                    break
            if removed_any:
                continue
                
            # 2. Swap
            if len(local_best_list) < 5:
                break
                
            # Pick a word to potentially remove
            rem_pos = random.randint(0, len(local_best_list) - 1)
            rem_idx = local_best_list[rem_pos]
            
            # Save state for rollback
            saved_list = local_best_list.copy()
            saved_cov = local_best_cov.copy()
            
            # Remove
            remove_coverage(local_best_cov, rem_idx)
            local_best_list.pop(rem_pos)
            
            # Find uncovered
            uncovered_mask = (local_best_cov == 0)
            
            if not np.any(uncovered_mask):
                # We improved by just removing
                local_best_len -= 1
                continue # Loop back to check for more removals
                
            # Find best add
            # gain = covers[add_idx][uncovered_mask].sum()
            gains = covers[:, :].dot(uncovered_mask)
            # Exclude words already in list
            for idx in local_best_list:
                gains[idx] = -1
            # Also exclude rem_idx (it's not in list, but we just removed it, 
            # adding it back is useless as it was the one we removed, 
            # unless it covers other uncovered words? No, we removed it, so it covers its own ball.
            # If we add it back, we are back to start. So set gain to -1 or low.
            gains[rem_idx] = -1
            
            best_add_idx = np.argmax(gains)
            
            if gains[best_add_idx] <= 0:
                # No improvement possible with single swap
                # Rollback
                local_best_list = saved_list
                local_best_cov = saved_cov
                continue
                
            # Apply add
            add_coverage(local_best_cov, best_add_idx)
            local_best_list.append(best_add_idx)
            
            # Now try to remove redundancies
            sub_removed = True
            while sub_removed:
                if time.time() > deadline:
                    break
                sub_removed = False
                sub_indices = list(range(len(local_best_list)))
                random.shuffle(sub_indices)
                for i in sub_indices:
                    if time.time() > deadline:
                        break
                    cw_idx = local_best_list[i]
                    if is_redundant(local_best_cov, cw_idx):
                        remove_coverage(local_best_cov, cw_idx)
                        local_best_list.pop(i)
                        local_best_len -= 1
                        sub_removed = True
                        break
                        
            if local_best_len < best_len_holder[0]:
                best_len_holder[0] = local_best_len
                best_cw_holder[0] = local_best_list.copy()
                
            # If no net improvement (len didn't decrease), we might want to keep the new configuration 
            # if it has "potential" (e.g. different coverage distribution), 
            # but for simplicity, if len is not smaller, we rollback to the previous best 
            # or keep the current one if it's the same? 
            # Actually, if len is same, we might be stuck. 
            # Let's accept the swap if it doesn't increase size, to allow further moves.
            # But we must ensure validity. It is valid.
            
            # To avoid infinite loops of swapping with same size, 
            # we can just keep the current state.
            
    # Run multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Perturb the best solution found so far
        if restart == 0:
            cw_list = initial_cw_idx_list.copy()
            coverage = current_coverage.copy()
        else:
            # Take best found so far
            cw_list = best_cw_holder[0].copy()
            coverage = np.zeros(total_words, dtype=np.int32)
            for idx in cw_list:
                add_coverage(coverage, idx)
            
            # Randomly remove 2-5 words to break symmetry
            num_remove = random.randint(2, 5)
            if len(cw_list) > num_remove:
                to_remove = random.sample(cw_list, num_remove)
                for idx in to_remove:
                    if idx in cw_list:
                        cw_list.remove(idx)
                        remove_coverage(coverage, idx)
                        
                # Repair: add words to cover uncovered
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    if is_valid(coverage):
                        break
                    uncovered_mask = (coverage == 0)
                    if not np.any(uncovered_mask):
                        break
                    gains = covers[:, :].dot(uncovered_mask)
                    for idx in cw_list:
                        gains[idx] = -1
                    best_add = np.argmax(gains)
                    if gains[best_add] > 0:
                        cw_list.append(best_add)
                        add_coverage(coverage, best_add)
                    else:
                        break
                        
        # Run local search
        perform_local_search(cw_list, coverage)
        
        # Update global best if local search found something better
        # (Already updated inside perform_local_search via best_len_holder)

    # Final check
    if best_cw_holder[0] is None:
        best_cw_holder[0] = initial_cw_idx_list
        
    return [idx_to_word(idx) for idx in best_cw_holder[0]]

# Dummy call to ensure no syntax errors if run directly
if __name__ == "__main__":
    # res = construct(3, 6, 1)
    # print(len(res))
    pass
