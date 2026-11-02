import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1_local_search()
    return _greedy_cover(q, n, radius)

def _construct_3_7_1_local_search():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # 1. Generate initial code using Greedy Set Cover
    all_words = list(itertools.product(range(3), repeat=7))
    N = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for all words
    # neighbors[i] is a list of indices of words at Hamming distance <= 1 from all_words[i]
    neighbors = []
    for w in all_words:
        nb = [word_to_idx[w]]
        for i in range(7):
            for d in [1, 2]:
                new_w = list(w)
                new_w[i] = (w[i] + d) % 3
                nb.append(word_to_idx[tuple(new_w)])
        neighbors.append(nb)
    
    # Greedy Set Cover to get initial valid code
    covers = [set(nb) for nb in neighbors]
    uncovered = set(range(N))
    code = []
    code_set = set()
    
    while uncovered:
        best = -1
        best_count = -1
        # Optimization: Only consider candidates that cover at least one uncovered word?
        # Actually, just iterate all. 2187 is small.
        for i in range(N):
            # If i is already in code, skip? No, we can pick any word.
            # But if i is in code, it's already added. We don't need to add it again.
            # Wait, the greedy loop adds words. We should not consider words already in code.
            if i in code_set:
                continue
            count = 0
            # Count how many uncovered words this candidate covers
            # Since N is small, we can just intersect sets
            # Or iterate covers[i] and check if in uncovered
            # covers[i] is a set. uncovered is a set.
            count = len(covers[i] & uncovered)
            if count > best_count:
                best_count = count
                best = i
        
        if best == -1:
            break
            
        code.append(best)
        code_set.add(best)
        uncovered -= covers[best]
        
    # 2. Local Search
    # State: code (list of indices), code_set (set of indices), cover_count (array of counts)
    cover_count = np.zeros(N, dtype=np.int32)
    for c in code:
        for nb in neighbors[c]:
            cover_count[nb] += 1
            
    # Helper to get uncovered words (count == 0)
    # We can maintain a set of uncovered words for O(1) check?
    # Or just check min(cover_count) >= 1.
    # Checking min is O(N). N=2187. Fast enough.
    
    def is_valid():
        return np.min(cover_count) >= 1
        
    # Optimization: Maintain a set of indices where count == 0?
    # If we do, we need to update it on every change.
    # Let's just use np.min for simplicity first, or a manual check if it's slow.
    # Actually, maintaining a `zero_count_set` is better.
    zero_count_set = set(np.where(cover_count == 0)[0])
    
    def update_counts(idx, delta):
        nonlocal zero_count_set
        for nb in neighbors[idx]:
            old_val = cover_count[nb]
            new_val = old_val + delta
            cover_count[nb] = new_val
            if old_val == 0 and new_val > 0:
                zero_count_set.discard(nb)
            elif old_val > 0 and new_val == 0:
                zero_count_set.add(nb)
                
    # Local Search Loop
    # We try to remove codewords. If we can't, we swap.
    # To avoid getting stuck, we track number of failed removals.
    
    current_size = len(code)
    best_size = current_size
    best_code = list(code)
    
    failed_removals = 0
    swaps_since_improvement = 0
    
    while time.time() - start_time < time_limit:
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
            swaps_since_improvement = 0
            failed_removals = 0
            
        # Pick a random codeword to try to remove
        if not code:
            break
            
        i = random.randint(0, len(code) - 1)
        c = code[i]
        
        # Tentatively remove c
        update_counts(c, -1)
        
        if len(zero_count_set) == 0:
            # Valid removal!
            code.pop(i)
            code_set.remove(c)
            failed_removals = 0
        else:
            # Invalid removal. Try to find a swap.
            # We need a c' such that zero_count_set is a subset of neighbors[c']
            # And c' not in code_set.
            
            # Find candidates
            # If zero_count_set is small, we can intersect neighbors of words in zero_count_set
            # Candidates must be in intersection of N(w) for all w in zero_count_set
            # Since zero_count_set is usually 1 or 2 elements, this is fast.
            
            candidates = None
            for w in zero_count_set:
                if candidates is None:
                    candidates = set(neighbors[w])
                else:
                    candidates &= set(neighbors[w])
            
            if candidates:
                # Filter out existing codewords
                valid_candidates = [c_new for c_new in candidates if c_new not in code_set]
                
                if valid_candidates:
                    # Pick a random valid candidate
                    c_new = random.choice(valid_candidates)
                    
                    # Check if adding c_new is valid?
                    # We know zero_count_set will be covered.
                    # Do we uncover anything else?
                    # No, adding c_new only increases counts.
                    # So it is always valid to swap if we find such c_new.
                    
                    # Perform Swap: Remove c, Add c_new
                    # We already did update_counts(c, -1).
                    # Now add c_new.
                    update_counts(c_new, 1)
                    
                    # Update code and code_set
                    code[i] = c_new
                    code_set.remove(c)
                    code_set.add(c_new)
                    
                    swaps_since_improvement += 1
                    failed_removals = 0 # Reset? Or keep? 
                    # Let's reset failed_removals to allow more swaps if we keep failing removals
                else:
                    # No valid swap candidate. Revert removal.
                    update_counts(c, 1)
                    failed_removals += 1
                    swaps_since_improvement += 1
            else:
                # No intersection, so no candidate can cover all uncovered words.
                # Revert removal.
                update_counts(c, 1)
                failed_removals += 1
                swaps_since_improvement += 1

        # If we have too many swaps without improvement, we might be stuck.
        # We can try a "kick": randomly add a word, then try to remove two?
        # Or just continue. The random swaps should eventually lead to a removable state.
        
        # To prevent infinite loops in a local minimum, if swaps_since_improvement > 1000:
        # We could restart with a slightly perturbed best_code?
        # For now, let's just keep going. The random nature of swaps should escape.
        
    # Return the best code found
    # Convert indices back to lists of ints
    return [list(all_words[idx]) for idx in best_code]

def _greedy_cover(q, n, radius):
    # Fallback for other parameters (not used in this specific task instance)
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = len(all_words)
    
    covers = []
    for w in all_words:
        covered = set()
        covered.add(word_to_idx[w])
        for r in range(1, radius + 1):
            for positions in itertools.combinations(range(n), r):
                for deltas in itertools.product(range(1, q), repeat=r):
                    new_w = list(w)
                    for pos, d in zip(positions, deltas):
                        new_w[pos] = (w[pos] + d) % q
                    covered.add(word_to_idx[tuple(new_w)])
        covers.append(covered)
    
    uncovered = set(range(N))
    code = []
    while uncovered:
        best = None
        best_count = -1
        for i in range(N):
            count = len(covers[i] & uncovered)
            if count > best_count:
                best_count = count
                best = i
        code.append(list(all_words[best]))
        uncovered -= covers[best]
    
    return code
