import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1_optimized(deadline)
    
    # Fallbacks for other cases
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4 and radius == 1:
            return tetracode
        
    # Generic fallback: return all words (valid format, worst case)
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1_optimized(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute balls for all words
    # weights[i] is the multiplier for the i-th digit (from right, 0-indexed)
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
                
    best_size = total
    best_selected_mask = np.zeros(total, dtype=bool)
    rng = np.random.default_rng(42)
    
    # 1. Greedy Initialization
    is_selected = np.zeros(total, dtype=bool)
    uncovered = np.ones(total, dtype=bool)
    num_uncovered = total
    
    # Vectorized Greedy
    while num_uncovered > 0:
        if time.time() > deadline - 20.0:
            break
            
        # Calculate gain for each word: number of uncovered words in its ball
        # uncovered[balls] is (total, ball_size) boolean array
        # Sum over axis 1 gives (total,) array of gains
        gains = np.sum(uncovered[balls], axis=1)
        
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        candidates = np.where(gains == max_gain)[0]
        w = candidates[rng.integers(len(candidates))]
        
        is_selected[w] = True
        # Update uncovered
        new_uncovered = uncovered[balls[w]]
        newly_covered = np.sum(new_uncovered)
        num_uncovered -= newly_covered
        uncovered[balls[w]] = False
        
    current_size = np.sum(is_selected)
    
    # 2. Local Search Loop
    while time.time() < deadline - 5.0:
        # Prune Phase
        pruned = True
        while pruned:
            pruned = False
            if time.time() > deadline - 5.0:
                break
                
            # Compute cover counts
            # cover_count[x] is number of selected words c such that x in ball[c]
            # Equivalent to sum(is_selected[balls[x]])
            cover_count = np.sum(is_selected[balls], axis=1)
            
            # A word c is removable if min(cover_count[ball[c]]) > 1
            # i.e., all neighbors of c are covered by at least 2 selected words
            min_counts = np.min(cover_count[balls], axis=1)
            removable = min_counts > 1
            
            if np.any(removable):
                # Remove one removable word
                rem_candidates = np.where(removable)[0]
                # Prefer removing words that are "less useful" (higher min count? or just random)
                c = rem_candidates[rng.integers(len(rem_candidates))]
                is_selected[c] = False
                current_size -= 1
                pruned = True
        
        if current_size < best_size:
            best_size = current_size
            best_selected_mask = is_selected.copy()
            
        if best_size <= 486:
            break
            
        # Swap Phase (1-opt improvement)
        # Try to swap a word c with a word w to enable a removal
        improved = False
        num_selected = np.sum(is_selected)
        
        # Get list of selected words
        selected_indices = np.where(is_selected)[0]
        
        if len(selected_indices) == 0:
            break
            
        # Shuffle to avoid bias
        rng.shuffle(selected_indices)
        
        for c in selected_indices:
            if time.time() > deadline - 5.0:
                break
                
            # Check if c is removable
            cover_count = np.sum(is_selected[balls], axis=1)
            min_count_c = np.min(cover_count[balls[c]])
            
            if min_count_c > 1:
                # Should have been pruned, but just in case
                is_selected[c] = False
                current_size -= 1
                improved = True
                continue
                
            # c is not removable. Find words uniquely covered by c
            # These are x in ball[c] with cover_count[x] == 1
            unique_covered = np.where(cover_count[balls[c]] == 1)[0]
            
            if len(unique_covered) == 0:
                continue
                
            # Try to find a replacement w
            # w must cover at least one of the unique_covered words
            # and w must not be in selected
            
            # Heuristic: Pick one unique covered word x
            # and find a word w that covers x
            # w can be any word in ball[x]
            
            # To be efficient, we can just try a few random candidates
            # or pick the one that covers the most other words
            
            x = unique_covered[rng.integers(len(unique_covered))]
            
            # Candidates for w are words in ball[x]
            candidates_w = balls[x]
            
            # Filter out already selected words
            # We want w not in is_selected
            # is_selected[candidates_w] gives boolean mask
            
            # We want to pick w that is NOT selected
            # If all are selected, we can't swap (would just duplicate)
            # Actually, if w is selected, swapping c with w does nothing (c removed, w stays)
            # So we must pick w NOT in selected.
            
            non_selected_mask = ~is_selected[candidates_w]
            
            if not np.any(non_selected_mask):
                continue
                
            valid_candidates = candidates_w[non_selected_mask]
            
            # Pick a random candidate
            w = valid_candidates[rng.integers(len(valid_candidates))]
            
            # Perform Swap: Remove c, Add w
            is_selected[c] = False
            is_selected[w] = True
            
            # Try to Prune
            # We need to check if this swap allows a removal
            # We do a quick prune pass
            pruned_after_swap = False
            while True:
                if time.time() > deadline - 5.0:
                    break
                cover_count = np.sum(is_selected[balls], axis=1)
                min_counts = np.min(cover_count[balls], axis=1)
                removable = min_counts > 1
                
                if np.any(removable):
                    rem_candidates = np.where(removable)[0]
                    # Remove one
                    d = rem_candidates[rng.integers(len(rem_candidates))]
                    is_selected[d] = False
                    current_size -= 1
                    pruned_after_swap = True
                else:
                    break
                    
            if pruned_after_swap:
                # Swap was beneficial (led to a removal)
                improved = True
                if current_size < best_size:
                    best_size = current_size
                    best_selected_mask = is_selected.copy()
                    if best_size <= 486:
                        break
            else:
                # Swap was not beneficial, revert
                is_selected[w] = False
                is_selected[c] = True
                
            if improved:
                break # Restart the loop with new state
        
        if not improved:
            # No improvement found in this round, try a "shake"
            # Randomly swap 5 words
            if current_size > 10:
                for _ in range(5):
                    if time.time() > deadline - 5.0:
                        break
                    sel = np.where(is_selected)[0]
                    if len(sel) < 2:
                        break
                    c1 = sel[rng.integers(len(sel))]
                    # Pick a random non-selected word
                    non_sel = np.where(~is_selected)[0]
                    if len(non_sel) == 0:
                        break
                    c2 = non_sel[rng.integers(len(non_sel))]
                    is_selected[c1] = False
                    is_selected[c2] = True

    # Construct result
    result = []
    for w in np.where(best_selected_mask)[0]:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    
    return result
