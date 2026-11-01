import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin from 150s limit

    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Generic fallback
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Precompute balls
    # weights for positions
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
    
    # Vectorized precomputation
    # balls[:, 0] is the word itself
    # balls[:, 1 + 2*pos : 2 + 2*pos+1] are the neighbors?
    # Actually, let's just fill it.
    # For each word w, ball is [w] + [w + (d - w_d) * weight for pos in range(n) for d in range(3) if d != w_d]
    
    # This is a bit tricky to vectorize fully without loops, but 6561 is small.
    # Let's use a loop for precomputation, it's fast enough once.
    
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
    best_set = None
    
    # Helper to convert index to list of digits
    def to_digits(idx):
        digits = []
        temp = idx
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        return digits

    # Strategy: Multiple restarts of greedy + local search (pruning + swaps)
    # The "hypergraph edge coloring" insight suggests viewing this as selecting
    # edges (balls) to cover vertices (words). The greedy approach is standard,
    # but local search (swaps) can escape local minima.
    
    while time.time() < deadline - 5.0:
        # 1. Randomized Greedy Construction
        cover_count = np.zeros(total, dtype=np.int32)
        selected = []
        uncovered_mask = np.ones(total, dtype=bool)
        
        while np.any(uncovered_mask):
            # Count how many uncovered words each candidate covers
            # uncovered_mask[balls] shape (total, 17)
            # sum along axis 1
            counts = uncovered_mask[balls].sum(axis=1)
            
            # Find max count
            max_count = np.max(counts)
            
            # If max_count is 0, we are stuck (should not happen if space is connected)
            if max_count == 0:
                break
            
            # Find all candidates with max count
            candidates = np.where(counts == max_count)[0]
            
            # Pick one randomly
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            
            # Update cover counts
            cover_count[balls[best_idx]] += 1
            
            # Update uncovered mask
            # A word is uncovered if cover_count is 0
            # We can just recompute uncovered_mask = (cover_count == 0)
            # Or update it: for w in balls[best_idx], if cover_count[w] was 0, it is now 1, so uncovered.
            # But cover_count[w] could be >1 if it was already covered.
            # So: new_uncovered = (cover_count == 0) is safest.
            uncovered_mask = (cover_count == 0)
            
            # Check time
            if time.time() > deadline - 10.0:
                break
        
        if np.any(uncovered_mask):
            # If we broke early due to time, this set might be invalid.
            # We should only consider it if it's valid.
            continue
        
        # 2. Pruning Phase
        # Remove codewords that are redundant.
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for c in selected:
                # Check if c is removable
                # min(cover_count[balls[c]]) > 1
                if np.min(cover_count[balls[c]]) > 1:
                    # Remove c
                    cover_count[balls[c]] -= 1
                    changed = True
                    # Do not add to new_selected
                else:
                    new_selected.append(c)
            selected = new_selected
            
        # 3. Local Search Phase (Swaps)
        # Try to replace a codeword c with d not in set, such that coverage is maintained.
        # This might enable further pruning.
        # We limit the number of swaps to keep it fast.
        
        improved = True
        max_swap_iterations = 50 # Limit to avoid getting stuck in long loops
        swap_iter = 0
        
        while improved and swap_iter < max_swap_iterations and time.time() < deadline - 5.0:
            improved = False
            swap_iter += 1
            # Shuffle to avoid bias
            c_list = list(selected)
            random.shuffle(c_list)
            
            for c in c_list:
                if time.time() > deadline - 5.0:
                    break
                if c not in selected:
                    continue
                
                # Find words covered ONLY by c
                # We need cover_count.
                # cover_count is maintained.
                
                # Check if c is removable (should have been pruned, but maybe state changed?)
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    selected.remove(c)
                    improved = True
                    break # Restart loop with new set
                
                # Find unique words
                # Words w in balls[c] where cover_count[w] == 1
                # We can vectorize this check partially or just loop over the 17 neighbors.
                # Since balls[c] is small (17), a loop is fine.
                unique_words = []
                for w in balls[c]:
                    if cover_count[w] == 1:
                        unique_words.append(w)
                
                if not unique_words:
                    # Should have been pruned
                    cover_count[balls[c]] -= 1
                    selected.remove(c)
                    improved = True
                    break
                
                # Try to find d
                # d must cover all unique_words
                # And d should not be in selected
                # Also, ideally d should be "good"
                
                # Optimization: Pre-compute for each word w, the list of codewords covering it?
                # We have balls. balls[w] is the ball of w.
                # The codewords covering w are those c such that w in balls[c].
                # This is the reverse graph.
                # Let's just check candidates from neighbors of the first unique word.
                
                candidates = set(balls[unique_words[0]])
                for w in unique_words[1:]:
                    candidates.intersection_update(balls[w])
                
                # Remove existing codewords from candidates
                selected_set = set(selected)
                candidates -= selected_set
                
                if candidates:
                    # Pick a random candidate
                    d = random.choice(list(candidates))
                    # Swap
                    selected.remove(c)
                    selected.append(d)
                    
                    # Update cover_count
                    cover_count[balls[c]] -= 1
                    cover_count[balls[d]] += 1
                    
                    improved = True
                    # Run prune after swap
                    # We can do a quick prune pass here
                    # For simplicity, we just mark improved and let the outer loop handle it?
                    # No, the outer loop restarts the shuffle.
                    # Let's do a quick prune to catch immediate redundancies.
                    # Actually, the main pruning loop is outside.
                    # We just break to restart the swap loop with the new set.
                    break
            
        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            # If we found 486, we can stop early?
            # The lower bound is 403, best known 486.
            # It's unlikely to go below 486 quickly, but possible.
            # Let's keep searching.
            
        if best_size <= 486:
            # We found a very good solution.
            # We can try a bit more to see if we can go lower, but 486 is the target.
            # Let's run a few more iterations just in case.
            pass

    # Convert best_set to list of digit lists
    if best_set is None:
        # Fallback: return all words (invalid for score but valid format)
        # Use the to_digits helper.
        return [to_digits(w) for w in range(total)]
        
    result = []
    for w in best_set:
        result.append(to_digits(w))
        
    return result
