import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    total_words = q ** n  # 2187
    shape = (q,) * n
    
    # Generate all words as numpy array (2187, 7)
    words_np = np.array(np.unravel_index(np.arange(total_words), shape)).T
    
    # Precompute ball masks
    ball_masks = np.zeros((total_words, total_words), dtype=bool)
    for i in range(total_words):
        w = words_np[i]
        ball_masks[i, i] = True
        for pos in range(n):
            for new_val in range(q):
                if new_val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = new_val
                idx = int(np.ravel_multi_index(new_w, shape))
                ball_masks[i, idx] = True
    
    # Greedy Set Cover
    def greedy_set_cover(perturb=0.0, seed=None, time_budget=20.0):
        uncovered = np.ones(total_words, dtype=bool)
        selected = []
        rng = np.random.default_rng(seed) if seed is not None else None
        deadline = start_time + time_budget
        
        while uncovered.any():
            if time.time() > deadline:
                break
            gains = (ball_masks & uncovered[np.newaxis, :]).sum(axis=1).astype(float)
            if perturb > 0 and rng is not None:
                gains += rng.random(total_words) * perturb
            best_idx = int(np.argmax(gains))
            if gains[best_idx] < 0.01:
                break
            selected.append(best_idx)
            uncovered &= ~ball_masks[best_idx]
        return selected
    
    # Optimized Local Search using coverage counts
    def local_search(selected_indices, time_budget=40.0):
        selected_set = set(selected_indices)
        # Initialize coverage counts
        cover_counts = np.zeros(total_words, dtype=int)
        for idx in selected_set:
            cover_counts[ball_masks[idx]] += 1
        
        deadline = start_time + time_budget
        improved = True
        
        # Phase 1: Remove redundant codewords
        while improved and (time.time() < deadline):
            improved = False
            for idx in list(selected_set):
                if time.time() > deadline:
                    break
                # Tentatively remove
                cover_counts[ball_masks[idx]] -= 1
                if not np.any(cover_counts == 0):
                    # Safe to remove
                    selected_set.remove(idx)
                    improved = True
                else:
                    # Restore
                    cover_counts[ball_masks[idx]] += 1
                    break # Restart loop with new set
        
        # Phase 2: 1-for-1 swap (try to replace a codeword with a better one to enable removals)
        # This is heuristic: try swapping each codeword with the one that maximizes "criticality" reduction
        while improved and (time.time() < deadline - 10):
            improved = False
            sel_list = list(selected_set)
            for idx1 in sel_list:
                if time.time() > deadline - 10:
                    break
                # Remove idx1
                cover_counts[ball_masks[idx1]] -= 1
                uncovered_mask = (cover_counts == 0)
                
                if not np.any(uncovered_mask):
                    # idx1 was redundant, remove it (should have been caught in Phase 1, but just in case)
                    selected_set.remove(idx1)
                    improved = True
                else:
                    # Find best replacement
                    # Gain of adding k is number of uncovered words it covers
                    gains = (ball_masks & uncovered_mask[np.newaxis, :]).sum(axis=1)
                    # Exclude current set
                    for k in selected_set:
                        gains[k] = -1
                    best_k = int(np.argmax(gains))
                    
                    if gains[best_k] == np.sum(uncovered_mask) and best_k not in selected_set:
                        # Swap
                        selected_set.remove(idx1)
                        selected_set.add(best_k)
                        # Update counts
                        cover_counts[ball_masks[idx1]] += 1 # Restore idx1 temporarily? No, we removed it.
                        # Wait, we subtracted idx1. Now we add best_k.
                        cover_counts[ball_masks[best_k]] += 1
                        improved = True
                    else:
                        # Restore idx1
                        cover_counts[ball_masks[idx1]] += 1
                        break # If no good swap found for this, try next? Or break to avoid infinite loop?
                        # Better to break inner loop and restart outer?
                        # Let's just break the for loop to restart the while loop with fresh state if no improvement
                        break
        
        return selected_set

    best_result = None
    best_size = float('inf')
    
    # Run multiple restarts
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 20:
            break
        
        if restart == 0:
            selected = greedy_set_cover(perturb=0.0, seed=None, time_budget=15.0)
        elif restart == 1:
            selected = greedy_set_cover(perturb=0.3, seed=42, time_budget=15.0)
        elif restart == 2:
            selected = greedy_set_cover(perturb=0.7, seed=123, time_budget=15.0)
        elif restart == 3:
            selected = greedy_set_cover(perturb=1.2, seed=456, time_budget=15.0)
        else:
            selected = greedy_set_cover(perturb=0.5, seed=restart * 7 + 1, time_budget=15.0)
        
        result_set = local_search(selected, time_budget=30.0)
        
        # Validate result
        if result_set:
            # Check coverage
            cov = np.zeros(total_words, dtype=bool)
            for idx in result_set:
                cov |= ball_masks[idx]
            if cov.all():
                if len(result_set) < best_size:
                    best_size = len(result_set)
                    best_result = result_set
                    if best_size <= 186:
                        break
    
    if best_result is None:
        # Fallback: Return a valid, though potentially large, code
        # Simple grid: cover all words
        best_result = set(range(total_words))
    
    # Convert to list of lists of ints
    return [[int(x) for x in words_np[i]] for i in best_result]
