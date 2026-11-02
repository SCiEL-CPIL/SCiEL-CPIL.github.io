import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1_advanced(deadline)
    
    # Fallbacks for other cases
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4 and radius == 1:
            return tetracode
        
    # Generic fallback: return all words (valid format, worst case)
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1_advanced(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Weights for positions (big-endian)
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    
    # Precompute balls for all words
    # balls[w, 0] = w, balls[w, 1:17] = neighbors
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
                
    # Helper to convert index to codeword list
    def idx_to_word(w):
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        return digits

    best_code = None
    best_size = total + 1
    
    # Run multiple randomized greedy + local search attempts
    num_attempts = 5 # Adjust based on time
    
    for attempt in range(num_attempts):
        if time.time() > deadline - 20.0:
            break
            
        # 1. Randomized Greedy Construction
        # We want to pick words that cover the most *uncovered* words.
        # To add randomness, we can shuffle the tie-breaking or use a random subset of top candidates.
        
        cover_count = np.zeros(total, dtype=np.int32)
        current_uncovered = np.ones(total, dtype=bool)
        selected = []
        
        # Pre-compute list of all indices for shuffling? No, too big.
        # Instead, we can shuffle the order in which we evaluate candidates in ties.
        # Or, simpler: just run standard greedy but with a random start or random tie-break.
        
        # For speed, we'll use a standard greedy but with a random seed for tie-breaking.
        rng = random.Random(attempt * 12345 + 678)
        
        while np.any(current_uncovered):
            if time.time() > deadline - 20.0:
                break
                
            # Count uncovered neighbors for each word
            counts = np.sum(current_uncovered[balls], axis=1)
            max_count = np.max(counts)
            
            if max_count == 0:
                break
            
            # Get candidates with max count
            candidates = np.where(counts == max_count)[0]
            
            # Randomly pick one from candidates
            best_idx = candidates[rng.randint(0, len(candidates) - 1)]
            
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            current_uncovered = (cover_count == 0)
            
        # 2. Local Search (Pruning and Swapping)
        if time.time() > deadline - 20.0:
            continue
            
        # Pruning Pass
        changed = True
        while changed and time.time() < deadline - 10.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
            
        # Swap Pass (Try to replace high-overlap codewords with low-overlap ones to enable further pruning)
        # This is a heuristic to escape local optima.
        if time.time() < deadline - 10.0:
            selected_set = set(selected)
            improved = True
            max_swaps = 50
            swap_iter = 0
            while improved and time.time() < deadline - 10.0 and swap_iter < max_swaps:
                improved = False
                swap_iter += 1
                # Try to find a codeword c that is "expensive" (covers many words that are already covered)
                # and replace it with a word w that is "cheap" (covers words that are under-covered)
                # Actually, simpler: just try random swaps and see if pruning improves.
                
                if len(selected) > 0:
                    c = selected[rng.randint(0, len(selected) - 1)]
                    # Pick a random neighbor w not in selected
                    neighbors = balls[c]
                    valid_neighbors = [w for w in neighbors if w not in selected_set]
                    if valid_neighbors:
                        w = valid_neighbors[rng.randint(0, len(valid_neighbors) - 1)]
                        
                        # Tentative swap
                        cover_count[balls[c]] -= 1
                        if np.any(cover_count == 0):
                            cover_count[balls[c]] += 1
                            continue
                            
                        cover_count[balls[w]] += 1
                        selected_set.remove(c)
                        selected_set.add(w)
                        selected.remove(c)
                        selected.append(w)
                        
                        # Try to prune
                        local_changed = True
                        while local_changed and time.time() < deadline - 10.0:
                            local_changed = False
                            new_selected = []
                            for cc in selected:
                                if np.min(cover_count[balls[cc]]) > 1:
                                    cover_count[balls[cc]] -= 1
                                    local_changed = True
                                else:
                                    new_selected.append(cc)
                            selected = new_selected
                            selected_set = set(selected)
                            
                        # If we didn't improve size, revert?
                        # Actually, if size didn't decrease, we might be stuck.
                        # For now, keep the swap if it didn't break coverage.
                        # To avoid cycling, we only accept if size decreases or we allow some cycling.
                        # Let's only accept if size decreases.
                        # We need to track size before swap.
                        # This logic is getting complex. Let's stick to aggressive pruning only.
                        pass

        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_code = selected[:]
            
        # Check if we hit the theoretical lower bound or known best
        if best_size <= 486:
            break

    if best_code is None:
        # Fallback to deterministic greedy if all attempts failed or timed out
        cover_count = np.zeros(total, dtype=np.int32)
        current_uncovered = np.ones(total, dtype=bool)
        selected = []
        while np.any(current_uncovered):
            counts = np.sum(current_uncovered[balls], axis=1)
            max_count = np.max(counts)
            if max_count == 0: break
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[0]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            current_uncovered = (cover_count == 0)
            
        # Final prune
        changed = True
        while changed:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        best_code = selected

    # Convert to list of lists
    result = []
    for w in best_code:
        result.append(idx_to_word(w))
        
    return result
