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
        # Direct sum with tetracode (Radius 2 for n>4, but valid format)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # For radius 1, this is not a covering code for n>4.
        # But for n=4, it is a covering code?
        # Tetracode min dist 4. Radius 1 covering?
        # Ball size 1+4*2=9. Tetracode size 9. 9*9=81=3^4.
        # Yes, Tetracode is a perfect code for (4,1).
        if n == 4 and radius == 1:
            return tetracode
        
    # Generic fallback: return all words (invalid for score but valid format)
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1_optimized(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Weights for positions
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    
    # Precompute balls for all words
    # balls[w] is a list of words in the ball of radius 1 around w
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
    
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
                
    # Strategy: 
    # 1. Greedy construction with structured tie-breaking
    # 2. Prune
    # 3. Local search (swap-based)
    
    ball_size = 1 + 2 * n  # 17
    
    # Step 1: Greedy Construction
    cover_count = np.zeros(total, dtype=np.int32)
    selected = []
    current_uncovered = np.ones(total, dtype=bool)
    
    # Use a deterministic but varied order for tie-breaking
    # Create a priority based on word structure
    word_priority = np.zeros(total, dtype=np.float64)
    for w in range(total):
        temp = w
        digits = []
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        # Count distinct digits and use as priority
        distinct = len(set(digits))
        # Also consider the "spread" of digits
        spread = max(digits) - min(digits) if digits else 0
        word_priority[w] = distinct * 100 + spread * 10 + (total - w)
    
    rng = np.random.default_rng(42)
    
    while np.any(current_uncovered) and time.time() < deadline - 15.0:
        # Count uncovered neighbors for each word
        counts = np.sum(current_uncovered[balls], axis=1)
        
        # Only consider words that cover at least one uncovered word
        valid_mask = counts > 0
        if not np.any(valid_mask):
            break
        
        max_count = np.max(counts[valid_mask])
        candidates = np.where((counts == max_count) & valid_mask)[0]
        
        if len(candidates) == 1:
            best_idx = candidates[0]
        else:
            # Use priority to break ties
            prios = word_priority[candidates]
            best_idx = candidates[np.argmax(prios)]
        
        selected.append(best_idx)
        
        # Update counts
        cover_count[balls[best_idx]] += 1
        current_uncovered = (cover_count == 0)
        
    # If we timed out during greedy, finish it
    if np.any(current_uncovered):
        while np.any(current_uncovered) and time.time() < deadline - 5.0:
            counts = np.sum(current_uncovered[balls], axis=1)
            max_count = np.max(counts)
            if max_count == 0: break
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[rng.integers(len(candidates))]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            current_uncovered = (cover_count == 0)
    
    # Step 2: Pruning
    # Try to remove codewords that are redundant.
    changed = True
    while changed and time.time() < deadline - 10.0:
        changed = False
        new_selected = []
        for c in selected:
            # Check if removing c leaves all its neighbors covered by others
            # i.e., cover_count[balls[c]] > 1 for all w in balls[c]
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                changed = True
            else:
                new_selected.append(c)
        selected = new_selected
    
    # Step 3: Local Search (Swap-based)
    # Try to improve by finding redundant pairs
    selected_list = list(selected)
    selected_set = set(selected_list)
    
    # Multiple passes of local search
    for _pass in range(20):
        if time.time() > deadline - 5.0:
            break
        
        improved = False
        
        # First pass: try to remove each codeword if redundant
        i = 0
        while i < len(selected_list):
            if time.time() > deadline - 5.0:
                break
            
            c = selected_list[i]
            
            # Check if c is redundant
            if np.min(cover_count[balls[c]]) > 1:
                # Remove c
                cover_count[balls[c]] -= 1
                selected_list.pop(i)
                selected_set.discard(c)
                improved = True
                # Don't increment i since we removed an element
            else:
                i += 1
        
        if not improved:
            # Second pass: try swap - remove c, add d that covers the gap
            for i, c in enumerate(selected_list):
                if time.time() > deadline - 5.0:
                    break
                
                # Temporarily remove c
                cover_count[balls[c]] -= 1
                uncovered_after_removal = np.where(cover_count == 0)[0]
                
                if len(uncovered_after_removal) == 0:
                    # Can remove without replacement
                    selected_list.pop(i)
                    selected_set.discard(c)
                    improved = True
                    break
                elif len(uncovered_after_removal) <= 3:
                    # Small gap: try to find a replacement
                    # Find words that cover all uncovered words
                    # For each uncovered word w, the candidates are balls[w]
                    # We need a word in the intersection of balls[w] for all uncovered w
                    
                    # Get candidate set
                    candidate_set = set(balls[uncovered_after_removal[0]].tolist())
                    for w in uncovered_after_removal[1:]:
                        candidate_set &= set(balls[w].tolist())
                    
                    # Check if any candidate is not already selected
                    for cw in candidate_set:
                        if cw not in selected_set:
                            # Add cw
                            selected_set.discard(c)
                            selected_set.add(cw)
                            cover_count[balls[cw]] += 1
                            selected_list[i] = cw
                            improved = True
                            break
                    
                    if improved:
                        break
                    else:
                        # Restore c
                        cover_count[balls[c]] += 1
                else:
                    # Too many uncovered, restore c
                    cover_count[balls[c]] += 1
        
        if not improved:
            break
    
    # Convert selected to list of digit lists
    result = []
    for w in selected_list:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
