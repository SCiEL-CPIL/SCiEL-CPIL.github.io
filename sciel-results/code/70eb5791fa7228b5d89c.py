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
        
    # Generic fallback: return all words (invalid for score but valid format)
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1_optimized(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Precompute balls and neighbor lists
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
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
    
    ball_size = 1 + 2 * n  # 17
    
    # Step 1: Greedy Construction with structured tie-breaking
    cover_count = np.zeros(total, dtype=np.int32)
    selected = []
    current_uncovered = np.ones(total, dtype=bool)
    
    # Priority: prefer words with more distinct digits (higher entropy)
    word_priority = np.zeros(total, dtype=np.float64)
    for w in range(total):
        temp = w
        digits = []
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        distinct = len(set(digits))
        spread = max(digits) - min(digits) if digits else 0
        word_priority[w] = distinct * 1000 + spread * 100 + (total - w)
    
    rng = np.random.default_rng(12345)
    
    while np.any(current_uncovered) and time.time() < deadline - 20.0:
        counts = np.sum(current_uncovered[balls], axis=1)
        valid_mask = counts > 0
        if not np.any(valid_mask):
            break
        
        max_count = np.max(counts[valid_mask])
        candidates = np.where((counts == max_count) & valid_mask)[0]
        
        if len(candidates) == 1:
            best_idx = candidates[0]
        else:
            # Randomized tie-breaking with bias toward high priority
            prios = word_priority[candidates]
            # Weighted random selection
            weights_sel = prios / (prios.sum() + 1e-9)
            best_idx = candidates[rng.choice(len(candidates), p=weights_sel)]
        
        selected.append(best_idx)
        cover_count[balls[best_idx]] += 1
        current_uncovered = (cover_count == 0)
    
    # Step 2: Aggressive Pruning
    selected_list = list(selected)
    changed = True
    while changed and time.time() < deadline - 15.0:
        changed = False
        i = 0
        while i < len(selected_list):
            c = selected_list[i]
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                selected_list.pop(i)
                changed = True
            else:
                i += 1
    
    # Step 3: Local Search - 1-swap improvements
    selected_set = set(selected_list)
    for _pass in range(50):
        if time.time() > deadline - 5.0:
            break
        
        improved = False
        
        # Try to remove each codeword
        i = 0
        while i < len(selected_list):
            if time.time() > deadline - 5.0:
                break
            c = selected_list[i]
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                selected_list.pop(i)
                selected_set.discard(c)
                improved = True
            else:
                i += 1
        
        if improved:
            continue
        
        # Try 1-swap: remove c, add d
        for i, c in enumerate(selected_list):
            if time.time() > deadline - 5.0:
                break
            
            cover_count[balls[c]] -= 1
            uncovered = np.where(cover_count == 0)[0]
            
            if len(uncovered) == 0:
                # Pure removal
                selected_list.pop(i)
                selected_set.discard(c)
                improved = True
                break
            elif len(uncovered) <= 4:
                # Find replacement that covers all uncovered
                candidate_set = set(balls[uncovered[0]].tolist())
                for w in uncovered[1:]:
                    candidate_set &= set(balls[w].tolist())
                
                # Try candidates in priority order
                for cw in sorted(candidate_set, key=lambda x: -word_priority[x]):
                    if cw not in selected_set:
                        selected_set.discard(c)
                        selected_set.add(cw)
                        cover_count[balls[cw]] += 1
                        selected_list[i] = cw
                        improved = True
                        break
                
                if improved:
                    break
                else:
                    cover_count[balls[c]] += 1
            else:
                cover_count[balls[c]] += 1
    
    # Convert to list of digit lists
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
