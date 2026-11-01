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
    
    # Precompute balls
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
    
    def greedy_construct(randomize=False):
        cover_count = np.zeros(total, dtype=np.int32)
        selected = []
        current_uncovered = np.ones(total, dtype=bool)
        
        while np.any(current_uncovered):
            if time.time() > deadline - 50.0:
                break
                
            counts = np.sum(current_uncovered[balls], axis=1)
            max_count = np.max(counts)
            
            if max_count == 0:
                break
            
            candidates = np.where(counts == max_count)[0]
            if randomize and len(candidates) > 1:
                best_idx = candidates[np.random.randint(len(candidates))]
            else:
                best_idx = candidates[0]
            
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            current_uncovered = (cover_count == 0)
        
        # Force completion
        if np.any(current_uncovered) and time.time() < deadline - 15.0:
            while np.any(current_uncovered):
                uncovered_indices = np.where(current_uncovered)[0]
                if len(uncovered_indices) == 0:
                    break
                best_idx = uncovered_indices[0]
                selected.append(best_idx)
                cover_count[balls[best_idx]] += 1
                current_uncovered = (cover_count == 0)
                if time.time() > deadline - 15.0:
                    break
        
        return selected, cover_count
    
    def prune(selected, cover_count):
        changed = True
        while changed and time.time() < deadline - 25.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        return selected, cover_count
    
    def local_search(selected, cover_count):
        selected_set = set(selected)
        
        for iteration in range(30):
            if time.time() > deadline - 8.0:
                break
            
            improved = False
            
            # Try moves and swaps
            for i in range(len(selected)):
                if time.time() > deadline - 8.0:
                    break
                c = selected[i]
                
                # Try moving to neighbors
                for w in balls[c]:
                    if w == c or w in selected_set:
                        continue
                    
                    cover_count[balls[c]] -= 1
                    if np.any(cover_count == 0):
                        cover_count[balls[c]] += 1
                        continue
                    
                    cover_count[balls[w]] += 1
                    
                    removable_d = None
                    for j in range(len(selected)):
                        if j == i:
                            continue
                        d = selected[j]
                        if np.min(cover_count[balls[d]]) > 1:
                            removable_d = d
                            break
                    
                    if removable_d is not None:
                        cover_count[balls[removable_d]] -= 1
                        selected[i] = w
                        selected_set.remove(c)
                        selected_set.add(w)
                        selected.remove(removable_d)
                        selected_set.remove(removable_d)
                        improved = True
                        break
                    else:
                        cover_count[balls[w]] -= 1
                        cover_count[balls[c]] += 1
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return selected, cover_count
    
    # Try multiple greedy constructions with different randomization
    best_selected = None
    best_count = float('inf')
    
    num_attempts = 3
    for attempt in range(num_attempts):
        if time.time() > deadline - 60.0:
            break
        
        selected, cover_count = greedy_construct(randomize=(attempt > 0))
        selected, cover_count = prune(selected, cover_count)
        
        if len(selected) < best_count:
            best_count = len(selected)
            best_selected = selected.copy()
            best_cover = cover_count.copy()
    
    # Use the best solution for local search
    if best_selected is not None:
        selected, cover_count = local_search(best_selected, best_cover)
        
        # Final prune
        selected, cover_count = prune(selected, cover_count)
    else:
        # Fallback
        selected, cover_count = greedy_construct(randomize=False)
        selected, cover_count = prune(selected, cover_count)
    
    # Final verification
    final_cover = np.zeros(total, dtype=np.int32)
    for c in selected:
        final_cover[balls[c]] += 1
    
    if np.any(final_cover == 0):
        missing = np.where(final_cover == 0)[0]
        for m in missing:
            selected.append(m)
            final_cover[balls[m]] += 1
    
    result = []
    for w in selected:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    
    return result
