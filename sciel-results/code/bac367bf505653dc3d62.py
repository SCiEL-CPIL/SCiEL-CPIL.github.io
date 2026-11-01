import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases
    import itertools
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute balls efficiently
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    # Build balls array: balls[w] = [w, neighbors...]
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
                balls[w, idx] = w + (new_digit - digit) * weight
                idx += 1
    
    # Build inverse mapping: for each word w, which ball centers cover it
    # inv[w] = list of center indices c such that w is in balls[c]
    # This allows O(ball_size) update when adding/removing a center
    inv = [[] for _ in range(total)]
    for c in range(total):
        for w in balls[c]:
            inv[w].append(c)
    
    # Precompute as numpy arrays for fast indexing
    inv_offsets = np.zeros(total + 1, dtype=np.int32)
    for w in range(total):
        inv_offsets[w + 1] = inv_offsets[w] + len(inv[w])
    inv_flat = np.zeros(inv_offsets[-1], dtype=np.int32)
    for w in range(total):
        start = inv_offsets[w]
        for i, c in enumerate(inv[w]):
            inv_flat[start + i] = c
    
    best_size = total
    best_set = None
    
    # Use multiple random restarts with greedy + pruning + local search
    np.random.seed(42)
    random.seed(42)
    
    def greedy_construct():
        """Fast greedy using inverse mapping for O(ball_size) per step."""
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered = np.ones(total, dtype=np.int32)  # 1 if uncovered
        
        while uncovered.sum() > 0:
            # For each potential codeword, count how many uncovered words it covers
            # Using vectorized approach: for each center c, sum uncovered over balls[c]
            # But we can be smarter: maintain a count array
            # counts[c] = number of uncovered words in balls[c]
            # When we add center c, for each w in balls[c]:
            #   if uncovered[w]: counts[d] -= 1 for all d in inv[w]
            #   uncovered[w] = 0
            
            # Efficient approach: compute counts via matrix-like operation
            # counts[c] = sum over w in balls[c] of uncovered[w]
            # = sum over w of uncovered[w] * (1 if w in balls[c] else 0)
            # Using inv: for each w with uncovered[w]=1, increment counts[d] for d in inv[w]
            
            # Reset counts and recompute from scratch (expensive but simple)
            # Better: maintain incrementally
            # Let's use the direct approach with numpy for clarity, 
            # but optimize by only computing for candidates
            
            # Fast path: use numpy to compute counts for all centers
            # counts[c] = uncovered[balls[c]].sum()
            counts = np.zeros(total, dtype=np.int32)
            # Vectorized: for each w where uncovered[w]=1, add 1 to all c in inv[w]
            unc_idx = np.where(uncovered > 0)[0]
            if len(unc_idx) == 0:
                break
            for w in unc_idx:
                start = inv_offsets[w]
                end = inv_offsets[w + 1]
                counts[inv_flat[start:end]] += 1
            
            max_count = np.max(counts)
            if max_count == 0:
                break
            
            # Pick randomly among max-count candidates for diversity
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            # Update cover_count and uncovered
            for w in balls[best_idx]:
                cover_count[w] += 1
                if cover_count[w] == 1:
                    uncovered[w] = 0
            
            if time.time() > deadline - 10.0:
                break
        
        if uncovered.sum() > 0:
            return None
        return selected
    
    def prune(selected, cover_count):
        """Iteratively remove codewords that are not needed."""
        changed = True
        while changed and time.time() < deadline - 3.0:
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
    
    def local_search_swap(selected, cover_count):
        """Try to replace one codeword with another to reduce size or improve coverage."""
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            for i in range(len(selected)):
                c = selected[i]
                # Check if removing c would leave any word uncovered
                min_cover = np.min(cover_count[balls[c]])
                if min_cover <= 1:
                    continue
                # c is removable; try to find a better replacement or just remove
                # For now, just remove (handled by prune)
                # Try swap: remove c, add some other center that covers more
                # This is complex; skip for now and rely on prune
                pass
        return selected, cover_count
    
    while time.time() < deadline - 5.0:
        # Randomized greedy construction
        selected = greedy_construct()
        
        if selected is None:
            continue
        
        # Build cover_count for pruning
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            cover_count[balls[c]] += 1
        
        # Pruning: iteratively remove codewords that are not needed
        selected, cover_count = prune(selected, cover_count)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
        
        # Early stop if we've found a very good solution and time is running low
        if time.time() > deadline - 8.0:
            break
    
    # Convert to list of digit lists
    if best_set is None:
        # Fallback: return a large valid set
        best_set = list(range(total))
    
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
