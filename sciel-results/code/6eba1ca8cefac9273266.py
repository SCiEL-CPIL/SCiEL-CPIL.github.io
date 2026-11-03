import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    total_words = q ** n
    
    if total_words > 10000 or radius > 1:
        # Fallback for larger spaces or general radius
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 7:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return [list(random.choices(range(q), k=n)) for _ in range(100)]

    # Bitset-based approach for small spaces (e.g., 3^7 = 2187)
    
    # 1. Generate all words
    all_words = [list(p) for p in itertools.product(range(q), repeat=n)]
    
    # 2. Precompute balls as bitmasks (Python ints)
    balls = [0] * total_words
    
    if radius == 1:
        for i, w in enumerate(all_words):
            mask = 0
            # Self
            mask |= (1 << i)
            # Neighbors
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        # Calculate neighbor index directly
                        neighbor_idx = 0
                        for k in range(n):
                            val = w[k] if k != pos else new_val
                            neighbor_idx = neighbor_idx * q + val
                        mask |= (1 << neighbor_idx)
            balls[i] = mask
    else:
        # General radius (slower)
        for i, w in enumerate(all_words):
            mask = 0
            for j, v in enumerate(all_words):
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    mask |= (1 << j)
            balls[i] = mask

    all_ones_mask = (1 << total_words) - 1
    
    best_code = None
    best_size = total_words + 1
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    num_restarts = 50 # More restarts, each is fast
    
    for _ in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        uncovered = all_ones_mask
        selected = []
        
        # Greedy Phase
        while uncovered:
            best_idx = -1
            max_cover = 0
            
            for i in range(total_words):
                c = (balls[i] & uncovered).bit_count()
                if c > max_cover:
                    max_cover = c
                    best_idx = i
                elif c == max_cover and c > 0:
                    # Tie break: random
                    if random.random() < 0.5:
                        best_idx = i
                        
            if best_idx == -1:
                break
                
            selected.append(best_idx)
            uncovered &= ~balls[best_idx]
            
        # Removal Phase
        changed = True
        while changed:
            changed = False
            for i in range(len(selected) - 1, -1, -1):
                # Check if selected[i] is redundant
                union_rest = 0
                for j in range(len(selected)):
                    if j != i:
                        union_rest |= balls[selected[j]]
                        
                if (union_rest & all_ones_mask) == all_ones_mask:
                    selected.pop(i)
                    changed = True
                    break
                    
        current_size = len(selected)
        if current_size < best_size:
            best_size = current_size
            best_code = [all_words[i] for i in selected]
            
    return best_code
