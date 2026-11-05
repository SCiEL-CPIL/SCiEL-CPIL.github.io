import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Handle edge cases or unexpected params with seed fallback
    if not (q == 3 and n == 8 and radius == 1):
        # Fallback to seed logic if it applies
        if n >= 4 and q == 3 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            # Generic fallback: return all words? (Expensive but valid)
            # Or just a small set if we know it works?
            # For safety, if we don't know, return a large set.
            # But the task is specific.
            pass

    N = q ** n
    
    # Precomputation
    def index_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    def word_to_index(w):
        idx = 0
        power = 1
        for i in range(n):
            idx += w[i] * power
            power *= q
        return idx

    # Precompute neighbors (list of lists) and masks (list of ints)
    neighbors = [[] for _ in range(N)]
    masks = [0] * N
    
    for i in range(N):
        w = index_to_word(i)
        mask = 1 << i
        neighbors[i].append(i)
        
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                w[pos] = new_val
                j = word_to_index(w)
                mask |= (1 << j)
                neighbors[i].append(j)
            w[pos] = original_val
        
        masks[i] = mask

    # 1. Greedy Set Cover
    uncovered = (1 << N) - 1
    code_indices = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        best_idx = -1
        best_count = 0
        
        for i in range(N):
            intersection = masks[i] & uncovered
            if intersection:
                # Use bit_count if available, else bin count
                try:
                    count = intersection.bit_count()
                except AttributeError:
                    count = bin(intersection).count('1')
                
                if count > best_count:
                    best_count = count
                    best_idx = i
        
        if best_idx == -1:
            break
            
        code_indices.append(best_idx)
        uncovered &= ~masks[best_idx]

    # 2. Local Search (Removal)
    # Initialize cover counts
    cover_counts = [0] * N
    for idx in code_indices:
        for j in neighbors[idx]:
            cover_counts[j] += 1

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle to avoid bias
        random.shuffle(code_indices)
        
        i = 0
        while i < len(code_indices):
            if time.time() > deadline:
                break
                
            idx = code_indices[i]
            
            # Try to remove idx
            valid = True
            for j in neighbors[idx]:
                cover_counts[j] -= 1
                if cover_counts[j] == 0:
                    valid = False
                    break
            
            if valid:
                # Check if any other count is 0? 
                # We only decremented counts for neighbors[idx].
                # If any of those became 0, it's invalid.
                # If all stayed > 0, it's valid.
                # We don't need to check the whole array, just the affected ones.
                code_indices.pop(i)
                improved = True
                # Do not increment i, because the next element shifted down
            else:
                # Restore counts
                for j in neighbors[idx]:
                    cover_counts[j] += 1
                i += 1

    # Convert back to words
    result = [index_to_word(i) for i in code_indices]
    return result
