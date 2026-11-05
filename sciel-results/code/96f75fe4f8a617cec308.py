import itertools
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            all_words = list(itertools.product(range(q), repeat=n))
            return all_words

    # Specific optimization for q=3, n=8, radius=1
    total_words = 3**8
    # Precompute neighbors for all words
    # neighbors[i] is a list of 17 indices
    neighbors = []
    for i in range(total_words):
        word = []
        tmp = i
        for _ in range(8):
            word.append(tmp % 3)
            tmp //= 3
        # word is reversed, let's fix or just use as is for consistency
        # Actually, let's use a standard base-3 conversion
        word = [0]*8
        tmp = i
        for j in range(8):
            word[j] = tmp % 3
            tmp //= 3
            
        neigh = [i]
        for pos in range(8):
            orig = word[pos]
            for val in range(3):
                if val != orig:
                    new_word = word[:]
                    new_word[pos] = val
                    # Convert back to index
                    idx = 0
                    for k in range(8):
                        idx = idx * 3 + new_word[k]
                    neigh.append(idx)
        neighbors.append(neigh)

    # Phase 1: Greedy Construction
    uncovered = np.ones(total_words, dtype=bool)
    code_indices = []
    
    # To speed up greedy, we can just iterate
    while uncovered.any():
        best_idx = -1
        best_count = -1
        # Iterate through all words to find the one covering most uncovered
        for w in range(total_words):
            if not uncovered[w]:
                continue # Optimization: if w itself is covered, its neighbors might still be uncovered?
                # Actually, if w is covered, it's in the neighborhood of some selected word.
                # But w's neighborhood might contain uncovered words.
                # So we must check all w.
            
            # Count uncovered in neighbors[w]
            # neighbors[w] is a list of 17 indices
            count = 0
            for idx in neighbors[w]:
                if uncovered[idx]:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = w
        
        if best_idx == -1:
            break
            
        code_indices.append(best_idx)
        # Update uncovered
        for idx in neighbors[best_idx]:
            uncovered[idx] = False

    # Phase 2: Peeling
    m = len(code_indices)
    if m == 0:
        return []
        
    # Build coverage array
    coverage = np.zeros(total_words, dtype=np.int32)
    for c in code_indices:
        for idx in neighbors[c]:
            coverage[idx] += 1
            
    active_set = set(code_indices)
    
    changed = True
    while changed:
        changed = False
        best_remove = -1
        best_score = 0
        
        # Find the most redundant codeword
        for c in active_set:
            # Check if removable: min coverage in neighborhood > 1
            min_cov = 999
            sum_cov = 0
            for idx in neighbors[c]:
                cov = coverage[idx]
                if cov <= 1:
                    min_cov = 0
                    break
                sum_cov += cov
            if min_cov > 1:
                # Score: higher sum_cov means more redundant
                if sum_cov > best_score:
                    best_score = sum_cov
                    best_remove = c
                    
        if best_remove != -1:
            # Remove best_remove
            active_set.remove(best_remove)
            for idx in neighbors[best_remove]:
                coverage[idx] -= 1
            changed = True

    # Convert indices back to words
    result = []
    for idx in active_set:
        word = [0]*8
        tmp = idx
        for j in range(8):
            word[j] = tmp % 3
            tmp //= 3
        result.append(word)
        
    return result
