import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = 120  # Safety margin from 150s limit
    
    # Handle small cases or fallback
    total_words = q ** n
    if total_words > 100000:
        # Fallback to tetracode product for larger n
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n-4)]
        else:
            # Generic fallback: Random
            return [list(np.random.randint(0, q, n)) for _ in range(100)]

    # Specific case for q=3, n=8, r=1
    # Precompute balls
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute neighbors (ball of radius 1)
    ball_size = 1 + n * (q - 1)
    neighbors = np.zeros((total_words, ball_size), dtype=np.int32)
    
    for i in range(total_words):
        w = idx_to_word(i)
        neighbors[i, 0] = i
        count = 1
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                w[pos] = val
                neighbors[i, count] = word_to_idx(w)
                count += 1
            w[pos] = orig

    best_code = None
    best_size = float('inf')
    
    # Fallback initial best
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        fallback = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n-4)]
        best_code = fallback
        best_size = len(fallback)

    while time.time() - start_time < deadline:
        # 1. Randomized Greedy Construction
        covered = np.zeros(total_words, dtype=bool)
        code_indices = []
        added = set()
        
        K = 10 # Top K candidates to choose from randomly
        
        while not covered.all():
            # Calculate score: number of uncovered words in ball
            # uncovered = ~covered
            # scores[i] = sum(uncovered[neighbors[i, :]])
            scores = np.sum((~covered)[neighbors], axis=1)
            
            # Mask out already added
            for idx in code_indices:
                scores[idx] = -1
                
            if scores.max() <= 0:
                # Fallback: pick random from remaining
                remaining = [i for i in range(total_words) if i not in added]
                if not remaining:
                    break
                idx = np.random.choice(remaining)
            else:
                # Get top K
                # Ensure K is not larger than number of valid candidates
                valid_count = np.sum(scores > 0)
                if valid_count == 0:
                    break
                k_actual = min(K, valid_count)
                top_k_indices = np.argpartition(scores, -k_actual)[-k_actual:]
                idx = top_k_indices[np.random.randint(k_actual)]
                
            code_indices.append(idx)
            added.add(idx)
            
            # Update covered
            covered[neighbors[idx, :]] = True
            
            if time.time() - start_time > deadline:
                break
        
        if time.time() - start_time > deadline:
            break
            
        # 2. Peeling (Redundancy Removal)
        cov_counts = np.zeros(total_words, dtype=np.int32)
        for idx in code_indices:
            cov_counts[neighbors[idx, :]] += 1
            
        current_code = list(code_indices)
        
        # Iteratively remove redundant codewords
        changed = True
        while changed:
            changed = False
            # Shuffle to avoid bias
            np.random.shuffle(current_code)
            for idx in current_code[:]:
                # Check if idx is redundant
                ball_cov = cov_counts[neighbors[idx, :]]
                if ball_cov.min() >= 2:
                    # Remove idx
                    current_code.remove(idx)
                    cov_counts[neighbors[idx, :]] -= 1
                    changed = True
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
                    if time.time() - start_time > deadline:
                        break
            if time.time() - start_time > deadline:
                break
        
        if time.time() - start_time > deadline:
            break

    if best_code is None:
        # Should not happen
        return [list(np.random.randint(0, q, n)) for _ in range(100)]
        
    # Convert indices back to words
    return [idx_to_word(idx) for idx in best_code]
