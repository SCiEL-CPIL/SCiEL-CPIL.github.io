# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    total = q ** n  # 2187 for q=3, n=7
    
    # Convert all words to integers 0..total-1
    # word[i] = sum(w[j] * q^(n-1-j))
    all_words = np.arange(total, dtype=np.int32)
    
    # Precompute digit vectors: shape (total, n)
    digits = np.zeros((total, n), dtype=np.int8)
    for j in range(n):
        divisor = q ** (n - 1 - j)
        digits[:, j] = (all_words // divisor) % q
    
    # Precompute: for each word index and each position, the list of words at distance 1
    # ball_mask[i] = boolean array of words within distance <= radius of word i
    # For radius=1: word i covers itself + n*(q-1) neighbors
    
    # Build neighborhood: for each word, which words are within distance 1
    # Use digit comparison
    # ball[i, j] = 1 if dist(word_i, word_j) <= radius
    
    # For efficiency, precompute the ball as a list of indices
    # ball_indices[i] = np.array of indices within distance 1 of i
    # This is 1 + 7*2 = 15 words per codeword
    
    # Precompute neighbor lists
    ball_indices = []
    for i in range(total):
        d = digits[i]
        neighbors = [i]
        for pos in range(n):
            for val in range(q):
                if val == d[pos]:
                    continue
                # Create neighbor
                nd = d.copy()
                nd[pos] = val
                # Convert back to index
                idx = 0
                for j in range(n):
                    idx = idx * q + nd[j]
                neighbors.append(idx)
        ball_indices.append(np.array(neighbors, dtype=np.int32))
    
    # Greedy set cover
    uncovered = np.ones(total, dtype=bool)
    code = []
    
    while np.any(uncovered):
        if time.time() > deadline - 10:
            break
        
        # For each candidate codeword, compute how many currently uncovered words it covers
        best_gain = -1
        best_idx = -1
        
        # Vectorized: for each candidate, count uncovered words in its ball
        # We'll sample or do full scan
        # Full scan: for each of 2187 candidates, count uncovered in ball
        # This is 2187 * 15 operations - fast enough
        
        for i in range(total):
            if not uncovered[i]:
                # Still can be a good codeword
                pass
            gain = np.sum(uncovered[ball_indices[i]])
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx == -1 or best_gain <= 0:
            break
        
        code.append(best_idx)
        # Update uncovered
        uncovered[ball_indices[best_idx]] = False
    
    # Local search: try to remove redundant codewords
    # A codeword is redundant if removing it doesn't uncover any word
    changed = True
    while changed:
        if time.time() > deadline - 5:
            break
        changed = False
        new_code = []
        for idx in code:
            # Check if this codeword is needed
            # Temporarily remove and check if all words still covered
            # Quick check: is there any word in its ball that is ONLY covered by this codeword?
            # Simpler: just try removing and recheck
            new_code.append(idx)
        
        # Try removing each codeword
        for i in range(len(code)):
            if time.time() > deadline - 5:
                break
            # Remove code[i] and check coverage
            temp_code = code[:i] + code[i+1:]
            # Check if all words are covered
            covered = np.zeros(total, dtype=bool)
            for c in temp_code:
                covered[ball_indices[c]] = True
            if np.all(covered):
                code = temp_code
                changed = True
                break
    
    # Try adding better codewords via local search
    # For each position in code, try replacing with a different codeword that gives same or better coverage
    for iteration in range(5):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(len(code)):
            if time.time() > deadline - 5:
                break
            # Current coverage without code[i]
            covered_without = np.zeros(total, dtype=bool)
            for j, c in enumerate(code):
                if j != i:
                    covered_without[ball_indices[c]] = True
            
            # Find best replacement for position i
            best_gain = np.sum(~covered_without)  # words only covered by code[i]
            best_replacement = code[i]
            
            for candidate in range(total):
                if candidate in code:
                    continue
                # Gain from adding candidate to covered_without
                gain = np.sum(~covered_without[ball_indices[candidate]])
                if gain > best_gain:
                    best_gain = gain
                    best_replacement = candidate
            
            if best_replacement != code[i]:
                code[i] = best_replacement
                improved = True
        
        if not improved:
            break
    
    # Final local search: try removing codewords again
    changed = True
    while changed:
        if time.time() > deadline - 3:
            break
        changed = False
        for i in range(len(code)):
            if time.time() > deadline - 3:
                break
            temp_code = code[:i] + code[i+1:]
            covered = np.zeros(total, dtype=bool)
            for c in temp_code:
                covered[ball_indices[c]] = True
            if np.all(covered):
                code = temp_code
                changed = True
                break
    
    # Verify coverage (should be valid)
    if len(code) > 0:
        covered = np.zeros(total, dtype=bool)
        for c in code:
            covered[ball_indices[c]] = True
        if not np.all(covered):
            # Fallback: use original construction
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n - 4)]
    
    # Convert back to list of lists
    result = []
    for idx in code:
        word = []
        for j in range(n):
            word.append(int(digits[idx, j]))
        result.append(word)
    
    return result
# EVOLVE-BLOCK-END
