import numpy as np
import time
import itertools

def construct(q, n, radius):
    # We assume q=3, n=8, radius=1 based on the task description, 
    # but the function signature takes q, n, radius.
    # The task specifically mentions K_3(8,1). 
    # If parameters differ, we might need a general fallback, 
    # but the prompt says "The evaluator passes this task's alphabet size q, length n and covering radius."
    # and the task is specifically about K_3(8,1).
    # I will assume q=3, n=8, radius=1 for the optimized path, 
    # and provide a basic greedy for others just in case, 
    # though the score is based on this specific instance.
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback: simple greedy for other parameters
        # This part is likely not executed for the main score, but good for robustness
        return _general_greedy(q, n, radius)

    # Optimized path for q=3, n=8, radius=1
    total_words = q ** n
    deadline = time.time() + 110.0 # Safety margin
    
    # Generate all words as indices 0 to total_words-1
    # We need to map index <-> word
    # To speed up covering set calculation, we can work with indices.
    # But calculating Hamming distance requires the word structure.
    
    # Let's precompute the covering sets for all words.
    # Word i corresponds to the ternary representation of i.
    
    # Helper to get word from index
    def index_to_word(idx):
        word = [0] * n
        for j in range(n):
            word[j] = idx % q
            idx //= q
        return word

    # Precompute covering sets
    # covering_sets[i] will be a list of indices of words within distance 1 of word i
    covering_sets = np.zeros((total_words, 1 + radius * n * (q - 1)), dtype=np.int32)
    # For q=3, n=8, radius=1, size is 1 + 8*2 = 17
    
    for i in range(total_words):
        w = index_to_word(i)
        neighbors = [i]
        for pos in range(n):
            for new_val in range(q):
                if new_val == w[pos]:
                    continue
                new_w = w[:]
                new_w[pos] = new_val
                # Convert new_w back to index
                idx = 0
                for k in range(n):
                    idx = idx * q + new_w[k]
                # Wait, the index calculation above:
                # word[j] = idx % q ; idx //= q
                # This means word[0] is least significant.
                # So index = w[0] + w[1]*q + w[2]*q^2 ...
                # Let's re-verify index_to_word
                # word[0] = idx % q
                # word[1] = (idx // q) % q
                # Yes.
                # So to get index from word:
                # idx = 0
                # for k in range(n-1, -1, -1): # if word[0] is LSB, we build from MSB?
                # No, if word[0] is LSB, then:
                # idx = w[0] + w[1]*q + ... + w[n-1]*q^(n-1)
                
                # Let's fix the index calculation in the loop
                idx_new = 0
                for k in range(n):
                    idx_new += new_w[k] * (q ** k)
                neighbors.append(idx_new)
        
        # Store in array
        covering_sets[i, :len(neighbors)] = neighbors
        # Note: len(neighbors) should be exactly 17 for q=3, n=8, r=1.
        # If not, we'd have issues. But it is.

    # Greedy Set Cover
    covered = np.zeros(total_words, dtype=bool)
    code = []
    uncovered_count = total_words
    
    while uncovered_count > 0:
        if time.time() > deadline:
            break
            
        # Calculate gain for each candidate
        # We want to pick the candidate i that covers the most *new* (uncovered) words.
        # new_covered = covering_sets[i] & ~covered
        # gain[i] = sum(new_covered)
        
        # Vectorized approach:
        # ~covered is a boolean array of shape (total_words,)
        # covering_sets is (total_words, 17)
        # We can index ~covered with covering_sets to get a (total_words, 17) boolean array
        # Then sum along axis 1.
        
        gains = (~covered)[covering_sets].sum(axis=1)
        
        # Find the best candidate
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            # This shouldn't happen if uncovered_count > 0
            # Because every uncovered word is a neighbor of itself, so gain for that word is at least 1.
            break
            
        # Add to code
        code.append(index_to_word(best_idx))
        
        # Update covered
        new_covered_indices = covering_sets[best_idx]
        # Only update those that are actually uncovered to keep count accurate? 
        # Or just set to True and recalculate count?
        # Setting to True is idempotent.
        # To update uncovered_count efficiently:
        # count how many were False before
        was_uncovered = ~covered[new_covered_indices]
        newly_covered_count = np.sum(was_uncovered)
        
        covered[new_covered_indices] = True
        uncovered_count -= newly_covered_count

    return code

def _general_greedy(q, n, radius):
    # Simple greedy for other parameters, just in case
    # This is slow, so we hope it's not called.
    total = q ** n
    words = [list(w) for w in itertools.product(range(q), repeat=n)]
    
    def hamming_dist(w1, w2):
        return sum(1 for a, b in zip(w1, w2) if a != b)
    
    # Precompute covering sets
    cover_sets = []
    for i, w in enumerate(words):
        cs = set()
        for j, v in enumerate(words):
            if hamming_dist(w, v) <= radius:
                cs.add(j)
        cover_sets.append(list(cs))
        
    covered = [False] * total
    code = []
    
    while True:
        if not any(not c for c in covered):
            break
        best_i = -1
        best_gain = -1
        for i in range(total):
            if any(cw == words[i] for cw in code):
                continue
            gain = 0
            for j in cover_sets[i]:
                if not covered[j]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i == -1:
            break
        code.append(words[best_i])
        for j in cover_sets[best_i]:
            covered[j] = True
            
    return code
