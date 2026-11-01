import time
import itertools

def construct(q, n, radius):
    """
    Constructs a covering code using a greedy set cover algorithm.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Parameters
    if q != 3 or n != 7 or radius != 1:
        # Fallback to the simple construction if parameters differ, though task is fixed
        # For robustness, we handle the specific case efficiently and fallback otherwise
        # But the task contract says evaluator passes q, n, radius for this specific problem.
        # We assume q=3, n=7, radius=1.
        pass

    total_words = q ** n
    
    # Precompute masks for all possible codewords
    # Each mask is an integer where bit i is 1 if the word corresponding to index i is covered.
    # We need a mapping from word to index.
    # Word is a tuple of n digits in 0..q-1.
    # Index is base-q integer representation.
    
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    # Precompute the ball masks for each candidate
    # This takes 2187 * 15 operations, which is fast.
    masks = [0] * total_words
    
    # Helper to get neighbors
    # For radius 1, neighbors are the word itself and all words differing in exactly 1 position.
    
    for c_idx in range(total_words):
        # Decode c_idx to word
        word = []
        temp = c_idx
        for i in range(n):
            word.append(temp % q)
            temp //= q
        word.reverse() # word[0] is most significant? 
        # Let's stick to a consistent mapping.
        # If idx = sum(w_i * q^(n-1-i)), then:
        # w_0 = idx // q^(n-1)
        # w_{n-1} = idx % q
        
        # Let's re-verify the mapping in word_to_idx.
        # idx = w_0 * q^(n-1) + ... + w_{n-1}
        # So decoding:
        # w_{n-1} = idx % q
        # w_{n-2} = (idx // q) % q
        # ...
        # w_0 = idx // q^(n-1)
        
        # My word_to_idx did: idx = idx*q + w[i].
        # So idx = w_0*q^(n-1) + ... + w_{n-1}.
        # This matches.
        
        # Re-decode correctly for mask generation
        c_word = [0] * n
        temp = c_idx
        for i in range(n-1, -1, -1):
            c_word[i] = temp % q
            temp //= q
            
        mask_val = 0
        # Add center
        mask_val |= (1 << c_idx)
        
        # Add neighbors
        for pos in range(n):
            for val in range(q):
                if val == c_word[pos]:
                    continue
                # Create neighbor
                n_word = c_word[:]
                n_word[pos] = val
                n_idx = 0
                for i in range(n):
                    n_idx = n_idx * q + n_word[i]
                mask_val |= (1 << n_idx)
                
        masks[c_idx] = mask_val

    # Greedy Set Cover
    # Universe mask: all bits set to 1
    universe_mask = (1 << total_words) - 1
    uncovered_mask = universe_mask
    
    selected_codewords = []
    
    while uncovered_mask != 0:
        if time.time() > deadline:
            # Fallback to a valid but potentially larger code if we run out of time
            # Since greedy is fast, this shouldn't happen.
            # We can break and return what we have? No, must be valid.
            # If we run out, we should probably just return a known valid code.
            # The direct sum code is valid.
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

        best_idx = -1
        best_count = 0
        
        # Iterate through all candidates
        # Optimization: We can skip candidates that are already fully covered? 
        # No, we need to find the one that covers the most *new* bits.
        
        # We can also prune: if a candidate's mask is a subset of another's? 
        # Not easy to check dynamically.
        
        for c_idx in range(total_words):
            # Count intersection with uncovered
            intersection = masks[c_idx] & uncovered_mask
            # Count bits
            # Python int.bit_count() is fast in 3.8+
            count = intersection.bit_count()
            
            if count > best_count:
                best_count = count
                best_idx = c_idx
                
        if best_idx == -1 or best_count == 0:
            # Should not happen if universe is not empty and masks are valid
            break
            
        # Select best_idx
        selected_codewords.append(best_idx)
        uncovered_mask &= ~masks[best_idx]
        
        # Optional: Early termination if we reach a known good bound?
        # The best known is 186. If we reach 186, we can stop?
        # Greedy might find 186 or slightly more.
        
    # Convert indices to codewords (lists)
    result = []
    for idx in selected_codewords:
        word = [0] * n
        temp = idx
        for i in range(n-1, -1, -1):
            word[i] = temp % q
            temp //= q
        result.append(word)
        
    return result
