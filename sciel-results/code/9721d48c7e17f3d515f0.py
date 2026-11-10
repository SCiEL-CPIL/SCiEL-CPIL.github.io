import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s margin for safety/overhead
    
    total_words = q ** n
    
    # Map word (tuple) to index
    # word = (w0, ..., w_{n-1})
    # idx = w0 * q^{n-1} + ... + w_{n-1}
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute coverage masks
    # masks[i] is an integer bitmask where bit j is 1 if word j is covered by codeword i
    # Note: We need to generate neighbors.
    
    # To generate neighbors of an index, it's easier to work with the word representation
    # or precompute transitions.
    
    # Let's generate all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    idx_map = {w: i for i, w in enumerate(all_words)}
    
    masks = [0] * total_words
    
    # For each potential codeword index, compute its coverage mask
    for i, w in enumerate(all_words):
        mask = 0
        # The codeword itself
        mask |= (1 << i)
        
        # Neighbors at distance 1
        # Change one position
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                # Create new word
                new_w = list(w)
                new_w[pos] = new_val
                new_idx = idx_map[tuple(new_w)]
                mask |= (1 << new_idx)
        masks[i] = mask

    covered = 0
    full_mask = (1 << total_words) - 1
    code_indices = []
    
    # To avoid re-picking the same codeword, we can keep a list of available indices
    # or just check if the gain is 0.
    # Since we want to minimize count, greedy is standard.
    
    while covered != full_mask:
        if time.time() > deadline:
            break
        
        best_gain = -1
        best_idx = -1
        
        # Iterate through all possible codewords
        # Optimization: We could skip ones already in code_indices, 
        # but checking gain is cheap.
        for i in range(total_words):
            if i in code_indices: 
                continue
            gain = bin(masks[i] & ~covered).count('1')
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx == -1:
            # Should not happen
            break
            
        code_indices.append(best_idx)
        covered |= masks[best_idx]
        
        # If best_gain is 0, we might be stuck, but theoretically greedy on full space shouldn't get stuck unless invalid params
        if best_gain == 0:
             break

    # Convert indices back to words
    result = [all_words[i] for i in code_indices]
    # Convert to lists as required
    return [list(w) for w in result]
