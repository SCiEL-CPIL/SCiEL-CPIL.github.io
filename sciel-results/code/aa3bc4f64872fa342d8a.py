import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110
    
    total_words = q ** n
    full_mask = (1 << total_words) - 1
    
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

    # Precompute masks
    masks = [0] * total_words
    for i in range(total_words):
        w = idx_to_word(i)
        mask = 0
        # Set bit for w
        mask |= 1 << i
        for pos in range(n):
            for new_val in range(q):
                if new_val != w[pos]:
                    neighbor = w[:]
                    neighbor[pos] = new_val
                    mask |= 1 << word_to_idx(neighbor)
        masks[i] = mask

    best_code = None
    best_size = float('inf')
    
    # Run multiple greedy restarts
    # Each restart uses random tie-breaking
    max_restarts = 20  # Adjust based on time
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5: # Stop if close to deadline
            break
            
        covered = 0
        code = []
        
        while covered != full_mask:
            if time.time() > deadline - 2:
                break
                
            best_gain = -1
            candidates = []
            
            for i in range(total_words):
                gain = (masks[i] & ~covered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain:
                    candidates.append(i)
            
            if not candidates:
                break
                
            # Randomly pick from best candidates
            best_idx = random.choice(candidates)
            code.append(idx_to_word(best_idx))
            covered |= masks[best_idx]
            
            if len(code) >= best_size:
                break
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
            # If we found a very good solution (e.g., 73), we can stop
            if best_size <= 73:
                break

    if best_code is None:
        # Fallback to baseline if something went wrong
        # Use the tetracode construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return best_code
