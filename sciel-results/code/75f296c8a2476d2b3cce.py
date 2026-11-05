import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety
    
    # 1. Generate initial code (Direct sum of tetracode)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Convert to tuples for hashing
    code = [tuple(c) for c in initial_code]
    
    # 2. Precompute neighbors for each codeword? 
    # No, we need to check coverage of the whole space.
    # Space size is 3^8 = 6561.
    # We can use an array of size 3**n to store counts.
    
    total_words = q ** n
    cover_count = [0] * total_words
    
    # Helper to convert word to index
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    # Helper to get neighbors (including self)
    def get_neighbors(word):
        neighbors = [word]
        for i in range(n):
            for v in range(q):
                if v != word[i]:
                    new_word = list(word)
                    new_word[i] = v
                    neighbors.append(tuple(new_word))
        return neighbors

    # Initialize counts
    for c in code:
        idx = word_to_idx(c)
        # Add 1 for the word itself
        cover_count[idx] += 1
        # Add 1 for each neighbor
        for i in range(n):
            for v in range(q):
                if v != c[i]:
                    new_word = list(c)
                    new_word[i] = v
                    n_idx = word_to_idx(tuple(new_word))
                    cover_count[n_idx] += 1

    # 3. Greedy Deletion Loop
    # We will try to remove codewords that are "redundant".
    # A codeword c is redundant if for all w in B(c, 1), cover_count[w] > 1.
    # Note: cover_count[w] includes c's contribution. So we need cover_count[w] >= 2.
    
    # To make it faster, we can precompute the list of indices for each codeword's ball.
    # But we also need to update counts.
    
    # Let's store the ball indices for each codeword in the current list.
    # Since the list changes, we might just recompute on the fly or store statically.
    # The set of neighbors of a word is static.
    
    # Precompute neighbor indices for all possible words? 
    # 6561 words * 17 neighbors = ~111k entries. This is small.
    # Let's precompute neighbor indices for all words in the space.
    neighbor_indices = {}
    for w in range(total_words):
        # Convert w to word
        word = [0] * n
        temp = w
        for i in range(n - 1, -1, -1):
            word[i] = temp % q
            temp //= q
        word = tuple(word)
        
        nbs = [w]
        for i in range(n):
            for v in range(q):
                if v != word[i]:
                    new_word = list(word)
                    new_word[i] = v
                    nbs.append(word_to_idx(tuple(new_word)))
        neighbor_indices[w] = nbs
        
    # Map codeword to its index
    code_indices = [word_to_idx(c) for c in code]
    
    # List of active codeword indices
    active = list(range(len(code)))
    
    # To allow removal, we need to know if a codeword can be removed.
    # We will iterate through active codewords.
    # If a codeword is removable, we remove it.
    
    # Optimization: Instead of checking all active codewords repeatedly,
    # we can just loop through the list and remove if possible.
    
    # We need to be careful with the order.
    # Let's do multiple passes.
    
    pass_count = 0
    while time.time() < deadline:
        changed = False
        # Iterate in reverse to allow removal from list
        for i in range(len(active) - 1, -1, -1):
            if time.time() > deadline:
                break
            c_idx = active[i]
            c_word_idx = code_indices[c_idx]
            nbs = neighbor_indices[c_word_idx]
            
            # Check if all neighbors have count > 1
            can_remove = True
            for nb_idx in nbs:
                if cover_count[nb_idx] <= 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove this codeword
                # Update counts
                for nb_idx in nbs:
                    cover_count[nb_idx] -= 1
                
                # Remove from active list
                active.pop(i)
                changed = True
                
        if not changed:
            # No more removals possible in this pass.
            # We could try a different heuristic or break.
            # Given the time, breaking is safer.
            break

    # Convert back to list of lists
    result = [list(code[i]) for i in active]
    
    return result
