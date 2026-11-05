import itertools
import time

def construct(q, n, radius):
    # Precompute word indices and balls
    # q=3, n=6
    # Total words = 729
    
    # Map tuple to index
    def word_to_idx(w):
        idx = 0
        for i, val in enumerate(w):
            idx += val * (q ** i)
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    # Generate all words
    all_words = [idx_to_word(i) for i in range(q ** n)]
    
    # Precompute balls (list of indices covered by each word)
    balls = []
    for w in all_words:
        covered = set()
        # The word itself
        covered.add(word_to_idx(w))
        # Words at distance 1
        for i in range(n):
            orig_val = w[i]
            for new_val in range(q):
                if new_val != orig_val:
                    new_w = w.copy()
                    new_w[i] = new_val
                    covered.add(word_to_idx(new_w))
        balls.append(list(covered))
        
    # For fast subset check: balls_set[i] is a set of indices covered by word i
    balls_set = [set(b) for b in balls]

    # Initial code: Direct sum of ternary tetracode and full space of length 2
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = h + list(tail)
            initial_code.append(word_to_idx(w))
            
    # Ensure distinct
    initial_code = list(set(initial_code))
    
    # Current code
    code = set(initial_code)
    
    # Compute coverage counts
    counts = [0] * (q ** n)
    for c in code:
        for w in balls[c]:
            counts[w] += 1
            
    # Local Search
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    while time.time() < deadline:
        improved = False
        # Try to remove
        for c in list(code):
            # Check if c is removable
            unique = [w for w in balls[c] if counts[w] == 1]
            if not unique:
                # Remove c
                code.remove(c)
                for w in balls[c]:
                    counts[w] -= 1
                improved = True
                break
        
        if improved:
            continue
            
        # Try to swap
        # Iterate over codewords
        # To make it faster, we can pick the codeword with the smallest unique set first
        # Or just iterate
        swap_done = False
        # Sort by size of unique set to find easy swaps first?
        # Let's just iterate
        for c in list(code):
            unique = [w for w in balls[c] if counts[w] == 1]
            if not unique:
                # Should have been removed above
                continue
                
            # Find a candidate c_prime that covers all unique words
            # c_prime must not be in code
            found_swap = False
            for c_prime in range(q ** n):
                if c_prime in code:
                    continue
                # Check if balls[c_prime] covers all words in unique
                # unique is small
                is_valid = True
                for w in unique:
                    if w not in balls_set[c_prime]:
                        is_valid = False
                        break
                if is_valid:
                    # Perform swap
                    code.remove(c)
                    code.add(c_prime)
                    for w in balls[c]:
                        counts[w] -= 1
                    for w in balls[c_prime]:
                        counts[w] += 1
                    found_swap = True
                    break
            if found_swap:
                improved = True
                swap_done = True
                break
                
        if not improved:
            # Stuck. Try random restarts or kicks?
            # For now, if stuck, we might try a random kick:
            # Remove a random codeword, add 2 random codewords, then try to minimize again?
            # Or just break if we've been stuck for a while.
            # Given the small search space, let's try a random kick.
            if len(code) > 70: # Only kick if we are not already very small
                # Pick a random codeword to remove
                c_to_remove = next(iter(code))
                # Find its unique words
                unique = [w for w in balls[c_to_remove] if counts[w] == 1]
                # This removal will uncover 'unique' words.
                # We need to add codewords to cover them.
                # Let's just add 2 random codewords and hope they cover some.
                # Actually, a better kick: Add a random codeword, then try to remove.
                # If we add a codeword, we increase the size.
                # Let's try: Add a random codeword, then run the removal loop.
                # This might escape local minima.
                c_new = None
                for _ in range(100):
                    c_new = int(time.time() * 1000) % (q ** n) # Pseudo random
                    if c_new not in code:
                        break
                if c_new is not None and c_new not in code:
                    code.add(c_new)
                    for w in balls[c_new]:
                        counts[w] += 1
                    # Don't break, let the loop continue and try to remove
                    improved = True # Force another iteration
            else:
                break

    # Return the list of codewords (as lists of integers)
    return [idx_to_word(c) for c in code]
