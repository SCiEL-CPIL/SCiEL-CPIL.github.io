import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = 145.0  # 5 seconds margin
    
    # Only handle the specific case q=3, n=6, radius=1 efficiently
    # Fallback for other cases
    if not (q == 3 and n == 6 and radius == 1):
        # Fallback to the original construction for general cases
        # This ensures the function doesn't crash if called with other params, 
        # though the task specifies the 6-match football pool (q=3, n=6, r=1)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # Adjust for general n if needed, but problem is specific
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Brute force for small n
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=6, radius=1
    # Total words = 3^6 = 729
    total_words = q ** n
    
    # Precompute the ball (neighbors) for each word as a bitmask
    # Index mapping: word (d0..d5) -> d0 + 3*d1 + ... + 3^5*d5
    # We use Python integers as bitsets. 729 bits fit easily in a Python int.
    
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

    # Precompute balls
    balls = [0] * total_words
    for idx in range(total_words):
        word = idx_to_word(idx)
        mask = 1 << idx
        # Add self
        # Add neighbors: change one digit
        for pos in range(n):
            for delta in [1, 2]:
                new_word = word[:]
                new_word[pos] = (word[pos] + delta) % q
                n_idx = word_to_idx(new_word)
                mask |= (1 << n_idx)
        balls[idx] = mask

    # Greedy Set Cover
    def greedy_cover(randomize=False):
        uncovered = (1 << total_words) - 1
        solution = []
        
        # To speed up, we can maintain a list of remaining candidates
        # But 729 is small, so scanning all is fine.
        
        while uncovered:
            best_idx = -1
            best_count = -1
            best_mask = 0
            
            # Iterate through all candidates
            for i in range(total_words):
                # Count bits in intersection
                inter = balls[i] & uncovered
                # bit_count is available in Python 3.8+
                count = inter.bit_count()
                
                # Add random noise if randomizing to break ties or explore
                if randomize:
                    count += random.random() * 0.1
                
                if count > best_count:
                    best_count = count
                    best_idx = i
                    best_mask = inter
                    if count == 0:
                        break # Should not happen if uncovered is not empty and balls cover everything
            
            if best_idx == -1:
                break # Should not happen
            
            solution.append(best_idx)
            uncovered &= ~balls[best_idx]
            
            # Deadline check inside loop?
            # The loop runs at most 73 times. 73 * 729 * bit_count is very fast.
            # No need for tight deadline check here, but good to have.
            if time.time() > deadline:
                break
        
        return solution

    # 1. Run standard greedy
    best_solution = greedy_cover(randomize=False)
    best_len = len(best_solution)
    
    # 2. Try to improve with randomized greedy
    # We have plenty of time (145s). Each greedy run takes < 0.01s.
    # We can run many iterations.
    iterations = 0
    while time.time() < deadline:
        sol = greedy_cover(randomize=True)
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
            if best_len <= 71: # Theoretical lower bound is 71, if we hit it, we stop
                break
        iterations += 1
        # To avoid infinite loop if something goes wrong, cap iterations
        if iterations > 10000:
            break

    # Convert indices back to words
    result = []
    for idx in best_solution:
        result.append(idx_to_word(idx))
        
    return result
