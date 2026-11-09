import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    # Fallback for non-standard parameters: return trivial valid code
    if q != 3 or n != 7 or radius != 1:
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    
    N = 3 ** n
    # Precompute ball masks as integers
    # We need to map index to digits and back quickly.
    # Since q=3, n=7, we can use base-3 arithmetic.
    
    # Precompute powers of 3
    powers = [1]
    for i in range(n):
        powers.append(powers[-1] * 3)
    
    # Function to get index from digits
    def digits_to_idx(digits):
        idx = 0
        for i in range(n):
            idx += digits[i] * powers[i]
        return idx
    
    # Precompute all ball masks
    # ball_masks[i] is an integer with bits set for all words in the ball of word i
    ball_masks = [0] * N
    
    # To speed up ball mask generation, we can precompute the neighbors.
    # For each index i, we can generate its 2*n neighbors.
    
    # Let's precompute the index of every word to allow O(1) lookup if we had a mapping,
    # but here we compute the index directly.
    
    # Optimization: Precompute the "delta" indices for each position and digit change.
    # For position p, changing digit d to d' adds (d'-d) * 3^p to the index.
    
    for i in range(N):
        mask = 1 << i
        # Extract digits
        temp = i
        digits = [0] * n
        for p in range(n):
            digits[p] = temp % 3
            temp //= 3
        
        for p in range(n):
            orig_val = digits[p]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                new_idx = i + (new_val - orig_val) * powers[p]
                mask |= (1 << new_idx)
        ball_masks[i] = mask

    # Convert ball_masks to a list of integers for fast bitwise operations
    # In Python, integers are arbitrary precision, so bit_count() is fast.
    
    # Greedy Set Cover with Randomization
    # We maintain a set of covered bits.
    # To speed up, we can use a list of remaining uncovered words?
    # No, bitmasks are better.
    
    # One optimization: if we find a word with gain == ball_size (15), we can pick it immediately?
    # Not necessarily, as we want to randomize ties.
    
    # Let's implement a fast greedy step.
    # We will shuffle the candidate list once at the start of each run?
    # Or shuffle every step? Shuffling 2187 items is cheap.
    
    def greedy_run():
        covered = 0
        code_indices = []
        # List of all indices
        indices = list(range(N))
        
        while covered != (1 << N) - 1:
            random.shuffle(indices)
            best_gain = -1
            best_idx = -1
            
            # We can break early if best_gain reaches the max possible gain (15)
            # But we want to randomize among max gains.
            # So we scan all, keeping track of max gain and the last one seen?
            # Or collect all max gain ones and pick one randomly.
            # Collecting is O(N) memory per step, which is fine.
            
            max_gain = -1
            candidates = []
            
            for idx in indices:
                gain = (ball_masks[idx] & ~covered).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    candidates = [idx]
                elif gain == max_gain:
                    candidates.append(idx)
            
            # Pick one randomly from candidates
            best_idx = random.choice(candidates)
            code_indices.append(best_idx)
            covered |= ball_masks[best_idx]
            
            # Optimization: If max_gain was 0, we are stuck (should not happen if valid)
            # If max_gain was 15, we covered a full ball.
            
        return code_indices

    # Baseline: Tetracode construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    # Tetracode has 9 words of length 4 covering radius 1.
    # We extend to length 7 by appending all 3^3 = 27 tails.
    # Total size: 9 * 27 = 243.
    seed_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            seed_code.append(h + list(tail))
    
    best_code = seed_code
    best_len = len(seed_code)
    
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer for 120s limit
    
    # Run greedy restarts
    run_count = 0
    while time.time() < deadline:
        code_indices = greedy_run()
        if len(code_indices) < best_len:
            best_len = len(code_indices)
            # Convert indices to words
            best_code = []
            for idx in code_indices:
                digits = [0] * n
                temp = idx
                for p in range(n):
                    digits[p] = temp % 3
                    temp //= 3
                best_code.append(digits)
        
        run_count += 1
        # Check time periodically
        if run_count % 10 == 0:
            if time.time() >= deadline:
                break

    return best_code
