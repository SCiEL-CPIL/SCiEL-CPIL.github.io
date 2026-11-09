import numpy as np
import time
import random
import itertools

def construct(q, n, radius):
    # Parameters from task
    # q=3, n=6, radius=1
    N = q ** n
    deadline = time.time() + 105.0  # 15s margin for safety
    
    # Precompute balls for all words in {0..q-1}^n
    # Represent words as indices 0..N-1
    # Convert index to base-q digits
    def idx_to_digits(idx):
        return [int(idx // (q ** i) % q) for i in range(n)]
    
    def digits_to_idx(digits):
        idx = 0
        for i, d in enumerate(digits):
            idx += d * (q ** i)
        return idx

    # Precompute ball membership as a list of lists of indices
    # balls[c] contains all indices within distance 1 from c
    balls = []
    ball_sets = []
    for idx in range(N):
        digits = idx_to_digits(idx)
        ball = [idx]
        for pos in range(n):
            for delta in range(1, q):
                new_digits = digits[:]
                new_digits[pos] = (digits[pos] + delta) % q
                ball.append(digits_to_idx(new_digits))
        balls.append(ball)
        ball_sets.append(set(ball))
        
    # Precompute a boolean membership matrix: in_ball[c][x] = True if x is in ball(c)
    # This allows O(1) membership checks
    in_ball = np.zeros((N, N), dtype=bool)
    for c in range(N):
        in_ball[c, balls[c]] = True

    # Initial code: Tetracode direct sum
    # Tetracode words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            initial_codewords.append(digits_to_idx(word))
            
    codewords = initial_codewords[:]
    cov = np.zeros(N, dtype=np.int32)
    
    for c in codewords:
        cov[balls[c]] += 1
        
    best_codewords = codewords[:]
    best_size = len(codewords)
    
    # Simulated Annealing Parameters
    # We want to minimize len(codewords).
    # We will perform moves: Delete, Swap.
    # Temperature T starts high and cools down.
    
    T = 1.0
    cooling = 0.997
    min_T = 1e-5
    
    while time.time() < deadline and T > min_T:
        # 1. Try to Delete a random codeword
        if len(codewords) > 0:
            i = random.randint(0, len(codewords) - 1)
            c = codewords[i]
            # Check if c is redundant: all words in ball(c) have coverage > 1
            if np.min(cov[balls[c]]) > 1:
                # Remove c
                cov[balls[c]] -= 1
                codewords.pop(i)
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = codewords[:]
                T *= cooling
                continue
                
        # 2. Try a Swap
        if len(codewords) == 0:
            break
            
        i = random.randint(0, len(codewords) - 1)
        c_old = codewords[i]
        c_new = random.randint(0, N - 1)
        
        if c_new == c_old:
            T *= cooling
            continue
            
        # Tentative removal of c_old
        cov[balls[c_old]] -= 1
        
        # Find uncovered words
        missing = np.where(cov < 1)[0]
        
        if len(missing) == 0:
            # c_old was redundant - just delete it
            codewords.pop(i)
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = codewords[:]
            T *= cooling
            continue
        else:
            # c_old is essential. Check if c_new covers all missing words.
            # Use the precomputed boolean matrix for O(1) membership checks
            # Check: in_ball[c_new, missing] should all be True
            if len(missing) <= 20:  # Only try if missing set is small enough
                if np.all(in_ball[c_new, missing]):
                    # Accept swap
                    cov[balls[c_new]] += 1
                    codewords[i] = c_new
                    T *= cooling
                    continue
            # Reject swap, revert
            cov[balls[c_old]] += 1
            T *= cooling

    # Return the best code found
    result = [idx_to_digits(c) for c in best_codewords]
    return result
