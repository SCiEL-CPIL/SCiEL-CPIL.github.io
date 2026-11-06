"""Program: find A, B (3 distinct ints in [1,128]) maximizing |A+B| with pairwise-coprime distinct sums."""
import itertools
import math
import time
import numpy as np

# EVOLVE-BLOCK-START
def _sieve(n):
    is_prime = np.ones(n + 1, dtype=bool)
    if n >= 0:
        is_prime[0] = False
    if n >= 1:
        is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:n + 1:i] = False
    return is_prime

def construct():
    t_start = time.time()
    DEADLINE = 75  # safety margin before 90s kill

    MAX_SUM = 256
    is_prime = _sieve(MAX_SUM)

    def score_of(A, B):
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        n = len(distinct)
        # Check pairwise coprime
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(distinct[i], distinct[j]) > 1:
                    return 0
        return n

    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score_of(best_A, best_B)

    max_d = 63  # since 2*d <= 127

    for d1 in range(1, max_d + 1):
        if time.time() - t_start > DEADLINE:
            break
        for d2 in range(1, max_d + 1):
            if math.gcd(d1, d2) != 1:
                continue
            if time.time() - t_start > DEADLINE:
                break
            
            a_max = 128 - 2 * d1
            b_max = 128 - 2 * d2
            if a_max < 1 or b_max < 1:
                continue
            
            offsets = np.array([i * d1 + j * d2 for i in range(3) for j in range(3)])
            
            for a in range(1, a_max + 1):
                if time.time() - t_start > DEADLINE:
                    break
                # Vectorized check for b
                # b ranges from 1 to b_max
                # sums = a + b + offsets
                # We need is_prime[sums] to be all True for some b
                # Create array of b values: 1, 2, ..., b_max
                b_vals = np.arange(1, b_max + 1)
                # sums shape: (9, b_max)
                sums = a + b_vals[np.newaxis, :] + offsets[:, np.newaxis]
                
                # Check bounds
                if sums.max() > MAX_SUM:
                    # If some sums are out of bounds, we need to mask them
                    # But since a <= 128-2d1 and b <= 128-2d2, max sum is 128+128=256.
                    # is_prime is size 257. So indices are safe.
                    pass
                
                # Check if all 9 are prime for any b
                # valid_b[j] is True if all 9 sums for b_vals[j] are prime
                # is_prime[sums] has shape (9, b_max)
                # We want: for all i, is_prime[sums[i, j]] is True
                # So: np.all(is_prime[sums], axis=0)
                valid_b_mask = np.all(is_prime[sums], axis=0)
                
                if np.any(valid_b_mask):
                    # Find the first valid b
                    b_idx = np.argmax(valid_b_mask)
                    b = b_vals[b_idx]
                    A = [a, a + d1, a + 2 * d1]
                    B = [b, b + d2, b + 2 * d2]
                    return [A, B]

    # Fallback random search
    import random
    rng = random.Random(42)
    if time.time() - t_start < DEADLINE - 5:
        for _ in range(50000):
            if time.time() - t_start > DEADLINE:
                break
            A = rng.sample(range(1, 129), 3)
            B = rng.sample(range(1, 129), 3)
            s = score_of(A, B)
            if s > best_score:
                best_score = s
                best_A = A
                best_B = B
                if s >= 9:
                    return [A, B]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
