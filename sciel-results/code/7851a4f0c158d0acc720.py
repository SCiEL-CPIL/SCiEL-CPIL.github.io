"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

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
    DEADLINE = 80  # safety margin before 90s kill

    MAX_SUM = 256
    is_prime = _sieve(MAX_SUM)

    def score_of(A, B):
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        n = len(distinct)
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(distinct[i], distinct[j]) > 1:
                    return 0
        return n

    # Fallback with known good score
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score_of(best_A, best_B)

    # Strategy 1: Arithmetic progressions A={a, a+d1, a+2d1}, B={b, b+d2, b+2d2}
    # with gcd(d1,d2)=1 ensures 9 distinct sums. Check all 9 are prime.
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

            # Offsets: i*d1 + j*d2 for i,j in {0,1,2}
            offsets = np.array([i * d1 + j * d2 for i in range(3) for j in range(3)])

            for a in range(1, a_max + 1):
                if time.time() - t_start > DEADLINE:
                    break
                # Vectorized check over all b values
                b_vals = np.arange(1, b_max + 1)
                # sums shape: (9, b_max)
                sums = a + b_vals[np.newaxis, :] + offsets[:, np.newaxis]

                # Ensure all sums are within [1, MAX_SUM]
                if sums.min() < 1 or sums.max() > MAX_SUM:
                    continue

                # Check if all 9 sums are prime for any b
                valid_b_mask = np.all(is_prime[sums], axis=0)

                if np.any(valid_b_mask):
                    b_idx = np.argmax(valid_b_mask)
                    b = int(b_vals[b_idx])
                    A = [a, a + d1, a + 2 * d1]
                    B = [b, b + d2, b + 2 * d2]
                    # Verify
                    s = score_of(A, B)
                    if s >= 9:
                        return [A, B]
                    elif s > best_score:
                        best_score = s
                        best_A = A
                        best_B = B

    # Strategy 2: Broader search if AP strategy didn't find 9
    if best_score < 9 and time.time() - t_start < DEADLINE - 5:
        # Try random sampling for high-score solutions
        import random
        rng = random.Random(42)
        for _ in range(100000):
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
