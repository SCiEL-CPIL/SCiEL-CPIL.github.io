"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import Counter

PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23)
MAX_N = 1000000

def construct():
    t0 = time.time()
    deadline = t0 + 85.0

    # Precompute the 23-smooth part for numbers 1..MAX_N
    # smooth[x] = product of p^v_p(x) for p in PRIMES
    size = MAX_N + 1
    smooth = [1] * size

    for p in PRIMES:
        for mult in range(p, size, p):
            val = mult
            while val % p == 0:
                smooth[mult] *= p
                val //= p

    # Slide a window of 24 over smooth[1001..1000000]
    # For n in [1000, 999976], look at smooth[n+1]..smooth[n+24]
    n_end = 999976  # n+24 <= 1000000

    # Initialize window for n=1000
    window = smooth[1001:10025]
    cnt = Counter(window)
    best_score = len(cnt)
    best_n = 1000

    # Slide window
    for n in range(1001, n_end + 1):
        if time.time() >= deadline:
            break
        old_val = smooth[n]
        cnt[old_val] -= 1
        if cnt[old_val] == 0:
            del cnt[old_val]

        new_val = smooth[n + 24]
        cnt[new_val] = cnt.get(new_val, 0) + 1

        cur_score = len(cnt)
        if cur_score < best_score:
            best_score = cur_score
            best_n = n
            if best_score == 1:
                break

    return best_n
# EVOLVE-BLOCK-END
