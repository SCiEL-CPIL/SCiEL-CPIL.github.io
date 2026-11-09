import numpy as np
import time

def construct():
    t_start = time.time()
    deadline = t_start + 88.0

    N = 10_000_000

    # Step 1: Sieve for smallest prime factor (compact)
    spf = np.zeros(N + 1, dtype=np.int32)
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            if i * i <= N:
                for j in range(i * i, N + 1, i):
                    if spf[j] == 0:
                        spf[j] = i

    if time.time() > deadline - 30.0:
        return [1, 1]

    # Step 2: Precompute sigma using multiplicative property (needed for fast lookup)
    sigma_arr = np.ones(N + 1, dtype=np.int64)
    p_pow = np.ones(N + 1, dtype=np.int64)
    p_sum = np.ones(N + 1, dtype=np.int64)

    for n in range(2, N + 1):
        if time.time() > deadline - 25.0:
            break
        p = spf[n]
        m = n // p
        if m % p == 0:
            p_pow[n] = p_pow[m] * p
            p_sum[n] = p_sum[m] + p_pow[n]
            sigma_arr[n] = sigma_arr[m] * p_sum[n] // p_sum[m]
        else:
            p_pow[n] = p
            p_sum[n] = 1 + p
            sigma_arr[n] = sigma_arr[m] * (1 + p)

    if time.time() > deadline - 15.0:
        return [1, 1]

    # Step 3: Search for best pair (a, b) with sigma(a) == sigma(b)
    # Minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # For each b, target a ≈ 1.5*b
    # Use direct sigma lookup (no grouping needed)

    best_num = 0  # numerator of |2a-3b|
    best_den = 1  # denominator = 2*b
    best_a = 1
    best_b = 1

    # Iterate over b from large to small (larger b gives smaller error for same |2a-3b|)
    # For each b, check a values near 1.5*b
    # The error is |2a-3b|/(2b), so for a given |2a-3b|=d, error = d/(2b)
    # We want small d and large b

    # Check b from N down to 1, but skip if we can't beat current best
    # Current best error = best_num / best_den
    # For b, minimum possible error with |2a-3b|=1 is 1/(2b)
    # So if 1/(2b) >= best_num/best_den, skip this b
    # i.e., best_den >= 2*b*best_num, i.e., b <= best_den/(2*best_num)

    # Start with b from N down
    # For each b, compute target_a = round(1.5*b) = 3b/2
    # Check a in range [target_a - W, target_a + W] where W is a window
    # For each a, check if sigma(a) == sigma(b) and a != b

    # Adaptive window: start small, expand if no good match found
    # But we need to be time-efficient

    # Strategy: 
    # Phase 1: For b from N down to 1, check a near 1.5*b with small window
    # Phase 2: If time remains, expand window for promising b values

    # Let's estimate: checking one b takes ~2*W sigma lookups
    # With W=20, that's ~40 lookups per b
    # For 10M b values, that's 400M lookups - too many
    # Need to be smarter

    # Better: only check b where 3*b is even (so 1.5*b is integer)
    # i.e., b is even. Then target_a = 3*b//2 is exact integer.
    # For odd b, target_a = (3*b)//2 or (3*b+1)//2, giving |2a-3b|=1

    # Actually, let's just iterate b and check a few candidates.
    # Key insight: for even b, a = 3b/2 gives |2a-3b|=0 (invalid), 
    #   so we need a = 3b/2 ± 1, giving |2a-3b| = 2
    # For odd b, a = (3b±1)/2... wait, 3b is odd, so a must make 2a-3b small
    #   a = floor(3b/2) gives 2a-3b = -1, |...| = 1
    #   a = ceil(3b/2) gives 2a-3b = 1, |...| = 1

    # So for odd b, we can get error = 1/(2b) which is very small for large b!
    # The question is: does sigma(a) == sigma(b) for some such a?

    # Let's iterate b from N down to 1, and for each b check:
    #   - If b is even: a candidates = 3b//2 - 1, 3b//2 + 1, 3b//2 - 2, 3b//2 + 2, ...
    #   - If b is odd: a candidates = (3b-1)//2, (3b+1)//2, (3b-3)//2, (3b+3)//2, ...
    #   Check sigma equality for each candidate

    # To limit work, use a window that grows with b (larger b = more candidates to check
    # but each has smaller error, so we can afford fewer checks)

    # Actually, let's just use a fixed small window and iterate all b.
    # With window W=5 (check 5 candidates on each side), that's 10 lookups per b.
    # 10M * 10 = 100M lookups. Each numpy int64 lookup is fast.
    # Should take maybe 5-10 seconds.

    W = 10  # window size

    for b in range(N, 0, -1):
        if time.time() > deadline - 3.0:
            break

        sb = sigma_arr[b]

        # Determine target a
        if b % 2 == 0:
            target = 3 * b // 2  # exact, |2a-3b|=0, skip this
            # Check target ± 1, ± 2, ..., ± W
            for d in range(1, W + 1):
                for sign in (1, -1):
                    a = target + sign * d
                    if a < 1 or a > N or a == b:
                        continue
                    if sigma_arr[a] == sb:
                        num = abs(2 * a - 3 * b)
                        if num == 0:
                            continue
                        den = 2 * b
                        # Compare with global best
                        if best_num == 0 or num * best_den < best_num * den:
                            best_num = num
                            best_den = den
                            best_a = a
                            best_b = b
                        break  # found a match for this d, move to next d
        else:
            # b is odd, 3b is odd
            # a = (3b-1)//2 gives 2a = 3b-1, 2a-3b = -1
            # a = (3b+1)//2 gives 2a = 3b+1, 2a-3b = 1
            target_lo = (3 * b - 1) // 2
            target_hi = (3 * b + 1) // 2
            # Check target_lo - d and target_hi + d for d = 0, 1, 2, ...
            for d in range(0, W + 1):
                for a in (target_lo - d, target_hi + d):
                    if a < 1 or a > N or a == b:
                        continue
                    if sigma_arr[a] == sb:
                        num = abs(2 * a - 3 * b)
                        if num == 0:
                            continue
                        den = 2 * b
                        # Compare with global best
                        if best_num == 0 or num * best_den < best_num * den:
                            best_num = num
                            best_den = den
                            best_a = a
                            best_b = b
                        break  # found match for this d
                else:
                    continue
                break  # found a match, stop checking larger d

    if best_a == 1 and best_b == 1:
        return [1, 1]

    return [int(best_a), int(best_b)]
