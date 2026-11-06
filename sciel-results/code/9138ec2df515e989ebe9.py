import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    # 1. Generate primes <= 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for p in range(2, int(limit**0.5) + 1):
        if is_prime[p]:
            for multiple in range(p*p, limit + 1, p):
                is_prime[multiple] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    N = len(primes)

    if N < 2:
        return primes

    # dp[i][j] = max length of valid sequence ending with primes[i], primes[j] (i < j)
    # prev[i][j] = index k such that the sequence ended with primes[k], primes[i]
    dp = [[0] * N for _ in range(N)]
    prev = [[-1] * N for _ in range(N)]

    # suf_max[j][k] = max(dp[k][j], dp[k+1][j], ..., dp[j-1][j]) for k in [0, j-1]
    # suf_arg[j][k] = index of the argmax in that range
    suf_max = [None] * N
    suf_arg = [None] * N

    max_len = 1
    best_i = 0
    best_j = 0

    for j in range(1, N):
        # Time guard: check periodically
        if time.time() > deadline:
            break

        # Compute dp[i][j] for all i < j
        for i in range(0, j):
            gap = primes[j] - primes[i]
            # Condition for non-decreasing gaps:
            # If sequence ends with ..., primes[k], primes[i], primes[j],
            # then (primes[i] - primes[k]) <= (primes[j] - primes[i])
            # i.e., primes[k] >= primes[i] - gap = 2*primes[i] - primes[j]
            threshold = primes[i] - gap

            # Find L: smallest index in [0, i-1] such that primes[L] >= threshold
            L = bisect.bisect_left(primes, threshold)

            if L < i:
                # We need max dp[k][i] for k in [L, i-1]
                # This is stored in suf_max[i][L] (if suf_max[i] exists)
                if suf_max[i] is not None:
                    val = suf_max[i][L]
                    if val > 0:
                        dp[i][j] = val + 1
                        prev[i][j] = suf_arg[i][L]
                    else:
                        dp[i][j] = 2
                        prev[i][j] = -1
                else:
                    dp[i][j] = 2
                    prev[i][j] = -1
            else:
                dp[i][j] = 2
                prev[i][j] = -1

            if dp[i][j] > max_len:
                max_len = dp[i][j]
                best_i = i
                best_j = j

        # Build suf_max[j] and suf_arg[j] for future use
        # suf_max[j][k] = max(dp[k][j], dp[k+1][j], ..., dp[j-1][j])
        sm = [0] * j
        sa = [0] * j
        if j > 0:
            sm[j-1] = dp[j-1][j]
            sa[j-1] = j-1
            for k in range(j-2, -1, -1):
                if dp[k][j] > sm[k+1]:
                    sm[k] = dp[k][j]
                    sa[k] = k
                else:
                    sm[k] = sm[k+1]
                    sa[k] = sa[k+1]
        suf_max[j] = sm
        suf_arg[j] = sa

    # Reconstruct sequence
    if max_len <= 1:
        return [primes[0]] if N > 0 else []

    seq = [primes[best_j]]
    curr_i = best_i
    curr_j = best_j
    while curr_i != -1:
        seq.append(primes[curr_i])
        prev_i = prev[curr_i][curr_j]
        curr_j = curr_i
        curr_i = prev_i

    seq.reverse()

    # Verify the sequence has non-decreasing gaps
    if len(seq) >= 2:
        gaps = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        for g in range(1, len(gaps)):
            if gaps[g] < gaps[g-1]:
                # Fallback: return a valid shorter sequence
                # Find the longest valid prefix
                valid_seq = [seq[0]]
                last_gap = 0
                for p in seq[1:]:
                    new_gap = p - valid_seq[-1]
                    if new_gap >= last_gap:
                        valid_seq.append(p)
                        last_gap = new_gap
                return valid_seq

    return seq
