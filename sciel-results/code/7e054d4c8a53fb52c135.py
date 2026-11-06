import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

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
    # prev[i][j] = index k such that the sequence ended with primes[k], primes[i] before primes[j]
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
        if time.time() > deadline:
            break

        for i in range(0, j):
            gap = primes[j] - primes[i]
            threshold = primes[i] - gap

            L = bisect.bisect_left(primes, threshold)

            if L < i:
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

        # Build suf_max[j] and suf_arg[j]
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

    if max_len <= 1:
        return [primes[0]] if N > 0 else []

    # Reconstruct: we have best_i, best_j as the last two elements
    # prev[best_i][best_j] gives the index k before best_i
    seq = [primes[best_j], primes[best_i]]
    cur_i = best_i
    cur_j = best_j
    while True:
        k = prev[cur_i][cur_j]
        if k == -1:
            break
        seq.append(primes[k])
        cur_j = cur_i
        cur_i = k

    seq.reverse()

    # Verify
    if len(seq) >= 2:
        gaps = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        valid = True
        for g in range(1, len(gaps)):
            if gaps[g] < gaps[g-1]:
                valid = False
                break
        if not valid:
            # Fallback: find longest valid prefix
            valid_seq = [seq[0]]
            last_gap = 0
            for p in seq[1:]:
                new_gap = p - valid_seq[-1]
                if new_gap >= last_gap:
                    valid_seq.append(p)
                    last_gap = new_gap
            return valid_seq

    return seq
