"""Cardinality-separated subset-sum set construction."""
import time

DEADLINE = 75.0


def _check_valid(dp, n):
    for k in range(n + 1):
        for k2 in range(k + 1, n + 1):
            if dp[k] & dp[k2]:
                return False
    return True


def construct():
    start = time.time()
    best = [1]
    best_n = 1

    # Greedy with smallest-first
    dp = [0] * 61
    dp[0] = 1
    current = []
    n = 0
    candidates = list(range(1, 257))

    while n < 60:
        if time.time() - start > DEADLINE:
            break
        added = False
        for v in candidates:
            if v in current:
                continue
            new_dp = dp[:]
            for k in range(n, 0, -1):
                new_dp[k] |= (new_dp[k - 1] << v)
            if _check_valid(new_dp, n + 1):
                dp = new_dp
                current.append(v)
                n += 1
                added = True
                if n > best_n:
                    best = list(current)
                    best_n = n
                break
        if not added:
            break

    return list(best)
