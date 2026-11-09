"""Combine MITM parity insight with clean DP + parent reconstruction for min-term divisor sum."""
import time
from itertools import combinations

N = 5040
TARGET = 4979
DEADLINE = 50.0  # safety margin under the 60s limit


def _divisors(n):
    ds = []
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            ds.append(i)
            if i * i != n:
                ds.append(n // i)
    return ds


def construct():
    t0 = time.time()
    divs = _divisors(N)
    even = [d for d in divs if d % 2 == 0]
    odd = [d for d in divs if d % 2 == 1]

    # DP: dp[s] = (min_count, parent_sum, divisor_used)
    # We process all divisors, tracking the minimum number of terms to reach each sum.
    INF = float('inf')
    dp_count = [INF] * (TARGET + 1)
    dp_parent = [(-1, -1)] * (TARGET + 1)  # (prev_sum, divisor)
    dp_count[0] = 0

    # Sort divisors descending for better pruning potential
    all_divs = sorted(divs, reverse=True)

    for d in all_divs:
        if d > TARGET:
            continue
        # Update in reverse to avoid using same divisor twice
        for s in range(TARGET, d - 1, -1):
            if dp_count[s - d] + 1 < dp_count[s]:
                dp_count[s] = dp_count[s - d] + 1
                dp_parent[s] = (s - d, d)

    if time.time() - t0 > DEADLINE:
        # Fallback: greedy
        rem = TARGET
        chosen = []
        for d in all_divs:
            if d <= rem:
                chosen.append(d)
                rem -= d
        if rem != 0:
            raise RuntimeError("no solution found")
        return chosen

    if dp_count[TARGET] == INF:
        # Fallback: greedy
        rem = TARGET
        chosen = []
        for d in all_divs:
            if d <= rem:
                chosen.append(d)
                rem -= d
        if rem != 0:
            raise RuntimeError("no solution found")
        return chosen

    # Reconstruct the solution
    chosen = []
    s = TARGET
    while s > 0:
        prev_s, d = dp_parent[s]
        chosen.append(d)
        s = prev_s

    # Verify
    assert len(set(chosen)) == len(chosen), "divisors not distinct"
    assert sum(chosen) == TARGET, f"sum {sum(chosen)} != {TARGET}"
    assert all(N % d == 0 for d in chosen), "not divisors"
    assert len(chosen) <= 256, "too many divisors"

    return chosen


if __name__ == "__main__":
    r = construct()
    print("count:", len(r), "sum:", sum(r))
    print(r)
