# EVOLVE-BLOCK-START
import time

def _divisors(n):
    ds = []
    for d in range(1, int(n ** 0.5) + 1):
        if n % d == 0:
            ds.append(d)
            if d * d != n:
                ds.append(n // d)
    return ds

def _greedy(n, target):
    rem = target
    chosen = []
    for d in range(n, 0, -1):
        if n % d == 0 and d <= rem:
            chosen.append(d)
            rem -= d
    return chosen if rem == 0 else None

def construct():
    start = time.time()
    deadline = start + 75.0  # safety margin under 90s wall-clock

    n = 5040
    target = 4979
    divs = sorted(_divisors(n), reverse=True)
    D = len(divs)

    best = _greedy(n, target)
    best_k = len(best) if best is not None else D + 1

    # Meet-in-the-middle: enumerate all subset sums for half the divisors.
    half = D // 2
    A = divs[:half]
    B = divs[half:]
    nA = len(A)
    nB = len(B)

    # Build (sum, bitmask) list for B (smaller half).
    B_items = [(0, 0)]
    for i in range(nB):
        v = B[i]
        cur = B_items
        B_items = cur + [(s + v, m | (1 << i)) for s, m in cur]

    # Group B by sum, keeping the minimum popcount mask per sum.
    from collections import defaultdict
    b_by_sum = {}
    for s, m in B_items:
        pc = m.bit_count()
        old = b_by_sum.get(s)
        if old is None or pc < old[0]:
            b_by_sum[s] = (pc, m)
    b_sums = sorted(b_by_sum.keys())

    # Iterate A subsets; for each, find a B subset whose sum completes target,
    # minimizing total count (popcount_A + popcount_B).
    nA_sub = 1 << nA
    for am in range(nA_sub):
        if time.time() > deadline:
            break
        # compute sum and popcount of A subset
        sA = 0
        pcA = 0
        x = am
        while x:
            low = x & (-x)
            idx = low.bit_length() - 1
            sA += A[idx]
            pcA += 1
            x ^= low
        if sA > target:
            continue
        need = target - sA
        if need not in b_by_sum:
            continue
        pcB, bm = b_by_sum[need]
        total = pcA + pcB
        if total < best_k:
            # reconstruct
            chosen = []
            x = am
            while x:
                low = x & (-x)
                idx = low.bit_length() - 1
                chosen.append(A[idx])
                x ^= low
            x = bm
            while x:
                low = x & (-x)
                idx = low.bit_length() - 1
                chosen.append(B[idx])
                x ^= low
            if len(set(chosen)) == total and sum(chosen) == target:
                best = chosen
                best_k = total
                if best_k == 1:
                    break

    # Deadline guard: return best valid result found.
    if best is not None:
        return best
    # Fallback: recompute greedy to guarantee a valid answer.
    return _greedy(n, target)
# EVOLVE-BLOCK-END
