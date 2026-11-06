import time
import random
from math import gcd

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 85
    n = limit

    # Precompute compatibility as bitmasks
    compat_mask = [0] * (n + 1)
    for i in range(1, n + 1):
        mask = 0
        for j in range(1, n + 1):
            if i * j // gcd(i, j) <= n:
                mask |= (1 << j)
        compat_mask[i] = mask

    def can_add(S_mask, x):
        return (S_mask & ~compat_mask[x]) == 0

    def complete(S_mask):
        for x in range(1, n + 1):
            if (S_mask & (1 << x)) == 0 and can_add(S_mask, x):
                S_mask |= (1 << x)
        return S_mask

    def mask_to_set(S_mask):
        return [i for i in range(1, n + 1) if (S_mask & (1 << i))]

    def popcount(x):
        return bin(x).count('1')

    def local_search(S_mask):
        S_mask = complete(S_mask)
        cur_size = popcount(S_mask)
        improved = True
        while improved and time.time() < deadline - 8:
            improved = False
            S_list = mask_to_set(S_mask)
            n_list = len(S_list)
            
            # 1-for-1 swaps
            for idx in range(n_list):
                if time.time() > deadline - 8:
                    break
                e = S_list[idx]
                S_minus = S_mask & ~(1 << e)
                for x in range(1, n + 1):
                    if x == e or (S_mask & (1 << x)):
                        continue
                    if can_add(S_minus, x):
                        S2 = complete(S_minus | (1 << x))
                        s2 = popcount(S2)
                        if s2 > cur_size:
                            S_mask = S2
                            cur_size = s2
                            improved = True
                            break

            # 1-for-2 swaps
            if time.time() < deadline - 12:
                for e in S_list:
                    if time.time() > deadline - 12:
                        break
                    S_minus = S_mask & ~(1 << e)
                    S2 = S_minus
                    added = 0
                    for x in range(1, n + 1):
                        if time.time() > deadline - 12:
                            break
                        if (S2 & (1 << x)) == 0 and can_add(S2, x):
                            S2 |= (1 << x)
                            added += 1
                            if added >= 2:
                                break
                    if popcount(S2) > cur_size:
                        S_mask = complete(S2)
                        cur_size = popcount(S_mask)
                        improved = True

        return S_mask

    best_mask = 0
    best_size = 0

    # Strategy 1: Greedy 1..n
    S = 0
    for x in range(1, n + 1):
        if can_add(S, x):
            S |= (1 << x)
    S = local_search(S)
    if popcount(S) > best_size:
        best_size = popcount(S)
        best_mask = S

    # Strategy 2: Greedy by compat count
    comp_counts = [popcount(compat_mask[i]) for i in range(n + 1)]
    order = sorted(range(1, n + 1), key=lambda x: -comp_counts[x])
    S = 0
    for x in order:
        if can_add(S, x):
            S |= (1 << x)
    S = local_search(S)
    if popcount(S) > best_size:
        best_size = popcount(S)
        best_mask = S

    # Strategy 3: Start with {1..k} for various k
    for k in range(18, 28):
        if time.time() > deadline - 25:
            break
        S = 0
        for x in range(1, k + 1):
            S |= (1 << x)
        S = complete(S)
        S = local_search(S)
        if popcount(S) > best_size:
            best_size = popcount(S)
            best_mask = S

    # Strategy 4: Random restarts
    rng = random.Random(98765)
    while time.time() < deadline - 8:
        S = 0
        seed_size = rng.randint(3, 12)
        candidates = list(range(1, 80))
        rng.shuffle(candidates)
        for x in candidates[:seed_size]:
            if can_add(S, x):
                S |= (1 << x)
        S = complete(S)
        S = local_search(S)
        if popcount(S) > best_size:
            best_size = popcount(S)
            best_mask = S

    return mask_to_set(best_mask)
