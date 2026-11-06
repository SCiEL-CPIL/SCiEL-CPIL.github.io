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
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if time.time() > deadline - 5:
                    return S_mask
                if (S_mask & (1 << x)) == 0 and can_add(S_mask, x):
                    S_mask |= (1 << x)
                    changed = True
        return S_mask

    def mask_to_set(S_mask):
        return [i for i in range(1, n + 1) if (S_mask & (1 << i))]

    def popcount(x):
        return bin(x).count('1')

    def local_search(S_mask):
        S_mask = complete(S_mask)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            S_list = mask_to_set(S_mask)
            current_size = popcount(S_mask)

            # 1-for-1 swaps
            for e in S_list:
                if time.time() > deadline - 10:
                    break
                S_minus = S_mask & ~(1 << e)
                for x in range(1, n + 1):
                    if time.time() > deadline - 10:
                        break
                    if x != e and (S_mask & (1 << x)) == 0 and can_add(S_minus, x):
                        S2 = complete(S_minus | (1 << x))
                        if popcount(S2) > current_size:
                            S_mask = S2
                            current_size = popcount(S2)
                            improved = True
                        break

            # 1-for-2 swaps
            if not improved and time.time() < deadline - 10:
                for e in S_list:
                    if time.time() > deadline - 10:
                        break
                    S_minus = S_mask & ~(1 << e)
                    S2 = S_minus
                    added = 0
                    for x in range(1, n + 1):
                        if time.time() > deadline - 10:
                            break
                        if (S2 & (1 << x)) == 0 and can_add(S2, x):
                            S2 |= (1 << x)
                            added += 1
                            if added >= 2:
                                break
                    if popcount(S2) > current_size:
                        S_mask = complete(S2)
                        current_size = popcount(S_mask)
                        improved = True

            # 2-for-3 swaps
            if not improved and time.time() < deadline - 20:
                for i in range(len(S_list)):
                    if time.time() > deadline - 20:
                        break
                    for j in range(i + 1, len(S_list)):
                        if time.time() > deadline - 20:
                            break
                        e1 = S_list[i]
                        e2 = S_list[j]
                        S_minus = S_mask & ~(1 << e1) & ~(1 << e2)
                        S2 = S_minus
                        added = 0
                        for x in range(1, n + 1):
                            if time.time() > deadline - 20:
                                break
                            if (S2 & (1 << x)) == 0 and can_add(S2, x):
                                S2 |= (1 << x)
                                added += 1
                                if added >= 3:
                                    break
                        if popcount(S2) > current_size:
                            S_mask = complete(S2)
                            current_size = popcount(S_mask)
                            improved = True
                            break
                    if improved:
                        break

        return S_mask

    best_mask = 0
    best_size = 0

    def try_strategy(S):
        nonlocal best_mask, best_size
        if time.time() > deadline - 10:
            return
        S = local_search(S)
        size = popcount(S)
        if size > best_size:
            best_size = size
            best_mask = S

    # Strategy 1: Greedy from small numbers
    S = 0
    for x in range(1, n + 1):
        if can_add(S, x):
            S |= (1 << x)
    try_strategy(S)

    # Strategy 2: Greedy by compatibility count (most compatible first)
    comp_counts = [bin(compat_mask[i]).count('1') for i in range(n + 1)]
    order = sorted(range(1, n + 1), key=lambda x: -comp_counts[x])
    S = 0
    for x in order:
        if time.time() > deadline - 30:
            break
        if can_add(S, x):
            S |= (1 << x)
    try_strategy(S)

    # Strategy 3: Force small numbers 1..k for various k
    for k in [10, 15, 20, 25, 30, 40, 50]:
        if time.time() > deadline - 15:
            break
        S = 0
        for x in range(1, k + 1):
            S |= (1 << x)
        # Try to add more
        for x in range(k + 1, n + 1):
            if time.time() > deadline - 15:
                break
            if can_add(S, x):
                S |= (1 << x)
        try_strategy(S)

    # Strategy 4: Random Restarts
    rng = random.Random(42)
    for _ in range(150):
        if time.time() > deadline - 10:
            break
        S = 0
        start_size = rng.randint(1, 10)
        for _ in range(start_size):
            x = rng.randint(1, n)
            S |= (1 << x)
        try_strategy(S)

    # Strategy 5: Random subsets of compatible small numbers
    for _ in range(100):
        if time.time() > deadline - 10:
            break
        S = 0
        # Pick random small numbers
        small_nums = list(range(1, 30))
        rng.shuffle(small_nums)
        for x in small_nums:
            if can_add(S, x):
                S |= (1 << x)
        # Then greedily add rest
        for x in range(1, n + 1):
            if time.time() > deadline - 10:
                break
            if (S & (1 << x)) == 0 and can_add(S, x):
                S |= (1 << x)
        try_strategy(S)

    # Final polish: try to add any remaining compatible elements
    if time.time() < deadline - 2:
        S = complete(best_mask)
        if popcount(S) > best_size:
            best_size = popcount(S)
            best_mask = S

    return mask_to_set(best_mask)
