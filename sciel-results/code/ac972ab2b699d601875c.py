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

    # Precompute: for each x, the list of numbers compatible with x (sorted)
    # Also precompute a "greedy order" for complete()
    def can_add(S_mask, x):
        return (S_mask & ~compat_mask[x]) == 0

    def complete(S_mask):
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if (S_mask & (1 << x)) == 0 and can_add(S_mask, x):
                    S_mask |= (1 << x)
                    changed = True
        return S_mask

    def mask_to_set(S_mask):
        return [i for i in range(1, n + 1) if (S_mask & (1 << i))]

    def popcount(m):
        return bin(m).count('1')

    def local_search(S_mask):
        S_mask = complete(S_mask)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            S_list = mask_to_set(S_mask)
            current_size = popcount(S_mask)

            # 1-for-1 swaps: remove e, try to add x, then complete
            for e in S_list:
                if time.time() > deadline - 5:
                    break
                S_minus = S_mask & ~(1 << e)
                for x in range(1, n + 1):
                    if time.time() > deadline - 5:
                        break
                    if x != e and (S_mask & (1 << x)) == 0 and can_add(S_minus, x):
                        S2 = complete(S_minus | (1 << x))
                        if popcount(S2) > current_size:
                            S_mask = S2
                            current_size = popcount(S_mask)
                            improved = True
                        break

            # 1-for-2 swaps: remove e, greedily add up to 2 new elements
            for e in S_list:
                if time.time() > deadline - 5:
                    break
                S_minus = S_mask & ~(1 << e)
                S2 = S_minus
                added = 0
                for x in range(1, n + 1):
                    if time.time() > deadline - 5:
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
            if time.time() < deadline - 15:
                S_list = mask_to_set(S_mask)
                for i in range(len(S_list)):
                    if time.time() > deadline - 15:
                        break
                    for j in range(i + 1, len(S_list)):
                        if time.time() > deadline - 15:
                            break
                        e1 = S_list[i]
                        e2 = S_list[j]
                        S_minus = S_mask & ~(1 << e1) & ~(1 << e2)
                        S2 = S_minus
                        added = 0
                        for x in range(1, n + 1):
                            if time.time() > deadline - 15:
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

    def update_best(S):
        nonlocal best_mask, best_size
        sz = popcount(S)
        if sz > best_size:
            best_size = sz
            best_mask = S

    # Strategy 1: Greedy from small numbers
    S = 0
    for x in range(1, n + 1):
        if can_add(S, x):
            S |= (1 << x)
    S = local_search(S)
    update_best(S)

    # Strategy 2: Greedy by compatibility count (most compatible first)
    comp_counts = [popcount(compat_mask[i]) for i in range(n + 1)]
    order = sorted(range(1, n + 1), key=lambda x: -comp_counts[x])
    S = 0
    for x in order:
        if time.time() > deadline - 40:
            break
        if can_add(S, x):
            S |= (1 << x)
    S = local_search(S)
    update_best(S)

    # Strategy 3: Structural seed - all numbers whose prime factors are <= 23
    # These are "23-smooth" numbers up to 512
    def is_smooth(x, max_prime):
        if x <= 1:
            return True
        d = 2
        while d * d <= x:
            while x % d == 0:
                if d > max_prime:
                    return False
                x //= d
            d += 1
        if x > 1 and x > max_prime:
            return False
        return True

    S = 0
    for x in range(1, n + 1):
        if is_smooth(x, 23):
            if can_add(S, x):
                S |= (1 << x)
    S = local_search(S)
    update_best(S)

    # Strategy 4: Seed with 1..k for various k, then complete
    for k in [16, 18, 20, 22, 24, 28, 32]:
        if time.time() > deadline - 30:
            break
        S = 0
        for x in range(1, k + 1):
            S |= (1 << x)
        # Complete greedily
        S = complete(S)
        S = local_search(S)
        update_best(S)

    # Strategy 5: Random Restarts with better initialization
    rng = random.Random(42)
    for trial in range(300):
        if time.time() > deadline - 5:
            break
        # Mix: start with a small structural core, then random additions
        core_size = rng.randint(3, 12)
        S = 0
        # Start with small numbers (more likely to be compatible)
        for _ in range(core_size):
            x = rng.randint(1, 64)
            if can_add(S, x):
                S |= (1 << x)
        # Add some random larger numbers
        for _ in range(rng.randint(0, 5)):
            x = rng.randint(1, n)
            if can_add(S, x):
                S |= (1 << x)
        S = complete(S)
        S = local_search(S)
        update_best(S)

    # Strategy 6: Random restarts with "divisor-rich" seeds
    for trial in range(100):
        if time.time() > deadline - 5:
            break
        S = 0
        # Pick a random "hub" number and include all its divisors
        hub = rng.randint(1, n)
        for d in range(1, hub + 1):
            if hub % d == 0 and can_add(S, d):
                S |= (1 << d)
        # Complete
        S = complete(S)
        S = local_search(S)
        update_best(S)

    return mask_to_set(best_mask)
