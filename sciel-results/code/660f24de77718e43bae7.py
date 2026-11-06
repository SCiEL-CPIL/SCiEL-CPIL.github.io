import time
import random
from math import gcd

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 75
    n = limit

    # Precompute compatibility as bitmasks
    # compat_mask[i] has bit j set if lcm(i, j) <= n
    # Note: bit 0 is unused, we use bits 1..n
    compat_mask = [0] * (n + 1)
    for i in range(1, n + 1):
        mask = 0
        for j in range(1, n + 1):
            if i * j // gcd(i, j) <= n:
                mask |= (1 << j)
        compat_mask[i] = mask

    # Helper: check if x can be added to S (represented as bitmask)
    # S is a bitmask. x is an integer.
    # x is compatible with S if compat_mask[x] covers all bits in S.
    # i.e., S & ~compat_mask[x] == 0
    def can_add(S_mask, x):
        return (S_mask & ~compat_mask[x]) == 0

    def complete(S_mask):
        # Greedily add all possible elements
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if time.time() > deadline - 5:
                    break
                if (S_mask & (1 << x)) == 0 and can_add(S_mask, x):
                    S_mask |= (1 << x)
                    changed = True
        return S_mask

    def mask_to_set(S_mask):
        return [i for i in range(1, n + 1) if (S_mask & (1 << i))]

    def local_search(S_mask):
        S_mask = complete(S_mask)
        current_size = bin(S_mask).count('1')
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            S_list = mask_to_set(S_mask)
            
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
                        s2_size = bin(S2).count('1')
                        if s2_size > current_size:
                            S_mask = S2
                            current_size = s2_size
                            improved = True
                        break

            # 1-for-2 swaps
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
                s2_size = bin(S2).count('1')
                if s2_size > current_size:
                    S_mask = complete(S2)
                    current_size = bin(S_mask).count('1')
                    improved = True

            # 2-for-3 swaps (heavier, but can jump local optima)
            # Only do this if we have time
            if time.time() < deadline - 20:
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
                        s2_size = bin(S2).count('1')
                        if s2_size > current_size:
                            S_mask = complete(S2)
                            current_size = bin(S_mask).count('1')
                            improved = True
                            break
                    if improved:
                        break

        return S_mask

    best_mask = 0
    best_size = 0

    # Strategy 0: Try divisor sets of highly composite numbers (known optimal candidates)
    # For n=512, numbers like 360, 420, 480, 504 have 24 divisors
    # All divisors of m have lcm <= m <= 512, so they form valid sets
    highly_composite_candidates = [360, 420, 480, 504, 240, 336, 384, 432, 468, 512]
    for m in highly_composite_candidates:
        if m > n:
            continue
        divisors = []
        for d in range(1, m + 1):
            if m % d == 0:
                divisors.append(d)
        # Verify all pairs have lcm <= 512 (should be true since lcm(a,b) | m for divisors)
        valid = True
        for a in divisors:
            for b in divisors:
                if a * b // gcd(a, b) > n:
                    valid = False
                    break
            if not valid:
                break
        if valid and len(divisors) > best_size:
            best_size = len(divisors)
            best_mask = 0
            for d in divisors:
                best_mask |= (1 << d)

    # Strategy 1: Greedy from small numbers
    S = 0
    for x in range(1, n + 1):
        if can_add(S, x):
            S |= (1 << x)
    S = local_search(S)
    if bin(S).count('1') > best_size:
        best_size = bin(S).count('1')
        best_mask = S

    # Strategy 2: Greedy by compatibility count (most compatible first)
    comp_counts = [bin(compat_mask[i]).count('1') for i in range(n + 1)]
    order = sorted(range(1, n + 1), key=lambda x: -comp_counts[x])
    S = 0
    for x in order:
        if time.time() > deadline - 30:
            break
        if can_add(S, x):
            S |= (1 << x)
    S = local_search(S)
    if bin(S).count('1') > best_size:
        best_size = bin(S).count('1')
        best_mask = S

    # Strategy 3: Random Restarts
    rng = random.Random(42)
    for _ in range(200):
        if time.time() > deadline - 10:
            break
        S = 0
        # Random start
        start_size = rng.randint(1, 10)
        for _ in range(start_size):
            x = rng.randint(1, n)
            S |= (1 << x)
        S = local_search(S)
        if bin(S).count('1') > best_size:
            best_size = bin(S).count('1')
            best_mask = S

    # Strategy 4: Specific seed based on analysis (1..23 + multiples)
    S = 0
    for x in range(1, 24):
        S |= (1 << x)
    # Try to add multiples of 23, 22, etc.
    for x in range(24, n + 1):
        if can_add(S, x):
            S |= (1 << x)
    S = local_search(S)
    if bin(S).count('1') > best_size:
        best_size = bin(S).count('1')
        best_mask = S

    # Strategy 5: Try to expand from the best divisor set found
    if best_size >= 20:
        S = best_mask
        # Try to add any remaining compatible elements
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for x in range(1, n + 1):
                if (S & (1 << x)) == 0 and can_add(S, x):
                    S |= (1 << x)
                    changed = True
        S = local_search(S)
        if bin(S).count('1') > best_size:
            best_size = bin(S).count('1')
            best_mask = S

    return mask_to_set(best_mask)
