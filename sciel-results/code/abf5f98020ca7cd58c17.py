"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    """
    Return a nonempty subset of [1,255] containing 255, with gcd(a,b)>1 for all distinct pairs.
    Maximize cardinality.
    """
    start_time = time.time()
    deadline = 55

    # Build candidate set
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    n = len(candidates)
    
    # Precompute incompatibility using numpy for speed
    import numpy as np
    compat = np.ones((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) == 1:
                compat[i, j] = False
                compat[j, i] = False

    idx_255 = candidates.index(255)

    def greedy_expand(core, all_indices):
        """Greedy expand from core, adding compatible elements in order of degree."""
        S = set(core)
        remaining = [i for i in all_indices if i not in S]

        def deg(x):
            c = 0
            for y in remaining:
                if y != x and compat[x, y]:
                    c += 1
            return c

        remaining.sort(key=lambda x: -deg(x))

        for x in remaining:
            if x in S:
                continue
            ok = True
            for y in S:
                if not compat[x, y]:
                    ok = False
                    break
            if ok:
                S.add(x)

        return S

    best_set = None
    best_size = 0

    # Strategy A: Greedy from {255}
    S_A = greedy_expand({idx_255}, list(range(n)))
    if len(S_A) > best_size:
        best_set = S_A
        best_size = len(S_A)

    # Strategy B: All multiples of 3
    core3 = {i for i in range(n) if candidates[i] % 3 == 0}
    S_B = greedy_expand(core3, list(range(n)))
    if len(S_B) > best_size:
        best_set = S_B
        best_size = len(S_B)

    # Strategy C: All multiples of 5
    core5 = {i for i in range(n) if candidates[i] % 5 == 0}
    S_C = greedy_expand(core5, list(range(n)))
    if len(S_C) > best_size:
        best_set = S_C
        best_size = len(S_C)

    # Strategy D: All multiples of 17
    core17 = {i for i in range(n) if candidates[i] % 17 == 0}
    S_D = greedy_expand(core17, list(range(n)))
    if len(S_D) > best_size:
        best_set = S_D
        best_size = len(S_D)

    if time.time() - start_time > deadline:
        return sorted(candidates[i] for i in best_set)

    # Local search on best_set with aggressive swaps
    S = set(best_set)
    improved = True
    while improved and (time.time() - start_time < deadline - 5):
        improved = False

        # Try adding any compatible element
        for i in range(n):
            if i in S:
                continue
            ok = True
            for y in S:
                if not compat[i, y]:
                    ok = False
                    break
            if ok:
                S.add(i)
                improved = True
                break

        if time.time() - start_time > deadline - 5:
            break

        # Try remove-one-add-many
        for rem in list(S):
            if rem == idx_255:
                continue
            S_t = S - {rem}
            for add in range(n):
                if add in S_t:
                    continue
                ok = True
                for y in S_t:
                    if not compat[add, y]:
                        ok = False
                        break
                if ok:
                    S_t.add(add)
            if len(S_t) > len(S):
                S = S_t
                improved = True
                break

    if len(S) > best_size:
        best_set = S
        best_size = len(S)

    if time.time() - start_time > deadline:
        return sorted(candidates[i] for i in best_set)

    # Random restarts
    import random
    random.seed(777)
    for _ in range(40):
        if time.time() - start_time > deadline - 3:
            break

        core = {idx_255}
        others = [i for i in range(n) if i != idx_255]
        random.shuffle(others)
        for i in others[:random.randint(0, 3)]:
            ok = True
            for y in core:
                if not compat[i, y]:
                    ok = False
                    break
            if ok:
                core.add(i)

        S_r = greedy_expand(core, list(range(n)))

        imp = True
        while imp and (time.time() - start_time < deadline - 3):
            imp = False
            for i in range(n):
                if i in S_r:
                    continue
                ok = True
                for y in S_r:
                    if not compat[i, y]:
                        ok = False
                        break
                if ok:
                    S_r.add(i)
                    imp = True
                    break
            if time.time() - start_time > deadline - 3:
                break
            for rem in list(S_r):
                if rem == idx_255:
                    continue
                S_t = S_r - {rem}
                for add in range(n):
                    if add in S_t:
                        continue
                    ok = True
                    for y in S_t:
                        if not compat[add, y]:
                            ok = False
                            break
                    if ok:
                        S_t.add(add)
                        break
                if len(S_t) > len(S_r):
                    S_r = S_t
                    imp = True
                    break

        if len(S_r) > best_size:
            best_set = S_r
            best_size = len(S_r)

    return sorted(candidates[i] for i in best_set)

# EVOLVE-BLOCK-END
