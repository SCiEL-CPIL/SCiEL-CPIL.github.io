import time
from math import gcd

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 75
    n = limit

    # Precompute LCM compatibility matrix
    compat = [[False] * (n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a, n + 1):
            l = (a * b) // gcd(a, b)
            c = l <= limit
            compat[a][b] = c
            compat[b][a] = c

    # Precompute divisors for each number
    divisors = [set() for _ in range(n + 1)]
    for d in range(1, n + 1):
        for m in range(d, n + 1, d):
            divisors[m].add(d)

    def full_set(M):
        """Given maximal elements M, return the full set (union of all divisors)."""
        S = set()
        for m in M:
            S.update(divisors[m])
        return S

    def can_add_maximal(M, x):
        """Check if x can be added to maximal set M (pairwise LCM compatible)."""
        for m in M:
            if not compat[x][m]:
                return False
        # Also check x doesn't divide any existing maximal (would not be maximal)
        # Actually for the representation, we want M to be an antichain.
        # But for simplicity, we allow any set and just compute full_set.
        # The key constraint is pairwise LCM <= limit.
        return True

    def is_valid_maximal_set(M):
        """Check all pairs in M have LCM <= limit."""
        M_list = list(M)
        for i in range(len(M_list)):
            for j in range(i + 1, len(M_list)):
                if not compat[M_list[i]][M_list[j]]:
                    return False
        return True

    def local_search_maximal(M):
        """Local search on maximal-element sets."""
        M = frozenset(M)
        if not is_valid_maximal_set(M):
            # Filter to only keep valid subset
            M_list = list(M)
            M_valid = set()
            for m in M_list:
                if all(compat[m][o] for o in M_valid):
                    M_valid.add(m)
            M = frozenset(M_valid)
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            S = full_set(M)
            current_size = len(S)
            
            # 1-for-1 swap: remove one maximal, try adding another
            for e in list(M):
                if time.time() > deadline - 10:
                    break
                M_minus = M - {e}
                for x in range(1, n + 1):
                    if time.time() > deadline - 10:
                        break
                    if x not in M and x != e and can_add_maximal(M_minus, x):
                        M2 = M_minus | {x}
                        if is_valid_maximal_set(M2):
                            S2 = full_set(M2)
                            if len(S2) > current_size:
                                M = M2
                                current_size = len(S2)
                                improved = True
                        break
            
            # 1-for-2 swap: remove one maximal, try adding two
            for e in list(M):
                if time.time() > deadline - 10:
                    break
                M_minus = M - {e}
                added = 0
                M2 = set(M_minus)
                for x in range(1, n + 1):
                    if time.time() > deadline - 10:
                        break
                    if x not in M2 and x != e and can_add_maximal(M2, x):
                        M2.add(x)
                        added += 1
                        if added >= 2:
                            break
                if len(M2) > len(M_minus):
                    if is_valid_maximal_set(M2):
                        S2 = full_set(M2)
                        S = full_set(M)
                        if len(S2) > len(S):
                            M = frozenset(M2)
                            improved = True
            
            # 2-for-2 swap: remove two maximals, try adding two
            M_list = list(M)
            for i in range(len(M_list)):
                if time.time() > deadline - 10:
                    break
                for j in range(i + 1, len(M_list)):
                    if time.time() > deadline - 10:
                        break
                    e1, e2 = M_list[i], M_list[j]
                    M_minus = M - {e1, e2}
                    added = 0
                    M2 = set(M_minus)
                    for x in range(1, n + 1):
                        if time.time() > deadline - 10:
                            break
                        if x not in M2 and x != e1 and x != e2 and can_add_maximal(M2, x):
                            M2.add(x)
                            added += 1
                            if added >= 2:
                                break
                    if len(M2) > len(M_minus):
                        if is_valid_maximal_set(M2):
                            S2 = full_set(M2)
                            S = full_set(M)
                            if len(S2) > len(S):
                                M = frozenset(M2)
                                improved = True
            
            # Try adding a new maximal element (if compatible)
            for x in range(1, n + 1):
                if time.time() > deadline - 10:
                    break
                if x not in M and can_add_maximal(M, x):
                    M2 = M | {x}
                    if is_valid_maximal_set(M2):
                        S2 = full_set(M2)
                        S = full_set(M)
                        if len(S2) > len(S):
                            M = M2
                            improved = True

        return full_set(M)

    best = set()

    # Seed strategies: start with various maximal-element sets
    seeds_maximal = []

    # Single highly composite numbers as maximal
    for m in [512, 480, 420, 360, 336, 288, 240, 210, 180, 168, 144, 120, 96, 72, 60, 48, 36, 24, 12, 6, 4, 2]:
        if m <= limit:
            seeds_maximal.append(frozenset({m}))

    # Pairs of compatible numbers
    for a in range(2, n + 1):
        for b in range(a + 1, n + 1):
            if compat[a][b]:
                seeds_maximal.append(frozenset({a, b}))
                if len(seeds_maximal) > 200:
                    break
        if len(seeds_maximal) > 200:
            break

    # Small ranges as maximals (each number is its own maximal)
    for upper in [4, 6, 8, 12, 16, 24, 32]:
        seeds_maximal.append(frozenset(range(1, upper + 1)))

    # Run local search on each seed
    for seed in seeds_maximal:
        if time.time() > deadline - 20:
            break
        S = local_search_maximal(seed)
        if len(S) > len(best):
            best = S

    # Random restarts on maximal-element representation
    import random
    rng = random.Random(42)
    for _ in range(100):
        if time.time() > deadline - 10:
            break
        # Random maximal set: pick random numbers that are pairwise compatible
        M = set()
        for x in range(1, n + 1):
            if rng.random() < 0.1:  # 10% chance to consider each number
                if all(compat[x][m] for m in M):
                    M.add(x)
        if len(M) > 0:
            S = local_search_maximal(frozenset(M))
            if len(S) > len(best):
                best = S

    # Final polish: try to add more elements to best
    if best:
        best_list = sorted(best)
        # Try adding any number that's compatible with all in best
        for x in range(1, n + 1):
            if x not in best and all(compat[x][y] for y in best):
                best.add(x)

    return sorted(best)
