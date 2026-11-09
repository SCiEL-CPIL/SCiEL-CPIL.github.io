"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    n = 128

    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]

    def lcm_lookup(a, b):
        return lcm_table[a][b]

    # Precompute prime-exponent signature for each number (tuple of (p, exp))
    # and count of distinct primes, and "rarity"
    def factor_exp(x):
        d = {}
        t = x
        p = 2
        while p * p <= t:
            if t % p == 0:
                c = 0
                while t % p == 0:
                    t //= p
                    c += 1
                d[p] = c
            p += 1
        if t > 1:
            d[t] = 1
        return d

    signatures = {}
    for x in range(1, n + 1):
        signatures[x] = tuple(sorted(factor_exp(x).items()))

    # Count how many numbers share each signature
    from collections import Counter
    sig_count = Counter(signatures.values())
    rarity = {x: sig_count[signatures[x]] for x in range(1, n + 1)}

    # Distinct prime factor count
    distinct_pf = {x: len(factor_exp(x)) for x in range(1, n + 1)}

    def is_conflict(selected, x):
        """Check if adding x to selected creates a bad triple."""
        m = len(selected)
        for i in range(m):
            a = selected[i]
            lab = lcm_lookup(a, x)
            for j in range(i + 1, m):
                b = selected[j]
                if lcm_lookup(a, b) == lab and lcm_lookup(b, x) == lab:
                    return True
        return False

    def greedy(ordering):
        selected = []
        for x in ordering:
            if time.time() > deadline:
                break
            if not is_conflict(selected, x):
                selected.append(x)
        return selected

    def local_improve(selected):
        """Try to swap out a selected element for an unselected one to allow growth.
        Returns a (possibly larger) selected list."""
        selected = list(selected)
        sel_set = set(selected)
        unselected = [x for x in range(1, n + 1) if x not in sel_set]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # For each unselected x, try to add it after removing one selected y
            for x in unselected:
                if time.time() > deadline:
                    break
                for idx in range(len(selected)):
                    y = selected[idx]
                    trial = selected[:idx] + selected[idx+1:]
                    if not is_conflict(trial, x):
                        new_selected = trial + [x]
                        # Check if this new set is a superset in potential: just accept if same size (swap)
                        # We accept the swap; then re-evaluate for further additions
                        # Try to also add more unselected elements greedily
                        sel_set2 = set(new_selected)
                        for z in unselected:
                            if z == x or z in sel_set2:
                                continue
                            if not is_conflict(new_selected, z):
                                new_selected.append(z)
                                sel_set2.add(z)
                        if len(new_selected) > len(selected):
                            selected = new_selected
                            sel_set = sel_set2
                            unselected = [x for x in range(1, n + 1) if x not in sel_set]
                            improved = True
                            break
                if improved:
                    break
        return selected

    # Build orderings
    random.seed(42)
    orderings = []

    # Natural
    orderings.append(list(range(1, n + 1)))
    # Reverse
    orderings.append(list(range(n, 0, -1)))
    # By distinct prime factors ascending (fewer = safer)
    orderings.append(sorted(range(1, n + 1), key=lambda x: (distinct_pf[x], x)))
    # By rarity ascending (rarer signatures first)
    orderings.append(sorted(range(1, n + 1), key=lambda x: (rarity[x], x)))
    # By value descending
    orderings.append(sorted(range(1, n + 1), key=lambda x: -x))
    # By (distinct_pf, value desc)
    orderings.append(sorted(range(1, n + 1), key=lambda x: (distinct_pf[x], -x)))

    # Random shuffles
    for _ in range(300):
        perm = list(range(1, n + 1))
        random.shuffle(perm)
        orderings.append(perm)

    best_result = None
    best_size = 0

    for ordering in orderings:
        if time.time() > deadline:
            break
        selected = greedy(ordering)
        # Local improvement
        selected = local_improve(selected)
        if len(selected) > best_size:
            best_size = len(selected)
            best_result = list(selected)
            if best_size >= n:
                break

    # Safety
    if not best_result:
        best_result = [1]

    return best_result
# EVOLVE-BLOCK-END
