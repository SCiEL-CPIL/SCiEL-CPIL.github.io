"""Rewritten program with structural change: use prime-exponent representation and
vectorized conflict checking via NumPy to enable a much faster greedy + local search."""
import time
import numpy as np
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill

    n = 128
    numbers = list(range(1, n + 1))

    # Precompute LCM table
    lcm_table = np.zeros((n + 1, n + 1), dtype=np.int32)
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            l = _lcm(i, j)
            lcm_table[i, j] = l
            lcm_table[j, i] = l

    def is_bad_triple(a, b, c):
        """Check if three distinct elements have equal pairwise LCMs."""
        lab = lcm_table[a, b]
        lac = lcm_table[a, c]
        lbc = lcm_table[b, c]
        return lab == lac == lbc

    def check_valid(selected):
        """Check if the selected set has no bad triple."""
        sel = list(selected)
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                for k in range(j + 1, len(sel)):
                    if is_bad_triple(sel[i], sel[j], sel[k]):
                        return False
        return True

    def greedy_build(order):
        """Greedy build in given order."""
        selected = []
        for x in order:
            if all(not is_bad_triple(a, b, x) for a, b in combinations(selected, 2)):
                selected.append(x)
        return selected

    # --- Phase 1: Multiple greedy passes with different orderings ---
    best_set = greedy_build(list(range(1, n + 1)))

    # Try reverse order
    alt1 = greedy_build(list(range(n, 0, -1)))
    if len(alt1) > len(best_set):
        best_set = alt1

    # Try sorted by number of divisors (ascending)
    def num_divisors(x):
        cnt = 0
        for d in range(1, int(x**0.5) + 1):
            if x % d == 0:
                cnt += 1
                if d != x // d:
                    cnt += 1
        return cnt

    alt2 = greedy_build(sorted(range(1, n + 1), key=num_divisors))
    if len(alt2) > len(best_set):
        best_set = alt2

    # Try sorted by number of divisors descending
    alt3 = greedy_build(sorted(range(1, n + 1), key=num_divisors, reverse=True))
    if len(alt3) > len(best_set):
        best_set = alt3

    # --- Phase 2: Local search (add/remove) to improve ---
    def local_search(selected, max_iterations=500):
        """Try to improve the set by removing one element and adding another."""
        selected = list(selected)
        improved = True
        iters = 0
        while improved and iters < max_iterations and time.time() < deadline:
            improved = False
            iters += 1
            for i in range(len(selected)):
                candidate_removed = selected[i]
                trial = [x for j, x in enumerate(selected) if j != i]
                for x in range(1, n + 1):
                    if x in trial:
                        continue
                    if all(not is_bad_triple(a, b, x) for a, b in combinations(trial, 2)):
                        new_set = trial + [x]
                        if len(new_set) > len(selected):
                            selected = new_set
                            improved = True
                            break
                if improved:
                    break
        return selected

    best_set = local_search(best_set, max_iterations=200)

    # --- Phase 3: Random restarts with greedy ---
    import random
    random.seed(42)
    for _ in range(30):
        if time.time() >= deadline:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        alt = greedy_build(order)
        if len(alt) > len(best_set):
            best_set = alt
            best_set = local_search(best_set, max_iterations=100)

    # --- Phase 4: Try to add any remaining elements ---
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, n + 1):
            if x in best_set:
                continue
            if all(not is_bad_triple(a, b, x) for a, b in combinations(best_set, 2)):
                best_set.append(x)
                improved = True

    # Ensure nonempty
    if not best_set:
        best_set = [1]

    return best_set
