"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    # Precompute gcd matrix for 1..128
    n = 128
    gcd_mat = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            gcd_mat[i][j] = gcd(i, j)
            gcd_mat[j][i] = gcd_mat[i][j]

    def is_bad_triple(a, b, c):
        """Check if three distinct elements have all three pairwise gcds equal."""
        g_ab = gcd_mat[a][b]
        g_ac = gcd_mat[a][c]
        g_bc = gcd_mat[b][c]
        return g_ab == g_ac == g_bc

    def can_add(selected, x):
        """Check if x can be added to selected without creating a bad triple."""
        for i in range(len(selected)):
            a = selected[i]
            for j in range(i + 1, len(selected)):
                b = selected[j]
                if is_bad_triple(a, b, x):
                    return False
        return True

    # Greedy construction with multiple orderings
    best_set = set()
    
    # Try multiple greedy strategies
    orderings = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
        # Primes first, then composites
        sorted(range(1, n + 1), key=lambda x: (0 if x > 1 and all(x % p != 0 for p in range(2, int(x**0.5)+1)) else 1, x)),
        # Even numbers last
        sorted(range(1, n + 1), key=lambda x: (x % 2, x)),
        # Odd numbers first
        sorted(range(1, n + 1), key=lambda x: (x % 2 == 0, x)),
    ]

    for order in orderings:
        if time.time() > deadline:
            break
        selected = []
        for x in order:
            if time.time() > deadline:
                break
            if can_add(selected, x):
                selected.append(x)
        if len(selected) > len(best_set):
            best_set = set(selected)

    # Local search: try to add elements not in best_set
    if time.time() < deadline:
        current_set = list(best_set)
        current_set.sort()
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, n + 1):
                if x in best_set:
                    continue
                if can_add(current_set, x):
                    current_set.append(x)
                    current_set.sort()
                    best_set = set(current_set)
                    improved = True
                    break

    # Swap-based local search: try removing one element and adding another
    if time.time() < deadline:
        current_set = list(best_set)
        current_set.sort()
        for _ in range(100):  # Limit iterations
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(current_set)):
                if time.time() > deadline:
                    break
                removed = current_set[i]
                temp_set = current_set[:i] + current_set[i+1:]
                for x in range(1, n + 1):
                    if x in best_set:
                        continue
                    if can_add(temp_set, x):
                        if x > removed:  # Prefer larger numbers? Not necessarily better
                            new_set = temp_set + [x]
                            new_set.sort()
                            best_set = set(new_set)
                            current_set = new_set
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break

    return list(best_set) if best_set else [1]
# EVOLVE-BLOCK-END
