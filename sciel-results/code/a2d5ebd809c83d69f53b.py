"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Precompute modular inverses for denominators 1 to 10 modulo 101
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, 101)

    # For each length k from 1 to 10, find all subsets of size k and their sums
    # best[r] = smallest subset (as tuple) that sums to r mod 101
    best = [None] * 101
    best_len = [1000] * 101

    # Generate subsets by increasing size to find minimum cardinality
    for k in range(1, 11):
        if time.time() > deadline:
            break
        for combo in combinations(range(1, 11), k):
            s = 0
            for x in combo:
                s = (s + inv[x]) % 101
            if best_len[s] > k:
                best_len[s] = k
                best[s] = combo
            # Early termination: if all residues are covered with length <= k
            if all(best_len[r] <= k for r in range(101)):
                break
        if all(best_len[r] <= k for r in range(101)):
            break

    # Construct the result
    result = []
    for r in range(101):
        if best[r] is None:
            # Fallback: find any subset that sums to r (should not happen)
            for mask in range(1, 1 << 10):
                s = 0
                for x in range(1, 11):
                    if mask & (1 << (x - 1)):
                        s = (s + inv[x]) % 101
                if s == r:
                    result.append([x for x in range(1, 11) if mask & (1 << (x - 1))])
                    break
            else:
                result.append([])
        else:
            result.append(list(best[r]))

    return result
# EVOLVE-BLOCK-END
