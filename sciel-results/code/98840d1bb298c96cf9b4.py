import itertools
import time

def construct():
    p = 101
    denominators = list(range(1, 11))
    n = 10
    deadline = time.time() + 85.0  # Safety margin before 90s limit

    # Precompute modular inverses
    inverses = [0] * (n + 1)
    for x in denominators:
        inverses[x] = pow(x, -1, p)

    # For each residue, find the smallest combination that achieves it
    best_size = [n + 1] * p
    best_combo = [None] * p

    for size in range(1, n + 1):
        for combo in itertools.combinations(denominators, size):
            if time.time() > deadline:
                break
            s = 0
            for d in combo:
                s += inverses[d]
            residue = s % p
            if best_size[residue] > size:
                best_size[residue] = size
                best_combo[residue] = combo

    # Build result
    result = []
    for r in range(p):
        if best_combo[r] is not None:
            result.append(list(best_combo[r]))
        else:
            # Fallback: use all denominators (shouldn't happen if search completes)
            result.append(denominators[:])

    return result
