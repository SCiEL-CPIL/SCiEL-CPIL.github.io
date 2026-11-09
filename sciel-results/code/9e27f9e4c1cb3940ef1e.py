import itertools

def construct():
    p = 101
    denominators = list(range(1, 11))
    n = 10

    # Precompute modular inverses
    inverses = [0] * (n + 1)
    for x in denominators:
        inverses[x] = pow(x, -1, p)

    # best[r] = minimum size that achieves residue r
    best_size = [n + 1] * p
    best_combo = [None] * p

    for size in range(1, n + 1):
        for combo in itertools.combinations(denominators, size):
            s = 0
            for d in combo:
                s += inverses[d]
            residue = s % p
            if best_size[residue] > size:
                best_size[residue] = size
                best_combo[residue] = combo

    result = []
    for r in range(p):
        if best_combo[r] is not None:
            result.append(list(best_combo[r]))
        else:
            result.append(denominators[:])

    return result
