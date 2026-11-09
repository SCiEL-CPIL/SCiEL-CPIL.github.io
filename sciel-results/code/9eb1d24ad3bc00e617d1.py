import itertools

def construct():
    p = 101
    denominators = list(range(1, 11))
    n = 10

    # Precompute modular inverses
    inverses = {x: pow(x, -1, p) for x in denominators}

    # For each residue, find the minimum-size subset
    # Process by size: for size 1, 2, ..., 10
    # best[r] = minimum size that achieves residue r
    best_size = [n + 1] * p  # initialize to n+1 (impossible)
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

    # All residues should be covered (1023 subsets, 101 residues)
    # Construct result
    result = []
    for r in range(p):
        if best_combo[r] is not None:
            result.append(list(best_combo[r]))
        else:
            # Fallback (shouldn't happen)
            result.append(denominators[:])

    return result
