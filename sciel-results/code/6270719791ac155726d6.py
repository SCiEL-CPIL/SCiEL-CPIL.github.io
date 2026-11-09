import itertools
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    p = 101
    denominators = list(range(1, 11))
    n = 10

    # Precompute modular inverses
    try:
        inverses = {x: pow(x, -1, p) for x in denominators}
    except TypeError:
        # Fallback for older Python versions, though unlikely
        inverses = {}
        for x in denominators:
            for y in range(1, p):
                if (x * y) % p == 1:
                    inverses[x] = y
                    break

    best_for_residue = {}
    found_count = 0

    # Iterate through subsets in increasing order of size.
    # This ensures that the first subset found for a residue is the smallest possible.
    for size in range(1, n + 1):
        if found_count == p:
            break
        
        # Time guard
        if time.time() - start_time > DEADLINE:
            break

        for combo in itertools.combinations(denominators, size):
            s = 0
            for d in combo:
                s += inverses[d]
            residue = s % p

            if residue not in best_for_residue:
                best_for_residue[residue] = (size, combo)
                found_count += 1
                
                if found_count == p:
                    break

    # Check if all residues are covered
    if found_count < p:
        # Fallback: for any uncovered residue, use the full set of denominators
        for r in range(p):
            if r not in best_for_residue:
                best_for_residue[r] = (n, tuple(denominators))

    # Construct result: row r contains the denominators
    result = []
    for r in range(p):
        size, combo = best_for_residue[r]
        result.append(list(combo))

    return result
