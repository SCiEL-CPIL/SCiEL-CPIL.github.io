"""Search-based program: find minimum max row length covering all residues mod 101."""
import time

# EVOLVE-BLOCK-START

def construct():
    t0 = time.time()
    deadline = 80.0  # safety margin before 90s kill

    P = 101
    D = list(range(1, 11))
    n = 10

    # Precompute modular inverses
    inv = [0] * (n + 1)
    for x in D:
        inv[x] = pow(x, -1, P)

    # For each residue, track the minimum-size subset that produces it
    # min_size[r] = smallest subset size achieving residue r
    # best_subset[r] = the actual list of denominators (smallest size)
    min_size = [11] * P  # 11 = infinity (max possible is 10)
    best_subset = [None] * P

    # Enumerate all non-empty subsets via bitmask
    for mask in range(1, 1 << n):
        size = 0
        residue = 0
        terms = []
        for i in range(n):
            if mask & (1 << i):
                x = i + 1
                size += 1
                residue = (residue + inv[x]) % P
                terms.append(x)

        if size < min_size[residue]:
            min_size[residue] = size
            best_subset[residue] = terms

    # Build result: row r gets best_subset[r]
    # If any residue is uncovered (should not happen with 1023 subsets),
    # fall back to using any available subset
    for r in range(P):
        if best_subset[r] is None:
            # This shouldn't happen, but handle gracefully
            best_subset[r] = D[:]  # use all denominators

    return [best_subset[r] for r in range(P)]

# EVOLVE-BLOCK-END
