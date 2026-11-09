import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    p = 101
    n = 10
    denominators = list(range(1, n + 1))

    # Precompute modular inverses for denominators 1..10
    inverses = [pow(x, -1, p) for x in denominators]

    # Total number of non-empty subsets
    total_masks = 1 << n

    # For each size k from 1 to n, collect all subsets of that size
    # and group by their residue (sum of inverses mod p)
    # best_for_residue[residue] = (size, mask) - track the smallest size

    # Start by checking size 1, then size 2, etc.
    # We want to minimize the maximum row length, so we need all residues
    # to be covered. We process sizes in increasing order.

    # First, let's find the minimum k such that all 101 residues are covered
    # by subsets of size <= k.

    # For each residue, find the smallest subset that achieves it
    best_for_residue = {}
    # We also need to know which residues are covered at each size level
    residues_covered_at_size = {k: set() for k in range(1, n + 1)}

    # Process subsets by increasing size
    for k in range(1, n + 1):
        if time.time() - start_time > DEADLINE:
            break

        for combo in combinations(range(n), k):
            mask = 0
            residue = 0
            for idx in combo:
                mask |= (1 << idx)
                residue = (residue + inverses[idx]) % p

            residues_covered_at_size[k].add(residue)

            if residue not in best_for_residue:
                best_for_residue[residue] = (k, mask)
            # If already covered, keep the smaller size (which is current k
            # since we process in increasing order, and k <= existing size)

    # Check which residues are still uncovered
    uncovered = [r for r in range(p) if r not in best_for_residue]
    if uncovered:
        # Fallback: assign all denominators to uncovered residues
        full_mask = total_masks - 1
        full_residue = sum(inverses) % p
        for r in uncovered:
            best_for_residue[r] = (n, full_mask)

    # Now, to minimize the maximum row length, we want to assign each residue
    # the smallest possible subset. But we can also try to reassign: if a residue
    # is covered by a subset of size k, we can use any subset of size <= k.
    # The key insight: we want to minimize max row length.
    # Since we process sizes in increasing order, best_for_residue already
    # has the minimum size for each residue.

    # Build result
    result = []
    for r in range(p):
        size, mask = best_for_residue[r]
        combo = []
        for i in range(n):
            if mask & (1 << i):
                combo.append(i + 1)
        result.append(combo)

    return result
