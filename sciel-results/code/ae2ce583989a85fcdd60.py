"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def _build_sigma(n_max):
    """Compute sigma(n) for all n in [1, n_max] using a divisor-sum sieve."""
    sigma = [0] * (n_max + 1)
    for d in range(1, n_max + 1):
        for m in range(d, n_max + 1, d):
            sigma[m] += d
    return sigma

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    n_max = 10_000_000
    target_num, target_den = 3, 2

    # Compute sigma for all n up to n_max
    sigma = _build_sigma(n_max)

    # Group indices by sigma value
    groups = defaultdict(list)
    for n in range(1, n_max + 1):
        groups[sigma[n]].append(n)

    # For each group, find pair (a, b) minimizing |a/b - 3/2|
    # i.e., minimize |2a - 3b| / (2b)  =>  minimize |2a - 3b| / b
    # For a fixed b, best a is the one in the group closest to 3b/2
    # We'll iterate over groups and use two-pointer or binary search

    best_error = float('inf')
    best_pair = (11, 6)  # fallback seed

    import bisect

    for s_val, idxs in groups.items():
        if len(idxs) < 2:
            continue
        if time.time() > deadline - 5:
            break

        # idxs is sorted (since we appended in order)
        # For each b in idxs, find a in idxs closest to 1.5*b
        # But we want to minimize |a/b - 3/2| = |2a - 3b|/(2b)
        # For fixed b, minimize |2a - 3b|/b => minimize |2a - 3b| (since b fixed)
        # Actually the score is float(|a/b - 3/2|), so we minimize |a/b - 3/2|
        # = |2a - 3b| / (2b)

        # Strategy: for each b, binary search for a closest to 3b/2
        # But doing this for all b in all groups could be slow.
        # Alternative: for each group, sort and use two pointers.

        # Two-pointer approach:
        # We want to minimize |2a - 3b| / (2b) over a, b in idxs, a != b (or a can equal b? 
        # If a == b, then a/b = 1, error = |1 - 3/2| = 1/2. Not optimal generally.)
        # Actually a and b just need sigma(a) == sigma(b), they can be equal.
        # But if a == b, error is 1/2, which is worse than seed's 1/3. So skip a==b.

        # For efficiency, for each b, find a in idxs nearest to 1.5*b.
        # Use bisect.

        for b in idxs:
            target_a = 1.5 * b
            # Find insertion point in idxs for target_a
            pos = bisect.bisect_left(idxs, int(target_a))
            # Check pos and pos-1
            candidates = []
            if pos < len(idxs):
                candidates.append(idxs[pos])
            if pos > 0:
                candidates.append(idxs[pos - 1])

            for a in candidates:
                if a == b:
                    continue
                # Compute error = |a/b - 3/2| = |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                den = 2 * b
                if num == 0:
                    # a/b = 3/2 exactly, error = 0, invalid
                    continue
                err = num / den
                if err < best_error:
                    best_error = err
                    best_pair = (a, b)

    return [best_pair[0], best_pair[1]]
# EVOLVE-BLOCK-END
