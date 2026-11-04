"""Covering design C(v,k,t) via strong-greedy max-gain block selection (bitset accelerated)."""
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin under 120s / 150s kill

    # Map every t-subset to an integer id; use a Python int as a bitset for covered.
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_index = {s: i for i, s in enumerate(all_subsets)}

    # pair_mask[x][a][b] = bitmask of t-subsets containing x, a, b (for t=3).
    # Generalize: for t-subsets, precompute mask of t-subsets containing a given (t-1)-tuple.
    # We'll build, for each point x, a dict from frozenset(other t-1 points) -> bitmask of
    # t-subsets containing x and those others.
    # For efficiency with t=3: mask3[x][a][b].
    # Build generic: masks[x] is a dict keyed by tuple of (t-1) other points -> bitmask.
    masks = [dict() for _ in range(v)]
    for idx, s in enumerate(all_subsets):
        bit = 1 << idx
        for i, x in enumerate(s):
            others = tuple(s[:i] + s[i+1:])
            masks[x][others] = masks[x].get(others, 0) | bit

    covered = 0  # bitset of covered t-subsets
    blocks = []

    # For a partial block (list/set B) and candidate x not in B, compute gain =
    # number of uncovered t-subsets that would become covered by adding x.
    # A t-subset {x} ∪ S (|S|=t-1, S ⊆ B) is newly covered iff it contains x and S ⊆ B.
    def gain_of_add(x, B_set):
        g = 0
        # iterate over (t-1)-subsets of B
        B_list = list(B_set)
        for S in itertools.combinations(B_list, t - 1):
            key = tuple(S)
            m = masks[x].get(key)
            if m is not None:
                g += bin(m & ~covered).count('1')
        return g

    def build_block_from(seed_points):
        # Grow a block from an initial set to size k by max-gain greedy.
        B = set(seed_points)
        while len(B) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in B:
                    continue
                g = gain_of_add(x, B)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x == -1:
                break
            B.add(best_x)
        return B

    while True:
        if time.time() > deadline:
            break

        # Find the best block overall: try seeding from each uncovered t-subset,
        # but to limit cost, evaluate a representative set of seeds.
        # Strong-greedy: pick the block with max total new coverage.
        # We enumerate candidate seeds = each uncovered t-subset (as the "forced" core),
        # grow each, and keep the best. To bound runtime, sample seeds.

        # Determine uncovered subsets
        uncov_mask = ((1 << n_subsets) - 1) & ~covered
        if uncov_mask == 0:
            break

        # Collect uncovered subset indices (bounded sample)
        # Find them by scanning bits
        uncovered_indices = []
        m = uncov_mask
        while m:
            # lowest set bit
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            uncovered_indices.append(idx)
            m ^= lsb
            if len(uncovered_indices) >= 400:  # sample cap
                break

        best_block = None
        best_gain_total = -1

        for idx in uncovered_indices:
            seed = all_subsets[idx]
            B = build_block_from(set(seed))
            # compute total new coverage of B
            total = 0
            for s in itertools.combinations(sorted(B), t):
                total += 1 if not (covered >> subset_to_index[s]) & 1 else 0
            if total > best_gain_total:
                best_gain_total = total
                best_block = B

        if best_block is None:
            break

        bl = sorted(best_block)
        # mark covered
        for s in itertools.combinations(bl, t):
            covered |= 1 << subset_to_index[s]
        blocks.append(bl)

    return blocks
