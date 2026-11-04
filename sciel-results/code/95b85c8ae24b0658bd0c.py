import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin under 120s / 150s kill

    # Map every t-subset to an integer id; use a Python int as a bitset for covered.
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_index = {s: i for i, s in enumerate(all_subsets)}

    # For each point x, precompute a dict: key = tuple of (t-1) other points (sorted)
    # value = bitmask of t-subset indices containing x and those others.
    masks = [dict() for _ in range(v)]
    for idx, s in enumerate(all_subsets):
        bit = 1 << idx
        for i, x in enumerate(s):
            others = tuple(sorted(s[:i] + s[i+1:]))
            masks[x][others] = masks[x].get(others, 0) | bit

    covered = 0  # bitset of covered t-subsets
    blocks = []
    full_mask = (1 << n_subsets) - 1

    def count_bits(n):
        return n.bit_count() if hasattr(n, 'bit_count') else bin(n).count('1')

    def gain_of_add(x, B_list, uncov):
        """Compute number of new t-subsets covered by adding x to block B_list."""
        g_mask = 0
        for S in itertools.combinations(B_list, t - 1):
            key = tuple(sorted(S))
            m = masks[x].get(key)
            if m is not None:
                g_mask |= m
        return count_bits(g_mask & uncov)

    def build_block_from(seed_points, uncov):
        """Grow a block from initial set to size k by max-gain greedy."""
        B = list(seed_points)
        B_set = set(B)
        available = [x for x in range(v) if x not in B_set]
        while len(B) < k:
            best_x = -1
            best_gain = -1
            for x in available:
                g = gain_of_add(x, B, uncov)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x == -1:
                break
            B.append(best_x)
            B_set.add(best_x)
            available.remove(best_x)
        return B

    def total_new_coverage(block, uncov):
        """Count how many t-subsets in block are newly covered using mask."""
        total_mask = 0
        for s in itertools.combinations(sorted(block), t):
            idx = subset_to_index[s]
            total_mask |= 1 << idx
        return count_bits(total_mask & uncov)

    while True:
        if time.time() > deadline:
            break

        # Check if all covered
        uncov = full_mask & ~covered
        if uncov == 0:
            break

        # Collect uncovered subset indices (with a cap to limit seed enumeration)
        uncovered_indices = []
        m = uncov
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            uncovered_indices.append(idx)
            m ^= lsb
            if len(uncovered_indices) >= 300:
                break

        # Strong-greedy: try building a block from each uncovered seed,
        # pick the one with maximum total new coverage.
        best_block = None
        best_gain_total = -1

        for idx in uncovered_indices:
            if time.time() > deadline:
                break
            seed = all_subsets[idx]
            B = build_block_from(list(seed), uncov)
            total = total_new_coverage(B, uncov)
            if total > best_gain_total:
                best_gain_total = total
                best_block = B

        if best_block is None:
            break

        bl = sorted(best_block)
        # Mark covered
        for s in itertools.combinations(bl, t):
            covered |= 1 << subset_to_index[s]
        blocks.append(bl)

    return blocks
