"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 110.0  # safety margin before 120s kill

    # Use bitmasks for speed
    # Each t-subset is represented as a bitmask of v bits
    # Each block is a bitmask of v bits

    # Precompute all t-subsets as masks
    all_t_subs = []
    for comb in itertools.combinations(range(v), t):
        mask = 0
        for p in comb:
            mask |= (1 << p)
        all_t_subs.append(mask)

    n_t_subs = len(all_t_subs)
    # uncovered: bitset of which t-subsets are still uncovered
    # Use a Python int as a big bitmask (n_t_subs bits)
    # Initially all uncovered
    uncovered_mask = (1 << n_t_subs) - 1

    # For each point x, precompute the list of t-subset indices that contain x
    # point_to_subs[x] = list of indices into all_t_subs
    point_to_subs = [[] for _ in range(v)]
    for i, mask in enumerate(all_t_subs):
        for x in range(v):
            if (mask >> x) & 1:
                point_to_subs[x].append(i)

    blocks = []
    blocks_set = set()

    while uncovered_mask != 0:
        if time.time() > DEADLINE:
            break

        # Find the first uncovered t-subset
        # Get the lowest set bit
        low_bit = uncovered_mask & (-uncovered_mask)
        uncovered_idx = low_bit.bit_length() - 1
        seed_comb = list(itertools.combinations(range(v), t))  # expensive but cached below
        # Actually let's just use the mask to get the subset
        seed_mask = all_t_subs[uncovered_idx]
        seed_points = [x for x in range(v) if (seed_mask >> x) & 1]

        # Build a block greedily starting from seed_points
        block = list(seed_points)
        block_mask = 0
        for p in block:
            block_mask |= (1 << p)

        # For each candidate point, compute how many uncovered t-subsets it would help cover
        # A t-subset S is covered by adding point x to block if:
        #   S is uncovered, S contains x, and S \ {x} is a subset of the current block
        #   i.e., S = {x} union R where R is a (t-1)-subset of block

        # Precompute: for the current block, which (t-1)-subsets are contained in it
        # A t-subset {x} + R where R subset block (|R| = t-1) is covered

        while len(block) < k:
            if time.time() > DEADLINE:
                break

            best_x = -1
            best_gain = -1

            # For each candidate point not in block
            for x in range(v):
                if (block_mask >> x) & 1:
                    continue

                # Count uncovered t-subsets that contain x and whose other t-1 points are in block
                # These are t-subsets S such that x in S and S \ {x} subset block
                gain = 0
                # Iterate over (t-1)-subsets of block
                if t >= 1:
                    for rest in itertools.combinations(block, t - 1):
                        # Check if the t-subset (rest + x) is uncovered
                        sub_mask = 0
                        for p in rest:
                            sub_mask |= (1 << p)
                        sub_mask |= (1 << x)
                        # Find index of this mask in all_t_subs
                        # We need a mapping from mask to index
                        # This is expensive to look up each time
                        # Alternative: use a dict
                        pass
                # This approach is too slow. Let's use a different strategy.
            break

        # Simpler greedy: use the original approach but with bitmask tracking
        # and a few optimizations
        block = list(seed_points)
        block_set = set(block)

        while len(block) < k:
            if time.time() > DEADLINE:
                break

            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Count how many uncovered t-subsets would be covered by adding x
                # A t-subset S is covered if:
                #   S is uncovered, x in S, and S \ {x} is a (t-1)-subset of block
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    # Build the t-subset rest + x
                    sub = tuple(sorted(rest + (x,)))
                    # Check if this t-subset is uncovered
                    # We need to check if it's in uncovered
                    # Compute its mask
                    m = 0
                    for p in sub:
                        m |= (1 << p)
                    # Find if this mask corresponds to an uncovered t-subset
                    # We can precompute a dict: mask -> index
                    pass
                # This is still slow. Let me use a completely different approach.
            break

        # Let me rewrite with a cleaner approach
        # Use the bitmask for uncovered, and for each candidate point, compute gain
        # by checking which uncovered t-subsets contain x and have their remaining
        # t-1 points in the current block.

        # Precompute a dict from mask to index for fast lookup
        mask_to_idx = {m: i for i, m in enumerate(all_t_subs)}

        block = list(seed_points)
        block_set = set(block)

        while len(block) < k:
            if time.time() > DEADLINE:
                break

            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                # For each (t-1)-subset of the current block, check if adding x
                # covers an uncovered t-subset
                if t >= 1:
                    for rest in itertools.combinations(block, t - 1):
                        sub = tuple(sorted(rest + (x,)))
                        m = 0
                        for p in sub:
                            m |= (1 << p)
                        idx = mask_to_idx.get(m, -1)
                        if idx >= 0 and (uncovered_mask >> idx) & 1:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x

            if best_x == -1:
                # Pick any point not in block
                for x in range(v):
                    if x not in block_set:
                        best_x = x
                        break

            if best_x >= 0:
                block.append(best_x)
                block_set.add(best_x)

        # Sort the block and check if it's new
        block.sort()
        block_tuple = tuple(block)
        if block_tuple not in blocks_set:
            blocks_set.add(block_tuple)
            blocks.append(block)

            # Update uncovered: remove all t-subsets contained in this block
            for sub in itertools.combinations(block, t):
                m = 0
                for p in sub:
                    m |= (1 << p)
                idx = mask_to_idx.get(m, -1)
                if idx >= 0:
                    uncovered_mask &= ~(1 << idx)

    return blocks
# EVOLVE-BLOCK-END
