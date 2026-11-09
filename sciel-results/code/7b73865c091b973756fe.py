"""Covering design C(v,k,t) via greedy set cover with efficient gain computation."""
# EVOLVE-BLOCK-START
import time
import random
import itertools


def construct(v, k, t):
    deadline = time.time() + 135.0
    rng = random.Random(42)

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # uncovered is a bytearray: 1 = still needs covering
    uncovered = bytearray(b'\x01' * n_tsubs)
    n_uncovered = n_tsubs

    # For each point x, precompute list of t-subset indices containing x
    # point_to_tsubs[x] = list of indices of t-subsets that contain x
    point_to_tsubs = [[] for _ in range(v)]
    for idx, ts in enumerate(all_tsubs):
        for x in ts:
            point_to_tsubs[x].append(idx)

    blocks = []

    def greedy_block(seed_tsub=None):
        """Build one block greedily, returning list of points."""
        if seed_tsub is not None:
            block = list(seed_tsub)
        else:
            # Pick the first uncovered t-subset as seed
            idx = 0
            while idx < n_tsubs and not uncovered[idx]:
                idx += 1
            if idx >= n_tsubs:
                return None
            block = list(all_tsubs[idx])

        # For the current partial block, track which (t-1)-subsets are "active"
        # i.e., which (t-1)-subsets of the block still need one more point to form a t-subset
        # We'll track the set of t-subset indices that are "almost covered" by the block
        # A t-subset is "almost covered" if it contains all but one point of the block
        # and that one point is not yet in the block.

        # More efficient: for each candidate point x, gain = number of uncovered t-subsets
        # that contain x and all other t-1 points from the current block.
        # We can compute this by: for each (t-1)-subset S of the current block,
        # the t-subsets containing S and x are those where the t-th element is x.
        # So gain(x) = sum over (t-1)-subsets S of block of [tsub_to_idx has S ∪ {x} and it's uncovered]

        block_set = set(block)

        while len(block) < k:
            # Compute gain for each candidate point
            best_x = -1
            best_gain = -1

            # For each (t-1)-subset of the current block, find which uncovered t-subsets extend it
            # and count per candidate point
            gains = {}
            for rest in itertools.combinations(block, t - 1):
                rest_set = set(rest)
                for x in range(v):
                    if x in rest_set:
                        continue
                    if x in block_set:
                        continue
                    ts = tuple(sorted(rest + (x,)))
                    idx = tsub_to_idx.get(ts, -1)
                    if idx >= 0 and uncovered[idx]:
                        gains[x] = gains.get(x, 0) + 1

            if not gains:
                # No point can cover any new t-subset; break
                break

            for x, g in gains.items():
                if g > best_gain or (g == best_gain and (best_x == -1 or rng.random() < 0.5)):
                    best_gain = g
                    best_x = x

            if best_x == -1:
                break

            block.append(best_x)
            block_set.add(best_x)

        return sorted(block)

    # Greedy set cover: repeatedly add the block that covers the most uncovered t-subsets
    # But constructing blocks greedily one at a time is the standard approach.
    # We'll use a randomized greedy approach: at each step, build a block seeded by a random
    # uncovered t-subset (or the first one), then pick the best extension.

    # Strategy: multiple passes. In each pass, build blocks until no more uncovered t-subsets.
    # Use randomization to try different orderings.

    best_blocks = None
    best_count = float('inf')

    while time.time() < deadline - 10.0:
        # Reset
        uncovered = bytearray(b'\x01' * n_tsubs)
        n_uncovered = n_tsubs
        blocks = []

        while n_uncovered > 0 and time.time() < deadline - 5.0:
            # Find an uncovered t-subset to seed the block
            # Use random sampling to find one
            seed_idx = rng.randint(0, n_tsubs - 1)
            # Walk forward from seed_idx to find an uncovered one
            attempts = 0
            while not uncovered[seed_idx] and attempts < 1000:
                seed_idx = (seed_idx + 1) % n_tsubs
                attempts += 1
            if not uncovered[seed_idx]:
                # Fallback: linear scan
                seed_idx = 0
                while seed_idx < n_tsubs and not uncovered[seed_idx]:
                    seed_idx += 1
                if seed_idx >= n_tsubs:
                    break

            seed_tsub = all_tsubs[seed_idx]
            block = greedy_block(seed_tsub)

            if block is None or len(block) < t:
                # Degenerate case; just take the seed
                block = sorted(seed_tsub)
                # Pad to k points if possible
                while len(block) < k:
                    x = rng.randint(0, v - 1)
                    if x not in block:
                        block.append(x)
                block = sorted(block)

            # Add the block and update uncovered
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(ts, -1)
                if idx >= 0 and uncovered[idx]:
                    uncovered[idx] = 0
                    n_uncovered -= 1

            blocks.append(block)

        # Check if we have a valid covering
        if n_uncovered == 0:
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = [list(b) for b in blocks]

            # Try local search: remove redundant blocks
            # For each block, check if removing it still leaves a valid covering
            i = 0
            while i < len(best_blocks):
                # Tentatively remove block i
                test_uncovered = bytearray(b'\x00' * n_tsubs)
                for j, b in enumerate(best_blocks):
                    if j == i:
                        continue
                    for ts in itertools.combinations(b, t):
                        idx = tsub_to_idx.get(ts, -1)
                        if idx >= 0:
                            test_uncovered[idx] = 1

                # Check if any t-subset is uncovered
                if not any(test_uncovered):
                    best_blocks.pop(i)
                    best_count = len(best_blocks)
                    # Don't increment i; check same index again
                else:
                    i += 1

            if time.time() > deadline - 20.0:
                break

    if best_blocks is None:
        # Fallback: use the last blocks we constructed
        if blocks:
            best_blocks = [list(b) for b in blocks]
        else:
            # Absolute fallback: brute force all k-subsets (will be huge, but must return something)
            # This is a last resort
            best_blocks = [list(c) for c in itertools.combinations(range(v), k)]

    return best_blocks
# EVOLVE-BLOCK-END
