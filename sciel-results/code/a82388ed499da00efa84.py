"""Covering design C(24,7,3) via randomized greedy with bitset acceleration and redundancy reduction."""
import itertools
import time
import random
from collections import defaultdict


def _subset_to_idx(subset, v, t):
    """Map a t-subset (sorted tuple) to a unique integer index."""
    # Use combinatorial number system for compact indexing
    # For C(24,3) = 2024 subsets, we can just use a dict or compute index
    # For speed, precompute a lookup table
    pass


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Precompute all t-subsets and assign indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubsets)
    tsub_index = {s: i for i, s in enumerate(all_tsubsets)}

    # Precompute for each point, which (t-1)-subsets it can extend
    # For each point x, and each (t-1)-subset S, x extends S to S∪{x}
    # We'll use a different approach: for a partial block B, the gain of adding x
    # is the number of uncovered t-subsets of the form (t-1)-subset of B plus x

    # Precompute: for each (t-1)-subset, list of points that extend it to a t-subset
    # And for each point x, for each (t-1)-subset S not containing x, the t-subset index
    # This is C(24,2) * 22 = 276 * 22 = 6072 entries - manageable

    # Actually, let's use a bitset approach for speed
    # uncovered is a bitset of n_tsubs bits
    # For a partial block B, gain(x) = popcount(uncovered & mask[B][x])
    # where mask[B][x] is the set of t-subsets that contain x and a (t-1)-subset of B

    # Precompute masks: for each subset S of points (up to size t-1), and each point x not in S,
    # mask[S][x] = set of t-subset indices for t-subsets = S' ∪ {x} where S' ⊆ S, |S'| = t-1
    # This is too many. Instead, compute on the fly but cache.

    # Better approach: represent uncovered as a Python int (bitset)
    # For a partial block B (sorted), and candidate x:
    #   gain = number of (t-1)-subsets S of B such that S∪{x} is uncovered
    # We can precompute for each pair (S, x) the index, then sum over S in C(B, t-1)

    # Precompute: idx_of[(s0, s1, ..., s_{t-2}, x)] -> index, for sorted (t-1)-tuple + x
    # For t=3: idx_of[(a, b, x)] where a < b, x not in {a,b}
    # Number of entries: C(24,2) * 22 = 6072

    idx_map = {}
    for i, s in enumerate(all_tsubsets):
        # s is a sorted tuple of length t
        for j in range(t):
            # (t-1)-subset is s without s[j], and the added point is s[j]
            rest = s[:j] + s[j+1:]
            x = s[j]
            # Store: key = (rest, x) -> i
            # But rest is a (t-1)-tuple, x is a point
            idx_map[(rest, x)] = i

    def greedy_once(seed):
        """One greedy run with randomization."""
        rng = random.Random(seed)
        # uncovered as a set of indices for simplicity, or use bitset
        # For speed, use a set of indices
        uncovered = set(range(n_tsubs))
        blocks = []

        while uncovered:
            if time.time() > deadline:
                return None  # timeout

            # Pick a random uncovered t-subset to start
            start_tsub = all_tsubsets[rng.choice(sorted(uncovered))]
            block = list(start_tsub)
            rng.shuffle(block)
            block.sort()

            while len(block) < k:
                # Find best point to add
                best_gain = -1
                best_x = -1
                best_x2 = -1  # for tie-breaking randomness
                block_set = set(block)

                for x in range(v):
                    if x in block_set:
                        continue
                    # Compute gain: number of (t-1)-subsets of block that with x form an uncovered t-subset
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        idx = idx_map.get((rest, x), -1)
                        if idx != -1 and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        best_x2 = -1
                    elif gain == best_gain and gain > 0:
                        # Collect for random tie-breaking
                        if best_x2 == -1:
                            best_x2 = x

                # Random tie-breaking
                if best_x2 != -1 and rng.random() < 0.5:
                    best_x = best_x2

                if best_gain <= 0:
                    # No point adds coverage; add a random point
                    candidates = [x for x in range(v) if x not in block_set]
                    if candidates:
                        best_x = rng.choice(candidates)
                    else:
                        break

                block.append(best_x)
                block.sort()

            blocks.append(block)

            # Remove all covered t-subsets
            for tsub in itertools.combinations(block, t):
                idx = tsub_index[tsub]
                if idx in uncovered:
                    uncovered.discard(idx)

            if time.time() > deadline:
                return None

        return blocks

    def reduce_redundancy(blocks):
        """Remove redundant blocks (blocks whose t-subsets are all covered by other blocks)."""
        if not blocks:
            return blocks

        # Build coverage: for each t-subset index, which blocks cover it
        coverage = defaultdict(list)
        for bi, block in enumerate(blocks):
            for tsub in itertools.combinations(block, t):
                idx = tsub_index[tsub]
                coverage[idx].append(bi)

        # A block is redundant if for every t-subset in it, there's another block covering it
        keep = [True] * len(blocks)
        for bi, block in enumerate(blocks):
            redundant = True
            for tsub in itertools.combinations(block, t):
                idx = tsub_index[tsub]
                covering = coverage[idx]
                if len(covering) == 1:
                    redundant = False
                    break
            if redundant:
                keep[bi] = False

        new_blocks = [b for i, b in enumerate(blocks) if keep[i]]

        # Repeat until no more reduction
        if len(new_blocks) < len(blocks):
            return reduce_redundancy(new_blocks)
        return new_blocks

    def verify(blocks):
        """Check if all t-subsets are covered."""
        covered = set()
        for block in blocks:
            for tsub in itertools.combinations(block, t):
                covered.add(tsub_index[tsub])
        return len(covered) == n_tsubs

    # Run multiple greedy attempts with different seeds
    best_blocks = None
    best_size = float('inf')

    # First, try a few deterministic-ish seeds
    seeds = list(range(100))
    rng_master = random.Random(42)
    rng_master.shuffle(seeds)

    for seed in seeds:
        if time.time() > deadline - 10:
            break

        blocks = greedy_once(seed)
        if blocks is None:
            break

        # Try redundancy reduction
        reduced = reduce_redundancy(blocks)

        if len(reduced) < best_size:
            best_size = len(reduced)
            best_blocks = reduced

        # Early stop if we hit the known best
        if best_size <= 76:
            break

    # If we didn't find a valid covering, return best effort
    if best_blocks is not None:
        return best_blocks

    # Fallback: single greedy run
    blocks = greedy_once(0)
    if blocks is not None:
        return blocks

    # Ultimate fallback: return empty (will be invalid, but shouldn't happen)
    return []
