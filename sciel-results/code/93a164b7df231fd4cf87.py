"""Covering design C(24,7,3) via randomized greedy with bitset acceleration and redundancy reduction."""
import itertools
import time
import random
from collections import defaultdict


def _subset_to_idx(subset, v, t):
    """Map a t-subset (sorted tuple) to a unique integer index."""
    pass


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Precompute all t-subsets and assign indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubsets)
    tsub_index = {s: i for i, s in enumerate(all_tsubsets)}

    # Precompute: for each (t-1)-subset (as a frozenset or tuple) and each point x not in it,
    # the bitset of t-subsets that contain x and that (t-1)-subset.
    # For t=3: (t-1)-subset is a 2-tuple. There are C(24,2)=276 such tuples.
    # For each, 22 points can extend it. Total: 276*22 = 6072 bitsets, each with 2024 bits.
    
    # Key: (a, b, x) where a<b, x not in {a,b} -> bitset int with bit i set if t-subset i = sorted({a,b,x})
    # We only need one bit per such triple (the t-subset {a,b,x}), so each mask has exactly 1 bit set.
    # Actually, for gain calculation: given partial block B and candidate x,
    # gain = number of (t-1)-subsets S of B such that S∪{x} is uncovered.
    # So we need: for each (t-1)-tuple S and point x not in S, a mask that has bit i set
    # iff the t-subset S∪{x} has index i. This is just a single bit.
    
    # Better: precompute idx_map[(rest_tuple, x)] = index, as before.
    # Then for gain: iterate over C(block, t-1) and check if idx is in uncovered.
    # With bitset: uncovered is an int. Check if bit idx is set: (uncovered >> idx) & 1.
    
    idx_map = {}
    for i, s in enumerate(all_tsubsets):
        for j in range(t):
            rest = s[:j] + s[j+1:]
            x = s[j]
            idx_map[(rest, x)] = i

    # Precompute for each point x, the list of (t-1)-tuples that don't contain x
    # and map to their index with x. This helps iterate candidates.
    # Actually, for the inner loop, we iterate over candidates x and compute gain.
    # The bottleneck is: for each x, iterate over C(block, t-1) tuples and check uncovered.
    
    # Alternative: precompute for each (t-1)-tuple rest, a dict from x to index.
    # rest_to_idx = {rest: {x: idx for x in range(v) if x not in rest}}
    rest_to_idx = {}
    for i, s in enumerate(all_tsubsets):
        for j in range(t):
            rest = s[:j] + s[j+1:]
            x = s[j]
            if rest not in rest_to_idx:
                rest_to_idx[rest] = {}
            rest_to_idx[rest][x] = i

    def greedy_once(seed):
        """One greedy run with randomization using bitset for uncovered."""
        rng = random.Random(seed)
        # uncovered as integer bitset: bit i set means t-subset i is uncovered
        uncovered = (1 << n_tsubs) - 1
        blocks = []

        while uncovered:
            if time.time() > deadline:
                return None

            # Pick a random uncovered t-subset to start
            # Find a random set bit in uncovered
            # Get list of set bit positions
            u = uncovered
            # Pick a random bit from the set bits
            # For efficiency, we can use: 
            #   positions = [i for i in range(n_tsubs) if (uncovered >> i) & 1]
            # But that's O(n_tsubs). Instead, sample:
            #   while True:
            #       i = rng.randrange(n_tsubs)
            #       if (uncovered >> i) & 1:
            #           break
            i = rng.randrange(n_tsubs)
            while not ((uncovered >> i) & 1):
                i = rng.randrange(n_tsubs)
            start_tsub = all_tsubsets[i]
            block = list(start_tsub)
            rng.shuffle(block)
            block.sort()

            while len(block) < k:
                # Find best point to add
                best_gain = -1
                best_x = -1
                best_x2 = -1
                block_set = set(block)
                
                # Precompute the (t-1)-subsets of block
                block_t1_subs = list(itertools.combinations(block, t - 1))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in block_t1_subs:
                        idx = idx_map.get((rest, x), -1)
                        if idx != -1 and ((uncovered >> idx) & 1):
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        best_x2 = -1
                    elif gain == best_gain and gain > 0:
                        if best_x2 == -1:
                            best_x2 = x

                if best_x2 != -1 and rng.random() < 0.5:
                    best_x = best_x2

                if best_gain <= 0:
                    candidates = [x for x in range(v) if x not in block_set]
                    if candidates:
                        best_x = rng.choice(candidates)
                    else:
                        break

                block.append(best_x)
                block.sort()

            blocks.append(block)

            # Remove all covered t-subsets from uncovered bitset
            for tsub in itertools.combinations(block, t):
                idx = tsub_index[tsub]
                uncovered &= ~(1 << idx)

            if time.time() > deadline:
                return None

        return blocks

    def reduce_redundancy(blocks):
        """Remove redundant blocks (blocks whose t-subsets are all covered by other blocks)."""
        if not blocks:
            return blocks

        coverage = defaultdict(list)
        for bi, block in enumerate(blocks):
            for tsub in itertools.combinations(block, t):
                idx = tsub_index[tsub]
                coverage[idx].append(bi)

        keep = [True] * len(blocks)
        for bi, block in enumerate(blocks):
            redundant = True
            for tsub in itertools.combinations(block, t):
                idx = tsub_index[tsub]
                if len(coverage[idx]) == 1:
                    redundant = False
                    break
            if redundant:
                keep[bi] = False

        new_blocks = [b for i, b in enumerate(blocks) if keep[i]]

        if len(new_blocks) < len(blocks):
            return reduce_redundancy(new_blocks)
        return new_blocks

    def verify(blocks):
        """Check if all t-subsets are covered."""
        covered_bits = 0
        for block in blocks:
            for tsub in itertools.combinations(block, t):
                idx = tsub_index[tsub]
                covered_bits |= (1 << idx)
        return covered_bits == ((1 << n_tsubs) - 1)

    # Run multiple greedy attempts with different seeds
    best_blocks = None
    best_size = float('inf')

    seeds = list(range(500))
    rng_master = random.Random(42)
    rng_master.shuffle(seeds)

    for seed in seeds:
        if time.time() > deadline - 5:
            break

        blocks = greedy_once(seed)
        if blocks is None:
            break

        reduced = reduce_redundancy(blocks)

        if len(reduced) < best_size:
            best_size = len(reduced)
            best_blocks = reduced

        if best_size <= 76:
            break

    if best_blocks is not None:
        return best_blocks

    blocks = greedy_once(0)
    if blocks is not None:
        return blocks

    return []
