"""Covering design C(v,k,t) via randomized greedy with fast triple-table + peeling."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Precompute all t-subsets and map each to a compact index
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_count = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # For each point x and each (t-1)-subset, precompute which t-subsets contain x
    # We'll build a fast structure: for a candidate block B, gain(x) = number of uncovered t-subsets
    # that contain x and are subsets of B ∪ {x}.
    # More efficiently: for each x, for each (t-1)-sub of B, the t-sub is (t-1)-sub + x.
    # We need to count how many of those are still uncovered.

    # Represent uncovered as a set of indices for O(1) lookup
    uncovered = set(range(tsub_count))

    def gain_with_block(block_set, block_list):
        """Compute gain for each candidate x not in block_set."""
        gains = {}
        # For each (t-1)-subset of block_list
        for rest in itertools.combinations(block_list, t - 1):
            for x in range(v):
                if x in block_set:
                    continue
                ts = tuple(sorted(rest + (x,)))
                idx = tsub_to_idx.get(ts)
                if idx is not None and idx in uncovered:
                    gains[x] = gains.get(x, 0) + 1
        return gains

    def greedy_cover(seed=None):
        """Run randomized greedy covering. Returns list of blocks (as sorted lists)."""
        if seed is not None:
            random.seed(seed)
        unc = set(range(tsub_count))
        blocks = []
        while unc:
            # Pick a random uncovered t-subset to start the block
            first_ts = list(unc)[random.randrange(len(unc))]
            block = list(first_ts)
            block_set = set(block)
            while len(block) < k:
                gains = {}
                for rest in itertools.combinations(block, t - 1):
                    for x in range(v):
                        if x in block_set:
                            continue
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsub_to_idx.get(ts)
                        if idx is not None and idx in unc:
                            gains[x] = gains.get(x, 0) + 1
                if not gains:
                    # No gain; pick a random point not in block
                    candidates = [x for x in range(v) if x not in block_set]
                    if not candidates:
                        break
                    block.append(random.choice(candidates))
                    block_set.add(block[-1])
                else:
                    max_gain = max(gains.values())
                    best = [x for x, g in gains.items() if g == max_gain]
                    chosen = random.choice(best)
                    block.append(chosen)
                    block_set.add(chosen)
            block.sort()
            blocks.append(block)
            # Remove all t-subsets covered by this block
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                unc.discard(idx)
            if time.time() > deadline:
                break
        return blocks, len(unc) == 0

    def peel(blocks):
        """Remove redundant blocks. Returns reduced list."""
        # For each block, check if removing it still covers all t-subsets
        # Build coverage count for each t-subset
        cover_count = [0] * tsub_count
        block_tsubs = []
        for b in blocks:
            tss = []
            for ts in itertools.combinations(b, t):
                idx = tsub_to_idx[ts]
                cover_count[idx] += 1
                tss.append(idx)
            block_tsubs.append(tss)

        # Try to remove blocks in random order
        order = list(range(len(blocks)))
        random.shuffle(order)
        keep = [True] * len(blocks)
        for bi in order:
            if not keep[bi]:
                continue
            # Check if all t-subsets in this block are covered by other kept blocks
            removable = True
            for idx in block_tsubs[bi]:
                if cover_count[idx] <= 1:
                    removable = False
                    break
            if removable:
                keep[bi] = False
                for idx in block_tsubs[bi]:
                    cover_count[idx] -= 1
        return [blocks[i] for i in range(len(blocks)) if keep[i]]

    # Multi-start randomized greedy
    best_blocks = None
    best_count = float('inf')
    num_starts = 8
    for s in range(num_starts):
        if time.time() > deadline - 20:
            break
        blocks, valid = greedy_cover(seed=42 + s)
        if not valid:
            continue
        # Peeling
        blocks = peel(blocks)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
        # Verify coverage after peeling
        # (peeling guarantees validity by construction)

    if best_blocks is None:
        # Fallback: deterministic greedy
        blocks, valid = greedy_cover(seed=0)
        if not valid:
            # Should not happen, but safety
            return []
        best_blocks = blocks

    # Final deadline check
    if time.time() > deadline:
        return best_blocks

    return best_blocks
# EVOLVE-BLOCK-END
