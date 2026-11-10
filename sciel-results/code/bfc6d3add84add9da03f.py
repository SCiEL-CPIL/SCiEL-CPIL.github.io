"""Covering design C(19,8,5) with efficient greedy and local improvement."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

DEADLINE_SECONDS = 118.0


def construct(v, k, t):
    start_time = time.time()
    random.seed(42)

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    idx_map = {s: i for i, s in enumerate(all_t_subsets)}

    # Precompute for each block (k-subset), which t-subsets it covers
    # We'll use a dictionary mapping frozenset(block) -> list of t-subset indices

    def block_t_indices(block):
        """Return list of indices of t-subsets contained in block."""
        return [idx_map[tuple(sorted(comb))] for comb in itertools.combinations(block, t)]

    # Greedy construction
    uncovered = np.ones(n_t, dtype=bool)
    blocks = []

    def add_block(block):
        nonlocal uncovered
        block = sorted(block)
        indices = block_t_indices(block)
        uncovered[indices] = False
        blocks.append(block)
        return sum(1 for i in indices if uncovered[i] == False and True)  # just return count

    # Precompute gains more efficiently
    # For a partial block B and candidate x, gain = number of uncovered t-subsets S
    # such that S ⊆ B ∪ {x} and x ∈ S
    # This means S = R ∪ {x} where R ⊆ B, |R| = t-1

    def compute_gain_fast(block, x, uncovered):
        gain = 0
        for rest in itertools.combinations(block, t - 1):
            s = tuple(sorted(rest + (x,)))
            si = idx_map.get(s, -1)
            if si >= 0 and uncovered[si]:
                gain += 1
        return gain

    # Build initial covering with greedy
    while uncovered.any():
        if time.time() - start_time > DEADLINE_SECONDS - 20:
            break

        # Find an uncovered t-subset to seed
        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) == 0:
            break
        start_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
        start_subset = all_t_subsets[start_idx]

        # Greedily build a block
        block = list(start_subset)
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = compute_gain_fast(block, x, uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)

        block.sort()
        # Add if it covers at least one new t-subset
        indices = block_t_indices(block)
        new_coverage = sum(1 for i in indices if uncovered[i])
        if new_coverage > 0:
            add_block(block)

    # Local improvement: try to remove or modify blocks
    def recompute_uncovered(blocks_list):
        unc = np.ones(n_t, dtype=bool)
        for b in blocks_list:
            indices = block_t_indices(b)
            unc[indices] = False
        return unc

    # Try to remove redundant blocks
    improved = True
    max_iter = 5
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        if time.time() - start_time > DEADLINE_SECONDS - 10:
            break

        for bi in range(len(blocks)):
            if time.time() - start_time > DEADLINE_SECONDS - 10:
                break
            # Try removing block bi
            remaining = blocks[:bi] + blocks[bi+1:]
            unc = recompute_uncovered(remaining)
            if not unc.any():
                # Can remove this block
                del blocks[bi]
                improved = True
                break

    # Try swapping points in blocks to improve coverage
    if time.time() - start_time < DEADLINE_SECONDS - 15:
        for _ in range(3):
            if time.time() - start_time > DEADLINE_SECONDS - 15:
                break
            for bi in range(len(blocks)):
                if time.time() - start_time > DEADLINE_SECONDS - 15:
                    break
                block = list(blocks[bi])
                best_swap = None
                best_gain = 0
                for si in range(len(block)):
                    for x in range(v):
                        if x in block:
                            continue
                        new_block = block[:]
                        new_block[si] = x
                        new_block.sort()
                        # Compute how many new t-subsets this covers
                        new_indices = set(block_t_indices(new_block))
                        old_indices = set(block_t_indices(block))
                        # Count unique coverage
                        unc = recompute_uncovered(blocks)
                        new_unique = sum(1 for i in new_indices if unc[i])
                        old_unique = sum(1 for i in old_indices if unc[i])
                        if new_unique > old_unique:
                            gain = new_unique - old_unique
                            if gain > best_gain:
                                best_gain = gain
                                best_swap = (si, x, new_block)
                if best_swap:
                    si, x, new_block = best_swap
                    blocks[bi] = new_block

    # Final deduplication
    seen = set()
    unique_blocks = []
    for b in blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)

    return unique_blocks
# EVOLVE-BLOCK-END
