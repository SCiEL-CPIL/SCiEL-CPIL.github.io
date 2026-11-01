"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations


def _build_bitsets(v, k, t):
    """Map every t-subset to an index and build a bitset for each k-subset (block).
    Returns (index_map, block_bits) where block_bits[b] is a Python int whose
    bit i is 1 iff t-subset i is contained in block b."""
    index_map = {}
    blocks = list(combinations(range(v), k))
    for b in blocks:
        for ts in combinations(b, t):
            index_map[ts] = len(index_map)
    n_t = len(index_map)
    block_bits = [0] * len(blocks)
    for bi, b in enumerate(blocks):
        mask = 0
        for ts in combinations(b, t):
            mask |= 1 << index_map[ts]
        block_bits[bi] = mask
    return index_map, block_bits, blocks


def _greedy(index_map, block_bits, blocks, randomize=False, seed=None):
    """Greedy set cover. Returns list of block indices."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    n_t = len(index_map)
    if n_t == 0:
        return []
    uncovered = (1 << n_t) - 1
    chosen = []
    while uncovered:
        # find block maximizing number of newly covered t-subsets
        best_gain = -1
        best_idx = -1
        candidates = range(len(block_bits))
        if randomize:
            candidates = list(candidates)
            rng.shuffle(candidates)
        for bi in candidates:
            g = bin(uncovered & block_bits[bi]).count('1')
            if g > best_gain:
                best_gain = g
                best_idx = bi
                if best_gain == 0:
                    break
        if best_gain <= 0:
            break
        chosen.append(best_idx)
        uncovered &= ~block_bits[best_idx]
    return chosen


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    index_map, block_bits, blocks = _build_bitsets(v, k, t)
    n_blocks = len(block_bits)

    # First pass: deterministic greedy
    chosen = _greedy(index_map, block_bits, blocks, randomize=False)
    best_count = len(chosen)
    best_chosen = chosen

    # Try randomized greedy with different seeds to potentially find shorter covers
    seed = 0
    while time.time() < deadline:
        chosen = _greedy(index_map, block_bits, blocks, randomize=True, seed=seed)
        if len(chosen) < best_count:
            best_count = len(chosen)
            best_chosen = chosen
        seed += 1
        # If we've hit the theoretical lower bound, stop
        if best_count <= 69:
            break

    return [list(blocks[i]) for i in best_chosen]
# EVOLVE-BLOCK-END
