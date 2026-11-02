"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def _count_uncovered_tsubsets(blocks, v, t):
    """Return a set of t-subsets (as frozensets) not covered by any block."""
    covered = set()
    for b in blocks:
        for c in itertools.combinations(b, t):
            covered.add(frozenset(c))
    all_t = set(frozenset(c) for c in itertools.combinations(range(v), t))
    return all_t - covered


def _greedy_cover(v, k, t, rng):
    """Fast greedy construction to get a valid starting covering."""
    uncovered = list(itertools.combinations(range(v), t))
    uncovered_set = set(frozenset(c) for c in uncovered)
    blocks = []
    while uncovered_set:
        # Pick a random uncovered t-subset as seed
        seed = rng.choice(list(uncovered_set))
        block = list(seed)
        rng.shuffle(block)
        while len(block) < k:
            best_x = None
            best_gain = -1
            # Candidate points not in block
            cands = [x for x in range(v) if x not in block]
            rng.shuffle(cands)
            for x in cands:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(list(rest) + [x])
                    if fs in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)
        block.sort()
        blocks.append(tuple(block))
        # Remove covered t-subsets
        for c in itertools.combinations(block, t):
            uncovered_set.discard(frozenset(c))
    return blocks


def _block_covers(block, t):
    return [frozenset(c) for c in itertools.combinations(block, t)]


def construct(v, k, t):
    rng = random.Random(12345)
    start_time = time.time()
    time_limit = 110.0  # leave 10s buffer

    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng)

    # Precompute for speed: map from frozenset(t-subset) to list of block indices covering it
    def build_index(blk_list):
        idx = {}
        for i, b in enumerate(blk_list):
            for fs in _block_covers(b, t):
                idx.setdefault(fs, []).append(i)
        return idx

    index = build_index(blocks)

    def is_valid(blk_list):
        # Check every t-subset is covered by at least one block
        idx = build_index(blk_list)
        total_t = len(list(itertools.combinations(range(v), t)))
        return len(idx) == total_t

    # Local search: try removing blocks, then swapping
    best_blocks = list(blocks)
    best_count = len(blocks)

    def try_remove(blk_list):
        """Try to remove a redundant block. Returns new list or None."""
        idx = build_index(blk_list)
        # Find blocks that are the sole cover for some t-subset? No, we want to remove blocks
        # A block is removable if every t-subset it covers is also covered by another block.
        for i in range(len(blk_list)):
            b = blk_list[i]
            removable = True
            for fs in _block_covers(b, t):
                covers = idx[fs]
                # Need at least one other block
                if len(covers) <= 1:
                    removable = False
                    break
            if removable:
                new_list = blk_list[:i] + blk_list[i+1:]
                return new_list
        return None

    def try_swap(blk_list, new_block):
        """Replace a block with new_block, check validity."""
        for i in range(len(blk_list)):
            candidate = blk_list[:i] + [new_block] + blk_list[i+1:]
            idx = build_index(candidate)
            total_t = len(list(itertools.combinations(range(v), t)))
            if len(idx) == total_t:
                return candidate
        return None

    # Simulated annealing-ish local search
    # We'll do iterative improvement: repeatedly try to remove, then perturb
    iteration = 0
    while time.time() - start_time < time_limit:
        iteration += 1
        # Try to remove a redundant block
        removed = try_remove(blocks)
        if removed is not None:
            blocks = removed
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = list(blocks)
            continue

        # No removable block; try a perturbation: replace one block with a random one
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            # Generate a random k-subset
            new_block = tuple(sorted(rng.sample(range(v), k)))
            candidate = blocks[:i] + [new_block] + blocks[i+1:]
            # Check validity
            idx = build_index(candidate)
            total_t = len(list(itertools.combinations(range(v), t)))
            if len(idx) == total_t:
                blocks = candidate
                # After swap, try to remove again
                continue

        # If stuck, do a 2-swap: replace two blocks
        if len(blocks) >= 2:
            i, j = rng.sample(range(len(blocks)), 2)
            nb1 = tuple(sorted(rng.sample(range(v), k)))
            nb2 = tuple(sorted(rng.sample(range(v), k)))
            candidate = blocks[:]
            candidate[i] = nb1
            candidate[j] = nb2
            idx = build_index(candidate)
            total_t = len(list(itertools.combinations(range(v), t)))
            if len(idx) == total_t:
                blocks = candidate
                continue

        # Occasionally re-seed with a new greedy cover if we've been stuck
        if iteration % 50 == 0:
            new_cover = _greedy_cover(v, k, t, rng)
            if len(new_cover) < len(blocks):
                blocks = new_cover
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
