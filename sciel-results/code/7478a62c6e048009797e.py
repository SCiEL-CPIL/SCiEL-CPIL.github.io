import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # Precompute
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Precompute for each t-subset, which blocks cover it
    # sub_to_blocks[s] = set of block indices that cover t-subset s
    sub_to_blocks = [set() for _ in range(n_t)]
    block_masks = [0] * n_blocks

    for bi, b in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(b, t):
            idx = t_sub_to_idx[sub]
            mask |= (1 << idx)
            sub_to_blocks[idx].add(bi)
        block_masks[bi] = mask

    full_mask = (1 << n_t) - 1

    def greedy_cover(seed=42):
        if seed is not None:
            random.seed(seed)

        uncovered = full_mask
        selected = []
        selected_set = set()
        counts = [0] * n_t  # coverage count per t-subset

        while uncovered:
            best_idx = -1
            best_gain = 0

            for i in range(n_blocks):
                if i in selected_set:
                    continue
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i

            if best_idx == -1:
                break

            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~block_masks[best_idx]

            # Update counts
            m = block_masks[best_idx]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb

            if time.time() > deadline:
                return None

        return selected

    def prune_redundant(blocks):
        """Remove blocks that don't cover any uniquely-covered t-subset."""
        if not blocks:
            return blocks

        counts = [0] * n_t
        for b_idx in blocks:
            m = block_masks[b_idx]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb

        essential = [False] * len(blocks)
        for i, b_idx in enumerate(blocks):
            m = block_masks[b_idx]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if counts[idx] == 1:
                    essential[i] = True
                    break
                m ^= lsb

        return [blocks[i] for i in range(len(blocks)) if essential[i]]

    def try_2for1(blocks, max_tries=100):
        """Try to replace 2 blocks with 1. Returns new list or None."""
        if len(blocks) < 2:
            return None

        n = len(blocks)
        for _ in range(max_tries):
            if time.time() > deadline:
                return None

            i, j = random.sample(range(n), 2)
            b1, b2 = blocks[i], blocks[j]

            mask12 = block_masks[b1] | block_masks[b2]
            # Find what's uniquely covered by b1 or b2 (not covered by any other selected block)
            others_mask = 0
            for idx in range(n):
                if idx != i and idx != j:
                    others_mask |= block_masks[blocks[idx]]

            unique = mask12 & ~others_mask
            if unique == 0:
                # One is redundant
                return [b for idx, b in enumerate(blocks) if idx != i]

            # Find a single block that covers all of 'unique'
            for bi in range(n_blocks):
                if bi == b1 or bi == b2:
                    continue
                if (unique & ~block_masks[bi]) == 0:
                    new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                    new_blocks.append(bi)
                    return new_blocks

        return None

    def try_remove_redundant(blocks):
        """Remove any single block that's redundant."""
        n = len(blocks)
        if n == 0:
            return None

        # Build coverage counts
        counts = [0] * n_t
        for b_idx in blocks:
            m = block_masks[b_idx]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb

        # Check each block
        for i, b_idx in enumerate(blocks):
            m = block_masks[b_idx]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if counts[idx] == 1:
                    break  # This block is essential
                m ^= lsb
            else:
                # No unique subset, remove it
                return blocks[:i] + blocks[i+1:]

        return None

    def local_search(blocks):
        """Iterative local search: 2-for-1, redundant removal."""
        improved = True
        max_iters = 50
        it = 0

        while improved and time.time() < deadline and it < max_iters:
            improved = False
            it += 1

            # Try to remove redundant blocks
            new_b = try_remove_redundant(blocks)
            if new_b is not None:
                blocks = new_b
                improved = True
                continue

            # Try 2-for-1
            new_b = try_2for1(blocks, max_tries=50)
            if new_b is not None:
                blocks = new_b
                improved = True

        return blocks

    def random_greedy_with_ties(seed=0):
        """Greedy with random tie-breaking for diversity."""
        if seed is not None:
            random.seed(seed)

        uncovered = full_mask
        selected = []
        selected_set = set()

        while uncovered:
            best_gain = 0
            candidates = []

            for i in range(n_blocks):
                if i in selected_set:
                    continue
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)

            if not candidates:
                break

            chosen = random.choice(candidates)
            selected.append(chosen)
            selected_set.add(chosen)
            uncovered &= ~block_masks[chosen]

            if time.time() > deadline:
                return None

        return selected

    best_blocks = None
    best_count = float('inf')

    # Multiple restarts with different seeds
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    for s in seeds:
        if time.time() > deadline:
            break

        blocks = greedy_cover(seed=s)
        if blocks is None:
            continue

        blocks = prune_redundant(blocks)
        blocks = local_search(blocks)

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

        if best_count <= 330:
            break

    # Also try random-tie greedy
    for s in [7, 17, 27]:
        if time.time() > deadline:
            break

        blocks = random_greedy_with_ties(seed=s)
        if blocks is None:
            continue

        blocks = prune_redundant(blocks)
        blocks = local_search(blocks)

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

        if best_count <= 330:
            break

    if best_blocks is None:
        blocks = greedy_cover(seed=None)
        if blocks:
            return [list(all_blocks[i]) for i in blocks]
        return []

    return [list(all_blocks[i]) for i in best_blocks]
