import itertools
import time
import random
import heapq
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Precompute all t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)

    # Map t-subset tuple to index
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Precompute bitmask for each block
    block_masks = [0] * n_blocks
    for i, b in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(b, t):
            idx = t_sub_to_idx[sub]
            mask |= (1 << idx)
        block_masks[i] = mask

    # Full mask of all t-subsets
    full_mask = (1 << n_t_subsets) - 1

    def greedy_cover_linear(seed=None):
        """Linear scan greedy - simpler and possibly faster for this size"""
        if seed is not None:
            random.seed(seed)

        uncovered = full_mask
        selected = []
        selected_set = set()

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
                elif gain == best_gain and gain > 0:
                    if random.random() < 0.3:
                        best_idx = i

            if best_idx == -1:
                break

            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~block_masks[best_idx]

            if time.time() > deadline:
                return None

        return selected

    def prune_redundant(blocks):
        """Remove blocks that are not essential."""
        if not blocks:
            return blocks

        counts = [0] * n_t_subsets

        for b_idx in blocks:
            mask = block_masks[b_idx]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb

        essential = [False] * len(blocks)
        for i, b_idx in enumerate(blocks):
            mask = block_masks[b_idx]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if counts[idx] == 1:
                    essential[i] = True
                    break
                m ^= lsb

        return [blocks[i] for i in range(len(blocks)) if essential[i]]

    def local_search_2for1(blocks):
        """Try to replace 2 blocks with 1 block."""
        if len(blocks) < 2:
            return None

        block_set = set(blocks)

        cover_mask = 0
        for b_idx in blocks:
            cover_mask |= block_masks[b_idx]

        max_attempts = 200
        attempts = 0

        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            i, j = random.sample(range(len(blocks)), 2)
            b1 = blocks[i]
            b2 = blocks[j]

            mask12 = block_masks[b1] | block_masks[b2]
            others_mask = cover_mask & ~mask12
            unique_mask = mask12 & ~others_mask

            if unique_mask == 0:
                # One of them is redundant, just remove it
                new_blocks = [b for idx, b in enumerate(blocks) if idx != i]
                return new_blocks

            for b3 in range(n_blocks):
                if b3 == b1 or b3 == b2:
                    continue
                if (unique_mask & ~block_masks[b3]) == 0:
                    new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                    new_blocks.append(b3)
                    return new_blocks

        return None

    def local_search_1for2(blocks):
        """Try to replace 1 block with 2 blocks (to enable further 2-for-1)."""
        if len(blocks) < 1:
            return None

        max_attempts = 100
        attempts = 0

        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            i = random.randrange(len(blocks))
            b1 = blocks[i]

            # Try to find two blocks b2, b3 such that their union covers b1's unique subsets
            # and doesn't add unnecessary blocks
            mask1 = block_masks[b1]

            # Find blocks that cover some of b1's subsets
            candidates = []
            for b_idx in range(n_blocks):
                if b_idx == b1:
                    continue
                overlap = (mask1 & block_masks[b_idx]).bit_count()
                if overlap > 0:
                    candidates.append((overlap, b_idx))

            if not candidates:
                continue

            # Sort by overlap descending
            candidates.sort(reverse=True)
            candidates = candidates[:50]  # Limit to top 50

            # Try pairs from candidates
            for a in range(len(candidates)):
                for b in range(a + 1, len(candidates)):
                    b2 = candidates[a][1]
                    b3 = candidates[b][1]
                    union_mask = block_masks[b2] | block_masks[b3]
                    if (mask1 & ~union_mask) == 0:
                        # b2 and b3 cover all of b1's subsets
                        new_blocks = blocks[:i] + blocks[i+1:]
                        new_blocks.append(b2)
                        new_blocks.append(b3)
                        return new_blocks

        return None

    best_blocks = None
    best_count = float('inf')

    # Run multiple restarts with different strategies
    num_restarts = 8
    for i in range(num_restarts):
        if time.time() > deadline:
            break

        # Alternate between greedy variants
        blocks = greedy_cover_linear(seed=i)

        if blocks is None:
            continue

        blocks = prune_redundant(blocks)

        # Local search loop
        improved = True
        max_ls_iters = 20
        ls_iter = 0
        while improved and time.time() < deadline and ls_iter < max_ls_iters:
            improved = False
            ls_iter += 1

            # Try 2-for-1 moves
            for _ in range(30):
                if time.time() > deadline:
                    break
                new_blocks = local_search_2for1(blocks)
                if new_blocks is not None:
                    blocks = new_blocks
                    improved = True
                    blocks = prune_redundant(blocks)
                    break

            if improved:
                continue

            # Try 1-for-2 moves (to enable future 2-for-1)
            if len(blocks) > 340 and random.random() < 0.3:
                new_blocks = local_search_1for2(blocks)
                if new_blocks is not None:
                    blocks = new_blocks
                    improved = True
                    blocks = prune_redundant(blocks)

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

        if best_count <= 335:
            break

    # Final polish: one more round of local search on the best solution
    if best_blocks is not None and time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            new_blocks = local_search_2for1(best_blocks)
            if new_blocks is not None:
                best_blocks = new_blocks
                improved = True
                best_blocks = prune_redundant(best_blocks)

    if best_blocks is None:
        blocks = greedy_cover_linear(seed=None)
        if blocks:
            return [list(all_blocks[i]) for i in blocks]
        return []

    return [list(all_blocks[i]) for i in best_blocks]
