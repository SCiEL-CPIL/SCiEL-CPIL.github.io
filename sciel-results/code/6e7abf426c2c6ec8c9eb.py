import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin before 150s kill

    # Pre-compute all t-subsets and map them to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}

    # Pre-compute all k-blocks and their t-subset coverage
    all_kblocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_kblocks)

    # For each block, compute which t-subsets it covers
    # block_tsubs[b] = list of t-subset indices covered by block b
    block_tsubs = []
    for b in all_kblocks:
        covered = []
        for sub in itertools.combinations(b, t):
            covered.append(tsub_to_idx[sub])
        block_tsubs.append(covered)

    # Greedy warm-start: find a good initial solution
    def greedy_seed(seed_val):
        rng = random.Random(seed_val)
        uncovered = set(range(n_tsubs))
        blocks = []

        while uncovered:
            # Pick a random uncovered t-subset to anchor the block
            anchor = rng.choice(list(uncovered))
            anchor_set = all_tsubs[anchor]

            # Find all blocks containing the anchor t-subset
            candidate_blocks = [b for b in all_kblocks if set(anchor_set).issubset(set(all_kblocks[b]))]

            if not candidate_blocks:
                # Should not happen, but fallback
                candidate_blocks = [rng.choice(all_kblocks)]

            # Pick the block that covers the most uncovered t-subsets
            best_block = None
            best_gain = -1
            for b in candidate_blocks:
                gain = sum(1 for idx in block_tsubs[b] if idx in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = b

            if best_block is None:
                best_block = candidate_blocks[0]

            blocks.append(list(best_block))

            # Remove covered t-subsets
            for idx in block_tsubs[all_kblocks.index(best_block)]:
                uncovered.discard(idx)

        return blocks

    # Local search: try to reduce block count via removal and swap
    def local_search(blocks, max_iters=500):
        blocks = [list(b) for b in blocks]
        improved = True
        iter_count = 0

        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1

            # Try removing each block
            for i in range(len(blocks)):
                if time.time() > deadline:
                    return blocks
                # Check if block i is redundant
                remaining = blocks[:i] + blocks[i+1:]
                if _is_covering(remaining, v, k, t, all_tsubs, tsub_to_idx):
                    blocks = remaining
                    improved = True
                    break

            if improved:
                continue

            # Try swapping: replace one block with another to maintain coverage
            for i in range(len(blocks)):
                if time.time() > deadline:
                    return blocks
                # Try replacing block i with a random other block
                for attempt in range(20):
                    j = random.randint(0, n_blocks - 1)
                    if all_kblocks[j] in [tuple(b) for b in blocks if b != blocks[i]]:
                        continue
                    new_blocks = blocks[:i] + [list(all_kblocks[j])] + blocks[i+1:]
                    if _is_covering(new_blocks, v, k, t, all_tsubs, tsub_to_idx):
                        # Check if this actually helps (it maintains coverage)
                        # Only accept if it might lead to further reductions
                        blocks = new_blocks
                        improved = True
                        break
                if improved:
                    break

        return blocks

    def _is_covering(blocks, v, k, t, all_tsubs, tsub_to_idx):
        """Check if blocks cover all t-subsets."""
        if not blocks:
            return n_tsubs == 0

        # Build coverage set
        covered = set()
        for b in blocks:
            for sub in itertools.combinations(b, t):
                covered.add(tsub_to_idx[sub])

        return len(covered) == n_tsubs

    # Main algorithm
    best_blocks = None
    best_count = float('inf')

    # Phase 1: Generate multiple greedy seeds
    n_seeds = 20
    for s in range(n_seeds):
        if time.time() > deadline - 30:  # Leave time for local search
            break
        blocks = greedy_seed(s)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]

    if best_blocks is None:
        # Fallback
        blocks = greedy_seed(0)
        return blocks

    # Phase 2: Local search on best solution
    if time.time() < deadline - 10:
        improved_blocks = local_search(best_blocks, max_iters=200)
        if len(improved_blocks) < best_count:
            best_blocks = improved_blocks
            best_count = len(best_blocks)

    # Phase 3: Try more seeds with remaining time, then local search each
    while time.time() < deadline - 15:
        s = random.randint(0, 10000)
        blocks = greedy_seed(s)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            if time.time() < deadline - 10:
                improved = local_search(best_blocks, max_iters=100)
                if len(improved) < best_count:
                    best_blocks = improved
                    best_count = len(best_blocks)

    return best_blocks
