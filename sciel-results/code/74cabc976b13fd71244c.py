import itertools
import random
import time
import sys
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using multi-start greedy with
    randomization and iterative redundancy removal + swap local search.
    """
    start_time = time.time()
    deadline = start_time + 135  # Safety margin for 150s limit

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Map each block to indices of t-subsets it covers
    block_to_tsubs = []
    for block in all_blocks:
        covered = [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_to_tsubs.append(covered)

    # Build inverse: for each t-subset, which blocks cover it
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for b_idx, t_indices in enumerate(block_to_tsubs):
        for t_idx in t_indices:
            tsub_to_blocks[t_idx].append(b_idx)

    def greedy_cover(seed=None):
        """Randomized greedy set cover. Returns list of block indices."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random

        uncovered = np.ones(n_tsubs, dtype=bool)
        blocks_used = []

        while np.any(uncovered):
            if time.time() > deadline - 5:
                break

            best_gain = -1
            best_b = -1

            # Pick a random uncovered t-subset to guide selection
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            target = rng.choice(uncovered_indices.tolist())

            # Candidate blocks: those covering the target t-subset
            candidates = tsub_to_blocks[target]
            if not candidates:
                break

            # From candidates, pick the one with highest gain (covers most uncovered)
            # Use randomization: with some probability pick random, otherwise pick best
            if rng.random() < 0.3:
                # Random selection from top candidates
                # First sort by potential gain (approximate: number of uncovered covered)
                scored = []
                for b_idx in candidates:
                    if not uncovered[0]:  # optimization: early exit if no uncovered
                        break
                    gain = 0
                    for t_idx in block_to_tsubs[b_idx]:
                        if uncovered[t_idx]:
                            gain += 1
                    scored.append((gain, b_idx))
                scored.sort(key=lambda x: -x[0])
                top_n = min(10, len(scored))
                chosen = rng.choice(scored[:top_n])[1]
            else:
                # Greedy: pick max gain
                best_gain = -1
                for b_idx in candidates:
                    gain = 0
                    for t_idx in block_to_tsubs[b_idx]:
                        if uncovered[t_idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b_idx
                chosen = best_b

            if chosen == -1:
                break

            blocks_used.append(chosen)
            for t_idx in block_to_tsubs[chosen]:
                uncovered[t_idx] = False

        return blocks_used

    def local_search(block_indices):
        """Iterative redundancy removal + swap to reduce block count."""
        if time.time() > deadline - 10:
            return block_indices

        blocks = list(block_indices)
        if not blocks:
            return blocks

        # Build coverage count
        tsub_count = np.zeros(n_tsubs, dtype=np.int32)
        for b_idx in blocks:
            for t_idx in block_to_tsubs[b_idx]:
                tsub_count[t_idx] += 1

        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 10:
                break

            n_cur = len(blocks)
            for i in range(n_cur):
                if time.time() > deadline - 10:
                    break

                b_idx = blocks[i]
                # Check if redundant
                redundant = True
                for t_idx in block_to_tsubs[b_idx]:
                    if tsub_count[t_idx] == 1:
                        redundant = False
                        break

                if redundant:
                    # Remove block
                    blocks.pop(i)
                    for t_idx in block_to_tsubs[b_idx]:
                        tsub_count[t_idx] -= 1
                    improved = True
                    break

            if improved:
                continue

            # Try swaps: replace one block with another to maintain coverage but allow further reductions
            # Find a block with critical t-subsets (covered only by it)
            for i in range(len(blocks)):
                if time.time() > deadline - 10:
                    break

                b_idx = blocks[i]
                critical = []
                for t_idx in block_to_tsubs[b_idx]:
                    if tsub_count[t_idx] == 1:
                        critical.append(t_idx)

                if not critical:
                    continue  # This block is redundant, should have been caught

                # Find a candidate block that covers all critical t-subsets
                # Start with blocks covering the first critical t-subset
                candidates = set(tsub_to_blocks[critical[0]])
                candidates.discard(b_idx)

                for t_idx in critical[1:]:
                    candidates &= set(tsub_to_blocks[t_idx])
                    if not candidates:
                        break

                if candidates:
                    # Try a random candidate swap
                    new_b = random.choice(list(candidates))
                    # Perform swap
                    for t_idx in block_to_tsubs[b_idx]:
                        tsub_count[t_idx] -= 1
                    for t_idx in block_to_tsubs[new_b]:
                        tsub_count[t_idx] += 1
                    blocks[i] = new_b
                    improved = True
                    break

        return blocks

    # Multi-start: try several greedy runs with different seeds
    best_blocks = None
    best_count = float('inf')

    num_starts = 5
    for trial in range(num_starts):
        if time.time() > deadline - 20:
            break

        seed = trial * 1000 + 42
        blocks = greedy_cover(seed=seed)

        if not blocks:
            continue

        # Local search
        blocks = local_search(blocks)

        # Verify coverage
        if len(blocks) < best_count:
            # Quick verification: ensure all t-subsets are covered
            covered = np.zeros(n_tsubs, dtype=bool)
            for b_idx in blocks:
                for t_idx in block_to_tsubs[b_idx]:
                    covered[t_idx] = True
            if np.all(covered):
                best_blocks = blocks
                best_count = len(blocks)

    if best_blocks is None or not best_blocks:
        # Fallback: simple greedy without randomization
        blocks = greedy_cover(seed=None)
        blocks = local_search(blocks)
        if blocks:
            best_blocks = blocks
        else:
            # Last resort: return empty (should not happen)
            best_blocks = []

    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    return final_blocks
