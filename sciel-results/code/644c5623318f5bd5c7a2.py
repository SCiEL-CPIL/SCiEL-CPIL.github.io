"""
Covering design C(19,8,5) solver using greedy construction + aggressive local search.
Focus: minimize block count by finding redundant blocks and using targeted swaps.
"""
import itertools
import random
import time
import numpy as np


def construct(v, k, t):
    start = time.time()
    deadline = start + 110  # 40s safety margin

    rng = random.Random(12345)

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute for each point, the t-subsets containing it (for later use)
    # But more importantly, precompute the list of t-subset indices for any k-block
    # We'll use a cache

    def block_to_tsub_indices(block):
        """Return list of t-subset indices contained in the block."""
        return [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]

    # Greedy construction
    def greedy_construct(seed=None, max_blocks=None):
        """
        Greedy set cover: repeatedly pick the block covering the most uncovered t-subsets.
        Uses randomization via seed for diversity.
        """
        if seed is not None:
            local_rng = random.Random(seed)
        else:
            local_rng = random.Random()

        uncovered = np.ones(n_tsubs, dtype=np.uint8)
        blocks = []
        block_indices_cache = {}

        # Precompute: for each possible k-block, its t-subset indices
        # C(19,8) = 75582 blocks, each has C(8,5)=56 t-subsets
        # That's 75582*56 = 4.2M entries - feasible to precompute

        # Actually, let's precompute all k-blocks and their t-subset indices
        all_kblocks = list(itertools.combinations(range(v), k))
        n_kblocks = len(all_kblocks)
        block_tsubs_arr = np.zeros((n_kblocks, 56), dtype=np.int32)  # C(8,5)=56 for k=8,t=5
        # But this is general... let me compute per block
        block_tsub_list = []
        for i, block in enumerate(all_kblocks):
            indices = [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]
            block_tsub_list.append(indices)

        # For speed, store as list of lists
        # Now greedy: at each step, find block with max new coverage

        while np.any(uncovered) and time.time() < deadline - 20:
            if max_blocks is not None and len(blocks) >= max_blocks:
                break

            # Find best block: maximize number of newly covered t-subsets
            # Use numpy for speed
            best_gain = -1
            best_idx = -1

            # Sample a subset of candidate blocks for speed
            # Try all blocks if n_kblocks is manageable, otherwise sample
            if n_kblocks <= 100000:
                # Compute gains for all blocks
                for i in range(n_kblocks):
                    gain = 0
                    for idx in block_tsub_list[i]:
                        if uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
            else:
                # Sample
                sample_size = min(5000, n_kblocks)
                candidates = local_rng.sample(range(n_kblocks), sample_size)
                for i in candidates:
                    gain = 0
                    for idx in block_tsub_list[i]:
                        if uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i

            if best_idx < 0 or best_gain <= 0:
                break

            blocks.append(list(all_kblocks[best_idx]))
            for idx in block_tsub_list[best_idx]:
                uncovered[idx] = 0

        return blocks

    def verify_coverage(blocks):
        """Check if all t-subsets are covered."""
        covered = np.zeros(n_tsubs, dtype=np.uint8)
        for block in blocks:
            for ts in itertools.combinations(block, t):
                covered[tsub_to_idx[ts]] = 1
        return np.all(covered == 1)

    def local_search(blocks, max_time=None):
        """
        Aggressive local search to reduce block count.
        Moves:
        1. Remove redundant blocks (those whose t-subsets are all covered by others)
        2. Swap a point in one block to cover a critical t-subset of another, then remove the other
        3. Two-block swap: modify two blocks to enable removal of a third
        """
        if max_time is None:
            max_time = deadline - 10

        # Build cover counts
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        block_tsubs = []
        for block in blocks:
            indices = block_to_tsub_indices(block)
            block_tsubs.append(indices)
            for idx in indices:
                cover_count[idx] += 1

        n_blocks = len(blocks)
        improved = True
        no_improve_count = 0

        while time.time() < max_time and no_improve_count < 100:
            improved = False

            # Phase 1: Remove redundant blocks
            for i in range(len(blocks)):
                if time.time() > max_time:
                    break
                redundant = True
                for idx in block_tsubs[i]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for idx in block_tsubs[i]:
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    block_tsubs.pop(i)
                    improved = True
                    no_improve_count = 0
                    break

            if improved:
                continue

            # Phase 2: Try to make a block redundant by modifying another
            # For each block i with critical t-subsets, try to cover them by modifying block j
            for i in range(len(blocks)):
                if time.time() > max_time:
                    break

                # Find critical t-subsets of block i
                criticals = [idx for idx in block_tsubs[i] if cover_count[idx] == 1]
                if not criticals:
                    continue

                # For each critical t-subset, try to cover it by modifying another block
                for crit_idx in criticals[:3]:  # Limit for speed
                    crit_ts = all_tsubs[crit_idx]
                    crit_set = set(crit_ts)

                    for j in range(len(blocks)):
                        if j == i or time.time() > max_time:
                            continue

                        block_j = blocks[j]
                        block_j_set = set(block_j)

                        # Skip if block_j already covers crit_idx
                        if crit_idx in block_tsubs[j]:
                            continue

                        # Find missing points
                        missing = crit_set - block_j_set
                        if len(missing) > 2:
                            continue

                        missing_list = list(missing)
                        extra = list(block_j_set - crit_set)
                        if len(extra) < len(missing_list):
                            continue

                        # Try all combinations of points to remove
                        for rem_combo in itertools.combinations(extra, len(missing_list)):
                            if time.time() > max_time:
                                break

                            rem_set = set(rem_combo)
                            new_block_j = [p for p in block_j if p not in rem_set] + missing_list
                            new_block_j.sort()
                            new_block_j_set = set(new_block_j)

                            # Compute new t-subsets
                            new_subs = set(block_to_tsub_indices(new_block_j))
                            old_subs_set = set(block_tsubs[j])

                            # Check validity: no t-subset loses its only cover
                            can_swap = True
                            for idx in old_subs_set:
                                if idx not in new_subs and cover_count[idx] == 1:
                                    can_swap = False
                                    break

                            if not can_swap:
                                continue

                            # Check if new block covers crit_idx
                            if crit_idx not in new_subs:
                                continue

                            # Perform swap
                            for idx in old_subs_set:
                                cover_count[idx] -= 1
                            for idx in new_subs:
                                cover_count[idx] += 1

                            blocks[j] = new_block_j
                            block_tsubs[j] = list(new_subs)

                            # Check if block i is now redundant
                            still_critical = False
                            for idx in block_tsubs[i]:
                                if cover_count[idx] == 1:
                                    still_critical = True
                                    break

                            if not still_critical:
                                # Remove block i
                                for idx in block_tsubs[i]:
                                    cover_count[idx] -= 1
                                blocks.pop(i)
                                block_tsubs.pop(i)
                                improved = True
                                no_improve_count = 0
                                break
                            else:
                                # Revert
                                for idx in new_subs:
                                    cover_count[idx] -= 1
                                for idx in old_subs_set:
                                    cover_count[idx] += 1
                                blocks[j] = sorted(block_j_set)
                                block_tsubs[j] = list(old_subs_set)

                        if improved:
                            break

                    if improved:
                        break

                if improved:
                    break

            if not improved:
                no_improve_count += 1

            # Phase 3: Random perturbation to escape local minima
            if no_improve_count > 20 and len(blocks) > 10 and time.time() < max_time:
                j = random.randint(0, len(blocks) - 1)
                block_j = blocks[j]
                old_subs_set = set(block_tsubs[j])

                # Replace one random point
                p_remove = random.choice(block_j)
                remaining = [p for p in range(v) if p != p_remove]
                p_add = random.choice(remaining)

                new_block_j = [p for p in block_j if p != p_remove] + [p_add]
                new_block_j.sort()

                new_subs = set(block_to_tsub_indices(new_block_j))

                # Check validity
                can_swap = True
                for idx in old_subs_set:
                    if idx not in new_subs and cover_count[idx] == 1:
                        can_swap = False
                        break

                if can_swap:
                    for idx in old_subs_set:
                        cover_count[idx] -= 1
                    for idx in new_subs:
                        cover_count[idx] += 1
                    blocks[j] = new_block_j
                    block_tsubs[j] = list(new_subs)
                    no_improve_count = 0

        return blocks

    # Main: try multiple greedy starts, keep best after local search
    best_blocks = None
    best_count = float('inf')

    # Try several different greedy constructions
    for seed in [None, 1, 2, 3, 4, 5]:
        if time.time() > deadline - 30:
            break

        blocks = greedy_construct(seed=seed)
        if not blocks:
            continue

        if not verify_coverage(blocks):
            continue

        # Local search
        blocks = local_search(blocks)

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        # Fallback: simple greedy
        best_blocks = greedy_construct()

    return best_blocks
