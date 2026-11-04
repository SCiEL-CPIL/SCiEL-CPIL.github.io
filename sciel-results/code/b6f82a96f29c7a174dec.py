import itertools
import time
import random
import sys
import math
import numpy as np

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    # Precompute all t-subsets and map to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute all blocks and their t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # For each block, store the list of t-subset indices it covers
    block_tsubs = []
    for b in all_blocks:
        tsubs_in_b = [tsub_index[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(tsubs_in_b)

    # Inverse: for each t-subset, which blocks cover it?
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for i, b_tsubs in enumerate(block_tsubs):
        for j in b_tsubs:
            tsub_blocks[j].append(i)

    # NumPy arrays
    block_tsubs_np = np.array(block_tsubs, dtype=np.int32)

    start_time = time.time()
    deadline = start_time + 145.0

    def check_time(budget=10.0):
        return time.time() - start_time > deadline - budget

    # --- Greedy initialization (fast) ---
    def greedy_init(seed=42):
        rng = random.Random(seed)
        uncovered = np.ones(num_tsubs, dtype=np.uint8)
        uncovered_count = num_tsubs
        selected = []

        while uncovered_count > 0:
            if check_time(10.0):
                break
            # Pick random uncovered t-subset
            while True:
                s_idx = rng.randint(0, num_tsubs - 1)
                if uncovered[s_idx]:
                    break
            candidates = tsub_blocks[s_idx]
            if not candidates:
                break
            cand_tsubs = block_tsubs_np[candidates]
            gains = np.sum(uncovered[cand_tsubs], axis=1)
            best_local = int(np.argmax(gains))
            best_idx = int(candidates[best_local])
            selected.append(best_idx)
            for j in block_tsubs[best_idx]:
                if uncovered[j]:
                    uncovered[j] = 0
                    uncovered_count -= 1
        return selected

    # --- Simulated Annealing ---
    def simulated_annealing(initial_blocks, seed=123):
        rng = random.Random(seed)
        blocks = list(initial_blocks)
        if not blocks:
            return blocks

        # Build cover count
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in blocks:
            cover_count[block_tsubs_np[b_idx]] += 1

        # Precompute for each block: the t-subsets it covers (for fast gain/loss)
        # We'll maintain cover_count and compute energy as:
        # energy = len(blocks) + penalty * num_uncovered
        # where num_uncovered = count of cover_count == 0

        # Initial energy
        num_uncovered = int(np.sum(cover_count == 0))
        penalty = 1000  # high penalty to force coverage
        current_energy = len(blocks) + penalty * num_uncovered
        best_energy = current_energy
        best_blocks = list(blocks)

        # Temperature schedule
        T_start = 50.0
        T_end = 0.1
        alpha = 0.995  # cooling rate
        T = T_start

        max_iters = 200000
        iter_count = 0

        while T > T_end and iter_count < max_iters:
            if check_time(15.0):
                break
            iter_count += 1

            # Move: swap one block
            if len(blocks) <= 1:
                break

            # Choose a block to remove
            remove_idx = rng.randint(0, len(blocks) - 1)
            remove_block = blocks[remove_idx]

            # Compute effect of removing this block
            # t-subsets that become uncovered (cover_count drops to 0)
            b_tsubs = block_tsubs_np[remove_block]
            # New cover count after removal
            new_cover = cover_count.copy()
            new_cover[b_tsubs] -= 1
            newly_uncovered = int(np.sum(new_cover[b_tsubs] == 0))
            # t-subsets that were uniquely covered (cover_count was 1)
            # Actually, newly_uncovered counts those that become 0

            # Choose a block to add (not already in set)
            # Sample from blocks that cover many of the newly uncovered
            # For simplicity, sample randomly from all blocks not in current set
            # But to be efficient, pick from blocks that intersect with uncovered t-subsets
            # Simple approach: pick random block
            add_block = rng.randint(0, num_blocks - 1)
            # Avoid adding same block
            if add_block == remove_block:
                # Try a few more times
                for _ in range(10):
                    add_block = rng.randint(0, num_blocks - 1)
                    if add_block != remove_block and add_block not in blocks:
                        break
                else:
                    continue

            # Check if add_block is already in blocks
            # For efficiency, we'll just check via a set
            # But maintaining a set is overhead. Let's just allow duplicates temporarily?
            # No, blocks must be distinct. Let's check.
            # Since num_blocks is ~75k and blocks is ~300, a linear check is ~300 ops, acceptable.
            if add_block in blocks:
                continue

            # Compute effect of adding add_block
            add_tsubs = block_tsubs_np[add_block]
            # t-subsets that become covered (were 0, now >=1)
            # After removal, some may be 0. After addition, they become >=1.
            # Newly covered = those that were 0 after removal and are in add_tsubs
            # But we need to compute the combined effect.
            # Let's compute the energy change directly.
            # Current state: cover_count, num_uncovered
            # After removal: new_cover, new_uncovered = num_uncovered + newly_uncovered
            # After addition: final_cover = new_cover + add_tsubs (increment)
            # final_uncovered = count of final_cover == 0

            # To avoid full array copy, let's compute delta.
            # This is complex. For simplicity, let's do the full computation but it's O(num_tsubs) per move.
            # That's too slow. Let's use a different move: 1-opt (remove only) and 1-opt (add only) separately?
            # Or, let's precompute the t-subset indices for each block and use them.

            # Alternative: use a simpler energy delta computation.
            # Let's just do the swap with full recompute of affected t-subsets.
            # The affected t-subsets are the union of tsubs in remove_block and add_block.
            # That's at most 56+56=112 t-subsets. We can update cover_count for just those.

            # Get the set of affected t-subset indices
            affected = set(b_tsubs.tolist()) | set(add_tsubs.tolist())
            affected_list = list(affected)

            # Save old cover counts for affected
            old_counts = cover_count[affected_list].copy()

            # Apply removal
            cover_count[b_tsubs] -= 1
            # Apply addition
            cover_count[add_tsubs] += 1

            # Compute new energy
            new_uncovered = int(np.sum(cover_count[affected_list] == 0))
            # But num_uncovered globally may have changed for t-subsets not in affected? No, only affected ones change.
            # So delta_uncovered = new_uncovered - int(np.sum(old_counts == 0))
            old_uncovered_affected = int(np.sum(old_counts == 0))
            delta_uncovered = new_uncovered - old_uncovered_affected

            new_energy = current_energy + delta_uncovered * penalty  # block count stays same (swap)

            # Accept or reject
            if new_energy < current_energy or rng.random() < math.exp(-(new_energy - current_energy) / T):
                # Accept
                current_energy = new_energy
                # Update blocks list
                blocks[remove_idx] = add_block
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_blocks = list(blocks)
            else:
                # Reject, revert
                cover_count[b_tsubs] += 1
                cover_count[add_tsubs] -= 1

            T *= alpha

        # Final reduction pass on best_blocks
        best_blocks = reduce_blocks(best_blocks)
        return best_blocks

    def reduce_blocks(block_indices):
        """Remove redundant blocks."""
        block_indices = list(block_indices)
        if not block_indices:
            return block_indices

        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in block_indices:
            cover_count[block_tsubs_np[b_idx]] += 1

        changed = True
        max_passes = 30
        for _ in range(max_passes):
            if check_time(5.0):
                break
            changed = False
            new_block_indices = []
            for b_idx in block_indices:
                b_tsubs = block_tsubs_np[b_idx]
                if np.any(cover_count[b_tsubs] == 1):
                    new_block_indices.append(b_idx)
                else:
                    changed = True
                    cover_count[b_tsubs] -= 1
            block_indices = new_block_indices
            if not changed:
                break
        return block_indices

    # Main: greedy init, then SA
    initial = greedy_init(seed=42)
    if not initial:
        # Fallback: return empty (should not happen)
        return []

    # Reduce initial
    initial = reduce_blocks(initial)

    best_blocks = list(initial)
    best_len = len(best_blocks)

    # Run SA with a few different seeds
    for sa_seed in range(5):
        if check_time(20.0):
            break
        result = simulated_annealing(initial, seed=sa_seed * 100 + 1)
        if result and len(result) < best_len:
            best_len = len(result)
            best_blocks = result

    # Final reduction
    best_blocks = reduce_blocks(best_blocks)

    return [list(all_blocks[b]) for b in best_blocks]
