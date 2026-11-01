import itertools
import time
import random
import numpy as np
from collections import Counter


def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using greedy set cover followed by
    local search with simulated annealing-style moves.
    """
    deadline = time.time() + 115.0  # Safety margin for 120s limit

    # 1. Generate all t-subsets and map them to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}

    # 2. Generate all k-subsets
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # 3. Precompute coverage masks for each k-subset
    ksub_masks = [0] * n_ksubs
    for j, ks in enumerate(all_ksubs):
        mask = 0
        for tsub in itertools.combinations(ks, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
        ksub_masks[j] = mask

    # Convert to NumPy array for vectorized operations
    ksub_masks_np = np.array(ksub_masks, dtype=object)

    # 4. Greedy set cover
    uncovered_mask = (1 << n_tsubs) - 1
    remaining = n_tsubs
    blocks = []
    used_blocks = set()
    block_indices = []  # parallel to blocks, stores indices into all_ksubs

    while remaining > 0:
        if time.time() > deadline:
            break

        best_j = -1
        best_gain = 0

        # Scan all candidates for maximum gain
        for j in range(n_ksubs):
            if j in used_blocks:
                continue
            common = uncovered_mask & ksub_masks[j]
            gain = common.bit_count()
            if gain > best_gain:
                best_gain = gain
                best_j = j

        if best_j == -1 or best_gain == 0:
            break

        blocks.append(list(all_ksubs[best_j]))
        block_indices.append(best_j)
        used_blocks.add(best_j)
        uncovered_mask &= ~ksub_masks[best_j]
        remaining -= best_gain

    # 5. Local search with simulated annealing
    if time.time() < deadline:
        # Initialize temperature
        T = 10.0
        T_min = 0.1
        cooling = 0.995
        max_iter = 10000

        # Current solution
        current_blocks = block_indices[:]
        current_used = set(current_blocks)

        # Count coverage for each t-subset
        cover_counts = np.zeros(n_tsubs, dtype=np.int32)
        for j in current_blocks:
            mask = ksub_masks[j]
            # Count bits set in mask
            temp = mask
            while temp:
                lsb = temp & (-temp)
                idx = lsb.bit_length() - 1
                cover_counts[idx] += 1
                temp ^= lsb

        uncovered = np.sum(cover_counts == 0)

        # Energy function: uncovered * 1000 + len(blocks)
        def energy(blocks_list, uncov_count):
            return uncov_count * 1000 + len(blocks_list)

        current_energy = energy(current_blocks, uncovered)
        best_blocks = current_blocks[:]
        best_energy = current_energy

        # Precompute for each k-subset: how many t-subsets it covers that are currently
        # only covered once (critical) vs covered multiple times
        # This is expensive to recompute, so we'll do it on-the-fly but efficiently

        rng = random.Random(42)

        for it in range(max_iter):
            if time.time() > deadline - 5:
                break

            # Choose a random move: add or remove a block
            if len(current_blocks) > 0 and rng.random() < 0.4:
                # Remove a random block
                rem_idx = rng.randrange(len(current_blocks))
                rem_block = current_blocks[rem_idx]
                new_blocks = current_blocks[:rem_idx] + current_blocks[rem_idx+1:]

                # Recompute uncovered count
                new_cover = np.zeros(n_tsubs, dtype=np.int32)
                for j in new_blocks:
                    mask = ksub_masks[j]
                    temp = mask
                    while temp:
                        lsb = temp & (-temp)
                        idx = lsb.bit_length() - 1
                        new_cover[idx] += 1
                        temp ^= lsb
                new_uncovered = np.sum(new_cover == 0)
                new_energy = energy(new_blocks, new_uncovered)
            else:
                # Add a random block not already in the solution
                candidates = [j for j in range(n_ksubs) if j not in current_used]
                if not candidates:
                    continue
                add_block = rng.choice(candidates)
                new_blocks = current_blocks + [add_block]
                new_used = current_used | {add_block}

                # Recompute uncovered count
                new_cover = np.zeros(n_tsubs, dtype=np.int32)
                for j in new_blocks:
                    mask = ksub_masks[j]
                    temp = mask
                    while temp:
                        lsb = temp & (-temp)
                        idx = lsb.bit_length() - 1
                        new_cover[idx] += 1
                        temp ^= lsb
                new_uncovered = np.sum(new_cover == 0)
                new_energy = energy(new_blocks, new_uncovered)

            # Accept or reject
            delta_E = new_energy - current_energy
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                current_blocks = new_blocks
                current_used = set(current_blocks)
                current_energy = new_energy

                # Recompute cover counts for current
                cover_counts = np.zeros(n_tsubs, dtype=np.int32)
                for j in current_blocks:
                    mask = ksub_masks[j]
                    temp = mask
                    while temp:
                        lsb = temp & (-temp)
                        idx = lsb.bit_length() - 1
                        cover_counts[idx] += 1
                        temp ^= lsb
                uncovered = np.sum(cover_counts == 0)

                if current_energy < best_energy:
                    best_blocks = current_blocks[:]
                    best_energy = current_energy

            T = max(T * cooling, T_min)

        # Use the best solution found
        if best_energy < energy(block_indices, np.sum(cover_counts == 0)):
            block_indices = best_blocks
            blocks = [list(all_ksubs[j]) for j in block_indices]
        else:
            blocks = [list(all_ksubs[j]) for j in block_indices]

    return blocks
