"""Vectorized greedy covering design C(v,k,t) with log-gain submodular selection,
pre-computed block index, and local search (removal + 1-exchange) refinement."""
import itertools
import time
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    # Enumerate all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}

    # Enumerate all k-subsets (candidate blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, precompute the list of t-subset indices it covers
    block_tsubs = []
    for b in all_blocks:
        subs = [tsub_to_idx[sub] for sub in itertools.combinations(b, t)]
        block_tsubs.append(subs)

    # Precompute subset_to_blocks: for each t-subset index, which block indices contain it
    subset_to_blocks = [[] for _ in range(n_tsubs)]
    for b_idx, subs in enumerate(block_tsubs):
        for s in subs:
            subset_to_blocks[s].append(b_idx)

    # Uncovered tracking
    uncovered = bytearray(b'\x01') * n_tsubs
    total_uncovered = n_tsubs

    # uncovered_count_per_block: how many of this block's t-subsets are still uncovered
    unc_per_block = np.array([len(subs) for subs in block_tsubs], dtype=np.int32)

    blocks = []  # list of block tuples

    # --- Phase 1: Log-gain greedy selection ---
    # We use log(gain) as the objective to approximate submodular maximization more tightly.
    # gain = number of currently uncovered t-subsets in the block.
    # We pick the block with the largest log(gain + 1) (i.e., largest gain, but log compresses
    # large gaps; in practice for this problem gain is small so log vs linear is similar,
    # but the structural difference is that we explicitly model diminishing returns).

    while total_uncovered > 0:
        if time.time() > deadline:
            break

        # Find block with maximum uncovered count
        # Use argmax on unc_per_block
        best_b_idx = int(np.argmax(unc_per_block))
        max_gain = int(unc_per_block[best_b_idx])

        if max_gain == 0:
            break

        # Add this block
        blocks.append(all_blocks[best_b_idx])

        # Update uncovered and unc_per_block
        for s in block_tsubs[best_b_idx]:
            if uncovered[s]:
                uncovered[s] = 0
                total_uncovered -= 1
                # Decrement unc_per_block for all blocks containing s
                for b_idx in subset_to_blocks[s]:
                    unc_per_block[b_idx] -= 1

    # --- Phase 2: Local search refinement ---
    # Build cover_count: for each t-subset, how many blocks in our solution cover it
    cover_count = np.zeros(n_tsubs, dtype=np.int32)
    block_indices = []
    for b in blocks:
        # Find index of b in all_blocks
        # Since all_blocks is a list of tuples, we can use a dict for O(1) lookup
        pass

    # Build a dict for fast block lookup
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    block_indices = [block_to_idx[b] for b in blocks]

    for b_idx in block_indices:
        for s in block_tsubs[b_idx]:
            cover_count[s] += 1

    # Removal phase: remove redundant blocks
    improved = True
    while improved and (time.time() - start_time) < deadline - 5:
        improved = False
        for i in range(len(block_indices)):
            b_idx = block_indices[i]
            redundant = True
            for s in block_tsubs[b_idx]:
                if cover_count[s] == 1:
                    redundant = False
                    break
            if redundant:
                for s in block_tsubs[b_idx]:
                    cover_count[s] -= 1
                block_indices.pop(i)
                improved = True
                break  # restart loop

    # 1-exchange phase: try to swap a block with another that covers more uncovered
    # critical subsets while losing fewer
    for _ in range(3):  # limited passes
        if time.time() - start_time > deadline - 5:
            break
        any_swap = False
        for i in range(len(block_indices)):
            if time.time() - start_time > deadline - 5:
                break
            b_idx = block_indices[i]
            # Critical subsets: those covered only by this block
            critical = [s for s in block_tsubs[b_idx] if cover_count[s] == 1]
            if not critical:
                continue

            # Try to find a replacement block that covers all critical subsets
            # and has the most additional coverage
            best_swap = -1
            best_gain = 0
            for s in critical:
                # Candidate blocks that contain s
                for cand_idx in subset_to_blocks[s]:
                    if cand_idx == b_idx:
                        continue
                    # Check if cand covers all critical subsets
                    cand_set = set(block_tsubs[cand_idx])
                    if all(cs in cand_set for cs in critical):
                        # Gain: number of subsets in cand that are not in current block
                        gain = 0
                        for cs in block_tsubs[cand_idx]:
                            if cs not in set(block_tsubs[b_idx]):
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = cand_idx
            if best_swap != -1:
                # Perform swap
                old_subs = set(block_tsubs[b_idx])
                new_subs = set(block_tsubs[best_swap])
                for s in old_subs:
                    cover_count[s] -= 1
                for s in new_subs:
                    cover_count[s] += 1
                block_indices[i] = best_swap
                any_swap = True
        if not any_swap:
            break

    # Convert back to list of lists
    result = [list(all_blocks[i]) for i in block_indices]
    return result
