import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}

    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Map each block to the list of triplet IDs it covers
    block_triplets = []
    for block in all_blocks:
        tids = []
        for triplet in itertools.combinations(block, t):
            tids.append(triplet_to_id[triplet])
        block_triplets.append(tids)

    # Map each triplet to the list of block IDs that cover it
    blocks_covering = [[] for _ in range(n_triplets)]
    for b_idx, tids in enumerate(block_triplets):
        for tid in tids:
            blocks_covering[tid].append(b_idx)

    def greedy_set_cover(seed=0):
        rng = random.Random(seed)
        covered_count = np.zeros(n_triplets, dtype=np.int32)
        # Initial score: number of triplets in the block (which is C(k,t))
        score = np.array([len(tids) for tids in block_triplets], dtype=np.int32)
        used_blocks = set()
        solution = []

        # We need to find the max score block efficiently.
        # Since scores only decrease, we can just scan.
        # To add randomness to tie-breaking, we can add a small random noise to scores initially?
        # Or just pick random among max.

        while True:
            if time.time() > deadline - 20:
                break

            # Find max score block not in used_blocks
            max_score = -1
            best_block_idx = -1
            candidates = []

            # Optimization: if many blocks are used, scanning all is still O(N_blocks)
            # 346k is small enough.
            for b_idx in range(n_blocks):
                if b_idx not in used_blocks:
                    s = score[b_idx]
                    if s > max_score:
                        max_score = s
                        best_block_idx = b_idx
                        candidates = [b_idx]
                    elif s == max_score:
                        candidates.append(b_idx)

            if best_block_idx == -1:
                break

            # Randomly pick from candidates to diversify
            chosen_block = rng.choice(candidates)
            solution.append(all_blocks[chosen_block])
            used_blocks.add(chosen_block)

            # Update coverage and scores
            newly_covered = []
            for tid in block_triplets[chosen_block]:
                if covered_count[tid] == 0:
                    newly_covered.append(tid)
                covered_count[tid] += 1

            for tid in newly_covered:
                for b_idx in blocks_covering[tid]:
                    score[b_idx] -= 1

            if np.all(covered_count > 0):
                break

        return solution

    def local_search_remove(blocks):
        # Re-compute covered_count from scratch for safety? Or maintain?
        # Maintaining is tricky if we restart. Let's re-compute.
        covered_count = np.zeros(n_triplets, dtype=np.int32)
        for block in blocks:
            for triplet in itertools.combinations(block, t):
                tid = triplet_to_id[triplet]
                covered_count[tid] += 1

        new_blocks = []
        changed = True
        while changed:
            changed = False
            next_blocks = []
            for block in blocks:
                # Check if block is redundant
                redundant = True
                for triplet in itertools.combinations(block, t):
                    tid = triplet_to_id[triplet]
                    if covered_count[tid] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove it
                    for triplet in itertools.combinations(block, t):
                        tid = triplet_to_id[triplet]
                        covered_count[tid] -= 1
                    changed = True
                else:
                    next_blocks.append(block)
            blocks = next_blocks
        return blocks

    def local_search_2for1(blocks, covered_count):
        """Try to replace 2 blocks with 1 block that covers all uniquely-covered triplets."""
        if time.time() > deadline - 10:
            return blocks, covered_count

        n_blocks = len(blocks)
        if n_blocks < 2:
            return blocks, covered_count

        # For each pair of blocks, find triplets uniquely covered by them
        # Then check if a single block can cover those triplets
        # This is expensive, so we'll sample pairs

        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            new_blocks = list(blocks)
            new_covered = covered_count.copy()

            # Sample some pairs to try
            rng = random.Random()
            pairs_to_try = min(100, n_blocks * (n_blocks - 1) // 2)
            tried_pairs = set()

            for _ in range(pairs_to_try):
                i, j = rng.sample(range(n_blocks), 2)
                if (i, j) in tried_pairs or (j, i) in tried_pairs:
                    continue
                tried_pairs.add((i, j))
                tried_pairs.add((j, i))

                # Find triplets uniquely covered by blocks i and j
                unique_to_pair = []
                for triplet in itertools.combinations(new_blocks[i], t):
                    tid = triplet_to_id[triplet]
                    if new_covered[tid] == 1:
                        unique_to_pair.append(tid)
                for triplet in itertools.combinations(new_blocks[j], t):
                    tid = triplet_to_id[triplet]
                    if new_covered[tid] == 1:
                        unique_to_pair.append(tid)

                if not unique_to_pair:
                    # Both blocks are redundant in some sense
                    continue

                # Check if any single block covers all these unique triplets
                # We need a block that contains all points in these triplets
                # The union of points in unique_to_pair must be contained in some block

                # Collect all points that must be in the new block
                required_points = set()
                for tid in unique_to_pair:
                    for p in all_triplets[tid]:
                        required_points.add(p)

                if len(required_points) > k:
                    continue

                # Find a block that contains all required_points
                # And check if it's different from i and j
                found = False
                for b_idx in range(len(all_blocks)):
                    if b_idx in (i, j):
                        continue
                    block = all_blocks[b_idx]
                    block_set = set(block)
                    if required_points.issubset(block_set):
                        # Check that this block doesn't introduce issues
                        # (it shouldn't, since we're replacing 2 with 1)
                        # Remove blocks i and j, add this block
                        # But we need to be careful about indices

                        # For simplicity, let's just mark this as a potential improvement
                        # and break out to apply it
                        found = True
                        replacement_block = block
                        break

                if found:
                    # Apply the replacement
                    # This is complex with index management, so let's do it carefully
                    # Remove blocks i and j (higher index first to preserve indices)
                    remove_indices = sorted([i, j], reverse=True)
                    for idx in remove_indices:
                        for triplet in itertools.combinations(new_blocks[idx], t):
                            tid = triplet_to_id[triplet]
                            new_covered[tid] -= 1
                        new_blocks.pop(idx)

                    # Add the replacement block
                    new_blocks.append(replacement_block)
                    for triplet in itertools.combinations(replacement_block, t):
                        tid = triplet_to_id[triplet]
                        new_covered[tid] += 1

                    # Verify all triplets are still covered
                    if np.all(new_covered > 0):
                        blocks = new_blocks
                        covered_count = new_covered
                        improved = True
                        break
                    else:
                        # Revert
                        new_blocks = list(blocks)
                        new_covered = covered_count.copy()

        return blocks, covered_count

    def local_search_1for1(blocks, covered_count):
        """Try to replace 1 block with another that covers all its uniquely-covered triplets."""
        if time.time() > deadline - 10:
            return blocks, covered_count

        n_blocks = len(blocks)
        if n_blocks < 1:
            return blocks, covered_count

        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            new_blocks = list(blocks)
            new_covered = covered_count.copy()

            # Try each block
            for i in range(n_blocks):
                if time.time() > deadline - 10:
                    break

                # Find triplets uniquely covered by block i
                unique_to_block = []
                for triplet in itertools.combinations(new_blocks[i], t):
                    tid = triplet_to_id[triplet]
                    if new_covered[tid] == 1:
                        unique_to_block.append(tid)

                if not unique_to_block:
                    # Block is redundant
                    for triplet in itertools.combinations(new_blocks[i], t):
                        tid = triplet_to_id[triplet]
                        new_covered[tid] -= 1
                    new_blocks.pop(i)
                    if np.all(new_covered > 0):
                        blocks = new_blocks
                        covered_count = new_covered
                        improved = True
                        break
                    else:
                        new_blocks = list(blocks)
                        new_covered = covered_count.copy()
                    continue

                # Find a replacement block that covers all unique_to_block
                required_points = set()
                for tid in unique_to_block:
                    for p in all_triplets[tid]:
                        required_points.add(p)

                if len(required_points) > k:
                    continue

                # Find a block that contains all required_points and is different from block i
                found = False
                for b_idx in range(len(all_blocks)):
                    if b_idx == i:
                        continue
                    block = all_blocks[b_idx]
                    block_set = set(block)
                    if required_points.issubset(block_set):
                        # Check if this block is already in the solution
                        if block in new_blocks:
                            continue

                        # Try the replacement
                        for triplet in itertools.combinations(new_blocks[i], t):
                            tid = triplet_to_id[triplet]
                            new_covered[tid] -= 1
                        new_blocks[i] = block
                        for triplet in itertools.combinations(block, t):
                            tid = triplet_to_id[triplet]
                            new_covered[tid] += 1

                        if np.all(new_covered > 0):
                            blocks = new_blocks
                            covered_count = new_covered
                            improved = True
                            found = True
                            break
                        else:
                            # Revert
                            for triplet in itertools.combinations(new_blocks[i], t):
                                tid = triplet_to_id[triplet]
                                new_covered[tid] -= 1
                            for triplet in itertools.combinations(block, t):
                                tid = triplet_to_id[triplet]
                                new_covered[tid] += 1
                            new_blocks[i] = blocks[i]

                if found:
                    break

        return blocks, covered_count

    def compute_covered_count(blocks):
        covered_count = np.zeros(n_triplets, dtype=np.int32)
        for block in blocks:
            for triplet in itertools.combinations(block, t):
                tid = triplet_to_id[triplet]
                covered_count[tid] += 1
        return covered_count

    best_blocks = None
    best_count = float('inf')

    # Run multiple greedy seeds
    num_seeds = 50
    for seed in range(num_seeds):
        if time.time() > deadline - 30:
            break
        blocks = greedy_set_cover(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_set_cover(0)

    # Local search polish
    best_blocks = local_search_remove(best_blocks)

    # Compute covered count for further local search
    covered_count = compute_covered_count(best_blocks)

    # Try 2-for-1 swaps
    if time.time() < deadline - 15:
        best_blocks, covered_count = local_search_2for1(best_blocks, covered_count)

    # Try 1-for-1 swaps
    if time.time() < deadline - 10:
        best_blocks, covered_count = local_search_1for1(best_blocks, covered_count)

    # Final removal pass
    if time.time() < deadline - 5:
        best_blocks = local_search_remove(best_blocks)

    return best_blocks

if __name__ == "__main__":
    pass
