import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)

    # Map t-subset to index
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Precompute all possible k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # For each block, precompute the list of t-subset indices it covers
    # This is expensive to compute for all 50388 blocks, but we only need it for the ones we pick?
    # No, in the greedy we construct blocks on the fly.
    # But in local search, we have specific blocks.

    best_blocks = None
    best_count = float('inf')

    # Helper to compute t-subset indices for a block
    def get_block_t_indices(block):
        return [t_sub_to_idx[sub] for sub in itertools.combinations(block, t)]

    # Faster greedy: pick uncovered t-subset, build block greedily
    def fast_greedy(seed_offset=0):
        uncovered = np.ones(num_t, dtype=bool)
        blocks = []

        while uncovered.any():
            if time.time() > deadline - 3.0:
                break

            # Pick a random uncovered t-subset to seed the block
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            seed_idx = uncovered_indices[seed_offset % len(uncovered_indices)]
            seed_subset = all_t_subsets[seed_idx]
            block = list(seed_subset)
            seed_offset += 1

            while len(block) < k:
                # Find x that covers most uncovered t-subsets
                best_x = -1
                best_gain = -1
                tied_candidates = []

                block_set = set(block)
                t_minus_1_subsets = list(itertools.combinations(block, t - 1))

                for x in range(v):
                    if x in block_set:
                        continue

                    gain = 0
                    # We can optimize this by noting that if we add x,
                    # the new t-subsets are {x} U rest for rest in C(block, t-1).
                    for rest in t_minus_1_subsets:
                        T = tuple(sorted(rest + (x,)))
                        if T in t_sub_to_idx:
                            idx = t_sub_to_idx[T]
                            if uncovered[idx]:
                                gain += 1

                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        tied_candidates = [x]
                    elif gain == best_gain:
                        tied_candidates.append(x)

                if best_x == -1:
                    break

                # Random tie-breaking
                if len(tied_candidates) > 1:
                    best_x = random.choice(tied_candidates)

                block.append(best_x)

            block.sort()
            blocks.append(block)

            # Remove covered t-subsets
            for sub in itertools.combinations(block, t):
                if sub in t_sub_to_idx:
                    uncovered[t_sub_to_idx[sub]] = False

        return blocks

    # Local search: try to remove redundant blocks, then try swaps
    def local_search(blocks, max_iterations=1000):
        if not blocks:
            return blocks

        n_blocks = len(blocks)
        block_t_indices = [get_block_t_indices(b) for b in blocks]
        coverage_count = np.zeros(num_t, dtype=np.int32)

        for idxs in block_t_indices:
            coverage_count[idxs] += 1

        # Iteratively remove redundant blocks
        changed = True
        while changed:
            changed = False
            if time.time() > deadline - 2.0:
                break

            # Find a removable block
            order = list(range(len(blocks)))
            random.shuffle(order)

            removable_idx = -1
            for i in order:
                # Check if block i is removable
                is_removable = True
                for sub_idx in block_t_indices[i]:
                    if coverage_count[sub_idx] <= 1:
                        is_removable = False
                        break
                if is_removable:
                    removable_idx = i
                    break

            if removable_idx != -1:
                # Remove block
                idxs_to_remove = block_t_indices[removable_idx]
                for sub_idx in idxs_to_remove:
                    coverage_count[sub_idx] -= 1

                blocks.pop(removable_idx)
                block_t_indices.pop(removable_idx)
                changed = True

        # Try block swaps: replace one block with another that covers more
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            n_blocks = len(blocks)
            if n_blocks == 0:
                break

            # Recompute coverage
            coverage_count = np.zeros(num_t, dtype=np.int32)
            for idxs in block_t_indices:
                coverage_count[idxs] += 1

            # Try removing each block and see if we can add a better one
            for i in range(n_blocks):
                if time.time() > deadline - 2.0:
                    break

                # Temporarily remove block i
                idxs_removed = block_t_indices[i]
                for sub_idx in idxs_removed:
                    coverage_count[sub_idx] -= 1

                # Find uncovered t-subsets
                uncovered_mask = coverage_count == 0
                if not uncovered_mask.any():
                    # All covered, can remove block i
                    blocks.pop(i)
                    block_t_indices.pop(i)
                    improved = True
                    break

                # Try to find a block that covers the most uncovered
                best_gain = 0
                best_block = None
                best_block_t_idxs = None

                # Sample some candidate blocks
                candidates = random.sample(all_blocks, min(500, len(all_blocks)))
                for cand in candidates:
                    cand_set = set(cand)
                    # Check if cand is different from removed block
                    if cand_set == set(blocks[i]):
                        continue

                    gain = 0
                    for sub in itertools.combinations(cand, t):
                        idx = t_sub_to_idx.get(sub, -1)
                        if idx >= 0 and uncovered_mask[idx]:
                            gain += 1

                    if gain > best_gain:
                        best_gain = gain
                        best_block = cand
                        best_block_t_idxs = [t_sub_to_idx[sub] for sub in itertools.combinations(cand, t)]

                if best_gain > 0 and best_block is not None:
                    # Replace block i with best_block
                    blocks[i] = list(best_block)
                    blocks[i].sort()
                    block_t_indices[i] = best_block_t_idxs
                    for sub_idx in best_block_t_idxs:
                        coverage_count[sub_idx] += 1
                    improved = True
                    break
                else:
                    # Restore coverage
                    for sub_idx in idxs_removed:
                        coverage_count[sub_idx] += 1

        return blocks

    # Multiple runs with different random seeds
    for run_id in range(500):  # Max 500 runs, but limited by time
        if time.time() > deadline - 5.0:
            break

        random.seed(run_id * 12345 + 67890)

        # Run fast greedy
        blocks = fast_greedy(seed_offset=run_id)

        if time.time() > deadline - 3.0:
            break

        # Local optimization
        blocks = local_search(blocks, max_iterations=500)

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]

    # If no blocks found, fallback to simple greedy
    if best_blocks is None:
        blocks = fast_greedy(seed_offset=0)
        best_blocks = blocks
        best_count = len(blocks)

    return best_blocks
