import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Safety margin before the 150s kill

    # Pre-compute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}

    # Pre-compute all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_blocks_total = len(all_k_blocks)

    # Map each t-subset to list of block indices containing it
    idx = [[] for _ in range(num_t)]

    # Map each block index to list of t-subset indices contained in it
    block_subs = [[] for _ in range(num_blocks_total)]

    # Build indices
    for b_idx, block in enumerate(all_k_blocks):
        for s in itertools.combinations(block, t):
            s_idx = t_subset_to_idx[s]
            idx[s_idx].append(b_idx)
            block_subs[b_idx].append(s_idx)

    # Convert to numpy arrays for faster operations where possible
    # block_subs_np: list of arrays
    block_subs_np = [np.array(bs, dtype=np.int32) for bs in block_subs]
    # idx_np: list of arrays
    idx_np = [np.array(i, dtype=np.int32) for i in idx]

    # Greedy Construction with symmetry breaking
    def greedy_construct():
        uncovered = np.ones(num_t, dtype=bool)
        uncovered_count = num_t
        blocks = []

        while uncovered_count > 0:
            # Pick a random uncovered subset
            uncovered_indices = np.where(uncovered)[0]
            s_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
            candidates = idx_np[s_idx]

            best_block = -1
            best_gain = -1

            for b in candidates:
                subs = block_subs_np[b]
                gain = np.count_nonzero(uncovered[subs])
                if gain > best_gain:
                    best_gain = gain
                    best_block = b

            if best_block != -1:
                blocks.append(best_block)
                subs = block_subs_np[best_block]
                newly_covered = subs[uncovered[subs]]
                uncovered[newly_covered] = False
                uncovered_count -= len(newly_covered)

        return blocks

    # Greedy Removal
    def greedy_remove(blocks):
        if not blocks:
            return blocks

        # Count coverage
        cover_counts = np.zeros(num_t, dtype=np.int32)
        for b in blocks:
            subs = block_subs_np[b]
            cover_counts[subs] += 1

        new_blocks = []
        for b in blocks:
            subs = block_subs_np[b]
            # Check if b is redundant (all its subsets are covered by others)
            if np.all(cover_counts[subs] > 1):
                # Remove b
                cover_counts[subs] -= 1
            else:
                new_blocks.append(b)

        return new_blocks

    # Local Search
    def local_search(blocks):
        blocks = greedy_remove(blocks)
        best_blocks = blocks[:]
        best_size = len(blocks)

        while time.time() - start_time < time_limit - 5.0:
            if not blocks:
                break

            # Pick a random block to remove
            b_idx_to_remove = random.randint(0, len(blocks) - 1)
            b = blocks[b_idx_to_remove]

            # Remove it
            temp_blocks = [blocks[i] for i in range(len(blocks)) if i != b_idx_to_remove]

            # Check if still valid
            cover_counts = np.zeros(num_t, dtype=np.int32)
            for b in temp_blocks:
                subs = block_subs_np[b]
                cover_counts[subs] += 1

            uncovered_mask = (cover_counts == 0)
            uncovered_count = np.count_nonzero(uncovered_mask)

            if uncovered_count == 0:
                # Valid without b
                blocks = temp_blocks
                blocks = greedy_remove(blocks)
                if len(blocks) < best_size:
                    best_size = len(blocks)
                    best_blocks = blocks[:]
                continue

            # Need to cover 'uncovered'
            # Pick random uncovered subset
            uncovered_indices = np.where(uncovered_mask)[0]
            s_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
            candidates = idx_np[s_idx]

            best_new_block = -1
            best_gain = -1

            for b in candidates:
                if b in temp_blocks:
                    continue
                subs = block_subs_np[b]
                gain = np.count_nonzero(uncovered_mask[subs])
                if gain > best_gain:
                    best_gain = gain
                    best_new_block = b

            if best_new_block != -1:
                # Try replacing
                new_blocks = temp_blocks + [best_new_block]
                # Run greedy remove on new set
                new_blocks = greedy_remove(new_blocks)

                if len(new_blocks) < len(blocks):
                    blocks = new_blocks
                    if len(blocks) < best_size:
                        best_size = len(blocks)
                        best_blocks = blocks[:]
                elif len(new_blocks) == len(blocks) and random.random() < 0.1:
                    # Accept equal size with some probability to escape local minima
                    blocks = new_blocks

        return best_blocks

    # Main logic
    random.seed(42)

    # Start with greedy construction
    blocks = greedy_construct()
    blocks = greedy_remove(blocks)

    # If we have time, do local search
    if time.time() - start_time < time_limit - 10.0:
        blocks = local_search(blocks)

    # Final greedy removal pass
    blocks = greedy_remove(blocks)

    # Convert block indices back to lists of points
    result = [list(all_k_blocks[b]) for b in blocks]
    return result
