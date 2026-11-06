import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 118.0

    points = range(12)

    # 1. Enumerate all 5-subsets and assign indices
    all_5_subsets = list(combinations(points, 5))
    n_5 = len(all_5_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_5_subsets)}

    # 2. Enumerate all 6-subsets (blocks)
    all_blocks = list(combinations(points, 6))
    n_blocks = len(all_blocks)

    # 3. For each block, compute its mask of 5-subsets
    block_masks = []
    blocks_covering = [[] for _ in range(n_5)]

    for b in all_blocks:
        mask = 0
        for s in combinations(b, 5):
            idx = subset_to_idx[s]
            mask |= (1 << idx)
            blocks_covering[idx].append(len(block_masks))
        block_masks.append(mask)

    # Symmetry breaking: Fix first block to contain 0,1,2,3,4,5
    # The automorphism group of S(5,6,12) is M12, which is transitive on points
    # We can fix the first block to be {0,1,2,3,4,5}
    first_block = (0, 1, 2, 3, 4, 5)
    first_block_idx = all_blocks.index(first_block)

    # Second block: fix to contain point 6 (not in first block) and 3 more from {6,7,8,9,10,11}
    # To break more symmetry, fix second block to be (6, 7, 8, 9, 10, 11)
    second_block = (6, 7, 8, 9, 10, 11)
    second_block_idx = all_blocks.index(second_block)

    # Initialize
    initial_uncovered = (1 << n_5) - 1
    initial_uncovered &= ~block_masks[first_block_idx]
    initial_uncovered &= ~block_masks[second_block_idx]

    selected_blocks = [first_block, second_block]
    selected_indices = {first_block_idx, second_block_idx}

    step_count = 0
    best_partial = [first_block, second_block]
    best_len = 2
    time_exceeded = False

    def check_time():
        nonlocal time_exceeded
        if time.time() > deadline:
            time_exceeded = True
            return True
        return False

    def solve(uncovered_mask):
        nonlocal step_count, best_partial, best_len, time_exceeded
        step_count += 1
        if check_time():
            return None

        if uncovered_mask == 0:
            if len(selected_blocks) > best_len:
                best_len = len(selected_blocks)
                best_partial = selected_blocks[:]
            return selected_blocks[:]

        # Find the 5-subset with minimum remaining values (MRV)
        min_count = float('inf')
        best_subset_idx = -1
        best_candidates = []

        m = uncovered_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            m ^= lsb

            # Count valid blocks for this idx
            count = 0
            candidates = []
            for b_idx in blocks_covering[idx]:
                if b_idx in selected_indices:
                    continue
                if block_masks[b_idx] & ~uncovered_mask == 0:
                    count += 1
                    candidates.append(b_idx)

            if count < min_count:
                min_count = count
                best_subset_idx = idx
                best_candidates = candidates

            if min_count == 0:
                return None # Dead end

        if not best_candidates:
            return None

        # Value Ordering: Sort candidates by how many new cells they cover (descending)
        # Use a lookahead: prefer blocks that cover more currently-uncovered 5-subsets
        best_candidates.sort(key=lambda b_idx: (block_masks[b_idx] & uncovered_mask).bit_count(), reverse=True)

        for b_idx in best_candidates:
            # Select this block
            new_uncovered = uncovered_mask & ~block_masks[b_idx]
            selected_blocks.append(all_blocks[b_idx])
            selected_indices.add(b_idx)

            result = solve(new_uncovered)
            if result is not None:
                return result

            # Backtrack
            selected_blocks.pop()
            selected_indices.remove(b_idx)

            if time_exceeded:
                return None

        # Update best partial if we are deeper than before
        if len(selected_blocks) > best_len:
            best_len = len(selected_blocks)
            best_partial = selected_blocks[:]
            
        return None

    result = solve(initial_uncovered)

    if result is None:
        return best_partial

    return result
