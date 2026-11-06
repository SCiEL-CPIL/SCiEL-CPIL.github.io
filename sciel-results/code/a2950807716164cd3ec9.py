import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    points = range(12)

    # 1. Enumerate all 5-subsets and assign indices
    all_5_subsets = list(combinations(points, 5))
    n_5 = len(all_5_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_5_subsets)}

    # 2. Enumerate all 6-subsets (blocks)
    all_blocks = list(combinations(points, 6))
    n_blocks = len(all_blocks)

    # 3. For each block, compute its mask of 5-subsets
    #    Also, create a list of blocks for each 5-subset index
    block_masks = []
    blocks_covering = [[] for _ in range(n_5)]

    for b in all_blocks:
        mask = 0
        for s in combinations(b, 5):
            idx = subset_to_idx[s]
            mask |= (1 << idx)
            blocks_covering[idx].append(len(block_masks))
        block_masks.append(mask)

    # 4. Solve Exact Cover
    # State: uncovered_mask (bits set for subsets that still need covering)
    # Initial state: all bits set (all 792 subsets need covering)
    # Goal: uncovered_mask == 0

    # Symmetry breaking:
    # Fix the first block to (0, 1, 2, 3, 4, 5).
    # Fix the second block to (6, 7, 8, 9, 10, 11) (the complement).
    # This is valid because the S(5,6,12) design is self-complementary and unique.
    
    first_block = (0, 1, 2, 3, 4, 5)
    second_block = (6, 7, 8, 9, 10, 11)
    
    first_block_idx = all_blocks.index(first_block)
    second_block_idx = all_blocks.index(second_block)

    # Initialize
    initial_uncovered = (1 << n_5) - 1
    initial_uncovered &= ~block_masks[first_block_idx]
    initial_uncovered &= ~block_masks[second_block_idx]

    selected_blocks = [first_block, second_block]

    step_count = 0

    def solve(uncovered_mask, selected_count):
        nonlocal step_count
        step_count += 1
        if step_count % 100 == 0:
            if time.time() > deadline:
                return None

        if uncovered_mask == 0:
            return selected_blocks[:]

        # Find the 5-subset with minimum remaining values (MRV)
        min_count = float('inf')
        best_subset_idx = -1
        best_candidates = []

        # Iterate over set bits in uncovered_mask
        m = uncovered_mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            m ^= lsb

            # Count valid blocks for this idx
            count = 0
            candidates = []
            for b_idx in blocks_covering[idx]:
                # Skip if this block is already selected?
                # The mask check `block_masks[b_idx] & ~uncovered_mask == 0` ensures
                # that the block only covers subsets that are currently uncovered.
                # If a block was already selected, its subsets are no longer in uncovered_mask.
                # So if we pick a previously selected block, its mask would be 0 w.r.t uncovered_mask?
                # No, if we pick a previously selected block, it would have `block_masks[b_idx] & ~uncovered_mask == 0` 
                # only if it covers NO uncovered subsets. But it covers subsets that are ALREADY covered.
                # So `block_masks[b_idx] & ~uncovered_mask` would be non-zero?
                # Wait. `uncovered_mask` has 0s for covered subsets.
                # `~uncovered_mask` has 1s for covered subsets.
                # If a block covers only covered subsets, `block_masks[b_idx] & ~uncovered_mask` is non-zero.
                # The condition is `block_masks[b_idx] & ~uncovered_mask == 0`.
                # This means the block covers NO covered subsets. i.e. it only covers uncovered subsets.
                # So we cannot re-select a block.
                
                if block_masks[b_idx] & ~uncovered_mask == 0:
                    count += 1
                    candidates.append(b_idx)

            if count < min_count:
                min_count = count
                best_subset_idx = idx
                best_candidates = candidates

            if min_count == 0:
                return None # Dead end

        # Try each candidate for the best_subset_idx
        for b_idx in best_candidates:
            # Select this block
            new_uncovered = uncovered_mask & ~block_masks[b_idx]
            selected_blocks.append(all_blocks[b_idx])

            result = solve(new_uncovered, selected_count + 1)
            if result is not None:
                return result

            # Backtrack
            selected_blocks.pop()

        return None

    result = solve(initial_uncovered, 2)

    if result is None:
        return []

    return result
