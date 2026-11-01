import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s
    
    # Precompute all t-subsets and their indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {s: i for i, s in enumerate(all_tsubs)}
    
    # Bitset representation: each block is a bitmask of which t-subsets it covers
    # We'll use Python integers as bitsets
    
    def block_bitmask(block):
        mask = 0
        for tsub in itertools.combinations(block, t):
            mask |= (1 << tsub_index[tsub])
        return mask
    
    # Greedy construction
    uncovered_mask = (1 << n_tsubs) - 1  # all t-subsets uncovered
    blocks = []
    block_masks = []
    
    while uncovered_mask:
        # Pick the first uncovered t-subset (lowest set bit)
        # Find the index of the lowest set bit
        lowest_set = uncovered_mask & (-uncovered_mask)
        idx = lowest_set.bit_length() - 1
        start_tsub = all_tsubs[idx]
        
        # Build a block containing start_tsub, greedily adding points
        block = list(start_tsub)
        
        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Count how many currently uncovered t-subsets would be covered
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    tsub = tuple(sorted(rest + (x,)))
                    bit = 1 << tsub_index[tsub]
                    if uncovered_mask & bit:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)
        
        block.sort()
        mask = block_bitmask(block)
        blocks.append(block)
        block_masks.append(mask)
        uncovered_mask &= ~mask
        
        if time.time() > deadline - 10:
            break
    
    # Remove redundant blocks (keep only if they cover at least one unique t-subset)
    # Do this in reverse order to potentially remove more
    n_blocks = len(blocks)
    if n_blocks > 0:
        # Compute for each block, which t-subsets it uniquely covers
        # First, compute coverage count for each t-subset
        coverage_count = [0] * n_tsubs
        for mask in block_masks:
            m = mask
            while m:
                lowest = m & (-m)
                idx = lowest.bit_length() - 1
                coverage_count[idx] += 1
                m &= m - 1
        
        # Now check each block: does it cover any t-subset with count 1?
        keep = [True] * n_blocks
        for i, mask in enumerate(block_masks):
            m = mask
            has_unique = False
            while m:
                lowest = m & (-m)
                idx = lowest.bit_length() - 1
                if coverage_count[idx] == 1:
                    has_unique = True
                    break
                m &= m - 1
            if not has_unique:
                keep[i] = False
                # Update coverage counts
                m = mask
                while m:
                    lowest = m & (-m)
                    idx = lowest.bit_length() - 1
                    coverage_count[idx] -= 1
                    m &= m - 1
        
        # Rebuild
        new_blocks = []
        new_masks = []
        for i in range(n_blocks):
            if keep[i]:
                new_blocks.append(blocks[i])
                new_masks.append(block_masks[i])
        blocks = new_blocks
        block_masks = new_masks
    
    # Variable Neighborhood Search (VNS) for improvement
    # Try to swap one block for a better one
    def count_uncovered():
        covered = 0
        for mask in block_masks:
            covered |= mask
        return n_tsubs - covered.bit_count()
    
    current_uncovered = count_uncovered()
    
    # VNS: try replacing each block with a greedily constructed one that covers more
    # We'll do this for a limited number of iterations
    vns_start = time.time()
    max_vns_time = deadline - vns_start - 5
    
    # Try to improve by replacing blocks
    # Strategy: for each block, try to find a replacement that covers more uncovered t-subsets
    for _ in range(50):  # limit iterations
        if time.time() > vns_start + max_vns_time:
            break
        
        # Recompute uncovered
        covered_mask = 0
        for mask in block_masks:
            covered_mask |= mask
        uncovered_mask = ((1 << n_tsubs) - 1) & ~covered_mask
        
        if uncovered_mask == 0:
            break
        
        # Pick a random block to replace
        i = random.randint(0, len(blocks) - 1)
        
        # Try to construct a new block that covers more uncovered t-subsets
        # Start from the least-covered uncovered t-subset
        lowest_set = uncovered_mask & (-uncovered_mask)
        idx = lowest_set.bit_length() - 1
        start_tsub = all_tsubs[idx]
        
        new_block = list(start_tsub)
        while len(new_block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in new_block:
                    continue
                gain = 0
                for rest in itertools.combinations(new_block, t - 1):
                    tsub = tuple(sorted(rest + (x,)))
                    bit = 1 << tsub_index[tsub]
                    if uncovered_mask & bit:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            new_block.append(best_x)
        
        new_block.sort()
        new_mask = block_bitmask(new_block)
        
        # Check if new block covers more uncovered t-subsets than old block
        old_gain = (block_masks[i] & uncovered_mask).bit_count()
        new_gain = (new_mask & uncovered_mask).bit_count()
        
        if new_gain > old_gain:
            blocks[i] = new_block
            block_masks[i] = new_mask
    
    # Final redundancy removal
    n_blocks = len(blocks)
    if n_blocks > 0:
        coverage_count = [0] * n_tsubs
        for mask in block_masks:
            m = mask
            while m:
                lowest = m & (-m)
                idx = lowest.bit_length() - 1
                coverage_count[idx] += 1
                m &= m - 1
        
        keep = [True] * n_blocks
        for i, mask in enumerate(block_masks):
            m = mask
            has_unique = False
            while m:
                lowest = m & (-m)
                idx = lowest.bit_length() - 1
                if coverage_count[idx] == 1:
                    has_unique = True
                    break
                m &= m - 1
            if not has_unique:
                keep[i] = False
                m = mask
                while m:
                    lowest = m & (-m)
                    idx = lowest.bit_length() - 1
                    coverage_count[idx] -= 1
                    m &= m - 1
        
        new_blocks = []
        for i in range(n_blocks):
            if keep[i]:
                new_blocks.append(blocks[i])
        blocks = new_blocks
    
    return blocks
