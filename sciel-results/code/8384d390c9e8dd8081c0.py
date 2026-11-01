import itertools
import random
import time

def construct(v, k, t):
    all_points = list(range(v))
    total_5subsets = list(itertools.combinations(all_points, t))
    n_5subsets = len(total_5subsets)
    
    # Map each 5-subset to an index
    subset_to_idx = {s: i for i, s in enumerate(total_5subsets)}
    
    # Precompute all possible blocks? 
    # C(19, 8) = 75582. This is small enough to precompute their 5-subsets?
    # Each block has C(8,5)=56 subsets.
    # 75582 * 56 ~ 4.2 million entries. 
    # Storing this as a list of lists of indices might use significant memory but is fast.
    # Let's try to generate on the fly or use a smarter approach.
    
    # Let's stick to on-the-fly for memory safety, but optimize the greedy step.
    
    def get_block_subsets_indices(block):
        """Return list of indices of 5-subsets contained in block"""
        subs = []
        for combo in itertools.combinations(block, t):
            idx = subset_to_idx.get(combo)
            if idx is not None:
                subs.append(idx)
        return subs

    def greedy_init():
        """Greedy Set Cover with randomized candidates"""
        blocks = []
        block_subs = [] # list of lists of subset indices for each block
        cover_count = [0] * n_5subsets
        uncovered = [i for i in range(n_5subsets)]
        
        rng = random.Random(42)
        
        # Precompute all blocks? 
        # Let's generate candidates on the fly.
        
        while uncovered:
            # Pick a random uncovered subset to focus on
            # Weighted pick: subsets covered by fewer blocks are harder? 
            # Initially all 0. Let's just pick random uncovered.
            target_idx = rng.choice(uncovered)
            target_subset = total_5subsets[target_idx]
            
            # Generate M candidates containing target_subset
            M = 20
            best_block = None
            best_count = -1
            best_subs = []
            
            # We need to fill k-t = 3 more points
            available_points = [p for p in all_points if p not in target_subset]
            
            for _ in range(M):
                # Pick 3 random points from available
                extra = rng.sample(available_points, k - t)
                candidate = tuple(sorted(list(target_subset) + extra))
                
                # Calculate how many uncovered subsets this candidate covers
                cand_subs = get_block_subsets_indices(candidate)
                count = 0
                for idx in cand_subs:
                    if cover_count[idx] == 0:
                        count += 1
                
                if count > best_count:
                    best_count = count
                    best_block = candidate
                    best_subs = cand_subs
            
            if best_block is None:
                # Fallback: just pick one
                extra = rng.sample(available_points, k - t)
                best_block = tuple(sorted(list(target_subset) + extra))
                best_subs = get_block_subsets_indices(best_block)
            
            # Add best_block
            blocks.append(best_block)
            block_subs.append(best_subs)
            
            new_uncovered = []
            for idx in range(n_5subsets):
                if idx in best_subs:
                    cover_count[idx] += 1
                if cover_count[idx] == 0:
                    new_uncovered.append(idx)
            
            uncovered = new_uncovered
            
        return blocks, block_subs, cover_count

    def prune(blocks, block_subs, cover_count):
        """Remove redundant blocks"""
        i = 0
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(blocks):
                redundant = True
                for idx in block_subs[i]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for idx in block_subs[i]:
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    block_subs.pop(i)
                    changed = True
                    # Do not increment i, check same index again
                else:
                    i += 1

    start_time = time.time()
    time_limit = 110
    
    # 1. Initialization
    blocks, block_subs, cover_count = greedy_init()
    prune(blocks, block_subs, cover_count)
    
    best_len = len(blocks)
    best_blocks = [list(b) for b in blocks]
    
    rng = random.Random(123)
    
    # 2. Local Search Loop
    # Strategy: Try to replace a block with another block that covers its unique subsets.
    # To find a good replacement, we can look at the unique subsets of the block to be removed.
    # If the union of points in unique subsets is <= k, we can form a block.
    # If it is > k, we can't remove it directly.
    # We can also try "swap" with a random block? No, that's weak.
    # Let's try to improve the block itself: 
    # For a block B, let U be the set of unique 5-subsets it covers.
    # We want a new block B' that covers U.
    # If we can find B' such that B' covers U and maybe more, and B' is not B, we can swap.
    # Actually, if B' covers U, then B is redundant if B' is added? No, B' must cover *all* unique subsets of B.
    
    while time.time() - start_time < time_limit:
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
        if not blocks:
            break
            
        i = rng.randint(0, len(blocks) - 1)
        
        # Find unique subsets of block i
        unique_subs = []
        for idx in block_subs[i]:
            if cover_count[idx] == 1:
                unique_subs.append(idx)
                
        if not unique_subs:
            # Block is redundant, remove it
            for idx in block_subs[i]:
                cover_count[idx] -= 1
            blocks.pop(i)
            block_subs.pop(i)
            continue
            
        # Calculate union of points in unique subsets
        union_points = set()
        for idx in unique_subs:
            union_points.update(total_5subsets[idx])
            
        if len(union_points) > k:
            # Cannot cover all unique subsets with a single block
            continue
            
        # We need to form a block containing union_points.
        # There are k - len(union_points) spots to fill.
        # We want to choose the fillers such that the new block covers as many *other* uncovered or singly covered subsets as possible?
        # Or just any valid block?
        # Let's try a few random fillers and pick the best one.
        
        available_points = [p for p in all_points if p not in union_points]
        num_fillers = k - len(union_points)
        
        best_new_block = None
        best_score = -1
        best_new_subs = []
        
        # Try M random fillers
        M = 10
        for _ in range(M):
            if num_fillers > 0:
                fillers = rng.sample(available_points, num_fillers)
            else:
                fillers = []
            
            new_block = tuple(sorted(list(union_points) + fillers))
            
            if new_block == blocks[i]:
                continue
                
            new_subs = get_block_subsets_indices(new_block)
            
            # Score: How many subsets does it cover? 
            # Ideally, it should cover all unique_subs.
            # It does by construction.
            # Let's check if it covers any *other* subsets that are currently covered by 0 or 1?
            # Actually, we just want to ensure it's a valid replacement.
            # A better score: number of subsets covered by new_block that are NOT covered by any other block 
            # (i.e., they become unique to new_block) minus the number of subsets that were unique to old_block 
            # and are NOT covered by new_block? 
            # Wait, if new_block covers all unique_subs of old_block, then removing old_block and adding new_block 
            # keeps the cover valid.
            # The "quality" of the move depends on whether new_block allows further pruning.
            # A simple heuristic: prefer new_block that covers more total subsets?
            # Or prefer new_block that overlaps less with other blocks?
            
            # Let's just pick the first valid one or randomize.
            # To avoid getting stuck, let's pick the one with the most total coverage?
            score = len(new_subs)
            
            if score > best_score:
                best_score = score
                best_new_block = new_block
                best_new_subs = new_subs
        
        if best_new_block is None:
            continue
            
        # Perform swap
        # Remove old block i
        for idx in block_subs[i]:
            cover_count[idx] -= 1
        blocks.pop(i)
        block_subs.pop(i)
        
        # Add new block
        # Check if new_block already exists?
        # If it does, we don't add it, we just keep the existing one.
        # This effectively reduces size by 1.
        exists = False
        for j, b in enumerate(blocks):
            if b == best_new_block:
                exists = True
                break
        
        if not exists:
            blocks.append(best_new_block)
            block_subs.append(best_new_subs)
            for idx in best_new_subs:
                cover_count[idx] += 1
        
        # Prune
        prune(blocks, block_subs, cover_count)
        
    return best_blocks
