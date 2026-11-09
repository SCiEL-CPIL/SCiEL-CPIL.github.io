import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Precompute all t-subsets as sorted tuples
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index for bitmask tracking
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute for each k-block, the list of t-subset indices it covers
    # But C(24,7) = 346104, which is large. Instead, compute on-the-fly with caching.
    
    # Better: precompute for each point x, the list of t-subsets containing x
    # Actually, let's use a different approach:
    # For each k-subset, we need to know which t-subsets it covers.
    # C(24,7) = 346104 blocks, each covering C(7,3) = 35 t-subsets.
    # Total: 346104 * 35 = 12,113,640 entries. Too many to precompute all.
    
    # Alternative: use bitmask over t-subsets. n_tsubs = C(24,3) = 2024.
    # 2024 bits = 253 bytes per block. For many blocks this is fine.
    
    # Let's use Python integers as bitmasks (arbitrary precision)
    
    def block_mask(block):
        """Return bitmask of t-subsets covered by this block."""
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            mask |= (1 << idx)
        return mask
    
    # Precompute masks for all possible k-blocks? C(24,7)=346104, each taking ~35 ops.
    # 346104 * 35 = ~12M ops. Might be too slow in Python.
    # Let's compute on demand with a cache.
    
    mask_cache = {}
    
    def get_block_mask(block):
        key = tuple(sorted(block))
        if key not in mask_cache:
            mask_cache[key] = block_mask(list(key))
        return mask_cache[key]
    
    def greedy_cover(randomize=False):
        """Greedy covering with optional randomization in tie-breaking."""
        # uncovered_mask: bit i is set if t-subset i is uncovered
        uncovered_mask = (1 << n_tsubs) - 1
        blocks = []
        
        while uncovered_mask:
            if randomize:
                # Pick a random uncovered t-subset to start the block
                # Find a random set bit
                bits = [i for i in range(n_tsubs) if (uncovered_mask >> i) & 1]
                if not bits:
                    break
                target_idx = random.choice(bits)
                target_ts = all_tsubs[target_idx]
            else:
                # Pick the first uncovered t-subset (lowest index)
                # Find lowest set bit
                target_idx = (uncovered_mask & (-uncovered_mask)).bit_length() - 1
                target_ts = all_tsubs[target_idx]
            
            # Start block with the t-subset's points
            block = list(target_ts)
            
            # Add points one at a time to maximize coverage of uncovered t-subsets
            while len(block) < k:
                best_gain = -1
                best_points = []
                for x in range(v):
                    if x in block:
                        continue
                    # Compute gain: how many currently uncovered t-subsets would be covered
                    # by adding x to the block
                    gain = 0
                    # For each (t-1)-subset of current block, check if adding x forms an uncovered t-subset
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsub_to_idx[ts]
                        if (uncovered_mask >> idx) & 1:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain and gain > 0:
                        best_points.append(x)
                
                if randomize and best_points:
                    block.append(random.choice(best_points))
                else:
                    block.append(best_points[0])
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered mask
            bm = get_block_mask(block)
            uncovered_mask &= ~bm
        
        return blocks
    
    def try_remove_blocks(blocks):
        """Try to remove redundant blocks."""
        n = len(blocks)
        if n == 0:
            return blocks
        
        # Compute coverage count for each t-subset
        cover_count = [0] * n_tsubs
        block_masks = [get_block_mask(b) for b in blocks]
        
        for bm in block_masks:
            # Count bits
            temp = bm
            while temp:
                lsb = temp & (-temp)
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                temp &= temp - 1
        
        # Try removing each block if it doesn't break coverage
        new_blocks = []
        for i in range(n):
            bm = block_masks[i]
            # Check if any t-subset has cover_count == 1 and is in this block
            redundant = True
            temp = bm
            while temp:
                lsb = temp & (-temp)
                idx = lsb.bit_length() - 1
                if cover_count[idx] == 1:
                    redundant = False
                    break
                temp &= temp - 1
            
            if redundant:
                # Remove this block
                temp = bm
                while temp:
                    lsb = temp & (-temp)
                    idx = lsb.bit_length() - 1
                    cover_count[idx] -= 1
                    temp &= temp - 1
            else:
                new_blocks.append(blocks[i])
        
        return new_blocks
    
    # Run multiple greedy restarts
    best_blocks = None
    best_len = float('inf')
    
    # First, run a deterministic greedy to get a baseline
    blocks = greedy_cover(randomize=False)
    blocks = try_remove_blocks(blocks)
    best_blocks = blocks
    best_len = len(blocks)
    
    # Now run randomized restarts
    random.seed(42)
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        blocks = greedy_cover(randomize=True)
        blocks = try_remove_blocks(blocks)
        if len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)
            # Check if we can do better with more removal passes
            for _ in range(3):
                new_blocks = try_remove_blocks(best_blocks)
                if len(new_blocks) < len(best_blocks):
                    best_blocks = new_blocks
                    best_len = len(best_blocks)
                else:
                    break
        
        # If we've hit the lower bound, stop early
        if best_len <= 69:
            break
    
    return best_blocks
