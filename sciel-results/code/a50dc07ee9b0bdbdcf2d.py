import itertools
import time
import random
from collections import defaultdict

def _tsubsets(block, t):
    return list(itertools.combinations(sorted(block), t))

def _greedy_construct(v, k, t, rng, randomize_factor=0.0):
    """
    Greedy construction.
    randomize_factor: 0.0 means always pick best. >0 means pick from top candidates with probability.
    """
    # Precompute all t-subsets of range(v)
    # This is C(19, 4) = 3876. Small enough.
    all_tsubs = list(itertools.combinations(range(v), t))
    
    # Track which t-subsets are uncovered
    uncovered = set(all_tsubs)
    
    blocks = []
    
    while uncovered:
        best_blocks = []
        max_gain = -1
        
        # We can't iterate over all C(19, 7) = 50388 blocks every time?
        # 50388 * 3876 is too big?
        # 50388 blocks * 35 tsubs/block = 1.7M operations per step.
        # Number of steps ~ 150.
        # 150 * 1.7M = 255M operations. Might be too slow in Python if not optimized.
        
        # Optimization: We only care about blocks that cover *some* uncovered t-subset.
        # Or, we can just iterate over all possible blocks?
        # C(19, 7) is 50,388.
        # For each block, count how many of its t-subsets are in `uncovered`.
        
        # Let's try iterating all blocks.
        # To speed up, we can precompute t-subsets for all blocks?
        # No, memory might be an issue but 50k * 35 is fine.
        
        # Actually, a faster greedy:
        # Pick a random uncovered t-subset S.
        # Consider all blocks containing S.
        # Pick the one that covers the most OTHER uncovered t-subsets.
        
        # Let's use the "Pick random uncovered, find best block containing it" strategy.
        # This is often faster and produces diverse results.
        
        if not uncovered:
            break
            
        # Pick a random uncovered t-subset
        target = rng.choice(list(uncovered))
        
        # Find all blocks containing target
        # Target is a tuple of t elements.
        # A block contains target if target is a subset of block.
        
        best_block = None
        max_gain = -1
        candidates = []
        
        # Iterate all possible blocks?
        # Or generate blocks containing target?
        # Number of blocks containing a specific t-subset in v points:
        # We need to choose k-t elements from v-t elements.
        # C(19-4, 7-4) = C(15, 3) = 455 blocks.
        # This is much faster!
        
        remaining_points = [x for x in range(v) if x not in target]
        
        for comb in itertools.combinations(remaining_points, k - t):
            block = tuple(sorted(target + comb))
            # Calculate gain: how many uncovered t-subsets does this block cover?
            gain = 0
            for ts in _tsubsets(block, t):
                if ts in uncovered:
                    gain += 1
            
            if gain > max_gain:
                max_gain = gain
                best_block = block
                candidates = [block]
            elif gain == max_gain:
                candidates.append(block)
        
        if not candidates:
            # Should not happen if uncovered is not empty
            break
            
        # Randomize selection from candidates
        if randomize_factor > 0 and len(candidates) > 1:
            # Pick randomly from top candidates?
            # Or weighted?
            # Let's just pick randomly to diversify
            block = rng.choice(candidates)
        else:
            block = best_block
            
        blocks.append(list(block))
        
        # Update uncovered
        for ts in _tsubsets(block, t):
            uncovered.discard(ts)
            
    return blocks

def _local_search(blocks, v, k, t, rng, deadline):
    """
    Local search to reduce block count.
    1. Remove redundant blocks.
    2. Try swaps.
    """
    # Precompute t-subsets for all blocks
    block_tsubs = [_tsubsets(b, t) for b in blocks]
    
    # Count coverage
    cover_count = defaultdict(int)
    for tsubs in block_tsubs:
        for ts in tsubs:
            cover_count[ts] += 1
            
    # Phase 1: Redundancy Removal
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Shuffle indices for random order
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                return blocks, cover_count, block_tsubs
            
            # Check if block i is redundant
            redundant = True
            for ts in block_tsubs[i]:
                if cover_count[ts] == 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove block i
                for ts in block_tsubs[i]:
                    cover_count[ts] -= 1
                blocks.pop(i)
                block_tsubs.pop(i)
                changed = True
                break # Restart scan
                
    # Phase 2: Swaps
    # Try to swap a block with a random new block if it allows removal of another block
    # Or just if it maintains validity?
    # If it maintains validity, size is same. We need to trigger redundancy.
    
    # Let's try:
    # 1. Pick a block B.
    # 2. Generate random B'.
    # 3. Replace B with B'.
    # 4. Check if covering is valid.
    # 5. If valid, run redundancy removal.
    # 6. If size decreased, keep. Else, revert.
    
    # This is expensive. Let's limit it.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                return blocks, cover_count, block_tsubs
                
            original_block = blocks[i]
            original_tsubs = block_tsubs[i]
            
            # Generate a random new block
            # To be effective, maybe pick a block that covers some "weakly covered" t-subsets?
            # For now, just random.
            new_block = tuple(sorted(rng.sample(range(v), k)))
            new_tsubs = _tsubsets(new_block, t)
            
            # Hypothetically replace
            # Check if all t-subsets of new_block are covered by others OR by new_block itself
            # Actually, we just need to check if the covering remains valid.
            # A t-subset is valid if cover_count[ts] > 0 after replacement.
            
            # Decrement old
            valid = True
            for ts in original_tsubs:
                cover_count[ts] -= 1
                if cover_count[ts] < 0:
                    valid = False
                    break
            
            if not valid:
                # Revert
                for ts in original_tsubs:
                    cover_count[ts] += 1
                continue
                
            # Increment new
            for ts in new_tsubs:
                cover_count[ts] += 1
                
            # Check if any t-subset is now uncovered (count 0)
            # We only need to check t-subsets that were in original_block?
            # No, we need to check if any t-subset in the *whole system* is 0.
            # But only t-subsets in original_block *lost* coverage.
            # T-subsets in new_block *gained* coverage.
            # So we only need to check if any ts in original_tsubs has count 0.
            # Wait, if a ts was covered ONLY by original_block, and it's not in new_block, it's now 0.
            # So yes, check original_tsubs.
            
            for ts in original_tsubs:
                if cover_count[ts] == 0:
                    valid = False
                    break
            
            if not valid:
                # Revert
                for ts in original_tsubs:
                    cover_count[ts] += 1
                for ts in new_tsubs:
                    cover_count[ts] -= 1
                continue
                
            # Valid swap!
            # Now try to remove redundant blocks
            blocks[i] = list(new_block)
            block_tsubs[i] = new_tsubs
            
            # Try redundancy removal
            removed_any = False
            indices2 = list(range(len(blocks)))
            rng.shuffle(indices2)
            
            for j in indices2:
                if time.time() > deadline:
                    # Keep the swap, but stop search
                    return blocks, cover_count, block_tsubs
                    
                redundant = True
                for ts in block_tsubs[j]:
                    if cover_count[ts] == 1:
                        redundant = False
                        break
                        
                if redundant:
                    for ts in block_tsubs[j]:
                        cover_count[ts] -= 1
                    blocks.pop(j)
                    block_tsubs.pop(j)
                    removed_any = True
                    break
                    
            if removed_any:
                improved = True
                break # Restart outer loop
                
            # If no removal, revert the swap?
            # If no removal, the size is the same.
            # We might want to keep it to explore different states, but it doesn't help reduce size directly.
            # However, it might enable future swaps.
            # For now, let's revert if no size reduction to avoid wandering.
            # Actually, let's keep it with some probability? No, let's revert to stay focused on reduction.
            
            # Revert
            blocks[i] = original_block
            block_tsubs[i] = original_tsubs
            for ts in original_tsubs:
                cover_count[ts] -= 1
            for ts in new_tsubs:
                cover_count[ts] += 1

    return blocks, cover_count, block_tsubs

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    rng = random.Random(42)
    
    best_blocks = None
    best_size = float('inf')
    
    # Multi-start
    num_starts = 10
    for i in range(num_starts):
        if time.time() > deadline:
            break
            
        # Vary randomization
        # Start 0: Deterministic
        # Start 1+: Randomized
        rand_factor = 0.0 if i == 0 else 1.0
        
        # Greedy
        blocks = _greedy_construct(v, k, t, rng, rand_factor)
        
        # Local Search
        blocks, _, _ = _local_search(blocks, v, k, t, rng, deadline)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            # If we hit a very good score, we can break early?
            # Lower bound is 131. Best known 151.
            # If we get <= 151, it's great.
            if best_size <= 151:
                break
                
    if best_blocks is None:
        # Fallback: Just return greedy
        blocks = _greedy_construct(v, k, t, rng, 0.0)
        return blocks
        
    return best_blocks
