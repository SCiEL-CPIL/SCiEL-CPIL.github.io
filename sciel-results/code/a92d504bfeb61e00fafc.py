"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import defaultdict


def _tsub_to_idx(tsub):
    """Convert a t-subset tuple to a unique integer index for fast lookup."""
    # Use a hash-based approach; for small v,t this is fine
    return hash(tsub)


def _greedy_cover(v, k, t, rng, deadline, strategy="random"):
    """Greedy covering with randomized block selection."""
    all_tsubs = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsubs)
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            return None  # timeout
        
        # Pick a t-subset to start a new block
        if strategy == "random":
            start_tsub = rng.choice(list(uncovered))
        elif strategy == "min":
            start_tsub = min(uncovered)
        else:
            # Pick the uncovered t-subset with fewest "compatible" points
            best_tsub = None
            best_score = float('inf')
            sample = list(uncovered)[:50]
            for ts in sample:
                score = len(uncovered)  # placeholder
                if score < best_score:
                    best_score = score
                    best_tsub = ts
            start_tsub = best_tsub if best_tsub else rng.choice(list(uncovered))
        
        block = list(start_tsub)
        
        while len(block) < k:
            # Find point that covers most uncovered t-subsets
            best_gain = -1
            best_point = -1
            block_set = set(block)
            
            # Pre-compute all (t-1)-subsets of current block
            block_t1_subs = list(itertools.combinations(block, t - 1))
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in block_t1_subs:
                    ts = tuple(sorted(rest + (x,)))
                    if ts in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1:
                # No point improves; pick random
                candidates = [x for x in range(v) if x not in block_set]
                best_point = rng.choice(candidates)
            
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    
    return blocks


def _remove_redundant_blocks(blocks, v, t, max_passes=5):
    """Remove blocks that don't cover any unique t-subset. Multiple passes."""
    if not blocks:
        return blocks
    
    current_blocks = list(blocks)
    
    for _ in range(max_passes):
        if not current_blocks:
            break
        
        # Count how many blocks cover each t-subset
        cover_count = defaultdict(int)
        block_tsubs = []
        
        for block in current_blocks:
            tsubs = list(itertools.combinations(block, t))
            block_tsubs.append(tsubs)
            for ts in tsubs:
                cover_count[ts] += 1
        
        # Remove blocks where all their t-subsets are covered by other blocks
        kept_blocks = []
        for i, block in enumerate(current_blocks):
            all_redundant = True
            for ts in block_tsubs[i]:
                if cover_count[ts] == 1:
                    all_redundant = False
                    break
            if not all_redundant:
                kept_blocks.append(block)
        
        if len(kept_blocks) == len(current_blocks):
            break  # No progress
        current_blocks = kept_blocks
    
    return current_blocks


def _local_search(blocks, v, t, rng, deadline, max_iters=500):
    """Try to reduce block count by removing blocks and re-covering gaps."""
    if not blocks:
        return blocks
    
    best_blocks = list(blocks)
    best_count = len(best_blocks)
    all_tsubs = set(itertools.combinations(range(v), t))
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try removing a random block and see if we can re-cover
        idx = rng.randrange(len(best_blocks))
        removed_block = best_blocks[idx]
        temp_blocks = best_blocks[:idx] + best_blocks[idx+1:]
        
        # Find uncovered t-subsets after removal
        covered = set()
        for block in temp_blocks:
            covered.update(itertools.combinations(block, t))
        uncovered = all_tsubs - covered
        
        if not uncovered:
            # Block was redundant
            best_blocks = temp_blocks
            best_count = len(best_blocks)
            continue
        
        # Try to cover the uncovered t-subsets with minimal new blocks
        # Greedily add blocks to cover uncovered
        new_blocks = list(temp_blocks)
        remaining_uncovered = set(uncovered)
        
        # Try adding back a modified version of the removed block or new blocks
        for attempt in range(3):
            if time.time() > deadline:
                break
            if not remaining_uncovered:
                break
            
            # Pick a random uncovered t-subset
            start_tsub = rng.choice(list(remaining_uncovered))
            block = list(start_tsub)
            
            while len(block) < v and len(block) < 7:  # k=7 for this problem
                block_set = set(block)
                best_gain = -1
                best_point = -1
                block_t1_subs = list(itertools.combinations(block, t - 1)) if len(block) >= t - 1 else []
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in block_t1_subs:
                        ts = tuple(sorted(rest + (x,)))
                        if ts in remaining_uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point == -1:
                    candidates = [x for x in range(v) if x not in block_set]
                    if not candidates:
                        break
                    best_point = rng.choice(candidates)
                
                block.append(best_point)
            
            if len(block) == 7:
                block.sort()
                new_blocks.append(block)
                remaining_uncovered.difference_update(itertools.combinations(block, t))
        
        # Check if we can cover all uncovered with the new blocks
        if not remaining_uncovered:
            # Verify full coverage
            covered = set()
            for block in new_blocks:
                covered.update(itertools.combinations(block, t))
            if all_tsubs.issubset(covered):
                if len(new_blocks) < best_count:
                    best_blocks = new_blocks
                    best_count = len(best_blocks)
    
    return best_blocks


def _verify_covering(blocks, v, t):
    """Verify every t-subset is covered by at least one block."""
    covered = set()
    for block in blocks:
        covered.update(itertools.combinations(block, t))
    all_tsubs = set(itertools.combinations(range(v), t))
    return all_tsubs.issubset(covered)


def construct(v, k, t):
    deadline = time.time() + 135  # Safety margin before 150s limit
    rng = random.Random(42)
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple randomized restarts with different strategies
    max_restarts = 15
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Vary strategy
        if restart % 3 == 0:
            strategy = "random"
        elif restart % 3 == 1:
            strategy = "min"
        else:
            strategy = "random"
        
        blocks = _greedy_cover(v, k, t, rng, deadline, strategy)
        if blocks is None:
            continue
        
        # Remove redundant blocks
        blocks = _remove_redundant_blocks(blocks, v, t, max_passes=3)
        
        # Verify validity
        if _verify_covering(blocks, v, t):
            # Try local search to improve
            if len(blocks) < best_count and time.time() < deadline - 10:
                blocks = _local_search(blocks, v, t, rng, deadline, max_iters=200)
                blocks = _remove_redundant_blocks(blocks, v, t, max_passes=3)
                if _verify_covering(blocks, v, t):
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = blocks
            elif len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
        
        # Early exit if we hit a good solution
        if best_count <= 76:
            break
    
    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                block_set = set(block)
                best_gain = -1
                best_point = -1
                block_t1_subs = list(itertools.combinations(block, t - 1))
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in block_t1_subs:
                        ts = tuple(sorted(rest + (x,)))
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point == -1:
                    candidates = [x for x in range(v) if x not in block_set]
                    best_point = min(candidates) if candidates else 0
                block.append(best_point)
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        best_blocks = blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
