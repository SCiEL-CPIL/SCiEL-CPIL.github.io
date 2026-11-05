# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    """
    Vectorized greedy covering design with randomized restarts and local search.
    """
    deadline = time.time() + 115  # 115s deadline to be safe within 120s
    
    # Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute for each t-subset index, the set of points in it
    t_sub_points = [set(ts) for ts in all_t_subsets]
    
    # Precompute: for each point x, which t-subsets contain x
    t_subs_containing_point = [[] for _ in range(v)]
    for i, ts in enumerate(all_t_subsets):
        for x in ts:
            t_subs_containing_point[x].append(i)
    
    def greedy_cover(seed=None):
        """Run greedy covering with a given random seed."""
        if seed is not None:
            random.seed(seed)
        
        # Uncovered mask: True if uncovered
        uncovered = np.ones(n_t, dtype=bool)
        uncovered_count = n_t
        
        blocks = []
        
        while uncovered_count > 0:
            if time.time() > deadline:
                break
            
            # Find the best block to add
            # Pick a random uncovered t-subset to start from
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            
            start_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
            start_ts = all_t_subsets[start_idx]
            block = list(start_ts)
            block_set = set(block)
            
            # Greedily add points
            while len(block) < k:
                best_x = -1
                best_gain = 0
                block_set = set(block)
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Count how many uncovered t-subsets would be covered by adding x
                    gain = 0
                    # For each (t-1)-subset of current block, check if {x} + that subset is uncovered
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in t_sub_to_idx:
                            idx = t_sub_to_idx[ts]
                            if uncovered[idx]:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x >= 0:
                    block.append(best_x)
                else:
                    # No point improves, pick random
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        block.append(random.choice(remaining))
                    break
            
            if len(block) < k:
                break
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for ts in itertools.combinations(block, t):
                if ts in t_sub_to_idx:
                    idx = t_sub_to_idx[ts]
                    if uncovered[idx]:
                        uncovered[idx] = False
                        uncovered_count -= 1
        
        return blocks, uncovered_count
    
    def remove_redundant(blocks):
        """Remove blocks that don't cover any unique t-subset."""
        if not blocks:
            return blocks
        
        # Count coverage for each t-subset
        cover_count = defaultdict(int)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts] += 1
        
        new_blocks = []
        for b in blocks:
            # Check if this block covers any t-subset that no other block covers
            essential = False
            for ts in itertools.combinations(b, t):
                if cover_count[ts] == 1:
                    essential = True
                    break
            if essential:
                new_blocks.append(b)
            else:
                # Decrement counts for this block's t-subsets
                for ts in itertools.combinations(b, t):
                    cover_count[ts] -= 1
        
        return new_blocks
    
    def local_search(blocks, max_iter=20):
        """Try to improve by replacing blocks."""
        if not blocks:
            return blocks
        
        best_blocks = list(blocks)
        best_count = len(best_blocks)
        
        for _ in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Pick a random block to try replacing
            idx = random.randint(0, len(best_blocks) - 1)
            old_block = best_blocks[idx]
            
            # Remove it
            best_blocks.pop(idx)
            
            # Find uncovered t-subsets
            cover_count = defaultdict(int)
            for b in best_blocks:
                for ts in itertools.combinations(b, t):
                    cover_count[ts] += 1
            
            uncovered = []
            for ts in all_t_subsets:
                if cover_count[ts] == 0:
                    uncovered.append(ts)
            
            if not uncovered:
                # No need to replace, put it back
                best_blocks.insert(idx, old_block)
                continue
            
            # Greedily construct a new block to cover as many uncovered as possible
            first_uncovered = uncovered[random.randint(0, len(uncovered) - 1)]
            new_block = list(first_uncovered)
            new_block_set = set(new_block)
            
            while len(new_block) < k:
                best_x = -1
                best_gain = 0
                new_block_set = set(new_block)
                
                for x in range(v):
                    if x in new_block_set:
                        continue
                    
                    gain = 0
                    for rest in itertools.combinations(new_block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in cover_count and cover_count[ts] == 0:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x >= 0:
                    new_block.append(best_x)
                else:
                    remaining = [x for x in range(v) if x not in new_block_set]
                    if remaining:
                        new_block.append(random.choice(remaining))
                    break
            
            if len(new_block) < k:
                best_blocks.insert(idx, old_block)
                continue
            
            new_block.sort()
            best_blocks.insert(idx, new_block)
            
            # Check if this is valid (all t-subsets covered)
            cover_count2 = defaultdict(int)
            for b in best_blocks:
                for ts in itertools.combinations(b, t):
                    cover_count2[ts] += 1
            
            valid = all(cover_count2[ts] > 0 for ts in all_t_subsets)
            
            if not valid:
                # Revert
                best_blocks.pop(idx)
                best_blocks.insert(idx, old_block)
        
        return best_blocks
    
    # Run multiple greedy restarts with different seeds
    best_blocks = None
    best_count = float('inf')
    
    n_restarts = 5  # Number of restarts
    for i in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        blocks, unc = greedy_cover(seed=i * 12345 + 67890)
        
        if unc == 0:  # Valid covering
            # Remove redundant blocks
            blocks = remove_redundant(blocks)
            
            # Local search
            blocks = local_search(blocks, max_iter=10)
            
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = list(blocks)
    
    # If no valid covering found, do one more greedy without time limit check
    if best_blocks is None:
        blocks, unc = greedy_cover(seed=42)
        if unc > 0:
            # Add blocks for remaining uncovered
            uncovered = np.ones(n_t, dtype=bool)
            for b in blocks:
                for ts in itertools.combinations(b, t):
                    if ts in t_sub_to_idx:
                        uncovered[t_sub_to_idx[ts]] = False
            
            while np.any(uncovered):
                if time.time() > deadline - 1:
                    break
                uncovered_indices = np.where(uncovered)[0]
                start_idx = uncovered_indices[0]
                start_ts = all_t_subsets[start_idx]
                block = list(start_ts)
                block_set = set(block)
                
                while len(block) < k:
                    best_x = -1
                    best_gain = 0
                    for x in range(v):
                        if x in block_set:
                            continue
                        gain = 0
                        for rest in itertools.combinations(block, t - 1):
                            ts = tuple(sorted(rest + (x,)))
                            if ts in t_sub_to_idx:
                                idx = t_sub_to_idx[ts]
                                if uncovered[idx]:
                                    gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_x >= 0:
                        block.append(best_x)
                        block_set.add(best_x)
                    else:
                        remaining = [x for x in range(v) if x not in block_set]
                        if remaining:
                            block.append(random.choice(remaining))
                            block_set.add(block[-1])
                        else:
                            break
                
                if len(block) == k:
                    block.sort()
                    blocks.append(block)
                    for ts in itertools.combinations(block, t):
                        if ts in t_sub_to_idx:
                            uncovered[t_sub_to_idx[ts]] = False
        
        best_blocks = blocks
        best_count = len(best_blocks)
    
    # Final cleanup: remove duplicates
    seen = set()
    unique_blocks = []
    for b in best_blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)
    
    return unique_blocks

# EVOLVE-BLOCK-END
