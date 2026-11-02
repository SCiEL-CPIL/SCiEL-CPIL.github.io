import itertools
import random
import time
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin for 120s limit
    
    # Pre-compute all t-subsets and map to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Pre-compute for each point, which t-subsets contain it
    point_tsubs = defaultdict(set)
    for i, ts in enumerate(all_tsubs):
        for p in ts:
            point_tsubs[p].add(i)
    
    # Helper to compute coverage of a block as a bitmask (using Python int for bitsets)
    def block_coverage_mask(block):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            mask |= (1 << idx)
        return mask
    
    # Helper to remove redundant blocks using coverage masks
    def remove_redundant_fast(blocks):
        if not blocks:
            return blocks
        
        # Compute coverage mask for each block
        block_masks = [block_coverage_mask(b) for b in blocks]
        full_mask = (1 << n_tsubs) - 1
        
        # Iteratively remove redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(block_masks)):
                # Check if block i is redundant
                if i == len(block_masks) - 1:
                    remaining_mask = 0
                    for j in range(i):
                        remaining_mask |= block_masks[j]
                else:
                    remaining_mask = 0
                    for j in range(len(block_masks)):
                        if j != i:
                            remaining_mask |= block_masks[j]
                
                if (remaining_mask & full_mask) == full_mask:
                    block_masks.pop(i)
                    blocks.pop(i)
                    changed = True
                    break
        
        return blocks
    
    # Local search: try to swap or remove blocks to reduce count
    def local_search(blocks, max_time=10):
        if not blocks:
            return blocks
        
        ls_start = time.time()
        block_masks = [block_coverage_mask(b) for b in blocks]
        full_mask = (1 << n_tsubs) - 1
        
        # Precompute for each t-subset, which blocks cover it
        tsub_block_list = [[] for _ in range(n_tsubs)]
        for bi, mask in enumerate(block_masks):
            for ts_idx, ts in enumerate(all_tsubs):
                if mask & (1 << ts_idx):
                    tsub_block_list[ts_idx].append(bi)
        
        improved = True
        while improved and time.time() - ls_start < max_time:
            improved = False
            
            # Try removing blocks
            for i in range(len(blocks)):
                # Check if block i can be removed
                remaining_mask = 0
                for j in range(len(block_masks)):
                    if j != i:
                        remaining_mask |= block_masks[j]
                
                if (remaining_mask & full_mask) == full_mask:
                    block_masks.pop(i)
                    blocks.pop(i)
                    # Update tsub_block_list
                    for ts_idx in range(n_tsubs):
                        if i in tsub_block_list[ts_idx]:
                            tsub_block_list[ts_idx].remove(i)
                        # Re-index blocks after i
                        for bi in tsub_block_list[ts_idx]:
                            if bi > i:
                                tsub_block_list[ts_idx][tsub_block_list[ts_idx].index(bi)] = bi - 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping: replace block i with a new block that covers more uncovered t-subsets
            for i in range(len(blocks)):
                if time.time() - ls_start > max_time:
                    break
                
                # Find t-subsets only covered by block i
                only_i_mask = 0
                for ts_idx in range(n_tsubs):
                    if len(tsub_block_list[ts_idx]) == 1 and tsub_block_list[ts_idx][0] == i:
                        only_i_mask |= (1 << ts_idx)
                
                if only_i_mask == 0:
                    continue
                
                # Try to find a replacement block that covers only_i_mask and has good overlap
                best_swap = None
                best_score = -1
                
                # Sample some candidate blocks
                random.seed(i * 1000 + len(blocks))
                candidates = []
                
                # Generate candidates containing some points from the current block
                current_block = set(blocks[i])
                for _ in range(50):
                    # Pick a random subset of 2-4 points from current block
                    n_keep = random.randint(2, 4)
                    keep = random.sample(list(current_block), min(n_keep, len(current_block)))
                    other_points = [p for p in range(v) if p not in set(keep)]
                    if len(other_points) < k - len(keep):
                        continue
                    fill = random.sample(other_points, k - len(keep))
                    candidate = tuple(sorted(set(keep) | set(fill)))
                    candidates.append(candidate)
                
                # Also try some random blocks
                for _ in range(30):
                    candidate = tuple(sorted(random.sample(range(v), k)))
                    candidates.append(candidate)
                
                for cand in candidates:
                    if cand == tuple(sorted(blocks[i])):
                        continue
                    
                    cand_mask = block_coverage_mask(cand)
                    
                    # Score: must cover only_i_mask, and maximize additional coverage
                    if (cand_mask & only_i_mask) != only_i_mask:
                        continue
                    
                    # Compute what would be lost if we remove block i without replacement
                    # and what we gain from the new block
                    gain = bin(cand_mask & ~block_masks[i]).count('1')
                    score = gain
                    
                    if score > best_score:
                        best_score = score
                        best_swap = (cand, cand_mask)
                
                if best_swap is not None:
                    cand, cand_mask = best_swap
                    blocks[i] = list(cand)
                    block_masks[i] = cand_mask
                    # Update tsub_block_list
                    for ts_idx in range(n_tsubs):
                        if i in tsub_block_list[ts_idx]:
                            tsub_block_list[ts_idx].remove(i)
                        if cand_mask & (1 << ts_idx):
                            tsub_block_list[ts_idx].append(i)
                    improved = True
                    break
        
        return blocks
    
    # Fast greedy using bitmask operations
    def greedy_fast(pivot_strategy='random'):
        uncovered_mask = (1 << n_tsubs) - 1
        blocks = []
        
        while uncovered_mask:
            if pivot_strategy == 'random':
                # Pick a random uncovered t-subset
                bits = []
                tmp = uncovered_mask
                while tmp:
                    lsb = tmp & (-tmp)
                    bits.append(lsb.bit_length() - 1)
                    tmp &= tmp - 1
                pivot_idx = random.choice(bits)
            else:
                # Pick the first uncovered t-subset
                pivot_idx = (uncovered_mask & (-uncovered_mask)).bit_length() - 1
            
            pivot_tsub = all_tsubs[pivot_idx]
            pivot_points = set(pivot_tsub)
            other_points = [p for p in range(v) if p not in pivot_points]
            
            best_block = None
            best_gain = -1
            
            # Try all combinations of (k-t) points from other_points
            for combo in itertools.combinations(other_points, k - t):
                block = tuple(sorted(pivot_points | set(combo)))
                
                # Compute gain using bitmask
                block_mask = block_coverage_mask(block)
                gain = bin(block_mask & uncovered_mask).count('1')
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = (block, block_mask)
            
            if best_block is not None:
                block, block_mask = best_block
                blocks.append(list(block))
                uncovered_mask &= ~block_mask
        
        return blocks
    
    # Run multiple iterations with different strategies
    best_blocks = None
    best_len = float('inf')
    
    iterations = 0
    while time.time() < deadline - 20:  # Leave 20s for local search
        # Use different pivot strategies
        if iterations % 3 == 0:
            blocks = greedy_fast('random')
        elif iterations % 3 == 1:
            blocks = greedy_fast('min')
        else:
            # Weighted random: prefer t-subsets with fewer covering blocks
            # For simplicity, just use random
            blocks = greedy_fast('random')
        
        # Remove redundant blocks
        blocks = remove_redundant_fast(blocks)
        
        # Apply local search if we have time
        if time.time() < deadline - 15:
            ls_time = min(5, deadline - 15 - time.time())
            blocks = local_search(blocks, max_time=ls_time)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        
        iterations += 1
    
    # Final local search on best solution
    if best_blocks is not None and time.time() < deadline - 5:
        ls_time = min(10, deadline - 5 - time.time())
        best_blocks = local_search(best_blocks, max_time=ls_time)
    
    if best_blocks is None:
        # Fallback: simple deterministic greedy
        uncovered_mask = (1 << n_tsubs) - 1
        blocks = []
        while uncovered_mask:
            pivot_idx = (uncovered_mask & (-uncovered_mask)).bit_length() - 1
            pivot_tsub = all_tsubs[pivot_idx]
            pivot_points = set(pivot_tsub)
            other_points = [p for p in range(v) if p not in pivot_points]
            
            best_block = None
            best_gain = -1
            
            for combo in itertools.combinations(other_points, k - t):
                block = tuple(sorted(pivot_points | set(combo)))
                block_mask = block_coverage_mask(block)
                gain = bin(block_mask & uncovered_mask).count('1')
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = (block, block_mask)
            
            if best_block is not None:
                block, block_mask = best_block
                blocks.append(list(block))
                uncovered_mask &= ~block_mask
        
        best_blocks = remove_redundant_fast(blocks)

    return best_blocks
