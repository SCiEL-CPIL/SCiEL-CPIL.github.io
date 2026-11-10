import itertools
import time
import sys
from collections import defaultdict

sys.setrecursionlimit(10000)

def construct(v, k, t):
    """
    Constructs a covering design using a constraint-based backtracking search 
    with a bottleneck heuristic and memoization.
    """
    start_time = time.time()
    time_limit = 115
    
    # Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute all k-subsets (blocks) and their covered t-subsets as bitsets
    all_blocks = list(itertools.combinations(range(v), k))
    
    # block_t_mask: dictionary mapping block tuple to integer bitset of covered t-subsets
    # t_subset_to_blocks: dictionary mapping t-subset index to list of blocks covering it
    block_t_mask = {}
    t_subset_to_blocks = defaultdict(list)
    
    for b in all_blocks:
        mask = 0
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
            t_subset_to_blocks[idx].append(b)
        block_t_mask[b] = mask
        
    # 1. Greedy Initial Solution
    uncovered_mask = (1 << num_t_subsets) - 1
    greedy_blocks = []
    
    # Pre-compute coverage counts for greedy
    # We want to pick block maximizing popcount(block_mask & uncovered_mask)
    # This is O(N_blocks * N_t_subsets/64) per step.
    # With N_blocks ~ 75k and N_t ~ 11k, this might be slow if done naively in Python.
    # Let's optimize: 
    # Instead of iterating all blocks every time, we can just do it. 
    # 335 steps * 75k blocks * (bit ops) is manageable? 
    # 25M bit ops. Python bit ops on large ints are C-speed. Should be fast enough.
    
    while uncovered_mask:
        best_block = None
        max_cov = 0
        for b in all_blocks:
            cov = (block_t_mask[b] & uncovered_mask).bit_count()
            if cov > max_cov:
                max_cov = cov
                best_block = b
        if best_block is None or max_cov == 0:
            break
        greedy_blocks.append(best_block)
        uncovered_mask &= ~block_t_mask[best_block]
        
    best_size = len(greedy_blocks)
    best_solution = list(greedy_blocks)
    
    # 2. Local Search (Hill Climbing) to improve greedy solution
    # Try swapping one block with another to reduce total count or improve coverage?
    # Actually, just trying to remove blocks is hard. 
    # Let's try a simple "drop" heuristic: if a block is redundant, remove it.
    # Then maybe swap?
    # Given time constraints, let's stick to a simple improvement: 
    # Re-run greedy but skip blocks that don't cover new things? No, greedy is greedy.
    # Let's just use the greedy result as the bound.
    
    # 3. Backtracking Search
    # State: uncovered_mask, current_blocks
    # Heuristic: Bottleneck. Find t-subset in uncovered_mask with fewest available blocks.
    
    # Precompute for each t-subset index, the list of blocks covering it.
    # We need to know which of those blocks are "available" (not used yet).
    # Since we are building a set of blocks, "available" means not in current_blocks.
    
    # Memoization: dict mapping uncovered_mask to min_blocks_used_to_reach_state
    # If we reach same uncovered_mask with >= blocks, prune.
    memo = {}
    
    def search(uncovered_mask, current_blocks):
        nonlocal best_size, best_solution
        
        if time.time() - start_time > time_limit:
            return False
        
        if len(current_blocks) >= best_size:
            return False
            
        if not uncovered_mask:
            if len(current_blocks) < best_size:
                best_size = len(current_blocks)
                best_solution = list(current_blocks)
            return True
        
        # Memoization check
        # If we've seen this uncovered_mask with <= blocks, we are in a worse or equal state
        if uncovered_mask in memo:
            if len(current_blocks) >= memo[uncovered_mask]:
                return False
        else:
            memo[uncovered_mask] = len(current_blocks)
            
        # Find bottleneck: t-subset index in uncovered_mask with fewest available blocks
        # Iterate through set bits of uncovered_mask
        min_avail = float('inf')
        bottleneck_idx = -1
        
        # Get list of uncovered indices
        # Efficiently iterate set bits
        u = uncovered_mask
        while u:
            # Get lowest set bit
            lsb = u & -u
            idx = lsb.bit_length() - 1
            u ^= lsb
            
            # Count available blocks for this idx
            # Available = blocks in t_subset_to_blocks[idx] that are NOT in current_blocks
            # Checking "not in current_blocks" is O(N_current). 
            # N_current is small (< 340). 
            # t_subset_to_blocks[idx] size is C(v-t, k-t) = C(14, 3) = 364.
            # So 364 * 340 checks per idx. 
            # Number of uncovered idxs can be large initially.
            # This might be slow.
            
            # Optimization: 
            # We can precompute a global count of how many times each block is used? No.
            # Let's just do the count.
            
            avail = 0
            for b in t_subset_to_blocks[idx]:
                if b not in current_blocks:
                    avail += 1
            if avail == 0:
                return False # Dead end
            
            if avail < min_avail:
                min_avail = avail
                bottleneck_idx = idx
                if avail == 1:
                    break # Can't get better than 1
                    
        if bottleneck_idx == -1:
            return False
            
        # Candidates: blocks covering bottleneck_idx
        candidates = []
        for b in t_subset_to_blocks[bottleneck_idx]:
            if b not in current_blocks:
                # Score: how many other uncovered t-subsets does it cover?
                score = (block_t_mask[b] & uncovered_mask).bit_count()
                candidates.append((score, b))
                
        candidates.sort(reverse=True)
        
        # Limit branching to top few to control explosion
        # With bottleneck=1, we only have 1 candidate.
        # With bottleneck=2, we have 2 candidates.
        # This heuristic naturally limits branching.
        
        for score, b in candidates:
            # Pruning: if len(current_blocks) + 1 >= best_size, skip
            if len(current_blocks) + 1 >= best_size:
                continue
                
            current_blocks.append(b)
            new_uncovered = uncovered_mask & ~block_t_mask[b]
            
            if search(new_uncovered, current_blocks):
                current_blocks.pop()
                return True # Propagate success up
                
            current_blocks.pop()
            
        return False
        
    search((1 << num_t_subsets) - 1, [])
    
    return [list(b) for b in best_solution]
