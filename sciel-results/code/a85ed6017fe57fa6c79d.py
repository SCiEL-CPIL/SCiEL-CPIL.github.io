import itertools
import random
import numpy as np
import time
import math
import heapq

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using Greedy Initialization 
    with Local Search (Redundancy Removal and Targeted Swaps).
    """
    start_time = time.time()
    time_limit = 115.0

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # Precompute block -> t-subset indices mapping
    # For each block, store the list of t-subset indices it covers
    block_t_indices = []
    for block in all_k_subsets:
        indices = []
        for ts in itertools.combinations(block, t):
            indices.append(t_to_idx[ts])
        block_t_indices.append(indices)
        
    # Precompute t-subset -> block indices mapping
    # t_to_blocks[t_idx] = list of block indices that cover this t-subset
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for t_idx in indices:
            t_to_blocks[t_idx].append(b_idx)

    def greedy_init(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        uncovered_set = set(range(num_t_subsets))
        current_blocks = []
        in_set = [False] * num_k_subsets
        
        while uncovered_set:
            # Find block with max gain
            best_b_idx = -1
            best_gain = -1
            # To speed up, we can sample candidates or iterate all.
            # With 75k blocks, iterating all is ~75k ops. 
            # 11628 t-subsets.
            # Let's iterate all blocks to find max gain.
            for b_idx in range(num_k_subsets):
                if in_set[b_idx]:
                    continue
                gain = 0
                for t_idx in block_t_indices[b_idx]:
                    if t_idx in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_b_idx = b_idx
                    # Tie-breaking: randomize by shuffling or just taking first max?
                    # To allow different restarts, we can add a small random noise or shuffle candidates.
                    # But for now, deterministic max gain.
                    
            if best_b_idx == -1:
                break # Should not happen if v,k,t are valid

            current_blocks.append(best_b_idx)
            in_set[best_b_idx] = True
            for t_idx in block_t_indices[best_b_idx]:
                coverage[t_idx] += 1
                if coverage[t_idx] == 1:
                    uncovered_set.discard(t_idx)
                    
        # Remove redundancies
        current_blocks, coverage = remove_redundant(current_blocks, coverage, in_set, block_t_indices)
        return current_blocks, coverage

    def remove_redundant(blocks, coverage, in_set, block_t_indices):
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                b_idx = blocks[i]
                is_redundant = True
                for t_idx in block_t_indices[b_idx]:
                    if coverage[t_idx] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    blocks.pop(i)
                    in_set[b_idx] = False
                    for t_idx in block_t_indices[b_idx]:
                        coverage[t_idx] -= 1
                    changed = True
                    break
        return blocks, coverage

    best_blocks = None
    best_size = float('inf')
    
    # Run multiple restarts
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        current_blocks, coverage = greedy_init(random_seed=r)
        in_set = [False] * num_k_subsets
        for b in current_blocks:
            in_set[b] = True
            
        current_size = len(current_blocks)
        current_uncovered = np.sum(coverage == 0)
        
        if current_uncovered == 0 and current_size < best_size:
            best_size = current_size
            best_blocks = current_blocks[:]
            
        # Local Search
        # We try to reduce size by finding swaps that allow removals
        # Or just direct removals if we can find them (greedy should have removed them, 
        # but maybe a swap creates a new redundancy)
        
        improved = True
        while improved:
            if time.time() - start_time > time_limit:
                break
            improved = False
            
            # 1. Try to remove redundant blocks
            old_size = len(current_blocks)
            current_blocks, coverage = remove_redundant(current_blocks, coverage, in_set, block_t_indices)
            new_size = len(current_blocks)
            if new_size < old_size:
                improved = True
                current_size = new_size
                if current_size < best_size:
                    best_size = current_size
                    best_blocks = current_blocks[:]
                continue # Restart loop to find more redundancies
            
            # 2. If no redundancy, try a swap
            # Pick a random block from the current set
            if not current_blocks:
                break
                
            i = random.randint(0, len(current_blocks) - 1)
            b_old = current_blocks[i]
            
            # Find t-subsets uniquely covered by b_old
            # These must be covered by the new block if we swap
            unique_ts = []
            for t_idx in block_t_indices[b_old]:
                if coverage[t_idx] == 1:
                    unique_ts.append(t_idx)
                    
            if not unique_ts:
                # b_old is redundant? Should have been caught.
                continue
                
            # Find a block b_new that covers all unique_ts
            # And is not in the set
            # To maximize chance of creating new redundancy, we want b_new to cover
            # as many other t-subsets as possible that are currently covered by only b_new or few others?
            # Actually, we just want to find *any* valid swap first.
            # Then we can try to optimize.
            
            candidates = []
            # Intersect lists of blocks covering each t in unique_ts
            # unique_ts is small (size of t-subsets in a block is C(8,5)=56, but unique ones are fewer)
            # Let's find intersection of t_to_blocks[t] for t in unique_ts
            if unique_ts:
                common_blocks = set(t_to_blocks[unique_ts[0]])
                for t_idx in unique_ts[1:]:
                    common_blocks.intersection_update(t_to_blocks[t_idx])
                
                # Filter out blocks already in set
                for b_new in common_blocks:
                    if not in_set[b_new]:
                        candidates.append(b_new)
                        
            if not candidates:
                continue
                
            # Pick a candidate
            # To be smart, pick one that maximizes "potential redundancy"
            # i.e., covers many t-subsets with coverage count == 1 (other than the unique ones?)
            # Or simply pick one randomly to explore.
            # Let's pick the one with max total coverage of t-subsets with count > 1?
            # No, we want to create new redundancies. A block that covers many t-subsets
            # that are currently covered by *only* it (after swap) is good?
            # Actually, if we swap b_old -> b_new, the coverage of t in b_new increases.
            # If t was covered by 1 block (which is b_new, since it wasn't in set), it becomes 2.
            # This makes b_new potentially redundant? No, it makes the *other* block covering t
            # potentially redundant? No, the block that was covering t is now covered by 2.
            # So the *other* block might become redundant if all its unique t-subsets are now covered by others.
            
            # Let's just pick a random candidate to allow exploration.
            b_new = random.choice(candidates)
            
            # Perform swap
            # Remove b_old
            for t_idx in block_t_indices[b_old]:
                coverage[t_idx] -= 1
            in_set[b_old] = False
            
            # Add b_new
            for t_idx in block_t_indices[b_new]:
                coverage[t_idx] += 1
            in_set[b_new] = True
            
            current_blocks[i] = b_new
            
            # Check validity (should be valid by construction of unique_ts)
            # But let's verify uncovered is 0
            # It should be, because we covered all unique_ts of b_old with b_new.
            # And no other t-subset became uncovered because we only decreased coverage of b_old's t-subsets.
            # If a t-subset in b_old was NOT unique, its count was >= 2, so it becomes >= 1.
            # If it WAS unique, it was in unique_ts, so b_new covers it, so it becomes >= 1.
            
            # Now, this swap might have created new redundancies.
            # The loop will catch them in the next iteration (step 1).
            improved = True # Force another iteration to check for redundancies
            
    if best_blocks is None:
        # Fallback
        current_blocks, coverage = greedy_init(random_seed=0)
        best_blocks = current_blocks

    final_blocks = [list(all_k_subsets[idx]) for idx in best_blocks]
    return final_blocks
