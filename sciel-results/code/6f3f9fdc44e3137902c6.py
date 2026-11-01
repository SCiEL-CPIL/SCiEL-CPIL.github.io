import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # Map each block to the indices of the t-subsets it covers
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_t_indices.append(indices)
    
    # Map each t-subset to the blocks that cover it
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)
    
    # Precompute block coverage as numpy arrays for speed
    block_t_arrs = [np.array(indices, dtype=np.int32) for indices in block_t_indices]
    
    def minimize(block_set, cover_counts):
        """Remove blocks that are no longer needed."""
        changed = True
        while changed:
            changed = False
            items = list(block_set)
            random.shuffle(items)
            for b_idx in items:
                if b_idx not in block_set:
                    continue
                can_remove = True
                for t_idx in block_t_indices[b_idx]:
                    if cover_counts[t_idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    block_set.remove(b_idx)
                    for t_idx in block_t_indices[b_idx]:
                        cover_counts[t_idx] -= 1
                    changed = True
        return block_set, cover_counts
    
    def greedy_construct(seed, sample_size=5000):
        """Greedy set cover with random sampling for speed."""
        rng = random.Random(seed)
        active_blocks = set()
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        uncovered = set(range(num_t_subsets))
        
        # Precompute candidate pool
        candidates = list(range(num_k_subsets))
        
        while uncovered and time.time() < deadline - 5.0:
            # Sample candidates for speed
            if len(candidates) > sample_size:
                sampled = rng.sample(candidates, sample_size)
            else:
                sampled = candidates
            
            best_block = -1
            max_gain = 0
            
            for b_idx in sampled:
                if b_idx in active_blocks:
                    continue
                # Count how many uncovered t-subsets this block covers
                gain = 0
                for t_idx in block_t_indices[b_idx]:
                    if t_idx in uncovered:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block = b_idx
            
            if best_block == -1 or max_gain == 0:
                break
            
            active_blocks.add(best_block)
            for t_idx in block_t_indices[best_block]:
                cover_count[t_idx] += 1
                if t_idx in uncovered:
                    uncovered.discard(t_idx)
        
        return active_blocks, cover_count
    
    def local_search(block_set, cover_counts, time_budget=30.0):
        """Local search with removal and swap operations."""
        end_time = time.time() + time_budget
        improved = True
        
        while improved and time.time() < end_time:
            improved = False
            current_blocks = list(block_set)
            random.shuffle(current_blocks)
            
            for B in current_blocks:
                if time.time() >= end_time:
                    break
                if B not in block_set:
                    continue
                
                # Find t-subsets uniquely covered by B
                uncovered_by_B = []
                for t_idx in block_t_indices[B]:
                    if cover_counts[t_idx] == 1:
                        uncovered_by_B.append(t_idx)
                
                if not uncovered_by_B:
                    # Try removal
                    block_set.remove(B)
                    for t_idx in block_t_indices[B]:
                        cover_counts[t_idx] -= 1
                    improved = True
                    continue
                
                # Sort by number of covering blocks to speed up intersection
                uncovered_by_B.sort(key=lambda x: len(t_subset_blocks[x]))
                
                candidates = None
                for t_idx in uncovered_by_B:
                    blocks_list = t_subset_blocks[t_idx]
                    if candidates is None:
                        candidates = set(blocks_list)
                    else:
                        candidates.intersection_update(blocks_list)
                    
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                # Filter candidates
                valid_candidates = []
                for c in candidates:
                    if c != B and c not in block_set:
                        valid_candidates.append(c)
                
                if not valid_candidates:
                    continue
                
                # Perform Swap
                B_new = random.choice(valid_candidates)
                
                block_set.remove(B)
                block_set.add(B_new)
                
                for t_idx in block_t_indices[B]:
                    cover_counts[t_idx] -= 1
                for t_idx in block_t_indices[B_new]:
                    cover_counts[t_idx] += 1
                
                # Minimize after swap
                block_set, cover_counts = minimize(block_set, cover_counts)
                improved = True
        
        return block_set, cover_counts
    
    # Multi-restart strategy
    best_blocks = None
    best_count = float('inf')
    
    # Try multiple seeds with different strategies
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]
    
    for seed in seeds:
        if time.time() > deadline - 10.0:
            break
        
        # Greedy construction
        blocks, cover_count = greedy_construct(seed)
        
        if not blocks:
            continue
        
        # Check validity
        if np.all(cover_count >= 1):
            # Minimize
            blocks, cover_count = minimize(blocks, cover_count)
            
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks.copy()
                best_cover = cover_count.copy()
                
                # Local search if time permits
                remaining = deadline - time.time() - 5.0
                if remaining > 10.0:
                    blocks, cover_count = local_search(blocks, cover_count, time_budget=min(remaining, 20.0))
                    if np.all(cover_count >= 1) and len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = blocks.copy()
                        best_cover = cover_count.copy()
    
    # If no valid solution found, do a final greedy
    if best_blocks is None:
        blocks, cover_count = greedy_construct(99999)
        best_blocks = blocks
        best_count = len(blocks)
    
    # Final minimization
    if best_blocks:
        # Recompute cover count
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in best_blocks:
            cover_count[block_t_arrs[b]] += 1
        
        best_blocks, cover_count = minimize(best_blocks, cover_count)
    
    result = [list(all_k_subsets[b_idx]) for b_idx in best_blocks]
    return result
