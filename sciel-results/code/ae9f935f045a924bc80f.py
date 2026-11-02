import itertools
import random
import time

def construct(v, k, t):
    # Precompute all t-subsets and k-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    
    # Map each k-subset to the list of indices of t-subsets it covers
    k_sub_to_t_idxs = []
    # Map each t-subset index to the list of k-subset indices that cover it
    t_idx_to_k_idxs = [[] for _ in range(len(all_t_subsets))]
    
    for k_idx, ks in enumerate(all_k_subsets):
        covered_t_idxs = []
        for ts in itertools.combinations(ks, t):
            idx = t_sub_to_idx[ts]
            covered_t_idxs.append(idx)
            t_idx_to_k_idxs[idx].append(k_idx)
        k_sub_to_t_idxs.append(covered_t_idxs)
        
    def greedy_cover():
        uncovered = set(range(len(all_t_subsets)))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset
            seed_idx = random.choice(list(uncovered))
            candidates = t_idx_to_k_idxs[seed_idx]
            
            best_block_idx = -1
            max_cov = 0
            
            # Evaluate candidates
            for b_idx in candidates:
                cov = 0
                # Count how many of its covered t-subsets are currently uncovered
                for t_idx in k_sub_to_t_idxs[b_idx]:
                    if t_idx in uncovered:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                break # Should not happen
            
            blocks.append(best_block_idx)
            
            # Update uncovered
            for t_idx in k_sub_to_t_idxs[best_block_idx]:
                uncovered.discard(t_idx)
                
        return blocks

    def local_search(block_indices):
        # block_indices is a list of indices into all_k_subsets
        # Convert to set for fast lookup
        block_set = set(block_indices)
        
        # Initialize coverage counts
        coverage = [0] * len(all_t_subsets)
        for b_idx in block_set:
            for t_idx in k_sub_to_t_idxs[b_idx]:
                coverage[t_idx] += 1
        
        improved = True
        while improved:
            improved = False
            
            # 1. Try to remove redundant blocks
            to_remove = []
            for b_idx in block_set:
                redundant = True
                for t_idx in k_sub_to_t_idxs[b_idx]:
                    if coverage[t_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(b_idx)
            
            for b_idx in to_remove:
                block_set.remove(b_idx)
                for t_idx in k_sub_to_t_idxs[b_idx]:
                    coverage[t_idx] -= 1
                improved = True
            
            # 2. Try to swap blocks
            current_blocks = list(block_set)
            for b_idx in current_blocks:
                # Temporarily remove b_idx
                for t_idx in k_sub_to_t_idxs[b_idx]:
                    coverage[t_idx] -= 1
                    
                # Find t-subsets that are now uncovered (coverage == 0)
                uncovered_critical = []
                for t_idx in k_sub_to_t_idxs[b_idx]:
                    if coverage[t_idx] == 0:
                        uncovered_critical.append(t_idx)
                        
                if not uncovered_critical:
                    # This means b_idx was redundant
                    block_set.remove(b_idx)
                    improved = True
                    continue
                
                # We need to find a block b_new that covers ALL of uncovered_critical
                # and is not in block_set (excluding b_idx which is effectively out)
                
                if len(uncovered_critical) > 1:
                    candidates = set(t_idx_to_k_idxs[uncovered_critical[0]])
                    for crit in uncovered_critical[1:]:
                        candidates &= set(t_idx_to_k_idxs[crit])
                    
                    candidates -= block_set
                else:
                    candidates = set(t_idx_to_k_idxs[uncovered_critical[0]])
                    candidates -= block_set
                    
                if not candidates:
                    # No replacement found, restore b_idx
                    for t_idx in k_sub_to_t_idxs[b_idx]:
                        coverage[t_idx] += 1
                    continue
                
                # Pick the best candidate
                best_new = -1
                max_score = -1
                for cand in candidates:
                    score = len(k_sub_to_t_idxs[cand])
                    if score > max_score:
                        max_score = score
                        best_new = cand
                        
                if best_new != -1:
                    # Perform swap
                    block_set.remove(b_idx)
                    block_set.add(best_new)
                    
                    # Update coverage:
                    # b_idx was already decremented.
                    # Add best_new
                    for t_idx in k_sub_to_t_idxs[best_new]:
                        coverage[t_idx] += 1
                        
                    improved = True
                else:
                    # Restore b_idx
                    for t_idx in k_sub_to_t_idxs[b_idx]:
                        coverage[t_idx] += 1

        return list(block_set)

    # Multi-start
    best_blocks = None
    best_len = float('inf')
    
    # Run greedy + local search multiple times
    start_time = time.time()
    num_starts = 10
    
    for i in range(num_starts):
        if time.time() - start_time > 100:
            break
            
        g_blocks = greedy_cover()
        l_blocks = local_search(g_blocks)
        
        if len(l_blocks) < best_len:
            best_len = len(l_blocks)
            best_blocks = l_blocks
            
    # Convert indices back to lists of points
    if best_blocks is None:
        # Fallback to simple greedy if something went wrong
        g_blocks = greedy_cover()
        best_blocks = g_blocks
        
    result = []
    for b_idx in best_blocks:
        result.append(list(all_k_subsets[b_idx]))
        
    return result
