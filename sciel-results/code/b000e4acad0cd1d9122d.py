import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0
    
    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    def nCr(n, r):
        if r > n: return 0
        if r == 0: return 1
        res = 1
        for i in range(r):
            res *= (n - i)
            res //= (i + 1)
        return res
            
    t_count = nCr(k, t)
    
    # Precompute: for each block, which t-subsets it covers
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]
            
    # Precompute inverse map: for each t-subset, which blocks contain it?
    # This is useful for local search swaps.
    # n_ts = 2024.
    # We can use a list of lists.
    blocks_containing_subset = [[] for _ in range(n_ts)]
    for i, block in enumerate(all_ks):
        for sub in itertools.combinations(block, t):
            idx = idx_map[sub]
            blocks_containing_subset[idx].append(i)
    
    best_solution = None
    best_score = float('inf')
    
    def aggressive_deletion(current_blocks, coverage_count):
        """
        Remove any block that is redundant (all its t-subsets are covered by other blocks).
        """
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 3.0:
                break
            improved = False
            new_blocks = []
            for b_idx in current_blocks:
                subs_in_block = block_t_indices[b_idx]
                # Check if all subsets in this block have coverage > 1
                if np.all(coverage_count[subs_in_block] > 1):
                    improved = True
                    coverage_count[subs_in_block] -= 1
                else:
                    new_blocks.append(b_idx)
            current_blocks = new_blocks
        return current_blocks

    def local_search_polish(current_blocks, coverage_count, max_iterations=200):
        """
        Try to reduce block count via swap operations.
        """
        if time.time() - start_time > time_limit - 5.0:
            return current_blocks, coverage_count
        
        best_blocks = list(current_blocks)
        best_blocks_set = set(best_blocks)
        best_coverage = coverage_count.copy()
        best_count = len(best_blocks)
        
        for iteration in range(max_iterations):
            if time.time() - start_time > time_limit - 5.0:
                break
            
            # Try removing each block
            removed_any = False
            new_blocks = list(best_blocks)
            new_coverage = best_coverage.copy()
            
            for i in range(len(new_blocks)):
                b_idx = new_blocks[i]
                subs_in_block = block_t_indices[b_idx]
                
                # Temporarily remove this block
                new_coverage[subs_in_block] -= 1
                
                # Check if coverage is still complete
                if np.all(new_coverage > 0):
                    # Safe to remove
                    new_blocks.pop(i)
                    removed_any = True
                    break
                else:
                    # Restore
                    new_coverage[subs_in_block] += 1
            
            if removed_any:
                best_blocks = new_blocks
                best_blocks_set = set(new_blocks)
                best_coverage = new_coverage
                best_count -= 1
                continue
            
            # Try swap: remove one block, add another
            if len(best_blocks) > 0:
                # Calculate unique coverage for each block
                contributions = []
                for b_idx in best_blocks:
                    subs = block_t_indices[b_idx]
                    unique_count = np.sum(best_coverage[subs] == 1)
                    contributions.append(unique_count)
                
                contributions = np.array(contributions)
                min_contrib = np.min(contributions)
                low_contrib_indices = np.where(contributions == min_contrib)[0]
                
                if len(low_contrib_indices) > 0:
                    # Pick a random low-contribution block to try removing
                    remove_idx = low_contrib_indices[random.randint(0, len(low_contrib_indices) - 1)]
                    remove_b_idx = best_blocks[remove_idx]
                    remove_subs = block_t_indices[remove_b_idx]
                    
                    # Find uncovered subsets after removal
                    temp_coverage = best_coverage.copy()
                    temp_coverage[remove_subs] -= 1
                    uncovered_mask = (temp_coverage == 0)
                    uncovered_indices = np.where(uncovered_mask)[0]
                    
                    if len(uncovered_indices) == 0:
                        # Can remove without adding
                        new_blocks = [b for i, b in enumerate(best_blocks) if i != remove_idx]
                        new_coverage = temp_coverage
                        best_blocks = new_blocks
                        best_coverage = new_coverage
                        best_count -= 1
                        continue
                    
                    # Try to find a block that covers all uncovered subsets
                    # Heuristic: Find candidates that cover the most uncovered subsets
                    # We need to cover len(uncovered_indices) subsets.
                    # If we find one that covers all, it's a perfect swap.
                    
                    # To speed up, we can filter candidates by checking if they contain 
                    # a specific "hard" uncovered subset.
                    # Pick a random uncovered subset 'target'
                    # Any candidate must contain 'target'.
                    # Precompute a map: subset_index -> list of blocks containing it?
                    # This is the inverse of block_t_indices.
                    # n_ts = 2024. n_ks = 346k.
                    # Each block has 35 subsets. Total entries = 346k * 35 = 12M.
                    # We can build this map once.
                    
                    # Since we are in a loop, building it every time is bad.
                    # But we can build it once in construct() and pass it?
                    # Or build it here if not cached?
                    
                    # Let's assume we have a global or closure variable 'blocks_containing_subset'
                    # We will create it in construct().
                    
                    # For now, let's just sample more intelligently.
                    # Pick a random uncovered subset 'target'
                    target = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
                    
                    # Get all blocks that contain 'target'
                    # We need the inverse map.
                    # Let's create it outside the function and use it here.
                    
                    # If not available, fallback to random sampling.
                    if 'blocks_containing_subset' in locals():
                        candidates = blocks_containing_subset[target]
                    else:
                        # Fallback: Random sampling
                        sample_size = min(500, n_ks)
                        existing_set = set(best_blocks)
                        candidates = []
                        while len(candidates) < sample_size:
                            cand = random.randint(0, n_ks - 1)
                            if cand not in existing_set:
                                candidates.append(cand)
                    
                    best_candidate = -1
                    best_overlap = -1
                    
                    # Filter candidates that are not in best_blocks
                    # And score them by overlap with uncovered_indices
                    for cand in candidates:
                        if cand in best_blocks_set or cand == remove_b_idx:
                            continue
                        cand_subs = block_t_indices[cand]
                        # Count how many uncovered_indices are in cand_subs
                        # np.intersect1d is slow.
                        # Use boolean mask:
                        # mask = np.zeros(n_ts, dtype=bool)
                        # mask[uncovered_indices] = True
                        # overlap = np.sum(mask[cand_subs])
                        
                        # This is faster:
                        overlap = np.isin(cand_subs, uncovered_indices).sum()
                        
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_candidate = cand
                            
                            if overlap == len(uncovered_indices):
                                break # Perfect match found
                    
                    if best_candidate >= 0 and best_overlap == len(uncovered_indices):
                        # Accept the swap
                        new_blocks = [b for i, b in enumerate(best_blocks) if i != remove_idx]
                        new_blocks.append(best_candidate)
                        new_blocks_set = set(new_blocks)
                        new_coverage = temp_coverage
                        new_coverage[block_t_indices[best_candidate]] += 1
                        best_blocks = new_blocks
                        best_blocks_set = new_blocks_set
                        best_coverage = new_coverage
        
        return best_blocks, best_coverage

    def run_greedy(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        blocks_indices = []
        
        # Track how many times each t-subset has been covered
        cover_freq = np.zeros(n_ts, dtype=np.int32)
        
        # NEW: Track point load (how many times each point has been used in selected blocks)
        point_load = np.zeros(v, dtype=np.int32)
        
        # We stop when all triples are covered at least once
        # To allow balancing, we might continue a bit? No, just cover all.
        
        while not np.all(cover_freq > 0):
            if time.time() - start_time > time_limit - 5.0:
                return None, None
            
            # Score: Sum of inverse frequencies of the triples in the block
            # This encourages covering rare triples.
            # 1/(1+freq) is 1 if freq=0, 0.5 if freq=1, etc.
            # We can vectorize this.
            # block_t_indices shape: (n_ks, t_count)
            # cover_freq shape: (n_ts,)
            # We want sum over axis 1 of 1/(1+cover_freq[block_t_indices])
            
            # Note: cover_freq is int. 1.0 / (1.0 + cover_freq) works.
            # This creates a (n_ks, t_count) array of floats.
            # This is 346k * 35 * 8 bytes = ~100MB. Acceptable.
            
            scores = np.sum(1.0 / (1.0 + cover_freq[block_t_indices]), axis=1)
            
            # Add a small penalty for high point load to distribute points?
            # Or just rely on the triple balancing.
            # Let's add a small penalty to encourage using under-used points.
            # point_load shape: (v,)
            # We need sum of point_load[p] for p in block.
            # This is hard to vectorize without a point-block incidence matrix.
            # We can precompute a point-block incidence matrix?
            # n_ks x v = 346k * 24 * 1 byte = 8MB.
            # Let's create it once outside? Or inside?
            # It's better to do it outside if possible, but we are in a function.
            # Let's just skip point load penalty for now to keep it fast, 
            # as triple balancing is the main goal for deletion.
            
            max_score = np.max(scores)
            if max_score == 0:
                break # Should not happen if not all covered
            
            # Get candidates with high score
            # Threshold: top 1% or top 100?
            # Let's take top 50 to keep it fast.
            # argsort is O(N log N).
            # We can use np.argpartition for top K.
            
            if n_ks > 50:
                # Get indices of top 50 scores
                top_indices = np.argpartition(scores, -50)[-50:]
                candidates = top_indices
            else:
                candidates = np.arange(n_ks)
            
            # Among candidates, pick the best one (max score)
            # Tie-breaking: random?
            # Let's pick the one with max score, break ties randomly.
            best_candidate = -1
            best_score_val = -1
            for cand in candidates:
                if scores[cand] > best_score_val:
                    best_score_val = scores[cand]
                    best_candidate = cand
                elif scores[cand] == best_score_val:
                    # Random tie break
                    if random.random() < 0.5:
                        best_candidate = cand
            
            blocks_indices.append(best_candidate)
            
            subs_to_cover = block_t_indices[best_candidate]
            cover_freq[subs_to_cover] += 1
            
            # Update point load
            for p in all_ks[best_candidate]:
                point_load[p] += 1
            
        current_blocks = list(blocks_indices)
        
        # Build coverage count
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
        
        # Aggressive deletion
        current_blocks = aggressive_deletion(current_blocks, coverage_count)
        
        # Rebuild coverage count after deletion
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
        
        # Local search polish
        current_blocks, coverage_count = local_search_polish(current_blocks, coverage_count)
        
        # Final deletion pass
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
        current_blocks = aggressive_deletion(current_blocks, coverage_count)
        
        return current_blocks, len(current_blocks)

    # Phase 1: Run multiple greedy restarts
    restart_count = 0
    while time.time() - start_time < time_limit - 10.0:
        seed = restart_count
        blocks, score = run_greedy(seed)
        
        if blocks is None:
            break
            
        if score < best_score:
            best_score = score
            best_solution = blocks
            
        restart_count += 1
    
    if best_solution is None:
        blocks, score = run_greedy(0)
        if blocks is not None:
            best_solution = blocks
            best_score = score
        else:
            return []
    
    # Phase 2: Additional polish on best solution
    if time.time() - start_time < time_limit - 5.0:
        # Rebuild coverage count for best solution
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in best_solution:
            coverage_count[block_t_indices[b_idx]] += 1
        
        # Run local search with more iterations
        polished_blocks, polished_coverage = local_search_polish(
            list(best_solution), coverage_count, max_iterations=500
        )
        
        if len(polished_blocks) < best_score:
            best_solution = polished_blocks
            best_score = len(polished_blocks)
        
        # Final aggressive deletion
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in best_solution:
            coverage_count[block_t_indices[b_idx]] += 1
        best_solution = aggressive_deletion(list(best_solution), coverage_count)
    
    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
