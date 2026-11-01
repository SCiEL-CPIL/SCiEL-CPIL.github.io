import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill

    # Precompute all t-subsets and map to indices
    all_ts = list(itertools.combinations(range(v), t))
    n_ts = len(all_ts)
    ts_index = {s: i for i, s in enumerate(all_ts)}

    # Precompute all k-subsets
    all_ks = list(itertools.combinations(range(v), k))
    n_ks = len(all_ks)
    
    # Number of t-subsets in a k-set
    t_count = int(np.prod([k - i for i in range(t)]) / np.prod([i + 1 for i in range(t)]))
    
    # Precompute for each k-subset, the indices of the t-subsets it contains
    # block_t_indices[i, j] is the index of the j-th t-subset in the i-th k-subset
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = ts_index[sub]

    def greedy_randomized(seed, top_k=10):
        """
        Randomized greedy set cover.
        Picks a block from the top 'top_k' gainers randomly.
        """
        rng = np.random.RandomState(seed)
        covered = np.zeros(n_ts, dtype=bool)
        selected_blocks = []
        
        while True:
            if time.time() > deadline - 5.0:
                return None
            
            # Calculate gains: number of uncovered triples covered by each block
            # covered[block_t_indices] gets the coverage status for all triples in all blocks
            # ~ inverts it to get "uncovered" status
            # .sum(axis=1) counts them per block
            gains = (~covered[block_t_indices]).sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                # If max gain is 0, we are stuck or done. 
                # If not all covered, this is a failure for this run.
                if not np.all(covered):
                    return None
                else:
                    break
            
            # Find indices of blocks with max gain
            # To add randomness and escape local optima, we pick from the top K
            # But strictly speaking, standard greedy picks max. 
            # "Randomized Top-K" picks from blocks with gain close to max.
            
            # Let's pick from blocks with gain == max_gain first. 
            # If there are many, pick randomly.
            # To be more aggressive in escaping, we can allow gain == max_gain - 1 if top_k is large?
            # The prompt suggests "weighted sum". 
            # Let's stick to: Pick randomly from candidates with max_gain.
            # If we want to explore more, we can pick from top 'top_k' gainers.
            
            # Get sorted gains? No, that's expensive.
            # Just find where gains == max_gain
            candidates = np.where(gains == max_gain)[0]
            
            if len(candidates) > top_k:
                # Sample from top candidates
                # Actually, if there are thousands of max gainers, any is fine.
                # If we want to diversify, we might want to pick from a broader set.
                # Let's just pick randomly from all max gainers.
                # This is effectively "randomized greedy".
                pass
            
            best_idx = candidates[rng.randint(len(candidates))]
            
            # Add block
            selected_blocks.append(best_idx)
            
            # Update covered
            covered[block_t_indices[best_idx]] = True
            
            if np.all(covered):
                break

        return selected_blocks

    def reduce_blocks(block_indices):
        """
        Remove redundant blocks.
        A block is redundant if every t-subset in it is covered by at least one other block.
        """
        if not block_indices:
            return []
            
        current_blocks = list(block_indices)
        
        # We iterate until no more blocks can be removed
        while True:
            if time.time() > deadline - 2.0:
                break
                
            # Count coverage for each t-subset
            # coverage_count[j] = number of blocks in current_blocks that cover t-subset j
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current_blocks:
                coverage_count[block_t_indices[b_idx]] += 1
            
            new_blocks = []
            removed_any = False
            
            for b_idx in current_blocks:
                # Check if this block is redundant
                # Get the t-subsets in this block
                subs_in_block = block_t_indices[b_idx]
                
                # If all these t-subsets have coverage_count > 1, the block is redundant
                # Note: coverage_count includes this block itself, so we need > 1
                if np.all(coverage_count[subs_in_block] > 1):
                    # Redundant: do not add to new_blocks
                    # Decrement coverage for these subsets
                    coverage_count[subs_in_block] -= 1
                    removed_any = True
                else:
                    # Keep it
                    new_blocks.append(b_idx)
            
            current_blocks = new_blocks
            
            if not removed_any:
                break
                
        return current_blocks

    def local_search_swap(block_indices, max_iterations=200):
        """
        Try to swap points in blocks to reduce total block count or improve structure.
        This is a heuristic local search.
        """
        current_blocks = list(block_indices)
        if not current_blocks:
            return current_blocks
        
        # Calculate initial coverage
        covered = np.zeros(n_ts, dtype=bool)
        for b_idx in current_blocks:
            covered[block_t_indices[b_idx]] = True
        
        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time.time() < deadline - 10.0:
            improved = False
            iterations += 1
            
            # Try to remove a block and replace it with a better one?
            # Or try to swap a point in a block to increase coverage of another block?
            
            # Simple strategy: For each block, try swapping one point with another.
            # If the new block covers more uncovered triples (or same, but allows removal of another), keep it.
            # This is complex. Let's stick to a simpler "shake" if we have time.
            # Actually, the prompt suggests "final high-precision polish pass".
            # Let's try a simple hill-climbing on the set of blocks.
            
            # For each block, try to change it to a different k-subset that covers more "rare" triples.
            # Define weight of a triple as 1/coverage_count.
            # We want to maximize sum of weights of covered triples.
            
            # Calculate weights
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current_blocks:
                coverage_count[block_t_indices[b_idx]] += 1
            
            # Avoid division by zero
            weights = np.zeros(n_ts, dtype=np.float64)
            mask = coverage_count > 0
            weights[mask] = 1.0 / coverage_count[mask]
            
            # Try to improve each block
            for i, b_idx in enumerate(current_blocks):
                old_block = all_ks[b_idx]
                old_score = weights[block_t_indices[b_idx]].sum()
                
                # Try a few random neighbors
                # Pick a random point in block and a random point outside
                for _ in range(10):
                    if time.time() > deadline - 10.0:
                        break
                    p_in = old_block[random.randint(0, len(old_block)-1)]
                    p_out = random.randint(0, v-1)
                    if p_out in old_block:
                        continue
                    
                    new_block_set = set(old_block)
                    new_block_set.remove(p_in)
                    new_block_set.add(p_out)
                    new_block = tuple(sorted(new_block_set))
                    
                    # Find index of new_block in all_ks
                    # This is slow if done naively. We don't have a direct map from tuple to index for k-subsets.
                    # We can precompute a map? 
                    # all_ks is a list of tuples.
                    # Let's skip this for now as it's O(N) to find index.
                    # Instead, just calculate gain directly?
                    # We need block_t_indices for the new block.
                    # We can compute the t-subsets of new_block and map them to indices.
                    
                    new_subs = []
                    for sub in itertools.combinations(new_block, t):
                        if sub in ts_index:
                            new_subs.append(ts_index[sub])
                        else:
                            # Should not happen
                            new_subs.append(0) 
                    
                    new_score = weights[np.array(new_subs)].sum()
                    
                    if new_score > old_score + 1e-6:
                        # Accept
                        current_blocks[i] = None # Placeholder, need index
                        # We need to return the actual block, not index, or update the structure.
                        # The rest of the code uses indices into all_ks.
                        # If we change the block, we need its index in all_ks.
                        # Let's build a map: tuple -> index
                        # This is expensive to build every time.
                        # Let's assume we don't do this swap in this version to keep it simple and safe.
                        pass
        
        return current_blocks

    best_blocks = None
    best_size = float('inf')
    
    # Run multiple greedy attempts
    # We can afford many runs due to speed of NumPy gain calculation
    num_runs = 0
    while time.time() < deadline - 10.0:
        seed = num_runs
        blocks = greedy_randomized(seed)
        
        if blocks is not None:
            # Reduce redundancy
            reduced = reduce_blocks(blocks)
            
            if len(reduced) < best_size:
                best_size = len(reduced)
                best_blocks = reduced
                
                # Early stop if we hit theoretical lower bound or known best
                if best_size <= 76:
                    # We could try to improve further, but 76 is the known best.
                    # Let's run a few more to see if we can get 75? Unlikely.
                    # But maybe we can get a consistent 76 faster?
                    pass
        
        num_runs += 1
        
        # If we've run enough, break
        if num_runs > 200: 
             break

    if best_blocks is not None:
        # Convert indices back to lists of points
        return [list(all_ks[i]) for i in best_blocks]
    
    # Fallback if no valid covering found (should not happen)
    # Return a valid covering using pure greedy (no randomness) if possible
    # Or just return empty list which will be invalid.
    # Let's try one deterministic run.
    blocks = greedy_randomized(0)
    if blocks:
        return [list(all_ks[i]) for i in blocks]
        
    return []
