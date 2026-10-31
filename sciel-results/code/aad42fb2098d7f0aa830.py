import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110
    
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    n_t_subsets = len(all_t_subsets)
    
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Pre-compute all k-subsets
    all_k_subsets = list(itertools.combinations(points, k))
    n_k_subsets = len(all_k_subsets)
    
    # Map block index to list of t-subset indices it covers
    block_t_indices = []
    # Also compute degree of each t-subset for criticality weighting
    degree = [0] * n_t_subsets
    
    for block in all_k_subsets:
        covered_ts = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            covered_ts.append(idx)
            degree[idx] += 1
        block_t_indices.append(covered_ts)
        
    # Criticality weights: 1/degree. If degree is 0, it's impossible, but C(v,k,t) exists.
    # We use these weights to guide the search towards covering "hard" subsets first.
    weights = [1.0 / d if d > 0 else 0.0 for d in degree]
    
    # Build reverse map: t_subset_index -> list of block indices
    ts_to_blocks = [[] for _ in range(n_t_subsets)]
    for b_idx, ts_list in enumerate(block_t_indices):
        for ts_idx in ts_list:
            ts_to_blocks[ts_idx].append(b_idx)

    def greedy_construct():
        coverage = [0] * n_t_subsets
        selected = []
        uncovered_count = n_t_subsets
        
        # Track weighted uncovered sum for better gain calculation?
        # Gain = sum of weights of newly covered subsets.
        
        while uncovered_count > 0 and time.time() - start_time < time_limit * 0.2:
            # Find uncovered subset with highest weight? Or just pick one?
            # To be fast, we can pick a random uncovered one, but greedy by weight is better.
            # However, finding the global max weight uncovered every step is O(N).
            # Let's just pick a random uncovered one, but evaluate candidates by weighted gain.
            
            # Find an uncovered subset
            # Optimization: maintain a list of uncovered indices? 
            # For v=19, t=4, N=3876. Iterating is fine.
            uncovered_indices = [i for i, c in enumerate(coverage) if c == 0]
            if not uncovered_indices:
                break
            target = random.choice(uncovered_indices)
            
            best_block = -1
            best_gain = -1.0
            
            for b_idx in ts_to_blocks[target]:
                gain = 0.0
                for ts_idx in block_t_indices[b_idx]:
                    if coverage[ts_idx] == 0:
                        gain += weights[ts_idx]
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                break
                
            selected.append(best_block)
            for ts_idx in block_t_indices[best_block]:
                if coverage[ts_idx] == 0:
                    uncovered_count -= 1
                coverage[ts_idx] += 1
                
        # Force cover remaining if time ran out
        if uncovered_count > 0:
            for i in range(n_t_subsets):
                if coverage[i] == 0:
                    candidates = ts_to_blocks[i]
                    if candidates:
                        b = candidates[0]
                        if b not in selected:
                            selected.append(b)
                            for ts_idx in block_t_indices[b]:
                                if coverage[ts_idx] == 0:
                                    uncovered_count -= 1
                                coverage[ts_idx] += 1
        return selected, coverage

    init_blocks, init_cov = greedy_construct()
    
    # Local Search Optimization
    # Objective: Minimize number of blocks.
    # Constraint: All t-subsets covered.
    
    def try_remove(block_list, cov, b_idx):
        """Returns True if block b_idx can be removed."""
        for ts in block_t_indices[b_idx]:
            if cov[ts] == 1:
                return False
        return True

    def perform_swap(block_list, cov, i):
        """
        Tries to swap block at index i with a better one.
        Returns True if a swap was made.
        """
        old_b = block_list[i]
        unique_ts = []
        for ts in block_t_indices[old_b]:
            if cov[ts] == 1:
                unique_ts.append(ts)
        
        if not unique_ts:
            # Can remove, but we are in swap mode? 
            # If we can remove, we should have done it in the delete pass.
            # If we are here, maybe we want to swap to free up something else?
            # For now, if no unique ts, it's redundant.
            return False

        # Find candidate blocks that cover all unique_ts
        # Intersection of ts_to_blocks[ts] for ts in unique_ts
        # Start with the smallest list to be fast
        unique_ts.sort(key=lambda x: len(ts_to_blocks[x]))
        if not unique_ts: return False
        
        candidates = set(ts_to_blocks[unique_ts[0]])
        for ts in unique_ts[1:]:
            candidates.intersection_update(ts_to_blocks[ts])
        
        candidates.discard(old_b)
        
        if not candidates:
            return False
        
        # Evaluate candidates
        best_new = -1
        best_score = -1.0
        
        # Sample if too many
        cand_list = list(candidates)
        if len(cand_list) > 20:
            cand_list = random.sample(cand_list, 20)
            
        for nb in cand_list:
            # Score: How many "new" weighted subsets does it cover?
            # Or simply: does it cover other low-coverage subsets?
            score = 0.0
            for ts in block_t_indices[nb]:
                if cov[ts] == 1:
                    # This is a "unique" coverage for the current set.
                    # If we swap, we lose old_b's coverage.
                    # We want to minimize the "damage" or maximize "help".
                    # Actually, if we swap, we must ensure unique_ts are still covered.
                    # Since nb covers all unique_ts, they remain covered.
                    # What about other subsets?
                    # If cov[ts] > 1, removing old_b doesn't uncover it.
                    # If cov[ts] == 1, and ts is NOT in unique_ts, it gets uncovered!
                    # Wait. unique_ts are the ones with cov==1 in old_b.
                    # If nb covers unique_ts, they are safe.
                    # Other ts in old_b with cov==1 are NOT in unique_ts? No, unique_ts is ALL ts in old_b with cov==1.
                    # So if nb covers all unique_ts, no subset becomes uncovered.
                    # The gain is how many *other* subsets it helps (cov goes up).
                    # This allows future removals.
                    if cov[ts] > 1:
                        score += weights[ts] * 0.1 # Small bonus for redundancy
                    elif cov[ts] == 1:
                        # This case shouldn't happen if nb covers all unique_ts and unique_ts is defined as all cov==1 in old_b.
                        # Unless nb does NOT cover some ts in old_b where cov[ts]==1?
                        # No, we intersected for ALL unique_ts.
                        pass
                    # Actually, the main benefit of a swap is if it allows removing ANOTHER block later.
                    # A good heuristic: Maximize the sum of weights of subsets it covers that have low coverage.
                    pass
            
            # Simpler Score: Maximize overlap with "hard" subsets.
            # Let's just pick the one with max weighted gain on currently covered subsets to build redundancy.
            s = 0.0
            for ts in block_t_indices[nb]:
                if cov[ts] >= 1:
                    s += weights[ts]
            if s > best_score:
                best_score = s
                best_new = nb

        if best_new != -1:
            # Apply swap
            for ts in block_t_indices[old_b]:
                cov[ts] -= 1
            for ts in block_t_indices[best_new]:
                cov[ts] += 1
            block_list[i] = best_new
            return True
        return False

    current_blocks = init_blocks
    current_cov = init_cov
    
    # Iterative improvement
    improved = True
    while improved and time.time() - start_time < time_limit * 0.9:
        improved = False
        
        # Pass 1: Deletions
        for i in range(len(current_blocks)):
            b_idx = current_blocks[i]
            if try_remove(current_blocks, current_cov, b_idx):
                for ts in block_t_indices[b_idx]:
                    current_cov[ts] -= 1
                current_blocks.pop(i)
                improved = True
                # Restart deletion pass
                break
        
        if time.time() - start_time > time_limit * 0.8:
            break
            
        # Pass 2: Swaps
        if not improved:
            # Try to swap blocks to enable future deletions
            # Random order to avoid bias
            indices = list(range(len(current_blocks)))
            random.shuffle(indices)
            for i in indices:
                if perform_swap(current_blocks, current_cov, i):
                    improved = True
                    # Don't break immediately, try more swaps?
                    # Or break to re-evaluate deletions?
                    # Let's do a few swaps then check deletions.
                    # For simplicity, break to check deletions.
                    break

    # Final cleanup
    changed = True
    while changed and time.time() - start_time < time_limit:
        changed = False
        for i in range(len(current_blocks)):
            b_idx = current_blocks[i]
            if try_remove(current_blocks, current_cov, b_idx):
                for ts in block_t_indices[b_idx]:
                    current_cov[ts] -= 1
                current_blocks.pop(i)
                changed = True
                break

    return [list(all_k_subsets[i]) for i in current_blocks]
