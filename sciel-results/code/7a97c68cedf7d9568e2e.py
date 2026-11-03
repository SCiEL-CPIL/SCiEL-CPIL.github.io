import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 145.0
    start_time = time.time()
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Map each t-subset to an index
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute coverage for all possible blocks
    # A block is a k-subset of {0,...,v-1}
    # For each block, compute which t-subsets it covers
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, store the set of t-subset indices it covers
    # Using a list of sets for fast intersection checks
    block_to_covered = []
    for block in all_blocks:
        covered = set()
        for ts in itertools.combinations(block, t):
            covered.add(t_subset_to_idx[ts])
        block_to_covered.append(covered)
    
    # Build reverse index: for each t-subset, which blocks cover it?
    # This is essential for the Pricing Subproblem and Greedy
    t_subset_to_blocks = [set() for _ in range(n_t)]
    for b_idx, covered in enumerate(block_to_covered):
        for ts_idx in covered:
            t_subset_to_blocks[ts_idx].add(b_idx)

    def greedy_cover(seed=0):
        """Fast greedy covering to get an initial valid solution."""
        rng = random.Random(seed)
        uncovered = set(range(n_t))
        blocks = []
        
        while uncovered:
            if time.time() > deadline - 20.0:
                break
            
            # Pick a random uncovered t-subset to guide the search
            start_idx = rng.choice(list(uncovered))
            
            # Find all blocks that cover this t-subset
            candidate_blocks = t_subset_to_blocks[start_idx]
            
            # Choose the block that covers the most uncovered t-subsets
            best_block = None
            best_gain = -1
            for b_idx in candidate_blocks:
                gain = len(block_to_covered[b_idx] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block is None:
                break
                
            blocks.append(all_blocks[best_block])
            for ts_idx in block_to_covered[best_block]:
                uncovered.discard(ts_idx)
        
        return blocks

    def local_search(blocks, max_iter=1000, seed=0):
        """
        Local search to minimize the number of blocks while maintaining coverage.
        Strategy:
        1. Start with a valid solution.
        2. Try to remove a block if the solution remains valid.
        3. If no block can be removed, try to swap a block with another that allows removal of 2 blocks? 
           Or just simple deletion first.
        4. To escape local optima, we can also try adding a block and then removing 2.
        But simple deletion is the most effective for set cover.
        
        We will implement a robust deletion loop with randomization.
        """
        rng = random.Random(seed)
        current_blocks = list(blocks)
        n_current = len(current_blocks)
        
        # Precompute coverage counts for each t-subset
        # coverage_count[i] = number of blocks in current_blocks that cover t-subset i
        coverage_count = np.zeros(n_t, dtype=np.int32)
        for b in current_blocks:
            for ts_idx in block_to_covered[all_blocks.index(b)]: # This is slow, O(N) lookup
                coverage_count[ts_idx] += 1
        
        # Better: maintain indices
        current_indices = [all_blocks.index(b) for b in current_blocks]
        # This is also slow. Let's assume the input blocks are from all_blocks or we can map them.
        # Since greedy returns all_blocks[i], we should pass indices or ensure mapping is fast.
        # Let's refactor to work with indices internally.
        
        # Re-initialize coverage count properly using indices
        coverage_count = np.zeros(n_t, dtype=np.int32)
        for idx in current_indices:
            for ts_idx in block_to_covered[idx]:
                coverage_count[ts_idx] += 1
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            if time.time() > deadline - 10.0:
                break
                
            # Shuffle indices to randomize deletion order
            rng.shuffle(current_indices)
            
            for i in range(len(current_indices)):
                idx_to_remove = current_indices[i]
                
                # Check if removing this block leaves any t-subset uncovered
                # i.e., for all ts in block_to_covered[idx_to_remove], coverage_count[ts] > 1
                can_remove = True
                for ts_idx in block_to_covered[idx_to_remove]:
                    if coverage_count[ts_idx] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove the block
                    current_indices.pop(i)
                    # Update coverage counts
                    for ts_idx in block_to_covered[idx_to_remove]:
                        coverage_count[ts_idx] -= 1
                    improved = True
                    break # Restart the loop from the beginning to check for more removals
        
        return [all_blocks[i] for i in current_indices]

    def local_search_with_swaps(blocks, max_iter=500, seed=0):
        """
        More aggressive local search.
        1. Try deletions.
        2. If stuck, try swaps: replace block A with block B such that we can then delete block C.
        This is complex. A simpler variant: 
        Randomly remove a block (even if invalid), then try to fix the uncovered triples by adding blocks.
        This is essentially a random restart of greedy on the remaining uncovered set.
        
        Let's try a different approach: Simulated Annealing on the set of blocks.
        State: set of block indices.
        Energy: number of uncovered triples + lambda * number of blocks.
        We want to minimize Energy.
        
        However, given the time constraint, simple deletion local search is very strong for set cover.
        Let's add a "perturbation" phase: randomly remove a few blocks, then run greedy to fill gaps, then delete again.
        """
        rng = random.Random(seed)
        current_indices = [all_blocks.index(b) for b in blocks]
        
        def get_coverage_count(indices):
            counts = np.zeros(n_t, dtype=np.int32)
            for idx in indices:
                for ts_idx in block_to_covered[idx]:
                    counts[ts_idx] += 1
            return counts
        
        current_counts = get_coverage_count(current_indices)
        
        best_indices = list(current_indices)
        best_size = len(best_indices)
        
        for _ in range(max_iter):
            if time.time() > deadline - 10.0:
                break
            
            # Perturbation: Remove 1-3 random blocks
            n_remove = rng.randint(1, 3)
            if len(current_indices) <= n_remove:
                break
                
            to_remove = rng.sample(current_indices, n_remove)
            to_remove_set = set(to_remove)
            
            # Apply removal
            new_indices = [idx for idx in current_indices if idx not in to_remove_set]
            
            # Update counts
            new_counts = current_counts.copy()
            for idx in to_remove:
                for ts_idx in block_to_covered[idx]:
                    new_counts[ts_idx] -= 1
            
            # Fix uncovered triples using greedy
            uncovered = np.where(new_counts == 0)[0]
            uncovered_set = set(uncovered.tolist())
            
            while uncovered_set:
                if time.time() > deadline - 10.0:
                    break
                
                # Pick a random uncovered triple
                ts_idx = rng.choice(list(uncovered_set))
                candidates = t_subset_to_blocks[ts_idx]
                
                best_gain = -1
                best_block = None
                for b_idx in candidates:
                    # Gain is number of uncovered triples it covers
                    # We can approximate or compute exactly
                    # Exact computation:
                    gain = 0
                    for ts in block_to_covered[b_idx]:
                        if ts in uncovered_set:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_block = b_idx
                
                if best_block is None:
                    break
                    
                # Add the block
                new_indices.append(best_block)
                for ts in block_to_covered[best_block]:
                    if ts in uncovered_set:
                        uncovered_set.discard(ts)
                    new_counts[ts] += 1
            
            if uncovered_set:
                # Failed to fix, revert
                continue
                
            # Now try to delete blocks from this new set
            # Run a quick deletion pass
            improved = True
            while improved:
                improved = False
                rng.shuffle(new_indices)
                for i in range(len(new_indices)):
                    idx = new_indices[i]
                    can_remove = True
                    for ts in block_to_covered[idx]:
                        if new_counts[ts] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        new_indices.pop(i)
                        for ts in block_to_covered[idx]:
                            new_counts[ts] -= 1
                        improved = True
                        break
            
            # Accept if better or equal (to allow moving in plateau)
            if len(new_indices) < best_size:
                best_indices = list(new_indices)
                best_size = len(best_indices)
                current_indices = best_indices
                current_counts = get_coverage_count(current_indices)
            else:
                # Revert to previous state for next perturbation
                # But we modified current_indices? No, we used new_indices.
                # We should keep current_indices as the best found so far to perturb from best.
                pass
        
        return [all_blocks[i] for i in best_indices]

    # 1. Get initial greedy solution
    best_blocks = greedy_cover(0)
    best_len = len(best_blocks)
    
    # 2. Apply local search to improve
    if time.time() < deadline - 30.0:
        ls_blocks = local_search(best_blocks, max_iter=100, seed=0)
        if len(ls_blocks) < best_len:
            best_len = len(ls_blocks)
            best_blocks = ls_blocks
    
    # 3. Try perturbation-based local search
    if time.time() < deadline - 15.0:
        ls_blocks2 = local_search_with_swaps(best_blocks, max_iter=50, seed=1)
        if len(ls_blocks2) < best_len:
            best_len = len(ls_blocks2)
            best_blocks = ls_blocks2

    # 4. Try more greedy starts with local search
    for seed in range(1, 10):
        if time.time() > deadline - 15.0:
            break
        
        init = greedy_cover(seed)
        if len(init) < best_len:
            init_ls = local_search(init, max_iter=50, seed=seed)
            if len(init_ls) < best_len:
                best_len = len(init_ls)
                best_blocks = init_ls

    # Final validation
    if best_blocks is None:
        return []
        
    covered = set()
    for block in best_blocks:
        for ts in itertools.combinations(block, t):
            covered.add(t_subset_to_idx[ts])
            
    if len(covered) != n_t:
        # Fallback to a valid greedy solution
        return greedy_cover(0)
        
    return best_blocks
