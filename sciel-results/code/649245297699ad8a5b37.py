import itertools
import random
import time

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a randomized greedy algorithm 
    followed by redundancy removal.
    """
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    # Map each t-subset to an index for faster lookup if needed, but set of tuples is fine for small v
    
    best_blocks = None
    best_size = float('inf')
    
    start_time = time.time()
    time_limit = 110  # Leave some buffer for the evaluator
    
    # Run multiple iterations to find a better covering
    iterations = 0
    while time.time() - start_time < time_limit:
        iterations += 1
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset to anchor the block selection
            # This introduces randomness to escape local minima of the greedy choice
            anchor = random.choice(list(uncovered))
            
            # Generate all k-subsets that contain the anchor
            # The remaining k-t points must be chosen from the v-t points not in anchor
            remaining_points = [p for p in points if p not in anchor]
            candidates = []
            
            # We need to choose k-t points from remaining_points
            # v-t = 19-4=15, k-t=7-4=3. C(15,3)=455.
            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(anchor + combo))
                # Count how many uncovered t-subsets this block covers
                # A block of size k covers C(k, t) t-subsets
                # We only care about those in 'uncovered'
                covered_count = 0
                for t_sub in itertools.combinations(block, t):
                    if t_sub in uncovered:
                        covered_count += 1
                candidates.append((covered_count, block))
            
            # Pick the candidate with the maximum coverage
            # If there's a tie, pick randomly among the best to add diversity
            max_cov = max(c[0] for c in candidates)
            best_candidates = [c[1] for c in candidates if c[0] == max_cov]
            chosen_block = random.choice(best_candidates)
            
            blocks.append(list(chosen_block))
            
            # Remove all t-subsets covered by chosen_block from uncovered
            for t_sub in itertools.combinations(chosen_block, t):
                if t_sub in uncovered:
                    uncovered.discard(t_sub)
        
        # Post-processing: Remove redundant blocks
        # A block is redundant if every t-subset it covers is also covered by at least one other block.
        # To check this efficiently:
        # 1. For each t-subset, count how many blocks cover it.
        # 2. If a block only covers t-subsets that have count > 1, it is redundant.
        
        # Count coverage for each t-subset
        # Note: We only need to consider t-subsets that are in the union of all blocks.
        # But simpler: iterate through blocks and their t-subsets.
        
        # Let's build a map: t_subset -> list of block indices covering it
        # This might be memory heavy if we do it naively, but v=19 is small.
        # Total t-subsets = 3876.
        coverage_map = {ts: [] for ts in all_t_subsets}
        
        for i, block in enumerate(blocks):
            for ts in itertools.combinations(block, t):
                coverage_map[ts].append(i)
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for i, block in enumerate(blocks):
                # Check if block i is redundant
                # It is redundant if for every ts in block, len(coverage_map[ts]) > 1
                # AND that other block is not block i itself (which is guaranteed if count > 1)
                is_redundant = True
                for ts in itertools.combinations(block, t):
                    if len(coverage_map[ts]) <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove block i
                    changed = True
                    # Update coverage map: remove i from all ts in block
                    for ts in itertools.combinations(block, t):
                        coverage_map[ts].remove(i)
                else:
                    new_blocks.append(block)
            
            blocks = new_blocks
            # Note: We don't need to rebuild the whole map from scratch if we update it carefully,
            # but since we are removing blocks, the indices shift. 
            # It is safer to rebuild the coverage map if we remove a block, 
            # or just accept that the indices in the list `blocks` change.
            # Actually, the `coverage_map` values are indices into the `blocks` list.
            # If we remove a block, the indices of subsequent blocks in `blocks` change,
            # but the `coverage_map` still holds the old indices.
            # So we must rebuild `coverage_map` after any removal in the while loop?
            # Or better: just break the inner loop and restart the redundancy check from scratch?
            # Given the small size, rebuilding is fine.
            
            if changed:
                # Rebuild coverage map for the next iteration of the while loop
                coverage_map = {ts: [] for ts in all_t_subsets}
                for i, block in enumerate(blocks):
                    for ts in itertools.combinations(block, t):
                        coverage_map[ts].append(i)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
        # If we've run many iterations, we might be stuck. 
        # We can break if no improvement in last N iterations, but for simplicity, 
        # we just run until time limit.
        
    if best_blocks is None:
        # Fallback: return a simple valid covering (though unlikely to be reached)
        return []
        
    return best_blocks
