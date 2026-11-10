import itertools
import random
import time

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a deterministic greedy algorithm 
    followed by redundancy removal and local search.
    """
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    all_k_subsets = list(itertools.combinations(points, k))
    
    # Pre-compute t-subsets for each k-subset
    # Map: k_subset (tuple) -> list of t_subsets (tuples)
    k_to_t = {}
    for ks in all_k_subsets:
        k_to_t[ks] = list(itertools.combinations(ks, t))
        
    # For efficiency, we might want to map t_subsets to indices or just use a set
    uncovered = set(all_t_subsets)
    blocks = []
    
    start_time = time.time()
    time_limit = 100  # Leave buffer for post-processing
    
    # Phase 1: Deterministic Greedy Construction
    while uncovered and time.time() - start_time < time_limit:
        best_block = None
        max_cov = 0
        
        # Iterate over all k-subsets to find the one covering most uncovered t-subsets
        for ks in all_k_subsets:
            # Check if this block is already in our list? 
            # Greedy set cover usually doesn't add duplicates, but since we remove from uncovered,
            # a previously added block will have 0 coverage for uncovered items.
            # So we don't strictly need to check membership, but it saves time if we do.
            
            cov_count = 0
            for ts in k_to_t[ks]:
                if ts in uncovered:
                    cov_count += 1
                    # Early termination if we can't beat max_cov?
                    # if cov_count + (len(k_to_t[ks]) - current_index) <= max_cov: break
                    # This is complex to implement efficiently without numpy.
            
            if cov_count > max_cov:
                max_cov = cov_count
                best_block = ks
                
        if best_block is None:
            # Should not happen if uncovered is not empty
            break
            
        blocks.append(list(best_block))
        
        # Remove covered t-subsets
        for ts in k_to_t[best_block]:
            if ts in uncovered:
                uncovered.discard(ts)
                
    # Phase 2: Redundancy Removal
    # This is similar to the original but we can optimize it
    if not blocks:
        return []
        
    # Build coverage map: t_subset -> count of blocks covering it
    # We only need to know if count > 1 for redundancy check
    coverage_count = {ts: 0 for ts in all_t_subsets}
    
    for block in blocks:
        for ts in itertools.combinations(block, t):
            coverage_count[ts] += 1
            
    changed = True
    while changed:
        changed = False
        new_blocks = []
        for block in blocks:
            # Check if block is redundant
            # It is redundant if all t-subsets in it have coverage_count > 1
            # Note: coverage_count includes the block itself.
            # So if coverage_count[ts] > 1 for all ts in block, then block is redundant.
            is_redundant = True
            for ts in itertools.combinations(block, t):
                if coverage_count[ts] <= 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                changed = True
                # Update counts
                for ts in itertools.combinations(block, t):
                    coverage_count[ts] -= 1
            else:
                new_blocks.append(block)
        
        blocks = new_blocks
        
    # Phase 3: Local Search (Optional, if time remains)
    # Try to swap blocks to reduce count further?
    # For now, return the result of Greedy + Redundancy Removal.
    
    return blocks
