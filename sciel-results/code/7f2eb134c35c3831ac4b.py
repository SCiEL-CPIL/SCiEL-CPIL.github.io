import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    n_t_subsets = len(all_t_subsets)
    
    # Precompute t-subsets for each potential k-block?
    # No, we build blocks dynamically.
    
    def greedy_cover(seed):
        random.seed(seed)
        uncovered = set(range(n_t_subsets))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset to start the block
            # This helps escape local minima compared to always picking min
            start_idx = random.choice(list(uncovered))
            start_ts = all_t_subsets[start_idx]
            
            block = list(start_ts)
            # Ensure we have a partial block. 
            # The seed program starts with 1 point. 
            # Here we start with t points. 
            # We need to expand to k points.
            
            while len(block) < k:
                # Calculate gain for each candidate point
                best_gain = -1
                best_candidates = []
                
                # Current block points
                block_set = set(block)
                
                # We need to count how many new t-subsets are covered by adding x
                # A t-subset is covered if it is a subset of block + {x}
                # It must include x and (t-1) points from block.
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    # Iterate over (t-1) subsets of current block
                    for rest in itertools.combinations(block, t - 1):
                        # Form the t-subset: rest + x
                        # Sort to match the tuple format in t_subset_to_idx
                        # Note: rest is a tuple of sorted ints, x is int.
                        # We need to merge and sort.
                        ts_list = list(rest)
                        ts_list.append(x)
                        ts_list.sort()
                        ts_tuple = tuple(ts_list)
                        
                        idx = t_subset_to_idx.get(ts_tuple, -1)
                        if idx != -1 and idx in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain and gain > 0:
                        best_candidates.append(x)
                    
                if not best_candidates:
                    # If no gain, pick a random point to avoid infinite loop
                    # This shouldn't happen if uncovered is not empty, 
                    # unless the block is already full or something weird.
                    available = [x for x in range(v) if x not in block_set]
                    if available:
                        block.append(random.choice(available))
                    else:
                        break
                else:
                    # Random tie-break
                    block.append(random.choice(best_candidates))
            
            block.sort()
            # Convert block to tuple for consistent hashing if needed, but list is fine
            # Update uncovered
            block_t_subsets = set()
            for ts in itertools.combinations(block, t):
                idx = t_subset_to_idx[ts]
                block_t_subsets.add(idx)
            
            blocks.append(block)
            uncovered.difference_update(block_t_subsets)
            
            # Time check
            if time.time() > deadline:
                break
        
        return blocks

    def prune(blocks):
        if not blocks:
            return blocks
        
        # Count coverage for each t-subset
        coverage_count = [0] * n_t_subsets
        block_t_subsets_list = []
        
        for b in blocks:
            t_subs = []
            for ts in itertools.combinations(b, t):
                idx = t_subset_to_idx[ts]
                coverage_count[idx] += 1
                t_subs.append(idx)
            block_t_subsets_list.append(t_subs)
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for i, b in enumerate(blocks):
                # Check if any t-subset in this block is covered only by this block
                is_redundant = True
                for idx in block_t_subsets_list[i]:
                    if coverage_count[idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove this block
                    for idx in block_t_subsets_list[i]:
                        coverage_count[idx] -= 1
                    changed = True
                else:
                    new_blocks.append(b)
            
            blocks = new_blocks
            # Re-index block_t_subsets_list? 
            # Actually, we can just rebuild it or keep it in sync.
            # Since we remove from blocks, we should remove from block_t_subsets_list too.
            # Let's rebuild for simplicity if changed, or filter in sync.
            if changed:
                 # Rebuild mapping for next iteration is safer
                 block_t_subsets_list = []
                 for b in blocks:
                     t_subs = []
                     for ts in itertools.combinations(b, t):
                         idx = t_subset_to_idx[ts]
                         t_subs.append(idx)
                     block_t_subsets_list.append(t_subs)
        
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple randomized greedy instances
    # Estimate time per run: ~50ms-100ms? 
    # Let's try up to 50 runs or until deadline.
    for i in range(100):
        if time.time() > deadline - 5: # Leave 5s for pruning
            break
            
        blocks = greedy_cover(i)
        if not blocks:
            continue
            
        # Prune
        pruned_blocks = prune(blocks)
        
        if len(pruned_blocks) < best_len:
            best_len = len(pruned_blocks)
            best_blocks = pruned_blocks
            
    if best_blocks is None:
        # Fallback to simple deterministic greedy if all fail or timeout
        # Re-run greedy_cover with seed 0 and no pruning if needed?
        # Or just return whatever we have.
        # If best_blocks is None, we haven't found any valid covering in time.
        # This is unlikely for small v.
        # Let's force one run without pruning if time allows.
        if time.time() < deadline:
             blocks = greedy_cover(0)
             if blocks:
                 best_blocks = blocks
             else:
                 return []
        
    return best_blocks
