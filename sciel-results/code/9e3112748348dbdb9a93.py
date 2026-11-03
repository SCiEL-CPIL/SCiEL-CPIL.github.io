import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 40s margin for safety

    # 1. Precompute
    points = range(v)
    t_subsets = list(itertools.combinations(points, t))
    num_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # Map each block to the indices of t-subsets it covers
    block_t_indices = []
    for b in all_blocks:
        idxs = [t_index[ts] for ts in itertools.combinations(b, t)]
        block_t_indices.append(idxs)
        
    # For fast lookup: which blocks contain a given t-subset?
    t_containing_blocks = [[] for _ in range(num_t)]
    for b_idx, b in enumerate(all_blocks):
        for idx in block_t_indices[b_idx]:
            t_containing_blocks[idx].append(b_idx)

    best_blocks = None
    best_size = float('inf')

    # Run multiple greedy attempts with different random seeds/orders
    # Since the problem is symmetric, standard greedy is deterministic.
    # We can randomize the tie-breaking or the initial order.
    
    for attempt in range(10):
        if time.time() > deadline:
            break
            
        # Reset state
        cover_count = [0] * num_t
        uncovered = [True] * num_t
        current_blocks = []
        
        # Greedy Phase
        # To make it non-deterministic and explore different solutions, 
        # we can randomize the order in which we check blocks when gains are equal,
        # or simply randomize the starting point of the scan.
        
        # A faster greedy: 
        # Instead of scanning all blocks, we can pick a random uncovered t-subset,
        # and look at all blocks containing it. Pick the one with max total gain.
        # This is O(num_uncovered * avg_blocks_per_t). 
        # avg_blocks_per_t = C(v-t, k-t) = C(15, 3) = 455.
        # So each step is ~455 * (cost to calc gain).
        
        while any(uncovered):
            if time.time() > deadline:
                break
            
            # Pick a random uncovered t-subset to guide the search
            # This reduces the number of blocks we need to evaluate for gain
            # from 50k to ~455.
            
            # Find an uncovered index
            # To avoid O(N) scan for an uncovered item every time, we can maintain a list of uncovered indices
            # But for simplicity and speed, let's just pick a random index and check.
            # If we pick a covered one, we retry.
            
            seed_idx = -1
            for _ in range(20):
                idx = random.randint(0, num_t - 1)
                if uncovered[idx]:
                    seed_idx = idx
                    break
            
            if seed_idx == -1:
                # Fallback: find first uncovered
                for i in range(num_t):
                    if uncovered[i]:
                        seed_idx = i
                        break
                if seed_idx == -1:
                    break # Should not happen if any(uncovered) is true
            
            # Evaluate all blocks containing seed_idx
            candidates = t_containing_blocks[seed_idx]
            
            best_block_idx = -1
            best_gain = 0
            
            for b_idx in candidates:
                # Calculate gain: number of uncovered t-subsets covered by b_idx
                gain = 0
                for idx in block_t_indices[b_idx]:
                    if uncovered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                break
                
            # Add block
            current_blocks.append(best_block_idx)
            for idx in block_t_indices[best_block_idx]:
                if uncovered[idx]:
                    uncovered[idx] = False
                cover_count[idx] += 1
            
        # Refinement Phase: Iterative Deletion
        # Try to remove blocks that don't cover any unique t-subset
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_blocks = []
            # Shuffle to break symmetry
            random.shuffle(current_blocks)
            for b_idx in current_blocks:
                # Check if any t-subset in this block is covered ONLY by this block
                can_remove = True
                for idx in block_t_indices[b_idx]:
                    if cover_count[idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove it
                    for idx in block_t_indices[b_idx]:
                        cover_count[idx] -= 1
                    changed = True
                else:
                    new_blocks.append(b_idx)
            current_blocks = new_blocks
            
            if len(current_blocks) < best_size:
                best_size = len(current_blocks)
                best_blocks = current_blocks[:]
                
        # Additional Refinement: Randomized Re-addition / Swap?
        # For now, let's stick to deletion. If we have time, we can try to swap pairs.
        # Given the time limit, the deletion pass is the most reliable improvement.
        
        if time.time() > deadline:
            break

    if best_blocks is None:
        return []
        
    return [list(all_blocks[b_idx]) for b_idx in best_blocks]
