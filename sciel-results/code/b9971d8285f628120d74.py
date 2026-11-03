import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a randomized greedy algorithm 
    with local search improvement.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    triple_index = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute for each point, the list of triples containing it? 
    # Or for each triple, the list of points in it?
    # We need to evaluate candidate blocks.
    
    def generate_candidate_blocks(triple):
        """Generate all k-subsets containing the given t-subset."""
        remaining_points = [p for p in points if p not in triple]
        # We need to choose k-t more points from the remaining v-t points.
        for comb in itertools.combinations(remaining_points, k - t):
            yield tuple(sorted(triple + comb))

    def greedy_cover(random_seed=0):
        rng = random.Random(random_seed)
        # coverage[i] = number of blocks covering triple i
        coverage = np.zeros(num_triples, dtype=np.int32)
        blocks = []
        block_set = set()
        
        # Uncovered triples
        # We can maintain a list of uncovered triple indices for fast access
        uncovered_indices = list(range(num_triples))
        
        while uncovered_indices:
            if time.time() > deadline:
                break
            
            # Pick a random uncovered triple to anchor the block
            # Picking the one with lowest coverage (which is 0) is standard.
            # Randomizing the choice among uncovered ones helps escape local minima.
            idx = rng.choice(uncovered_indices)
            anchor_triple = all_triples[idx]
            
            best_block = None
            best_gain = -1
            
            # Iterate through all candidate blocks containing the anchor triple
            for block in generate_candidate_blocks(anchor_triple):
                if block in block_set:
                    continue
                
                gain = 0
                # Calculate how many uncovered triples this block covers
                for sub in itertools.combinations(block, t):
                    ti = triple_index[sub]
                    if coverage[ti] == 0:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    # Early stopping? If gain is max possible (35), we can stop.
                    if gain == 35:
                        break
            
            if best_block is None:
                # Should not happen if v >= k
                break
                
            # Add the block
            blocks.append(best_block)
            block_set.add(best_block)
            
            # Update coverage
            for sub in itertools.combinations(best_block, t):
                ti = triple_index[sub]
                coverage[ti] += 1
                
            # Remove newly covered triples from uncovered list
            # This is O(N) per step, which might be slow.
            # Better: Keep a list and filter, or use a boolean array and scan.
            # Since num_triples is ~2000, scanning is fast.
            new_uncovered = []
            for u_idx in uncovered_indices:
                if coverage[u_idx] == 0:
                    new_uncovered.append(u_idx)
            uncovered_indices = new_uncovered
            
            if not uncovered_indices:
                break
                
        return blocks

    def local_search(blocks):
        """
        Attempts to reduce the number of blocks by removing redundant ones 
        or swapping blocks to enable further removals.
        """
        if not blocks:
            return blocks
        
        # Precompute coverage for the current set of blocks
        coverage = np.zeros(num_triples, dtype=np.int32)
        for block in blocks:
            for sub in itertools.combinations(block, t):
                coverage[triple_index[sub]] += 1
        
        block_set = set(blocks)
        current_blocks = list(blocks)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Remove redundant blocks
            i = 0
            while i < len(current_blocks):
                block = current_blocks[i]
                # Check if this block is the only one covering any triple
                is_redundant = True
                for sub in itertools.combinations(block, t):
                    ti = triple_index[sub]
                    if coverage[ti] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove it
                    for sub in itertools.combinations(block, t):
                        coverage[triple_index[sub]] -= 1
                    current_blocks.pop(i)
                    block_set.discard(block)
                    improved = True
                    # Do not increment i, check the next block at the same index
                else:
                    i += 1
            
            # 2. Try swaps
            if not improved:
                # Try to swap a block with another to potentially allow removal
                # This is more expensive. We limit this phase.
                # Simple swap: Pick a block B. Remove it. Find uncovered triples U.
                # If U can be covered by a single new block B', swap B with B'.
                # This keeps the block count same, but might change the structure.
                # To actually reduce count, we need to find a swap that makes another block redundant.
                # This is complex. Let's stick to redundancy removal for now, 
                # as randomized greedy with multiple restarts is often sufficient.
                pass

        return current_blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple randomized greedy instances
    # The number of iterations depends on time available.
    # Each greedy run takes roughly T_greedy.
    # We estimate T_greedy by running one instance and measuring time.
    
    # First run to estimate time
    t0 = time.time()
    blocks_0 = greedy_cover(0)
    t1 = time.time()
    t_greedy = t1 - t0
    
    if len(blocks_0) < best_len:
        best_blocks = blocks_0
        best_len = len(blocks_0)
        
    # Apply local search to the best so far
    if best_blocks:
        improved_blocks = local_search(best_blocks)
        if len(improved_blocks) < best_len:
            best_blocks = improved_blocks
            best_len = len(improved_blocks)
            
    # Estimate how many more runs we can do
    if t_greedy > 0:
        # Reserve 10s for local search on the final best
        time_left = deadline - time.time()
        # We don't know the cost of local search yet, assume it's small compared to greedy
        # or just use 50% of time for greedy, 50% for local search on best
        # Let's say we use 70% for greedy, 30% for local search on the final best
        time_for_greedy = (time_left * 0.7)
        num_extra_runs = int(time_for_greedy / t_greedy)
    else:
        num_extra_runs = 100
        
    # Cap the number of runs to avoid excessive memory or time if t_greedy is very small
    num_extra_runs = min(num_extra_runs, 500)
    
    for i in range(1, num_extra_runs + 1):
        if time.time() > deadline - 10.0: # Stop 10s before deadline to allow final local search
            break
            
        seed = i
        blocks = greedy_cover(seed)
        
        if len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)
            
    # Final local search on the best block set found
    if best_blocks and time.time() < deadline - 5.0:
        improved_blocks = local_search(best_blocks)
        if len(improved_blocks) < best_len:
            best_blocks = improved_blocks
            best_len = len(improved_blocks)
            
    if best_blocks is None:
        # Fallback: simple greedy
        best_blocks = greedy_cover(0)
        
    return [list(b) for b in best_blocks]
