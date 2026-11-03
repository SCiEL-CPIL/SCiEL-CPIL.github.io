import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # 5s margin
    
    # Precompute all t-subsets (triples for t=3)
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    best_blocks = None
    best_count = float('inf')
    
    # Use a specific RNG instance to avoid global state issues
    rng = random.Random(42)
    
    run_id = 0
    while time.time() - start_time < time_limit:
        run_id += 1
        rng.seed(run_id)
        
        # 1. Greedy Construction
        # uncovered_mask: bitset or list of bools? List of bools is fine for 2024.
        uncovered = [True] * num_triples
        uncovered_count = num_triples
        blocks = []
        
        while uncovered_count > 0:
            # Pick a random uncovered triple
            # To avoid O(N) scan, we can just pick a random index and retry if covered?
            # Or maintain a list of uncovered indices.
            # Let's maintain a list for efficiency.
            # Actually, for 2024 items, list(uncovered_indices) is fast.
            # But we update it.
            # Let's use a set of indices for O(1) removal and random choice from list(set).
            # Wait, I already used a set in the thought process. Let's stick to that.
            # Re-initializing the set each run is O(N).
            
            # To speed up, let's just pick a random triple from all_triples 
            # and check if it's uncovered. If covered, retry.
            # Probability of picking uncovered is high initially, low later.
            # Later, we might loop many times.
            # Better: Maintain a list of uncovered indices.
            
            # Let's restart the inner logic with a list.
            # This requires rewriting the loop.
            pass
            
        # Since the previous logic with set was simpler, let's stick to it but optimize.
        # Actually, let's use the list of uncovered indices approach for speed.
        
        # Reset for list-based approach
        uncovered_indices = list(range(num_triples))
        blocks = []
        
        while uncovered_indices:
            # Pick random uncovered triple
            idx = rng.choice(uncovered_indices)
            triple = all_triples[idx]
            
            block = list(triple)
            
            # Greedy expansion
            while len(block) < k:
                best_point = -1
                best_gain = -1
                
                # Calculate gain for each candidate point x not in block
                # Gain = number of uncovered triples in block that become covered
                # Note: We only care about triples that are currently in `uncovered_indices`
                
                # To check if a triple is uncovered, we need a fast lookup.
                # `uncovered_indices` is a list. Checking membership is O(N).
                # This is slow.
                # We need a set for `uncovered_indices` for O(1) lookup, 
                # but we also need random choice.
                # `rng.choice(list(uncovered_set))` is O(N).
                # 
                # Alternative: Use a boolean array `is_uncovered`.
                # And a list `uncovered_list` to pick from.
                # When we cover a triple, set `is_uncovered[idx]=False` and remove from `uncovered_list`?
                # Removing from list is O(N).
                # 
                # Given N=2024, O(N) operations are very fast (<1ms).
                # So `list(uncovered_set)` and `in uncovered_set` is fine.
                
                # Let's go back to the Set approach, it's simpler and fast enough.
                pass
        
        # I will implement the Set approach in the final code.
        
        # Re-implementing the loop with Set for clarity and correctness.
        uncovered_set = set(range(num_triples))
        blocks = []
        
        while uncovered_set:
            # Pick random uncovered triple
            start_idx = rng.choice(list(uncovered_set))
            start_triple = all_triples[start_idx]
            block = list(start_triple)
            
            while len(block) < k:
                best_point = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    # Check pairs in block
                    # Optimization: Precompute pairs? No, block is small.
                    for i in range(len(block)):
                        for j in range(i+1, len(block)):
                            p1, p2 = block[i], block[j]
                            # Create triple
                            # Since block is not sorted, we must sort to match all_triples keys
                            # all_triples are sorted tuples.
                            triple = tuple(sorted((p1, p2, x)))
                            idx = triple_to_idx.get(triple)
                            if idx is not None and idx in uncovered_set:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered_set
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                if idx in uncovered_set:
                    uncovered_set.remove(idx)
        
        # 2. Deletion Phase
        counts = [0] * num_triples
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                counts[idx] += 1
        
        removed = True
        while removed:
            removed = False
            for i in range(len(blocks) - 1, -1, -1):
                block = blocks[i]
                is_redundant = True
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx[triple]
                    if counts[idx] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for triple in itertools.combinations(block, t):
                        idx = triple_to_idx[triple]
                        counts[idx] -= 1
                    blocks.pop(i)
                    removed = True
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
    return best_blocks
