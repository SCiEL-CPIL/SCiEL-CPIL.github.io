import time
import itertools
import random

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: 10 seconds before the 120s hard limit (or 150s if that's the real one, prompt says 150s kill, 120s evaluator? "The evaluator KILLS the program after 150s wall-clock... Each evaluation must finish within 120 s." Let's assume 120s is the hard constraint for the score, 150s is kill. We will use 110s to be safe.)
    deadline = start_time + 110.0
    
    # Precompute all t-subsets
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    
    best_blocks = None
    best_len = float('inf')
    
    # Helper: Check if a block covers any unique triples given the current set of blocks
    # This is handled in local search
    
    def random_greedy_cover():
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triple to start
            # Using next(iter(uncovered)) is deterministic, so we shuffle or pick random
            # For efficiency, we can just pick one, but randomizing helps diversity
            # However, picking the "most constrained" might be better. 
            # Let's stick to random for now, or maybe the first one for speed? 
            # Random is better for escaping local optima.
            
            # To speed up random choice, we can't easily index a set.
            # But 2024 items is small.
            seed_triple = random.choice(list(uncovered))
            
            block = list(seed_triple)
            
            # Extend block to size k
            while len(block) < k:
                # Calculate gain for each candidate point
                best_gain = -1
                best_points = []
                
                # Pre-compute pairs in current block for faster gain calculation?
                # block size is small (max 6).
                block_set = set(block)
                
                for x in all_points:
                    if x in block_set:
                        continue
                    
                    gain = 0
                    # For each pair (u, v) in block, check if (x, u, v) is in uncovered
                    # Note: (x, u, v) is a t-subset. Since t=3, we need pairs from block.
                    # If t > 3, we'd need combinations of block of size t-1.
                    # Here t=3, so we iterate over combinations of block of size 2.
                    
                    # Optimization: 
                    # The triples covered by adding x are { (x, u, v) | (u, v) in C(block, 2) }
                    # We need to count how many of these are in `uncovered`.
                    
                    for pair in itertools.combinations(block, t - 1):
                        # Create the tuple
                        # Ensure sorted order for set lookup
                        triple = tuple(sorted((x,) + pair))
                        if triple in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain:
                        best_points.append(x)
                
                # Pick randomly from best_points
                if best_points:
                    block.append(random.choice(best_points))
                else:
                    # Fallback: pick any point not in block
                    # This shouldn't happen if uncovered is not empty and v > k
                    available = [x for x in all_points if x not in block_set]
                    if available:
                        block.append(random.choice(available))
                    else:
                        break # Should not happen
            
            # Sort block for consistency
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for triple in itertools.combinations(block, t):
                uncovered.discard(tuple(triple))
        
        return blocks

    def local_search_remove_redundant(blocks):
        if not blocks:
            return blocks
        
        # Map each triple to the list of block indices that cover it
        triple_to_blocks = {triple: [] for triple in all_t_subsets}
        
        for i, block in enumerate(blocks):
            for triple in itertools.combinations(block, t):
                triple_to_blocks[tuple(triple)].append(i)
        
        changed = True
        # We need to be careful with indices when removing blocks.
        # It's safer to iterate and mark for removal, then filter.
        
        while changed:
            changed = False
            to_remove = []
            
            for i, block in enumerate(blocks):
                # Check if any triple in this block is covered ONLY by this block
                is_redundant = True
                for triple in itertools.combinations(block, t):
                    if len(triple_to_blocks[tuple(triple)]) == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    to_remove.append(i)
            
            if to_remove:
                # Remove blocks
                remove_set = set(to_remove)
                new_blocks = []
                for i, block in enumerate(blocks):
                    if i not in remove_set:
                        new_blocks.append(block)
                
                # Update triple_to_blocks
                # We need to remove the indices from the lists
                # Since indices change, it's easier to rebuild or update carefully.
                # Rebuilding is O(N * K * T) which is small.
                triple_to_blocks = {triple: [] for triple in all_t_subsets}
                for i, block in enumerate(new_blocks):
                    for triple in itertools.combinations(block, t):
                        triple_to_blocks[tuple(triple)].append(i)
                
                blocks = new_blocks
                changed = True
                
        return blocks

    # Main loop
    while time.time() < deadline:
        blocks = random_greedy_cover()
        blocks = local_search_remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
            # Optional: If we hit a very good number, we might want to run more local search?
            # But time is tight.
            
    return best_blocks if best_blocks is not None else []
