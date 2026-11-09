import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    points = range(v)
    all_triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    def generate_random_block():
        return tuple(sorted(random.sample(points, k)))
        
    def count_uncovered(block, uncovered_set):
        count = 0
        for triple in itertools.combinations(block, t):
            if triple_to_idx[triple] in uncovered_set:
                count += 1
        return count

    best_blocks = None
    best_count = float('inf')
    
    # We will run a few iterations (restarts)
    # Each iteration:
    # 1. Generate a pool of candidate blocks.
    # 2. Greedy select from pool.
    # 3. Local search removal.
    
    max_restarts = 10 # Adjust based on speed
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # 1. Generate Pool
        # Size of pool: enough to have good choices. 1000-2000 blocks.
        pool_size = 2000
        pool = set()
        while len(pool) < pool_size:
            pool.add(generate_random_block())
            
        # 2. Greedy Selection
        uncovered = set(range(n_triples))
        selected_blocks = []
        
        # To speed up greedy, we can precompute the "gain" of each block in the pool?
        # But gain changes as uncovered set shrinks.
        # So we must recalculate gain for remaining pool blocks each step.
        # Pool size 2000, steps ~100. 2000 * 100 * 35 (triples per block) = 7M ops. Fast.
        
        remaining_pool = list(pool)
        
        while uncovered:
            if time.time() > deadline:
                break
                
            best_block = None
            best_gain = -1
            
            for block in remaining_pool:
                # Optimization: Skip if block covers 0 new triples? 
                # We can check if any triple in block is in uncovered.
                # If not, skip.
                # count_uncovered does this.
                gain = count_uncovered(block, uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    if best_gain == 35: # Max possible
                        break
            
            if best_block is None:
                # If no block in pool covers anything (should not happen if pool is large enough and uncovered not empty)
                # Generate new blocks
                new_pool = set()
                while len(new_pool) < 500:
                    new_pool.add(generate_random_block())
                remaining_pool.extend(new_pool)
                continue
                
            selected_blocks.append(best_block)
            remaining_pool.remove(best_block)
            
            # Update uncovered
            for triple in itertools.combinations(best_block, t):
                idx = triple_to_idx[triple]
                if idx in uncovered:
                    uncovered.remove(idx)
                    
        # 3. Local Search: Removal
        # Count coverage
        coverage_count = [0] * n_triples
        for block in selected_blocks:
            for triple in itertools.combinations(block, t):
                coverage_count[triple_to_idx[triple]] += 1
                
        changed = True
        while changed:
            changed = False
            for i in range(len(selected_blocks) - 1, -1, -1):
                if time.time() > deadline:
                    break
                block = selected_blocks[i]
                can_remove = True
                for triple in itertools.combinations(block, t):
                    if coverage_count[triple_to_idx[triple]] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for triple in itertools.combinations(block, t):
                        coverage_count[triple_to_idx[triple]] -= 1
                    selected_blocks.pop(i)
                    changed = True
                    
        # Update best
        if len(selected_blocks) < best_count:
            best_count = len(selected_blocks)
            best_blocks = selected_blocks.copy()
            
    return [list(b) for b in best_blocks]
