import itertools
import random
import heapq

def construct(v, k, t):
    points = range(v)
    all_triples = list(itertools.combinations(points, t))
    triple_to_index = {trip: i for i, trip in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Pre-compute all blocks and their triple lists
    # And reverse mapping: triple -> list of blocks
    all_blocks = []
    block_to_triples = []
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    for block in itertools.combinations(points, k):
        block_idx = len(all_blocks)
        all_blocks.append(block)
        trips = []
        for trip in itertools.combinations(block, t):
            idx = triple_to_index[trip]
            trips.append(idx)
            triple_to_blocks[idx].append(block_idx)
        block_to_triples.append(trips)
        
    num_blocks = len(all_blocks)
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple restarts with different random seeds for triple selection
    num_restarts = 20
    
    for restart in range(num_restarts):
        random.seed(restart) # Deterministic for reproducibility, but varies per restart
        
        uncovered = set(range(num_triples))
        blocks = []
        
        while uncovered:
            # Pick a triple to cover. 
            # Strategy: Pick the triple that is covered by the fewest blocks? 
            # This is a standard heuristic for set cover (Most Constrained Variable).
            # However, checking the degree of every uncovered triple is O(N_uncovered * deg).
            # deg is approx 6000. N_uncovered starts at 2024.
            # 2024 * 6000 = 12M ops just to pick the triple.
            # Then finding best block is another 6000 * 35 ops.
            # This might be slow.
            
            # Alternative: Just pick a random uncovered triple.
            # This is faster and often works well with many restarts.
            target_idx = random.choice(list(uncovered))
            
            # Find best block covering target_idx
            best_block_idx = -1
            max_covered = -1
            
            for b_idx in triple_to_blocks[target_idx]:
                # Count how many uncovered triples this block covers
                covered = 0
                for t_idx in block_to_triples[b_idx]:
                    if t_idx in uncovered:
                        covered += 1
                
                if covered > max_covered:
                    max_covered = covered
                    best_block_idx = b_idx
                    if covered == len(block_to_triples[b_idx]):
                        break # Can't do better than covering all its triples
            
            if best_block_idx != -1:
                blocks.append(best_block_idx)
                # Update uncovered
                for t_idx in block_to_triples[best_block_idx]:
                    uncovered.discard(t_idx)
        
        # Refinement: Remove redundant blocks
        # Count coverage for each triple
        triple_count = [0] * num_triples
        for b_idx in blocks:
            for t_idx in block_to_triples[b_idx]:
                triple_count[t_idx] += 1
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b_idx in blocks:
                # Check if block is essential
                essential = False
                for t_idx in block_to_triples[b_idx]:
                    if triple_count[t_idx] == 1:
                        essential = True
                        break
                if not essential:
                    # Remove block
                    for t_idx in block_to_triples[b_idx]:
                        triple_count[t_idx] -= 1
                    changed = True
                else:
                    new_blocks.append(b_idx)
            blocks = new_blocks
            
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(all_blocks[b]) for b in blocks]
            
    return best_blocks
