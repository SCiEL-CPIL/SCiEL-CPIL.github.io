import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    all_t_subsets_set = set(all_t_subsets)
    
    # Precompute coverage
    potential_blocks = list(itertools.combinations(points, k))
    
    # Map each block to the set of t-subsets it covers
    block_coverage = {}
    # Map each t-subset to the list of blocks that cover it
    t_subset_to_blocks = {s: [] for s in all_t_subsets}
    
    for b in potential_blocks:
        # Generate all t-subsets of b
        cov = set(itertools.combinations(b, t))
        block_coverage[b] = cov
        for s in cov:
            t_subset_to_blocks[s].append(b)
            
    best_blocks = None
    best_count = float('inf')
    
    def run_greedy(seed):
        random.seed(seed)
        uncovered = set(all_t_subsets)
        
        if not uncovered:
            return []
        
        # Initialize scores: number of uncovered t-subsets covered by block
        # Initially all are uncovered, so score = len(coverage)
        scores = {b: len(cov) for b, cov in block_coverage.items()}
        
        blocks_chosen = []
        
        while uncovered:
            # Find block with max score
            max_score = -1
            candidates = []
            for b, s in scores.items():
                if s > max_score:
                    max_score = s
                    candidates = [b]
                elif s == max_score:
                    candidates.append(b)
            
            if not candidates or max_score == 0:
                # If max_score is 0 but uncovered is not empty, something is wrong
                # Or we picked all blocks?
                break
                
            b_star = random.choice(candidates)
            blocks_chosen.append(b_star)
            
            # Update scores and uncovered
            cov_star = block_coverage[b_star]
            for s in cov_star:
                if s in uncovered:
                    # s is now covered
                    # Decrement score of all blocks covering s (except b_star? 
                    # b_star's score doesn't matter anymore as it's picked, 
                    # but logically it also loses this point)
                    for b_prime in t_subset_to_blocks[s]:
                        scores[b_prime] -= 1
                    uncovered.discard(s)
                    
        # Post-processing: Remove redundant blocks
        # Count occurrences of each t-subset in the chosen blocks
        t_count = {s: 0 for s in all_t_subsets}
        for b in blocks_chosen:
            for s in block_coverage[b]:
                t_count[s] += 1
                
        final_blocks = []
        for b in blocks_chosen:
            is_redundant = True
            for s in block_coverage[b]:
                if t_count[s] == 1:
                    is_redundant = False
                    break
            if not is_redundant:
                final_blocks.append(b)
                
        return final_blocks

    # Run multiple restarts
    seed = 42
    while time.time() < deadline:
        res = run_greedy(seed)
        if len(res) < best_count:
            best_count = len(res)
            best_blocks = res
        seed += 1
        
    if best_blocks is None:
        # Fallback if no solution found (should not happen)
        return []
        
    return [list(b) for b in best_blocks]
