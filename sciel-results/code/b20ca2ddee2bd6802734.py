import itertools
import random
import time
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using a randomized greedy set cover
    with multiple restarts and local improvement.
    """
    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # For each block, compute which t-subsets it covers
    block_tsubs = []
    for block in all_blocks:
        tsubs_in_block = [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs.append(tsubs_in_block)
    
    # For each t-subset, which blocks cover it
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for b_idx, tsubs in enumerate(block_tsubs):
        for s_idx in tsubs:
            tsub_to_blocks[s_idx].append(b_idx)
    
    def greedy_cover(randomize=False, seed=None):
        """Run greedy set cover, optionally with random tie-breaking."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        # cover_count[s] = 1 if covered, 0 otherwise
        cover_count = [0] * n_tsubs
        # block_gain[b] = number of uncovered t-subsets covered by block b
        block_gain = [len(tsubs) for tsubs in block_tsubs]
        
        selected = []
        remaining = set(range(n_blocks))
        
        while True:
            # Check if all covered
            if all(cover_count[s] > 0 for s in range(n_tsubs)):
                break
            
            # Pick block with max gain (with random tie-breaking if randomize)
            best_block = -1
            best_gain = -1
            if randomize:
                # Collect all blocks with max gain
                max_gain = -1
                for b in remaining:
                    if block_gain[b] > max_gain:
                        max_gain = block_gain[b]
                candidates = [b for b in remaining if block_gain[b] == max_gain]
                if candidates:
                    best_block = rng.choice(candidates)
                    best_gain = max_gain
            else:
                for b in remaining:
                    if block_gain[b] > best_gain:
                        best_gain = block_gain[b]
                        best_block = b
            
            if best_block == -1 or best_gain <= 0:
                break
            
            selected.append(best_block)
            remaining.discard(best_block)
            
            # Update gains
            for s_idx in block_tsubs[best_block]:
                if cover_count[s_idx] == 0:
                    cover_count[s_idx] = 1
                    for b in tsub_to_blocks[s_idx]:
                        if b in remaining:
                            block_gain[b] -= 1
        
        return selected
    
    # Run greedy cover multiple times with different seeds and take the best
    best_solution = None
    best_len = float('inf')
    
    # Deterministic first
    sol = greedy_cover(randomize=False)
    best_solution = sol
    best_len = len(sol)
    
    # Randomized restarts
    num_restarts = 10
    for i in range(num_restarts):
        sol = greedy_cover(randomize=True, seed=i)
        if len(sol) < best_len:
            best_solution = sol
            best_len = len(sol)
    
    # Local improvement: try to replace blocks to reduce count
    # For each block in the solution, try removing it and see if we can re-cover
    def local_improve(solution):
        """Try to improve the solution by local search."""
        solution = list(solution)
        improved = True
        while improved:
            improved = False
            for i in range(len(solution)):
                # Try removing block i
                new_solution = solution[:i] + solution[i+1:]
                if len(new_solution) >= len(solution):
                    continue
                
                # Check if new_solution still covers all t-subsets
                # Build coverage for new_solution
                covered = [0] * n_tsubs
                for b in new_solution:
                    for s_idx in block_tsubs[b]:
                        covered[s_idx] = 1
                
                if all(covered[s] > 0 for s in range(n_tsubs)):
                    solution = new_solution
                    improved = True
                    break
        
        return solution
    
    # Apply local improvement to the best solution
    best_solution = local_improve(best_solution)
    
    return [list(all_blocks[b]) for b in best_solution]
