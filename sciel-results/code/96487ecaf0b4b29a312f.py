import itertools
import random
import time

def construct(v, k, t):
    all_t_subsets = list(itertools.combinations(range(v), t))
    
    # Precompute t-subsets for all possible k-subsets? 
    # No, memory. 
    # But we can precompute the list of all k-subsets to avoid generating them repeatedly in restarts?
    # No, we generate specific ones based on seed.
    
    def get_t_subsets(block):
        return list(itertools.combinations(block, t))

    best_blocks = None
    best_size = float('inf')
    
    # Estimate time per run
    # v=19, k=7, t=4
    # ~150 steps
    # ~455 candidates/step
    # ~35 t-subsets/candidate
    # ~2.4M checks/run
    # Python speed: ~5M-10M simple ops/sec.
    # So ~0.3-0.5s per run.
    # 120s limit allows ~200-400 runs.
    # Let's try 100 runs.
    
    num_restarts = 100
    
    for _ in range(num_restarts):
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset
            # Using random.choice on a list view is faster than tuple conversion?
            # random.choice requires a sequence.
            # list(uncovered) is O(N).
            # We can maintain a list of uncovered?
            # But we need O(1) removal. Set is good for removal.
            # For random selection, we can use random.sample(list(uncovered), 1)[0]
            # Or just random.choice(tuple(uncovered))
            
            seed_subset = random.choice(tuple(uncovered))
            
            seed_set = set(seed_subset)
            remaining_points = [p for p in range(v) if p not in seed_set]
            need_points = k - t
            
            best_block = None
            max_covered = 0
            max_possible = 35 # C(7,4)
            
            for combo in itertools.combinations(remaining_points, need_points):
                # Merge and sort
                block = tuple(sorted(seed_subset + combo))
                
                count = 0
                # Unroll loop for t=4? No, keep generic.
                for ts in itertools.combinations(block, t):
                    if ts in uncovered:
                        count += 1
                        
                if count > max_covered:
                    max_covered = count
                    best_block = block
                    if max_covered == max_possible:
                        break
                        
            if best_block is None:
                break
            
            blocks.append(list(best_block))
            
            # Update uncovered
            for ts in itertools.combinations(best_block, t):
                uncovered.discard(ts)
                
        # Local Search: Remove redundant blocks
        coverage_count = {}
        for b in blocks:
            for ts in itertools.combinations(tuple(b), t):
                coverage_count[ts] = coverage_count.get(ts, 0) + 1
        
        i = 0
        while i < len(blocks):
            b = tuple(blocks[i])
            redundant = True
            for ts in itertools.combinations(b, t):
                if coverage_count[ts] <= 1:
                    redundant = False
                    break
            if redundant:
                for ts in itertools.combinations(b, t):
                    coverage_count[ts] -= 1
                blocks.pop(i)
            else:
                i += 1
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks[:]
            
    return best_blocks
