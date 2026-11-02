import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute for each t-subset, the list of points NOT in it?
    # Not needed, we generate blocks on the fly.
    
    def get_blocks_containing(ts):
        ts_set = set(ts)
        remaining = [x for x in range(v) if x not in ts_set]
        for comb in itertools.combinations(remaining, k - t):
            yield tuple(sorted(ts_set.union(comb)))

    best_blocks = None
    best_score = float('inf')
    
    # Number of restarts
    # We estimate greedy takes ~0.1s, local search ~0.01s per restart.
    # So we can do ~1000 restarts in 10s.
    # Let's try 200 restarts to be safe.
    
    for seed in range(500):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        covered = np.zeros(n_t, dtype=np.int8)
        blocks = []
        
        # Greedy Phase
        while True:
            uncovered_idx = np.where(covered == 0)[0]
            if len(uncovered_idx) == 0:
                break
                
            # Pick a random uncovered triple
            target_idx = uncovered_idx[random.randint(0, len(uncovered_idx) - 1)]
            target_ts = all_t_subsets[target_idx]
            
            best_gain = -1
            best_block = None
            
            # Search for best block
            # Optimization: If we find a block that covers ALL uncovered triples 
            # (unlikely), we can break.
            for block in get_blocks_containing(target_ts):
                gain = 0
                # Iterate over t-subsets of the block
                for sub in itertools.combinations(block, t):
                    idx = t_subset_to_idx[sub]
                    if covered[idx] == 0:
                        gain += 1
                        # Early exit if gain > best_gain and we just need max?
                        # No, we need the max.
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is not None:
                blocks.append(best_block)
                for sub in itertools.combinations(best_block, t):
                    idx = t_subset_to_idx[sub]
                    covered[idx] += 1
        
        # Local Search Phase
        # Try to remove blocks
        improved = True
        # Limit local search iterations to avoid spending too much time
        max_ls_iters = 100
        ls_iter = 0
        while improved and ls_iter < max_ls_iters:
            improved = False
            ls_iter += 1
            # Shuffle to randomize removal order
            random.shuffle(blocks)
            for i in range(len(blocks)):
                block = blocks[i]
                can_remove = True
                for sub in itertools.combinations(block, t):
                    idx = t_subset_to_idx[sub]
                    if covered[idx] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    blocks.pop(i)
                    for sub in itertools.combinations(block, t):
                        idx = t_subset_to_idx[sub]
                        covered[idx] -= 1
                    improved = True
                    break
        
        current_score = len(blocks)
        if current_score < best_score:
            best_score = current_score
            best_blocks = blocks.copy()
            
            # Early termination if we hit the lower bound or a very good score
            if best_score <= 70:
                break

    if best_blocks is None:
        return []
    return [list(b) for b in best_blocks]
