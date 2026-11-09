import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 110.0  # 10s safety margin
    
    points = list(range(v))
    
    # 1. Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    # 2. Generate all t-subsets and map to indices
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    # 3. Precompute block -> t-subsets mapping
    # block_subs[i] is a list of t-subset indices covered by block i
    block_subs = []
    for b in all_blocks:
        subs = []
        for s in itertools.combinations(b, t):
            subs.append(t_idx[s])
        block_subs.append(subs)
        
    # 4. Precompute t-subset -> blocks mapping
    # sub_blocks[j] is a list of block indices that cover t-subset j
    sub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_subs):
        for si in subs:
            sub_blocks[si].append(i)

    def run_greedy(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        # gain[i] = number of currently uncovered t-subsets covered by block i
        gain = np.array([len(subs) for subs in block_subs], dtype=np.int32)
        is_covered = np.zeros(N_t, dtype=bool)
        uncovered_count = N_t
        selected_blocks = []
        
        while uncovered_count > 0 and time.time() < DEADLINE:
            # Find best block
            # To add randomization, we can pick randomly from the top few or add noise
            # Let's pick the max gain. If there are ties, np.argmax picks the first one.
            # To randomize, we can add small noise.
            if random_seed is not None:
                # Add small noise to break ties and explore
                noise = np.random.uniform(0, 0.1, size=N_blocks)
                scores = gain + noise
            else:
                scores = gain
                
            best_idx = np.argmax(scores)
            
            if gain[best_idx] == 0:
                # No more uncovered subsets can be covered? Should not happen if valid cover exists
                break
                
            selected_blocks.append(best_idx)
            gain[best_idx] = -1  # Disable this block
            
            # Update gains for all blocks that share subsets with best_idx
            for si in block_subs[best_idx]:
                if not is_covered[si]:
                    is_covered[si] = True
                    uncovered_count -= 1
                    # Update all blocks that cover this subset
                    for b_idx in sub_blocks[si]:
                        if gain[b_idx] > 0:
                            gain[b_idx] -= 1
                            
        return selected_blocks

    def try_reduce(block_indices):
        if not block_indices:
            return block_indices
            
        # Count coverage for each t-subset
        cov_count = np.zeros(N_t, dtype=np.int32)
        for b_idx in block_indices:
            for si in block_subs[b_idx]:
                cov_count[si] += 1
                
        # Try to remove blocks
        # We iterate through the list and try to remove each one
        # If removing it makes any cov_count[si] == 0, we cannot remove it
        # To be efficient, we can check if min(cov_count[si] for si in block_subs[b_idx]) > 1
        # But we need to update cov_count if we do remove it.
        
        # A simpler approach: 
        # Iterate through blocks. 
        # If a block is redundant (all its subsets are covered by >1 blocks), remove it.
        # We need to be careful: removing one block might make another redundant.
        # So we loop until no changes.
        
        changed = True
        while changed and time.time() < DEADLINE:
            changed = False
            # Iterate in reverse to keep indices stable? Or just use a list and filter.
            # Let's create a new list of blocks to keep.
            new_blocks = []
            for b_idx in block_indices:
                # Check if b_idx is redundant
                redundant = True
                for si in block_subs[b_idx]:
                    if cov_count[si] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove it
                    changed = True
                    for si in block_subs[b_idx]:
                        cov_count[si] -= 1
                else:
                    new_blocks.append(b_idx)
            if changed:
                block_indices = new_blocks
                
        return block_indices

    best_sol = None
    best_len = float('inf')
    
    # Run deterministic greedy first
    sol = run_greedy(random_seed=None)
    sol = try_reduce(sol)
    best_sol = sol
    best_len = len(sol)
    
    # Run randomized restarts
    attempt = 0
    while time.time() < DEADLINE - 5.0:
        sol = run_greedy(random_seed=attempt)
        if len(sol) < best_len:
            sol = try_reduce(sol)
            if len(sol) < best_len:
                best_len = len(sol)
                best_sol = sol
        attempt += 1
        
    # Convert indices back to blocks
    return [all_blocks[i] for i in best_sol]
