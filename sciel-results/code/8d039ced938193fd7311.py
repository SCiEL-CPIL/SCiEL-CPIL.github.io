import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map each block to a bitmask of the triplets it covers
    block_masks = [0] * n_blocks
    block_triplets = [[] for _ in range(n_blocks)] # List of triplet IDs for each block
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        t_ids = []
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
            t_ids.append(tid)
        block_masks[b_idx] = mask
        block_triplets[b_idx] = t_ids
        
    # Precompute triplet_to_blocks: for each triplet, list of blocks covering it
    # This is used for efficient swap candidate finding
    triplet_to_blocks = [[] for _ in range(n_triplets)]
    for b_idx in range(n_blocks):
        for tid in block_triplets[b_idx]:
            triplet_to_blocks[tid].append(b_idx)

    def greedy_set_cover(seed=0):
        rng = random.Random(seed)
        uncovered = (1 << n_triplets) - 1
        solution_indices = []
        solution_set = set()
        
        while uncovered:
            if time.time() > deadline - 30:
                break
                
            best_gain = -1
            best_idx = -1
            candidates = []
            
            for b_idx in range(n_blocks):
                if b_idx in solution_set:
                    continue
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = b_idx
                    candidates = [b_idx]
                elif gain == best_gain and gain > 0:
                    candidates.append(b_idx)
                    
            if best_idx == -1:
                break
                
            chosen_idx = rng.choice(candidates)
            solution_indices.append(chosen_idx)
            solution_set.add(chosen_idx)
            uncovered &= ~block_masks[chosen_idx]
            
        return solution_indices

    def local_search(initial_indices, max_iter=5000):
        # Work with a set of indices for O(1) lookup
        sol_set = set(initial_indices)
        counts = np.zeros(n_triplets, dtype=np.int32)
        
        # Initialize counts
        for idx in sol_set:
            for tid in block_triplets[idx]:
                counts[tid] += 1
                
        best_sol = list(sol_set)
        best_count = len(sol_set)
        
        no_improve_count = 0
        rng = random.Random()
        
        for i in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # 1. Reduction Phase: Remove redundant blocks
            reduced = False
            to_remove = []
            for idx in list(sol_set):
                redundant = True
                for tid in block_triplets[idx]:
                    if counts[tid] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(idx)
            
            for idx in to_remove:
                for tid in block_triplets[idx]:
                    counts[tid] -= 1
                sol_set.remove(idx)
                reduced = True
                
            if reduced:
                no_improve_count = 0
                if len(sol_set) < best_count:
                    best_count = len(sol_set)
                    best_sol = list(sol_set)
                    # Continue reducing
                    continue 
            else:
                no_improve_count += 1
                
            # 2. Swap Phase: Try to swap a block to enable future reduction
            # Pick a random block from solution
            if not sol_set:
                break
                
            # Try a few random swaps
            swap_made = False
            attempts = 0
            max_attempts = 10
            while attempts < max_attempts and not swap_made:
                attempts += 1
                idx_a = rng.choice(list(sol_set))
                # Get exclusive triplets of A
                exc_tids = []
                for tid in block_triplets[idx_a]:
                    if counts[tid] == 1:
                        exc_tids.append(tid)
                        
                if not exc_tids:
                    # Should have been removed in reduction
                    continue
                    
                # Find a block B not in sol_set that covers all exc_tids
                # Strategy: Pick the triplet in exc_tids with fewest covering blocks
                # to minimize search space
                # Actually, just pick the first one and filter
                target_tid = exc_tids[0]
                candidates_b = triplet_to_blocks[target_tid]
                
                # Shuffle to diversify
                rng.shuffle(candidates_b)
                
                found_b = -1
                for b_idx in candidates_b:
                    if b_idx in sol_set:
                        continue
                    # Check if b_idx covers all other exc_tids
                    valid = True
                    for tid in exc_tids:
                        if tid == target_tid:
                            continue
                        # Check if b_idx covers tid
                        # We can check bit in mask
                        if not (block_masks[b_idx] & (1 << tid)):
                            valid = False
                            break
                    if valid:
                        found_b = b_idx
                        break
                        
                if found_b != -1:
                    # Perform swap: Remove A, Add B
                    for tid in block_triplets[idx_a]:
                        counts[tid] -= 1
                    sol_set.remove(idx_a)
                    
                    for tid in block_triplets[found_b]:
                        counts[tid] += 1
                    sol_set.add(found_b)
                    
                    swap_made = True
                    no_improve_count = 0
                    # After swap, check if B is now redundant (or A's removal made others redundant)
                    # The outer loop will handle reduction in next iteration
                    break
            if not swap_made:
                # Stuck. Perturb: Add a random block
                # This increases count, but might break symmetry
                # Only do this if we haven't improved for a while
                if no_improve_count > 20:
                    # Pick a random block not in solution
                    # To avoid O(n) check, we can just pick a random index and check
                    # But n is 346k, so it's fine
                    while True:
                        b_idx = rng.randint(0, n_blocks - 1)
                        if b_idx not in sol_set:
                            break
                    for tid in block_triplets[b_idx]:
                        counts[tid] += 1
                    sol_set.add(b_idx)
                    no_improve_count = 0
                    # We accept the increase in count temporarily
                    # The reduction phase will try to remove it or others
                    
        return best_sol

    best_blocks = None
    best_count = float('inf')
    
    num_restarts = 10
    for seed in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Greedy start
        init_indices = greedy_set_cover(seed)
        
        # Local search
        refined_indices = local_search(init_indices)
        
        if len(refined_indices) < best_count:
            best_count = len(refined_indices)
            best_blocks = [all_blocks[i] for i in refined_indices]
            
    if best_blocks is None:
        # Fallback
        init_indices = greedy_set_cover(0)
        best_blocks = [all_blocks[i] for i in init_indices]
        
    return best_blocks

if __name__ == "__main__":
    pass
