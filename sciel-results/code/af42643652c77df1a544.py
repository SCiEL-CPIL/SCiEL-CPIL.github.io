import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Precomputation
    # Map t-subsets to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    num_t = len(t_subsets)
    
    # Map k-subsets to indices
    k_subsets = list(itertools.combinations(range(v), k))
    # We don't need k_to_idx if we just store the combinations, 
    # but we need to iterate over them efficiently.
    # Let's store t_indices for each k-subset.
    
    # Precompute t_indices for each k-subset
    # k_subsets[i] is a tuple of points.
    # We need the list of t-subset indices contained in k_subsets[i].
    t_indices_in_block = []
    for k_block in k_subsets:
        t_indices = []
        for t_block in itertools.combinations(k_block, t):
            t_indices.append(t_to_idx[t_block])
        t_indices_in_block.append(t_indices)
        
    # Precompute blocks containing each t-subset
    # blocks_containing[t_idx] = list of k_block indices
    blocks_containing = [[] for _ in range(num_t)]
    for b_idx, k_block in enumerate(k_subsets):
        for t_block in itertools.combinations(k_block, t):
            t_idx = t_to_idx[t_block]
            blocks_containing[t_idx].append(b_idx)

    best_solution = None
    best_len = float('inf')
    
    # 2. Main Loop: Randomized Greedy with Local Search
    while time.time() < deadline:
        uncovered = set(range(num_t))
        solution = []
        
        while uncovered:
            # Pick a random uncovered t-subset
            # Using random.choice on a set is O(n) in Python if not converted to list, 
            # but set iteration is fast. However, random.choice needs a sequence.
            # For speed, we can just pick an arbitrary one or use a list.
            # Converting set to list every step is O(n). 
            # Let's maintain a list of uncovered? No, removal is hard.
            # Let's just use next(iter(uncovered)) for speed, or random.sample.
            # random.sample(uncovered, 1)[0] is okay for small n, but n starts at 2024.
            # Let's just pick the first one for now to save time, or randomize the order of t_subsets initially.
            # Better: Shuffle the list of all t_indices once per restart?
            # No, uncovered changes.
            # Let's use a simple random pick.
            t_idx = random.choice(list(uncovered)) # This is O(N)
            
            # Find best block containing t_idx
            best_gain = -1
            best_b_idx = -1
            candidates = blocks_containing[t_idx]
            
            # Optional: Randomize candidate order to break ties differently
            # random.shuffle(candidates) # Too slow if done every step
            
            for b_idx in candidates:
                # Calculate gain: number of uncovered t-subsets in this block
                gain = 0
                for t_i in t_indices_in_block[b_idx]:
                    if t_i in uncovered:
                        gain += 1
                        if gain > best_gain:
                            # Early exit? No, we need max.
                            pass
                
                if gain > best_gain:
                    best_gain = gain
                    best_b_idx = b_idx
            
            # Add best block
            solution.append(best_b_idx)
            # Update uncovered
            for t_i in t_indices_in_block[best_b_idx]:
                uncovered.discard(t_i)
        
        # Local Search: Remove redundant blocks
        # Convert solution to set for fast lookup?
        # Check if any block is redundant.
        # A block is redundant if all its t-subsets are covered by other blocks.
        
        # To check redundancy efficiently:
        # 1. Count how many blocks cover each t-subset.
        # 2. If a t-subset is covered by exactly 1 block, that block is essential.
        # 3. If all t-subsets of a block are covered by >1 blocks, it might be redundant.
        
        # Let's do a simple pass:
        # For each block in solution, check if it can be removed.
        # This is O(|solution| * |block_t_count| * |solution|) if done naively.
        # |solution| ~ 80. |block_t_count| = 35.
        # 80 * 35 * 80 = 224,000 ops. Very fast.
        
        changed = True
        while changed:
            changed = False
            new_solution = []
            for b_idx in solution:
                # Check if b_idx is redundant
                redundant = True
                for t_i in t_indices_in_block[b_idx]:
                    # Is t_i covered by any OTHER block in solution?
                    covered_by_other = False
                    for other_b in solution:
                        if other_b == b_idx:
                            continue
                        if t_i in t_indices_in_block[other_b]:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        redundant = False
                        break
                
                if not redundant:
                    new_solution.append(b_idx)
                    changed = True
                # If redundant, we drop it. 
                # Note: Dropping one might make others redundant, hence the while loop.
            
            solution = new_solution
            
        if len(solution) < best_len:
            best_len = len(solution)
            best_solution = solution[:]
            
        # If we hit a very good solution, we might want to stop early?
        # No, keep trying to beat it.
        
    # Return the list of blocks (as lists of ints)
    if best_solution is None:
        # Fallback: return empty or a trivial cover if something went wrong
        return []
        
    return [list(k_subsets[b_idx]) for b_idx in best_solution]

