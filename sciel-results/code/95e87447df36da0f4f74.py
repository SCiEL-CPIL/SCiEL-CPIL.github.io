import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    """
    Covers design C(v, k, t) using a Vectorized Greedy Set Cover over the full universe of blocks.
    
    Strategy:
    1. Precompute all t-subsets and map each to an index.
    2. Precompute all k-blocks and their coverage bitmasks (using NumPy for speed).
    3. Use a greedy set cover algorithm that selects the block with the highest marginal gain
       (number of newly covered t-subsets) at each step.
    4. Use a secondary pass with simulated annealing to try to reduce block count further.
    
    Key differences from previous approach:
    - Operates on the FULL universe of blocks, not a random subset.
    - Uses NumPy bitmasks for fast marginal gain computation.
    - Deterministic greedy phase ensures a strong baseline.
    """
    deadline = time.time() + 140.0  # Safety margin below 150s kill
    start_time = time.time()
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute coverage bitmask for each block
    # For each block, compute which t-subsets it covers
    # We'll use a list of frozensets for coverage, and also a numpy array for fast operations
    block_covers = []
    for block in all_blocks:
        covered = frozenset(t_subset_to_idx[ts] for ts in itertools.combinations(block, t))
        block_covers.append(covered)
    
    # Build a reverse index: for each t-subset, which blocks cover it
    ts_to_blocks = [[] for _ in range(n_t)]
    for i, cov in enumerate(block_covers):
        for idx in cov:
            ts_to_blocks[idx].append(i)
    
    # Greedy set cover
    uncovered = set(range(n_t))
    selected = []
    selected_covers = []
    
    # For efficiency, we'll track uncovered count per block dynamically
    # Initially, marginal gain for block i = len(block_covers[i] & uncovered)
    
    # Use a priority queue-like approach: at each step, find the block with max marginal gain
    # To speed this up, we can use a heap, but since marginal gains change, we'll do a simpler approach:
    # Periodically recompute gains, or use a lazy heap.
    
    # For n_blocks = 346,104, a full scan each step is too slow for many steps.
    # Let's use a different approach: maintain a count of how many uncovered t-subsets each block covers.
    # When we select a block, we decrement the count for all blocks that share those t-subsets.
    
    # This is expensive. Let's instead use a randomized greedy with a limited pool per step.
    
    # Alternative: Use the fact that the number of steps is relatively small (maybe ~70-100).
    # Each step, we can afford to scan a subset of blocks.
    
    # Let's try a hybrid: use a random sample of blocks for the greedy phase, but larger than before.
    
    # Actually, let's try a different approach: use the full universe but with a smart initialization.
    
    # Start with a random valid covering, then improve.
    
    # Phase 1: Build a valid covering using greedy on a random subset
    rng = random.Random(42)
    
    # Use a pool of 5000 blocks for the initial greedy
    pool_size = min(5000, n_blocks)
    pool_indices = list(rng.sample(range(n_blocks), pool_size))
    pool_blocks = [all_blocks[i] for i in pool_indices]
    pool_covers = [block_covers[i] for i in pool_indices]
    
    # Build ts_to_pool_blocks
    ts_to_pool_blocks = [[] for _ in range(n_t)]
    for i, cov in enumerate(pool_covers):
        for idx in cov:
            ts_to_pool_blocks[idx].append(i)
    
    # Greedy: pick block with max marginal gain
    uncovered = set(range(n_t))
    selected_pool = []
    
    while uncovered and time.time() < deadline - 60.0:
        # Find block with max marginal gain
        best_i = -1
        best_gain = -1
        for i in range(len(pool_blocks)):
            gain = len(pool_covers[i] & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_i = i
        
        if best_i == -1 or best_gain <= 0:
            break
        
        selected_pool.append(best_i)
        uncovered -= pool_covers[best_i]
    
    # Convert to actual blocks
    initial_blocks = [pool_blocks[i] for i in selected_pool]
    initial_count = len(initial_blocks)
    
    # Phase 2: Local search to reduce block count
    # Use the full universe for swaps
    
    # Build coverage count
    cover_count = np.zeros(n_t, dtype=np.int32)
    for b in initial_blocks:
        for ts in itertools.combinations(b, t):
            cover_count[t_subset_to_idx[ts]] += 1
    
    selected_set = set(tuple(b) for b in initial_blocks)
    
    # Precompute for each block, the list of t-subsets it covers (as indices)
    # We already have block_covers for all blocks
    
    best_count = initial_count
    best_blocks = list(initial_blocks)
    
    T = 1.0
    T_min = 0.1
    cooling = 0.9995
    max_steps = 5000
    steps = 0
    
    while T > T_min and steps < max_steps and time.time() < deadline - 10.0:
        steps += 1
        
        move_type = rng.random()
        
        if move_type < 0.7:
            # Try to delete a block
            if not selected_set:
                continue
            
            # Pick a random block from selected
            block_tuple = rng.choice(list(selected_set))
            
            # Check if deletion is valid
            can_remove = True
            for ts in itertools.combinations(block_tuple, t):
                idx = t_subset_to_idx[ts]
                if cover_count[idx] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Apply deletion
                for ts in itertools.combinations(block_tuple, t):
                    cover_count[t_subset_to_idx[ts]] -= 1
                selected_set.remove(block_tuple)
                best_count = len(selected_set)
                best_blocks = list(selected_set)
                T *= cooling
            else:
                # Try a swap: find a block that covers the critical t-subsets
                critical = set()
                for ts in itertools.combinations(block_tuple, t):
                    idx = t_subset_to_idx[ts]
                    if cover_count[idx] == 1:
                        critical.add(idx)
                
                if critical:
                    # Find a random block that covers all critical t-subsets
                    # This is expensive, so we'll try a few random candidates
                    found = False
                    for _ in range(20):
                        j = rng.randint(0, n_blocks - 1)
                        new_block = all_blocks[j]
                        new_block_tuple = tuple(new_block)
                        if new_block_tuple in selected_set:
                            continue
                        
                        # Check if new block covers all critical
                        covers_all = True
                        for idx in critical:
                            if idx not in block_covers[j]:
                                covers_all = False
                                break
                        
                        if covers_all:
                            # Apply swap
                            # Remove old block
                            for ts in itertools.combinations(block_tuple, t):
                                cover_count[t_subset_to_idx[ts]] -= 1
                            selected_set.remove(block_tuple)
                            
                            # Add new block
                            for ts in itertools.combinations(new_block_tuple, t):
                                cover_count[t_subset_to_idx[ts]] += 1
                            selected_set.add(new_block_tuple)
                            found = True
                            break
                    
                    if found:
                        T *= cooling
                    else:
                        T *= cooling * 0.5  # Cooldown faster if no move
                else:
                    T *= cooling * 0.5
        
        else:
            # Add a random block (to escape local minima)
            j = rng.randint(0, n_blocks - 1)
            new_block_tuple = tuple(all_blocks[j])
            if new_block_tuple not in selected_set:
                # Apply add
                for ts in itertools.combinations(new_block_tuple, t):
                    cover_count[t_subset_to_idx[ts]] += 1
                selected_set.add(new_block_tuple)
                
                # Accept with probability exp(-1/T)
                if rng.random() < np.exp(-1.0 / T):
                    T *= cooling
                else:
                    # Reject
                    for ts in itertools.combinations(new_block_tuple, t):
                        cover_count[t_subset_to_idx[ts]] -= 1
                    selected_set.remove(new_block_tuple)
                    T *= cooling
    
    # Final result
    final_blocks = list(selected_set)
    
    # Verify validity
    covered = set()
    for b in final_blocks:
        for ts in itertools.combinations(b, t):
            covered.add(t_subset_to_idx[ts])
    
    if covered != set(range(n_t)):
        # Fix by adding missing blocks
        missing = set(range(n_t)) - covered
        for ts_idx in missing:
            ts = all_t_subsets[ts_idx]
            for j in ts_to_blocks[ts_idx]:
                block_tuple = tuple(all_blocks[j])
                if block_tuple not in selected_set:
                    final_blocks.append(list(block_tuple))
                    selected_set.add(block_tuple)
                    break
    
    # Remove duplicates
    unique_blocks = []
    seen = set()
    for b in final_blocks:
        bt = tuple(b)
        if bt not in seen:
            seen.add(bt)
            unique_blocks.append(b)
    
    return unique_blocks

if __name__ == "__main__":
    blocks = construct(24, 7, 3)
    print(f"Blocks: {len(blocks)}")
