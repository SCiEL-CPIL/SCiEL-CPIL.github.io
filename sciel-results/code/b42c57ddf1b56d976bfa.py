import itertools
import random
import time

def construct(v, k, t):
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)

    def greedy_cover(rng_seed):
        rng = random.Random(rng_seed)
        uncovered = [True] * num_t_subsets
        gain = [len(block_t_indices[b]) for b in range(num_k_subsets)]
        
        selected_blocks = []
        
        # Optimization: Use a max-heap or just scan? Scanning 50k is fast.
        # But we can maintain a list of active blocks.
        # For simplicity and speed in Python, scanning is okay if N is 50k.
        # 150 selections * 50k = 7.5M comparisons. ~0.5s in PyPy, ~1-2s in CPython.
        # We have 120s. So we can do 20 runs.
        
        while True:
            max_gain = -1
            best_block = -1
            # Find max gain
            # To allow random tie-breaking, we can collect all maxes?
            # Or just pick one randomly from the top few?
            # Simpler: just find max. If we want randomness, we can add small noise to gain?
            # No, noise breaks the logic.
            # Instead, if multiple blocks have max_gain, pick one randomly.
            
            # Efficient max finding with tie handling:
            # Scan once, track max_val and a list of indices with max_val.
            # If list is large, picking random is O(1) after scan.
            
            max_val = -1
            candidates = []
            for b_idx in range(num_k_subsets):
                g = gain[b_idx]
                if g > max_val:
                    max_val = g
                    candidates = [b_idx]
                elif g == max_val:
                    candidates.append(b_idx)
            
            if max_val == 0:
                break
                
            best_block = rng.choice(candidates)
            selected_blocks.append(best_block)
            
            for t_idx in block_t_indices[best_block]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    for b_idx in t_subset_blocks[t_idx]:
                        gain[b_idx] -= 1
                        
        # Minimalization
        cover_count = [0] * num_t_subsets
        for b_idx in selected_blocks:
            for t_idx in block_t_indices[b_idx]:
                cover_count[t_idx] += 1
                
        active_blocks = set(selected_blocks)
        indices_to_check = list(range(len(selected_blocks)))
        rng.shuffle(indices_to_check)
        
        for i in indices_to_check:
            b_idx = selected_blocks[i]
            if b_idx not in active_blocks:
                continue
            can_remove = True
            for t_idx in block_t_indices[b_idx]:
                if cover_count[t_idx] <= 1:
                    can_remove = False
                    break
            if can_remove:
                active_blocks.remove(b_idx)
                for t_idx in block_t_indices[b_idx]:
                    cover_count[t_idx] -= 1
                    
        return [list(all_k_subsets[b_idx]) for b_idx in active_blocks]

    # Run multiple times with different seeds
    best_blocks = None
    best_size = float('inf')
    
    # Time check: 120s limit.
    # Precomputation: ~1s.
    # One run: ~2s.
    # 50 runs: 100s.
    # Let's try 30 runs.
    
    import time
    start_time = time.time()
    max_time = 110 # Leave 10s buffer
    
    for seed in range(100):
        if time.time() - start_time > max_time:
            break
        blocks = greedy_cover(seed)
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            # Early stop if we hit the theoretical lower bound?
            # Lower bound is 131. If we hit 131, stop.
            if best_size <= 131:
                break
                
    if best_blocks is None:
        # Fallback to single deterministic run if no time left or error
        best_blocks = greedy_cover(0)
        
    return best_blocks
