import itertools
import time
import numpy as np

_PRECOMPUTED = {}

def _get_precomputed(v, k, t):
    key = (v, k, t)
    if key in _PRECOMPUTED:
        return _PRECOMPUTED[key]
    
    # Generate all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Generate all k-subsets (blocks)
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)
    
    # Precompute indices of t-subsets contained in each k-subset
    # C(k, t) is the number of t-subsets in a k-subset
    c_k_t = len(list(itertools.combinations(range(k), t)))
    block_t_indices = np.zeros((n_k, c_k_t), dtype=np.int32)
    
    for i, block in enumerate(all_k_subs):
        # Find indices of all t-subsets of this block
        indices = [t_sub_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_t_indices[i, :] = indices
        
    _PRECOMPUTED[key] = (n_t, n_k, block_t_indices, all_k_subs)
    return _PRECOMPUTED[key]

def construct(v, k, t):
    start_time = time.time()
    time_limit = 120.0  # 120 seconds wall clock limit
    
    # Fallback for non-standard parameters or if precomputation is too large
    # For v=19, k=8, t=5: n_t=11628, n_k=75582. This is manageable.
    # If v is larger, this might fail. But task is specific.
    
    try:
        n_t, n_k, block_t_indices, all_k_subs = _get_precomputed(v, k, t)
    except Exception:
        # Fallback to simple greedy if precomputation fails or is too big
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Use a fixed seed for reproducibility in the first run, then vary
    # Or just use system random.
    
    while True:
        elapsed = time.time() - start_time
        # Check if we have enough time for another run (assume 0.5s per run + margin)
        if elapsed + 0.5 > time_limit * 0.95:
            break
        
        # Initialize covered array
        covered = np.zeros(n_t, dtype=bool)
        current_blocks = []
        
        # Greedy loop
        while not covered.all():
            # Calculate gains: for each block, how many uncovered t-subsets does it cover?
            # covered[indices] gives a boolean matrix (n_k, c_k_t)
            # We want sum of True values in each row for UNCOVERED subsets.
            # ~covered[indices] gives True where NOT covered.
            
            gains = (~covered[block_t_indices]).sum(axis=1)
            
            # If all gains are 0, we are stuck (should not happen if covered.all() is False)
            if gains.max() == 0:
                break
            
            # Add small random noise to break ties and explore
            # Noise magnitude: 0.1 is small compared to typical gains
            noise = np.random.random(n_k) * 0.1
            noisy_gains = gains + noise
            
            best_idx = np.argmax(noisy_gains)
            
            # Update covered
            block_indices = block_t_indices[best_idx]
            covered[block_indices] = True
            
            # Record the block
            current_blocks.append(all_k_subs[best_idx])
            
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = current_blocks.copy()
            
        # Optional: Stop early if we hit the lower bound? 
        # Lower bound is 264. If we get 264, we can stop.
        if best_len <= 264:
            break

    if best_blocks is None:
        # Fallback if loop didn't run (e.g. time out immediately)
        # Just run one deterministic greedy
        covered = np.zeros(n_t, dtype=bool)
        current_blocks = []
        while not covered.all():
            gains = (~covered[block_t_indices]).sum(axis=1)
            best_idx = np.argmax(gains)
            block_indices = block_t_indices[best_idx]
            covered[block_indices] = True
            current_blocks.append(all_k_subs[best_idx])
        best_blocks = current_blocks

    return [list(b) for b in best_blocks]
