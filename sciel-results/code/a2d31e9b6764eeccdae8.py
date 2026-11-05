import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute tsubset indices for each block
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    block_indices = []
    for b in all_blocks:
        block_indices.append([tsubset_to_idx[ts] for ts in itertools.combinations(b, t)])
    
    # Precompute: for each tsubset, which blocks cover it (transpose)
    # This is needed for efficient gain updates
    tsubset_to_blocks = [[] for _ in range(n_tsubsets)]
    for bi, idxs in enumerate(block_indices):
        for idx in idxs:
            tsubset_to_blocks[idx].append(bi)
    
    best_blocks = None
    
    trial = 0
    while time.time() < deadline:
        trial += 1
        blocks = _weighted_greedy(v, k, t, n_tsubsets, n_blocks, block_indices, 
                                    tsubset_to_blocks, all_blocks, deadline, trial)
        if blocks is not None and (best_blocks is None or len(blocks) < len(best_blocks)):
            best_blocks = blocks
        if best_blocks and len(best_blocks) <= 76:
            break
    
    if best_blocks is None:
        blocks = _weighted_greedy(v, k, t, n_tsubsets, n_blocks, block_indices,
                                    tsubset_to_blocks, all_blocks, deadline, -1)
        if blocks is not None:
            best_blocks = blocks
        else:
            best_blocks = []
    
    return best_blocks

def _weighted_greedy(v, k, t, n_tsubsets, n_blocks, block_indices, tsubset_to_blocks, all_blocks, deadline, seed_val):
    rng = random.Random(seed_val if seed_val >= 0 else time.time())
    
    uncovered = np.ones(n_tsubsets, dtype=bool)
    
    # Compute degrees: how many blocks cover each tsubset
    degrees = np.array([len(tsubset_to_blocks[i]) for i in range(n_tsubsets)], dtype=np.float64)
    weights = 1.0 / (degrees + 1.0)
    
    # Compute initial gains
    gains = np.zeros(n_blocks, dtype=np.float64)
    for bi in range(n_blocks):
        idxs = block_indices[bi]
        gains[bi] = np.sum(weights[idxs])
    
    blocks_selected = []
    
    while np.any(uncovered):
        if time.time() > deadline:
            return None
        
        # Find best block with random tie-breaking
        # Only consider blocks that can cover at least one uncovered tsubset
        active = gains > 0
        if not np.any(active):
            return None
        
        noise = rng.random(n_blocks) * 1e-6
        gains_noisy = gains + noise
        gains_noisy[~active] = -1
        
        best_bi = int(np.argmax(gains_noisy))
        
        blocks_selected.append(list(all_blocks[best_bi]))
        
        # Update: for each newly covered tsubset, subtract its weight from gains of all blocks covering it
        newly_covered = block_indices[best_bi]
        for idx in newly_covered:
            if uncovered[idx]:
                uncovered[idx] = False
                w = weights[idx]
                for other_bi in tsubset_to_blocks[idx]:
                    gains[other_bi] -= w
    
    return blocks_selected
