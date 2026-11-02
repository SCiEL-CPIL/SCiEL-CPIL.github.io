"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys

def construct(v, k, t):
    deadline = time.time() + 110.0  # Safety margin
    
    # 1. Precompute all t-subsets and map them to indices
    t_sets = list(itertools.combinations(range(v), t))
    t_set_index = {ts: i for i, ts in enumerate(t_sets)}
    num_t_sets = len(t_sets)
    
    # 2. Precompute all k-blocks and the t-sets they cover
    # For v=19, k=8, C(19,8) = 75582. This is small enough to precompute.
    all_blocks = list(itertools.combinations(range(v), k))
    block_cover_indices = []
    for block in all_blocks:
        indices = []
        for ts in itertools.combinations(block, t):
            indices.append(t_set_index[ts])
        block_cover_indices.append(indices)
    
    num_blocks = len(all_blocks)
    
    # 3. Greedy Set Cover with Randomized Tie-Breaking
    best_solution = None
    best_size = float('inf')
    
    # We will run multiple greedy passes.
    # To diversify, we can randomly perturb the "gain" or the order of blocks.
    # A simple strategy: run standard greedy, then run greedy with slight noise.
    
    # Standard Greedy
    solution = _greedy_cover(v, k, t, t_set_index, all_blocks, block_cover_indices, num_t_sets, deadline, rng_seed=42)
    if solution:
        best_size = len(solution)
        best_solution = solution
    
    # Try a few randomized restarts if time permits
    # The greedy step is O(num_blocks * num_t_sets) in worst case if not optimized, 
    # but with precomputed lists and bitsets/sets it's faster.
    # Let's optimize the greedy step using a set of uncovered indices.
    
    if time.time() < deadline - 10.0:
        for seed in range(1, 10):
            if time.time() > deadline - 5.0:
                break
            sol = _greedy_cover(v, k, t, t_set_index, all_blocks, block_cover_indices, num_t_sets, deadline, rng_seed=seed + 100, randomize=True)
            if sol and len(sol) < best_size:
                best_size = len(sol)
                best_solution = sol
    
    if best_solution is None:
        # Fallback: should not happen, but return empty or a dummy valid cover
        # A trivial cover is to pick all blocks? No, that's too big.
        # Just return the first greedy result or empty if it failed.
        return []
        
    return best_solution

def _greedy_cover(v, k, t, t_set_index, all_blocks, block_cover_indices, num_t_sets, deadline, rng_seed=42, randomize=False):
    rng = random.Random(rng_seed)
    
    # Use a list of booleans or a set for uncovered
    # A set of integers (indices) is efficient for small N (11628)
    uncovered = set(range(num_t_sets))
    selected_blocks = []
    
    # Precompute a list of (block_index, cover_set) for quick access
    # We will maintain a count of how many times each t-set is covered by *potential* blocks?
    # No, standard greedy: pick block that covers most uncovered.
    
    # To speed up, we can maintain an array `uncovered_count` for each block?
    # That's hard to update dynamically.
    # Instead, since num_blocks ~ 75k and num_t_sets ~ 11k, 
    # iterating all blocks each step is 75k * 11k ops ~ 800M ops per step? No.
    # Intersection of two sets of size ~100 (avg cover per block) is fast.
    # C(8,5) = 56 t-sets per block.
    # So each block covers 56 t-sets.
    # Intersection of set of 56 ints with set of uncovered is fast.
    
    # However, 75,000 blocks * 500 steps (approx) = 37.5M intersections.
    # Each intersection is O(56). Total ~ 2B ops. Might be too slow in Python.
    
    # Optimization: Only consider blocks that cover at least one uncovered t-set.
    # We can maintain a list of active blocks.
    # Initially all blocks are active.
    # When we pick a block, we remove it from active? No, other blocks might still cover uncovered sets.
    # But we can prune blocks that cover 0 uncovered sets.
    
    # Let's use a different strategy:
    # Maintain `uncovered` as a set.
    # Maintain `block_uncovered_count`? No, that's hard to update.
    
    # Alternative: Use NumPy bitsets?
    # 11628 bits = 182 64-bit words.
    # 75582 blocks * 182 words = 13M words. Memory ~ 100MB. OK.
    # But Python overhead for list of arrays is high.
    
    # Let's stick to Python sets but optimize the loop.
    # We can skip blocks that have no intersection with `uncovered`.
    
    # To make it faster, we can pre-sort blocks by their "potential" or just iterate.
    # If it's too slow, we might need to limit the search space or use a heuristic to pick candidates.
    
    # Let's try a simple heuristic: 
    # 1. Pick a random uncovered t-set.
    # 2. Only consider blocks that cover this t-set.
    # 3. Among those, pick the one that covers the most *other* uncovered t-sets.
    # This reduces the number of blocks to check per step from 75k to ~1000 (since each t-set is in C(16, 3) blocks? No, C(v-t, k-t) = C(14, 3) = 364 blocks).
    # This is a huge speedup!
    
    # Precompute: for each t-set index, which blocks cover it?
    t_set_to_blocks = [[] for _ in range(num_t_sets)]
    for b_idx, block in enumerate(all_blocks):
        for ts_idx in block_cover_indices[b_idx]:
            t_set_to_blocks[ts_idx].append(b_idx)
    
    # Greedy Loop
    while uncovered:
        if time.time() > deadline - 2.0:
            return None # Timeout
        
        # Pick a random uncovered t-set to focus on
        # If randomize, pick random; else pick min index for determinism?
        # Picking random helps escape local minima in greedy choices?
        # Standard greedy picks the block with global max gain.
        # "Greedy by random pivot" picks a random uncovered set, then best block covering it.
        # This is often nearly as good and much faster.
        
        target_tset_idx = rng.choice(list(uncovered)) if randomize else min(uncovered)
        
        candidate_block_indices = t_set_to_blocks[target_tset_idx]
        
        best_block_idx = -1
        max_gain = 0
        
        for b_idx in candidate_block_indices:
            # Calculate gain: how many uncovered t-sets does this block cover?
            # block_cover_indices[b_idx] is a list of indices.
            # We need intersection with `uncovered`.
            # Since lists are small (56), we can do:
            gain = 0
            for ts_idx in block_cover_indices[b_idx]:
                if ts_idx in uncovered:
                    gain += 1
            
            if gain > max_gain:
                max_gain = gain
                best_block_idx = b_idx
        
        if best_block_idx == -1:
            # Should not happen if uncovered is not empty
            break
        
        # Add best block to solution
        selected_blocks.append(all_blocks[best_block_idx])
        
        # Update uncovered
        for ts_idx in block_cover_indices[best_block_idx]:
            if ts_idx in uncovered:
                uncovered.remove(ts_idx)
        
        if not uncovered:
            break
    
    return selected_blocks

# EVOLVE-BLOCK-END
