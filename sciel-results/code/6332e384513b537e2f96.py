import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    n_t_subsets = len(all_t_subsets)
    
    # Helper to get triple indices for a block
    def get_block_triple_indices(block):
        return [t_subset_to_idx[tuple(sub)] for sub in itertools.combinations(block, t)]
    
    best_blocks = None
    best_len = float('inf')
    
    while time.time() < deadline:
        # 1. Randomized Greedy Construction
        coverage = np.zeros(n_t_subsets, dtype=np.int32)
        blocks = []
        uncovered_count = n_t_subsets
        
        # List of uncovered indices to pick from randomly
        # Maintaining a list is O(N) to update, but picking is O(1)
        # Since N=2024, we can just use np.where(coverage==0) if needed, but that's O(N) per step.
        # Better: keep a list of uncovered indices.
        uncovered_indices = list(range(n_t_subsets))
        
        while uncovered_count > 0:
            if time.time() > deadline:
                break
            
            # Pick a random uncovered triple
            # To ensure we don't pick a covered one, we pick a random index from the 'uncovered_indices' list
            # But 'uncovered_indices' might contain indices that became covered?
            # No, we only add to blocks, so coverage only increases.
            # So if an index is in 'uncovered_indices', it is still uncovered.
            
            pick_idx = random.randint(0, len(uncovered_indices) - 1)
            target_triple_idx = uncovered_indices[pick_idx]
            
            # Remove from list
            uncovered_indices.pop(pick_idx)
            
            target_triple = all_t_subsets[target_triple_idx]
            
            # Generate candidates containing target_triple
            target_set = set(target_triple)
            remaining_points = [p for p in range(v) if p not in target_set]
            
            best_gain = -1
            best_block = None
            
            # Optimization: Sample candidates?
            # C(21, 4) = 5985.
            # Let's try to speed this up.
            # If we sample 100 candidates, we might miss the best one, but it's much faster.
            # Given 110s, and ~100 blocks per run, 100 * 5985 * 35 ops = 21M ops per run.
            # If we do 10 runs, 210M ops. Might be tight in Python.
            # Let's sample 200 candidates.
            
            # Sample combinations
            # itertools.combinations doesn't support sampling directly.
            # We can generate a random permutation of remaining_points and take the first few?
            # Or just iterate all if it's fast enough?
            # Let's try iterating all first, but break early if we find a "perfect" gain?
            # Max gain is 35.
            
            # Actually, let's just iterate all. 5985 is small.
            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(target_triple + combo))
                gain = 0
                for sub in itertools.combinations(block, t):
                    if coverage[t_subset_to_idx[sub]] == 0:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    
            if best_block is None:
                break
                
            blocks.append(best_block)
            
            # Update coverage
            block_indices = get_block_triple_indices(best_block)
            new_uncovered_indices = []
            for idx in block_indices:
                if coverage[idx] == 0:
                    # It was uncovered, now covered
                    # We need to remove it from uncovered_indices if it's there
                    # But we already popped the target.
                    # Other indices in this block might also be in uncovered_indices.
                    pass
                coverage[idx] += 1
            
            # Update uncovered_indices: remove any index that is now covered (coverage > 0)
            # This is O(len(uncovered_indices)) which is ~2000.
            # We do this ~100 times. 200k ops. Fine.
            new_uncovered_indices = []
            for idx in uncovered_indices:
                if coverage[idx] == 0:
                    new_uncovered_indices.append(idx)
            uncovered_indices = new_uncovered_indices
            uncovered_count = len(uncovered_indices)

        # 2. Redundancy Pruning
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b in blocks:
                b_indices = get_block_triple_indices(b)
                redundant = True
                for idx in b_indices:
                    if coverage[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    for idx in b_indices:
                        coverage[idx] -= 1
                    changed = True
                else:
                    new_blocks.append(b)
            blocks = new_blocks
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
            
    return best_blocks
