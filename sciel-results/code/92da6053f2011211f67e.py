import itertools
import numpy as np
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for safety

    # 1. Precompute t-subsets and their indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    n_t = len(t_subsets)

    # 2. Precompute k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)

    # 3. Precompute block_t_indices: (n_k, C(k, t))
    t_in_k = list(itertools.combinations(range(k), t))
    n_t_in_k = len(t_in_k)

    block_t_indices = np.empty((n_k, n_t_in_k), dtype=np.int32)

    # Filling the array
    for i, block in enumerate(k_subsets):
        for j, lc in enumerate(t_in_k):
            block_t_indices[i, j] = t_sub_to_idx[tuple(block[p] for p in lc)]

    # 4. Calculate Weights for Weighted Greedy
    # Frequency of each t-subset across all k-blocks
    t_counts = np.bincount(block_t_indices.ravel(), minlength=n_t)
    t_counts[t_counts == 0] = 1
    t_weights = 1.0 / t_counts.astype(np.float64)

    # 5. Helper: Remove redundant blocks efficiently
    def remove_redundant(remaining, block_t_indices, n_t):
        if not remaining:
            return remaining
        
        changed = True
        while changed:
            if time.time() > deadline - 2.0:
                break
            changed = False
            
            # Calculate coverage counts for current remaining blocks
            coverage_count = np.zeros(n_t, dtype=np.int32)
            for bi in remaining:
                coverage_count[block_t_indices[bi]] += 1
            
            # Identify all redundant blocks in a single pass
            to_remove = []
            for bi in remaining:
                ts = block_t_indices[bi]
                if np.all(coverage_count[ts] > 1):
                    to_remove.append(bi)
            
            if to_remove:
                remove_set = set(to_remove)
                remaining = [bi for bi in remaining if bi not in remove_set]
                changed = True
                
        return remaining

    # 6. Helper: Greedy repair for uncovered sets
    def greedy_repair(uncovered, selected_blocks, rng):
        """
        Greedily add blocks to cover the 'uncovered' t-subsets.
        Returns the updated list of selected blocks.
        """
        while True:
            if time.time() > deadline - 5.0:
                break
            if not uncovered.any():
                break

            # Calculate weighted gains
            gains = (uncovered[block_t_indices] * t_weights[block_t_indices]).sum(axis=1)
            
            # Only consider blocks not already in selected? 
            # Actually, adding a duplicate block is wasteful but harmless for coverage logic if we dedupe later.
            # But for efficiency, we should prefer new blocks.
            # However, the current 'selected_blocks' list might contain duplicates if we're not careful.
            # Let's assume 'selected_blocks' is a list of unique indices.
            # To avoid re-selecting, we could mask them out, but checking membership is O(N).
            # Given N is ~75k and we do this in a loop, let's just pick the best gain.
            # If it picks an existing block, gain is 0 for already covered parts, so it's less likely 
            # unless it's the only one covering a specific subset.
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break

            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]

            # Pick one randomly to encourage diversity
            best_idx = rng.choice(candidates)

            selected_blocks.append(best_idx)

            # Update uncovered
            uncovered[block_t_indices[best_idx]] = False

        return selected_blocks

    # 7. Main Loop: Iterated Local Search
    best_solution = None
    best_size = float('inf')
    rng = random.Random(42)

    # Initial Greedy Solution
    uncovered = np.ones(n_t, dtype=bool)
    selected_blocks = []
    
    # Run initial greedy
    while True:
        if time.time() > deadline - 10.0:
            break
        if not uncovered.any():
            break
        gains = (uncovered[block_t_indices] * t_weights[block_t_indices]).sum(axis=1)
        max_gain = np.max(gains)
        if max_gain == 0:
            break
        candidates = np.where(gains == max_gain)[0]
        best_idx = rng.choice(candidates)
        selected_blocks.append(best_idx)
        uncovered[block_t_indices[best_idx]] = False

    # Clean up initial solution
    selected_blocks = remove_redundant(selected_blocks, block_t_indices, n_t)
    if selected_blocks:
        best_size = len(selected_blocks)
        best_solution = [k_subsets[i] for i in selected_blocks]
    
    # Maintain a set of selected block indices for O(1) lookup in local search
    current_selected = list(selected_blocks)
    current_set = set(current_selected)
    
    # Precompute block_to_tsubs for faster local search updates if needed?
    # block_t_indices is already an array. Accessing block_t_indices[i] is fast.

    # Local Search Parameters
    max_iterations = 5000
    iteration = 0

    while time.time() < deadline - 5.0:
        iteration += 1
        
        # 1. Perturbation (Shake)
        # Remove a random subset of blocks
        shake_size = max(1, len(current_selected) // 15)
        # Ensure we don't remove too many
        shake_size = min(shake_size, len(current_selected))
        
        if shake_size > 0:
            # Randomly pick indices to remove
            to_remove_indices = rng.sample(range(len(current_selected)), shake_size)
            to_remove_blocks = [current_selected[i] for i in to_remove_indices]
            
            # Update current_selected
            remove_set = set(to_remove_blocks)
            current_selected = [b for b in current_selected if b not in remove_set]
            current_set.difference_update(remove_set)

        # 2. Calculate Uncovered
        # We need to know which t-subsets are no longer covered.
        # Efficient way: Recalculate coverage from scratch for the current set?
        # Or maintain a count array?
        # Given the perturbation size is small, recalculating coverage counts for the *remaining* set
        # and identifying zeros is robust.
        
        # Optimization: Instead of full recalc, we can track counts.
        # But for simplicity and correctness within time, let's do a fast check.
        # Since N_blocks is ~75k and we have ~300 blocks, iterating 300 blocks * 126 t-sets is ~37k ops.
        # Creating a zero array of 11628 is cheap.
        
        coverage_count = np.zeros(n_t, dtype=np.int32)
        for bi in current_selected:
            coverage_count[block_t_indices[bi]] += 1
        
        uncovered = (coverage_count == 0)
        
        # 3. Repair
        if uncovered.any():
            current_selected = greedy_repair(uncovered, current_selected, rng)
            
            # Deduplicate if greedy added existing blocks (unlikely but possible)
            # And remove redundants
            current_selected = list(set(current_selected))
            current_selected = remove_redundant(current_selected, block_t_indices, n_t)
            current_set = set(current_selected)
            
            # Check if we have a valid cover (should be, by construction of repair)
            # But remove_redundant might have removed too much? No, it only removes if redundant.
            # So it should still be a valid cover.

        # 4. Acceptance / Update Best
        current_size = len(current_selected)
        
        if current_size < best_size:
            best_size = current_size
            best_solution = [k_subsets[i] for i in current_selected]
            
        # Optional: Simulated Annealing style acceptance
        # If current_size > best_size, we might still accept it to escape local minima?
        # The prompt suggests ILS. Standard ILS usually accepts if better, or with probability if worse.
        # Given the tight time limit, simple "accept if better or equal" is safer.
        # However, if we only accept better, we get stuck.
        # Let's try: if current_size <= best_size + 2, accept (keep as current state for next iteration).
        # If current_size > best_size + 2, revert to best_solution.
        
        if current_size > best_size + 5:
            # Revert to best
            current_selected = [k_subsets.index(i) if i in best_solution else 0 for i in best_solution] # This is slow/wrong.
            # Better to keep indices in best_solution
            # We stored best_solution as list of tuples (blocks). We need indices.
            # Let's change best_solution storage to indices.
            
            # Correction: Store best_solution as indices.
            pass

    # Re-evaluating storage:
    # It is better to store `best_indices` and `best_solution_blocks` separately or just indices.
    
    # Let's restart the logic with cleaner state management.
    
    # --- Revised Implementation ---
    
    # Reset
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Precompute t-subsets and their indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    n_t = len(t_subsets)

    # 2. Precompute k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)

    # 3. Precompute block_t_indices: (n_k, C(k, t))
    t_in_k = list(itertools.combinations(range(k), t))
    n_t_in_k = len(t_in_k)

    block_t_indices = np.empty((n_k, n_t_in_k), dtype=np.int32)
    for i, block in enumerate(k_subsets):
        for j, lc in enumerate(t_in_k):
            block_t_indices[i, j] = t_sub_to_idx[tuple(block[p] for p in lc)]

    # 4. Calculate Weights
    t_counts = np.bincount(block_t_indices.ravel(), minlength=n_t)
    t_counts[t_counts == 0] = 1
    t_weights = 1.0 / t_counts.astype(np.float64)

    def remove_redundant(remaining):
        if not remaining:
            return remaining
        changed = True
        while changed:
            if time.time() > deadline - 2.0:
                break
            changed = False
            coverage_count = np.zeros(n_t, dtype=np.int32)
            for bi in remaining:
                coverage_count[block_t_indices[bi]] += 1
            to_remove = []
            for bi in remaining:
                ts = block_t_indices[bi]
                if np.all(coverage_count[ts] > 1):
                    to_remove.append(bi)
            if to_remove:
                remove_set = set(to_remove)
                remaining = [bi for bi in remaining if bi not in remove_set]
                changed = True
        return remaining

    def greedy_repair(uncovered, selected_blocks, rng):
        while True:
            if time.time() > deadline - 5.0:
                break
            if not uncovered.any():
                break
            gains = (uncovered[block_t_indices] * t_weights[block_t_indices]).sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best_idx = rng.choice(candidates)
            selected_blocks.append(best_idx)
            uncovered[block_t_indices[best_idx]] = False
        return selected_blocks

    rng = random.Random(42)
    
    # Initial Solution
    uncovered = np.ones(n_t, dtype=bool)
    current_indices = []
    while True:
        if time.time() > deadline - 10.0:
            break
        if not uncovered.any():
            break
        gains = (uncovered[block_t_indices] * t_weights[block_t_indices]).sum(axis=1)
        max_gain = np.max(gains)
        if max_gain == 0:
            break
        candidates = np.where(gains == max_gain)[0]
        best_idx = rng.choice(candidates)
        current_indices.append(best_idx)
        uncovered[block_t_indices[best_idx]] = False

    current_indices = remove_redundant(current_indices)
    best_indices = current_indices.copy()
    best_size = len(current_indices)

    # Local Search Loop
    while time.time() < deadline - 5.0:
        # Perturbation
        shake_size = max(1, len(current_indices) // 15)
        shake_size = min(shake_size, len(current_indices))
        
        if shake_size > 0:
            to_remove_idx = rng.sample(range(len(current_indices)), shake_size)
            to_remove_set = set(current_indices[i] for i in to_remove_idx)
            current_indices = [b for b in current_indices if b not in to_remove_set]

        # Calculate Uncovered
        coverage_count = np.zeros(n_t, dtype=np.int32)
        for bi in current_indices:
            coverage_count[block_t_indices[bi]] += 1
        uncovered = (coverage_count == 0)

        # Repair
        if uncovered.any():
            current_indices = greedy_repair(uncovered, current_indices, rng)
            # Deduplicate
            current_indices = list(set(current_indices))
            # Remove redundants
            current_indices = remove_redundant(current_indices)

        # Acceptance
        if len(current_indices) < best_size:
            best_size = len(current_indices)
            best_indices = current_indices.copy()
        elif len(current_indices) <= best_size + 3:
            # Accept even if slightly worse to allow exploration
            pass
        else:
            # Revert to best
            current_indices = best_indices.copy()

    if not best_indices:
        return []

    return [list(k_subsets[i]) for i in best_indices]
