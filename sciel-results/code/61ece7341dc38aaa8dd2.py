import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s

    # Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    # masks[i] is an integer bitmask
    masks = [0] * num_k_subsets
    # t_to_k[j] is list of k-subset indices that cover t-subset j
    t_to_k = [[] for _ in range(num_t_subsets)]
    
    for i, ks in enumerate(k_subsets):
        mask = 0
        for ts in itertools.combinations(ks, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
            t_to_k[idx].append(i)
        masks[i] = mask
        
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    best_blocks = None
    best_count = float('inf')
    
    # Strategy:
    # 1. Fast vectorized greedy to get a good initial solution quickly.
    # 2. Prune redundant blocks.
    # 3. Local Search (WalkSAT-style) to minimize block count further.
    
    # --- Phase 1: Fast Greedy Construction ---
    # We use a simple loop but with optimized gain updates.
    # Since num_k_subsets is ~75k and num_t_subsets is ~11628, bit operations are fast.
    
    def greedy_construct(randomize=True, seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered = (1 << num_t_subsets) - 1
        current_blocks = []
        used_blocks = set()
        
        # gain[i] = number of uncovered t-subsets covered by k-subset i
        gain = [0] * num_k_subsets
        for i in range(num_k_subsets):
            gain[i] = popcount(masks[i]) # Initially all uncovered
        
        while uncovered:
            if time.time() > deadline:
                return None, False
            
            # Find max gain
            max_gain = 0
            candidates = []
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                if gain[i] > max_gain:
                    max_gain = gain[i]
                    candidates = [i]
                elif gain[i] == max_gain and gain[i] > 0:
                    candidates.append(i)
            
            if max_gain == 0:
                return current_blocks, False # Should not happen for valid parameters
            
            # Random choice among max gain candidates
            if randomize and candidates:
                chosen_idx = random.choice(candidates)
            else:
                chosen_idx = candidates[0] if candidates else -1
            
            if chosen_idx == -1:
                return current_blocks, False
                
            current_blocks.append(chosen_idx)
            used_blocks.add(chosen_idx)
            
            # Update uncovered and gains
            newly_covered = uncovered & masks[chosen_idx]
            uncovered &= ~masks[chosen_idx]
            
            # Efficient gain update using t_to_k
            nc = newly_covered
            while nc:
                lsb = nc & -nc
                idx_j = lsb.bit_length() - 1
                nc ^= lsb
                for i in t_to_k[idx_j]:
                    if i not in used_blocks:
                        gain[i] -= 1
                        if gain[i] < 0: gain[i] = 0 # Safety

        return current_blocks, True

    def prune_redundant(block_indices):
        """
        Remove blocks that are redundant.
        Returns a new list of block indices.
        """
        if not block_indices:
            return []
        
        # Count coverage
        cover_counts = [0] * num_t_subsets
        for idx in block_indices:
            mask = masks[idx]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                cover_counts[j] += 1
                m ^= lsb
        
        changed = True
        current_blocks = list(block_indices)
        while changed:
            changed = False
            new_blocks = []
            for idx in current_blocks:
                # Check if removing idx keeps all counts >= 1
                mask = masks[idx]
                m = mask
                redundant = True
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    if cover_counts[j] == 1:
                        redundant = False
                        break
                    m ^= lsb
                if redundant:
                    # Remove idx
                    # Decrement counts
                    m = mask
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        cover_counts[j] -= 1
                        m ^= lsb
                    changed = True
                else:
                    new_blocks.append(idx)
            current_blocks = new_blocks
        return current_blocks

    # --- Phase 2: Local Search (WalkSAT-style for Set Cover) ---
    # Objective: Minimize number of blocks.
    # Constraint: All t-subsets must be covered.
    # Move: Remove a block if it doesn't break coverage. Add a block if it helps.
    # We focus on "Remove" moves primarily, as greedy tends to overshoot.
    # If removal breaks coverage, we can try to "Swap" or "Add" then "Remove" others.
    # Simplest effective local search: Try to remove each block one by one. If it fails, keep it.
    # This is basically prune_redundant but applied iteratively with random order.
    
    def local_search_improve(block_indices, max_iters=1000):
        if not block_indices:
            return []
        
        current_blocks = list(block_indices)
        current_size = len(current_blocks)
        
        # Precompute coverage counts
        cover_counts = [0] * num_t_subsets
        for idx in current_blocks:
            mask = masks[idx]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                cover_counts[j] += 1
                m ^= lsb
        
        # Helper to check if a block is removable
        def is_removable(idx):
            mask = masks[idx]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if cover_counts[j] == 1:
                    return False
                m ^= lsb
            return True
        
        # Helper to remove a block
        def remove_block(idx):
            nonlocal current_blocks
            mask = masks[idx]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                cover_counts[j] -= 1
                m ^= lsb
            current_blocks.remove(idx)
            
        # Helper to add a block
        def add_block(idx):
            nonlocal current_blocks
            mask = masks[idx]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                cover_counts[j] += 1
                m ^= lsb
            current_blocks.append(idx)

        # WalkSAT-like loop
        # We try to remove blocks. If we get stuck, we might try to add a block that covers
        # many "critical" t-subsets (those covered exactly once) to enable removal of multiple blocks?
        # That's complex. Let's stick to random order removal + occasional random swaps if stuck.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle blocks to break symmetry
            random.shuffle(current_blocks)
            
            for idx in current_blocks[:]:  # Iterate over a copy
                if time.time() > deadline:
                    break
                if is_removable(idx):
                    remove_block(idx)
                    improved = True
                    current_size -= 1
                    
        return current_blocks

    # Main Loop: Run multiple restarts
    # We interleave different strategies:
    # 1. Pure greedy (deterministic)
    # 2. Randomized greedy
    # 3. Randomized greedy + Local Search
    
    num_restarts = 20000  # Limit by time, not count
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Choose strategy based on restart index
        use_ls = (restart % 2 == 0)  # Even restarts use local search
        
        blocks, success = greedy_construct(randomize=(restart % 2 != 0), seed=restart)
        
        if not success or blocks is None:
            continue
            
        if use_ls:
            blocks = local_search_improve(blocks)
            
        # Final prune just in case
        blocks = prune_redundant(blocks)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [k_subsets[i] for i in blocks]
            
            # If we hit a very good solution, we might want to spend more time on local search
            # But for now, just continue to find even better ones.
            
    if best_blocks is None:
        # Fallback to simple greedy if randomization failed (unlikely)
        uncovered = (1 << num_t_subsets) - 1
        current_blocks = []
        used_blocks = set()
        while uncovered:
            max_gain = 0
            best_block_idx = -1
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                gain = popcount(masks[i] & uncovered)
                if gain > max_gain:
                    max_gain = gain
                    best_block_idx = i
            if max_gain == 0:
                break
            current_blocks.append(k_subsets[best_block_idx])
            used_blocks.add(best_block_idx)
            uncovered &= ~masks[best_block_idx]
        best_blocks = [list(b) for b in current_blocks]

    return best_blocks
