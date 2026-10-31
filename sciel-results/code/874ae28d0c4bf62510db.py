import itertools
import random
import time
import math
import sys

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115  # Leave margin
    
    all_points = list(range(v))
    all_tsubsets = list(itertools.combinations(all_points, t))
    tsubset_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    n_tsubsets = len(all_tsubsets)
    
    # Precompute all k-blocks and their covered t-subsets
    all_blocks = list(itertools.combinations(all_points, k))
    block_to_tsubs = {}
    tsub_to_blocks = {i: [] for i in range(n_tsubsets)}
    
    for b in all_blocks:
        covered_tsubs = []
        for ts in itertools.combinations(b, t):
            idx = tsubset_index[ts]
            covered_tsubs.append(idx)
            tsub_to_blocks[idx].append(b)
        block_to_tsubs[b] = covered_tsubs

    def grasp_init(rcl_size=5):
        """
        Greedy Randomized Adaptive Search Procedure initialization.
        rcl_size: Number of top candidates to choose from.
        """
        uncovered = [1] * n_tsubsets
        n_uncovered = n_tsubsets
        blocks = []
        block_set = set()
        
        while n_uncovered > 0:
            # Calculate gains for all blocks
            gains = []
            for b in all_blocks:
                if b in block_set:
                    continue
                gain = 0
                for ts_idx in block_to_tsubs[b]:
                    if uncovered[ts_idx]:
                        gain += 1
                gains.append((gain, b))
            
            if not gains or gains[0][0] == 0:
                break # No improvement possible, should not happen if valid
                
            # Sort by gain descending
            gains.sort(key=lambda x: -x[0])
            
            # Select from top rcl_size
            rcl = gains[:min(rcl_size, len(gains))]
            # Random selection from RCL
            _, best_block = random.choice(rcl)
            
            blocks.append(best_block)
            block_set.add(best_block)
            
            for ts_idx in block_to_tsubs[best_block]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = 0
                    n_uncovered -= 1
                    
        return blocks

    def local_search_sa(init_blocks, time_budget):
        """
        Simulated Annealing local search.
        Returns best blocks found.
        """
        current_blocks = list(init_blocks)
        current_block_set = set(current_blocks)
        
        # Coverage counts
        coverage = [0] * n_tsubsets
        for b in current_blocks:
            for ts_idx in block_to_tsubs[b]:
                coverage[ts_idx] += 1
        
        current_energy = sum(1 for c in coverage if c == 0)
        
        best_blocks = current_blocks[:]
        best_energy = current_energy
        best_size = len(current_blocks)
        
        # SA Parameters
        T = 5.0
        T_min = 0.01
        alpha = 0.99
        
        start_ls = time.time()
        
        # Helper: Remove block b
        def try_remove(b):
            nonlocal current_blocks, current_block_set, coverage, current_energy, best_blocks, best_energy, best_size
            delta = 0
            for ts_idx in block_to_tsubs[b]:
                if coverage[ts_idx] == 1:
                    delta += 1
            
            # Acceptance logic
            # delta = change in energy (uncovered count)
            # If delta < 0: Energy decreases (improvement)
            # If delta == 0: Energy same. If current_energy==0, size decreases (strict improvement).
            # If delta > 0: Energy increases.
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if current_energy == 0:
                    accept = True # Size reduction
                else:
                    accept = random.random() < 0.5 # Exploration
            else:
                accept = random.random() < math.exp(-delta / T)
            
            if not accept:
                return False
            
            # Apply
            try:
                idx = current_blocks.index(b)
            except ValueError:
                return False
            
            current_blocks.pop(idx)
            current_block_set.remove(b)
            for ts_idx in block_to_tsubs[b]:
                coverage[ts_idx] -= 1
            current_energy += delta
            
            if current_energy == 0 and len(current_blocks) < best_size:
                best_size = len(current_blocks)
                best_energy = 0
                best_blocks = current_blocks[:]
            elif current_energy < best_energy:
                best_energy = current_energy
                best_blocks = current_blocks[:]
                
            return True

        def try_swap(b_old, b_new):
            nonlocal current_blocks, current_block_set, coverage, current_energy, best_blocks, best_energy, best_size
            
            # Calculate delta energy
            affected = set(block_to_tsubs[b_old])
            affected.update(block_to_tsubs[b_new])
            
            old_energy_part = 0
            new_energy_part = 0
            
            for ts in affected:
                old_c = coverage[ts]
                new_c = old_c
                if ts in block_to_tsubs[b_old]:
                    new_c -= 1
                if ts in block_to_tsubs[b_new]:
                    new_c += 1
                
                if old_c == 0:
                    old_energy_part += 1
                if new_c == 0:
                    new_energy_part += 1
                    
            delta = new_energy_part - old_energy_part
            
            # Acceptance
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.2
            else:
                accept = random.random() < math.exp(-delta / T)
            
            if not accept:
                return False
            
            # Apply
            try:
                idx = current_blocks.index(b_old)
            except ValueError:
                return False
            
            current_blocks[idx] = b_new
            current_block_set.remove(b_old)
            current_block_set.add(b_new)
            
            for ts_idx in block_to_tsubs[b_old]:
                coverage[ts_idx] -= 1
            for ts_idx in block_to_tsubs[b_new]:
                coverage[ts_idx] += 1
                
            current_energy += delta
            
            if current_energy == 0 and len(current_blocks) < best_size:
                best_size = len(current_blocks)
                best_energy = 0
                best_blocks = current_blocks[:]
            elif current_energy < best_energy:
                best_energy = current_energy
                best_blocks = current_blocks[:]
                
            return True

        while (time.time() - start_ls) < time_budget:
            move_type = random.random()
            
            if move_type < 0.5:
                # Remove
                if len(current_blocks) > 0:
                    b = random.choice(current_blocks)
                    try_remove(b)
            else:
                # Swap
                if len(current_blocks) > 0:
                    b_old = random.choice(current_blocks)
                    b_new = random.choice(all_blocks)
                    if b_new not in current_block_set:
                        try_swap(b_old, b_new)
            
            T *= alpha
            if T < T_min:
                T = T_min
                
        return best_blocks

    # Main Loop: Iterated Local Search / GRASP + SA
    best_result = None
    best_size = float('inf')
    
    iterations = 5
    time_per_iter = 20.0
    
    for i in range(iterations):
        current_time = time.time()
        remaining = time_limit - (current_time - start_time)
        if remaining < 5:
            break
        
        if i == 0:
            # Deterministic Greedy for first start
            init_blocks = grasp_init(rcl_size=1)
        else:
            # Randomized Greedy
            rcl = 5 + i * 2
            init_blocks = grasp_init(rcl_size=rcl)
        
        # Allocate time for SA
        sa_time = min(time_per_iter, remaining - 2)
        
        result = local_search_sa(init_blocks, sa_time)
        
        if len(result) < best_size:
            best_size = len(result)
            best_result = result
            
        # Check time
        if (time.time() - start_time) > time_limit - 2:
            break
            
    if best_result is None:
        # Fallback to simple greedy
        best_result = grasp_init(rcl_size=1)
        
    return [list(b) for b in best_result]
