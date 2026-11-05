import numpy as np
import itertools
import time
import random
import sys

def construct(v, k, t):
    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block_to_subsets and subset_to_blocks
    block_to_subsets = [[] for _ in range(num_blocks)]
    subset_to_blocks = [[] for _ in range(num_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        for s in itertools.combinations(block, t):
            s_idx = subset_to_idx[s]
            block_to_subsets[b_idx].append(s_idx)
            subset_to_blocks[s_idx].append(b_idx)
    
    block_sizes = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)
    
    # Convert to numpy arrays for faster operations
    block_subsets_np = [np.array(subs, dtype=np.int32) for subs in block_to_subsets]
    subset_blocks_np = [np.array(blocks, dtype=np.int32) for blocks in subset_to_blocks]
    
    def run_greedy_full(random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered = np.ones(num_subsets, dtype=bool)
        gains = block_sizes.copy().astype(np.int32)
        
        selected_blocks = []
        
        while True:
            if not np.any(uncovered):
                break
                
            # Add noise to break ties and explore
            noise = rng.uniform(0, 0.5, size=num_blocks).astype(np.float64)
            scores = gains.astype(np.float64) + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_subsets_np[best_idx]
            # Find newly covered subsets
            newly_covered_mask = uncovered[b_subs]
            newly_covered = b_subs[newly_covered_mask]
            
            if len(newly_covered) == 0:
                continue
                
            for s in newly_covered:
                uncovered[s] = False
                for b in subset_blocks_np[s]:
                    gains[b] -= 1
                    
        return selected_blocks

    def run_greedy_partial(uncovered_mask, random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            return []
            
        gains = np.zeros(num_blocks, dtype=np.int32)
        for s_idx in uncovered_indices:
            for b_idx in subset_blocks_np[s_idx]:
                gains[b_idx] += 1
                
        selected_blocks = []
        current_uncovered = uncovered_mask.copy()
        
        while True:
            if not np.any(current_uncovered):
                break
                
            noise = rng.uniform(0, 0.5, size=num_blocks).astype(np.float64)
            scores = gains.astype(np.float64) + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_subsets_np[best_idx]
            newly_covered_mask = current_uncovered[b_subs]
            newly_covered = b_subs[newly_covered_mask]
            
            if len(newly_covered) == 0:
                continue
                
            for s in newly_covered:
                current_uncovered[s] = False
                for b in subset_blocks_np[s]:
                    gains[b] -= 1
                    
        return selected_blocks

    def minimize(blocks_indices, random_seed=None):
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        
        if not current_blocks:
            return current_blocks
            
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            for s_idx in block_subsets_np[b_idx]:
                cover_count[s_idx] += 1
                
        # Repeatedly remove redundant blocks
        changed = True
        while changed:
            changed = False
            redundant_blocks = []
            
            for i, b_idx in enumerate(current_blocks):
                is_redundant = True
                for s_idx in block_subsets_np[b_idx]:
                    if cover_count[s_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    redundant_blocks.append(i)
                    
            if not redundant_blocks:
                break
                
            # Remove all redundant blocks at once for efficiency
            for remove_pos in sorted(redundant_blocks, reverse=True):
                b_idx_to_remove = current_blocks[remove_pos]
                
                for s_idx in block_subsets_np[b_idx_to_remove]:
                    cover_count[s_idx] -= 1
                    
                current_blocks.pop(remove_pos)
                changed = True
            
        return current_blocks

    def perturb_and_repair(blocks_indices, random_seed=None, num_perturb=2):
        """
        Remove a few random blocks, repair with greedy, then minimize.
        Much faster than SA/LS.
        """
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        if len(current_blocks) <= num_perturb:
            return current_blocks
            
        # Compute coverage
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            for s_idx in block_subsets_np[b_idx]:
                cover_count[s_idx] += 1
        
        # Remove random blocks
        remove_indices = rng.sample(range(len(current_blocks)), num_perturb)
        blocks_to_remove = set(current_blocks[i] for i in remove_indices)
        
        # Update cover count
        for b_idx in blocks_to_remove:
            for s_idx in block_subsets_np[b_idx]:
                cover_count[s_idx] -= 1
                
        # Find uncovered
        uncovered_mask = (cover_count == 0).astype(bool)
        
        # Repair
        repair_res = run_greedy_partial(uncovered_mask, random_seed + 1)
        
        # Combine
        new_set = [b for b in current_blocks if b not in blocks_to_remove]
        new_set.extend(repair_res)
        
        # Minimize
        min_res = minimize(new_set, random_seed + 2)
        
        return min_res

    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    trial = 0
    while time.time() < deadline:
        # Strategy 1: Full Greedy + Minimize
        greedy_res = run_greedy_full(trial)
        min_res = minimize(greedy_res, trial)
        
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
        
        # Strategy 2: Perturb Best + Repair + Minimize
        if best_blocks is not None and len(best_blocks) > 5 and time.time() < deadline - 5:
            p_res = perturb_and_repair(best_blocks, trial + 1000, num_perturb=2)
            
            if len(p_res) < best_len:
                best_len = len(p_res)
                best_blocks = p_res

        trial += 1
        
    if best_blocks is None:
        greedy_res = run_greedy_full(0)
        best_blocks = minimize(greedy_res, 0)
        
    return [list(all_blocks[i]) for i in best_blocks]
