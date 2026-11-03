"""Covering design C(19,7,4) using Greedy + Local Search with NumPy."""
import random
import itertools
import numpy as np
from collections import defaultdict
import time

def _precompute_blocks(v, k, t):
    """Precompute all blocks and their t-subset indices."""
    all_points = range(v)
    blocks = list(itertools.combinations(all_points, k))
    n_blocks = len(blocks)
    
    # Map t-subset to index
    t_subs = list(itertools.combinations(all_points, t))
    t_sub_index = {ts: i for i, ts in enumerate(t_subs)}
    n_t_subs = len(t_subs)
    
    # block_t_indices[i] is a list of t-subset indices covered by block i
    block_t_indices_list = []
    for b in blocks:
        indices = [t_sub_index[ts] for ts in itertools.combinations(b, t)]
        block_t_indices_list.append(indices)
    
    # Convert to numpy array for fast access
    block_t_indices_arr = np.array(block_t_indices_list, dtype=np.int32)
    
    # Precompute for each t-subset, which blocks cover it
    t_sub_to_blocks = defaultdict(list)
    for b_idx, indices in enumerate(block_t_indices_list):
        for t_idx in indices:
            t_sub_to_blocks[t_idx].append(b_idx)
    
    return blocks, block_t_indices_arr, t_sub_to_blocks, n_blocks, n_t_subs

def construct(v, k, t):
    start_time = time.time()
    blocks, block_t_indices, t_sub_to_blocks, n_blocks, n_t_subs = _precompute_blocks(v, k, t)
    
    # coverage_counts: how many blocks in current solution cover each t-subset
    coverage_counts = np.zeros(n_t_subs, dtype=np.int32)
    # in_solution: boolean mask for blocks
    in_solution = np.zeros(n_blocks, dtype=bool)
    
    solution_blocks = [] # List of block indices
    
    # 1. Greedy Initialization
    uncovered_mask = np.ones(n_t_subs, dtype=bool)
    
    while np.any(uncovered_mask):
        # Vectorized coverage count for all blocks
        idx = block_t_indices
        vals = uncovered_mask[idx]
        sums = vals.sum(axis=1)
        
        best_block_idx = np.argmax(sums)
        if sums[best_block_idx] == 0:
            break
            
        # Add block
        solution_blocks.append(best_block_idx)
        in_solution[best_block_idx] = True
        
        # Update coverage counts and uncovered mask
        t_idxs = block_t_indices[best_block_idx]
        coverage_counts[t_idxs] += 1
        uncovered_mask[t_idxs] = False
        
        if time.time() - start_time > 5:
            break

    # 2. Local Search: Remove and Swap
    best_len = len(solution_blocks)
    best_solution = solution_blocks.copy()
    
    def remove_block(b_idx):
        if b_idx not in solution_blocks: return False
        t_idxs = block_t_indices[b_idx]
        if np.any(coverage_counts[t_idxs] == 1):
            return False
        coverage_counts[t_idxs] -= 1
        in_solution[b_idx] = False
        solution_blocks.remove(b_idx)
        return True

    def swap_block(A_idx, B_idx):
        if B_idx in solution_blocks or A_idx not in solution_blocks:
            return False
        
        t_idxs_A = block_t_indices[A_idx]
        t_idxs_B = block_t_indices[B_idx]
        
        essential_mask = (coverage_counts[t_idxs_A] == 1)
        essential_t_idxs = t_idxs_A[essential_mask]
        
        if len(essential_t_idxs) > 0:
            b_set = set(t_idxs_B.tolist())
            if not all(t in b_set for t in essential_t_idxs):
                return False
        
        coverage_counts[t_idxs_A] -= 1
        in_solution[A_idx] = False
        solution_blocks.remove(A_idx)
        
        coverage_counts[t_idxs_B] += 1
        in_solution[B_idx] = True
        solution_blocks.append(B_idx)
        
        return True

    no_improvement_count = 0
    
    while time.time() - start_time < 110:
        improved = False
        
        # Pass 1: Remove redundant blocks
        for b_idx in solution_blocks[:]:
            if remove_block(b_idx):
                improved = True
                no_improvement_count = 0
                break
        
        if improved:
            if len(solution_blocks) < best_len:
                best_len = len(solution_blocks)
                best_solution = solution_blocks.copy()
            continue
        
        # Pass 2: Try swaps
        sol_copy = solution_blocks[:]
        random.shuffle(sol_copy)
        
        for A_idx in sol_copy:
            if time.time() - start_time > 110:
                break
                
            t_idxs_A = block_t_indices[A_idx]
            essential_mask = (coverage_counts[t_idxs_A] == 1)
            essential_t_idxs = t_idxs_A[essential_mask]
            
            if len(essential_t_idxs) == 0:
                continue
            
            candidates = None
            for t_idx in essential_t_idxs:
                blocks_covering_t = t_sub_to_blocks[t_idx]
                if candidates is None:
                    candidates = set(blocks_covering_t)
                else:
                    candidates.intersection_update(blocks_covering_t)
                
                if not candidates:
                    break
            
            if not candidates:
                continue
            
            candidates = [b for b in candidates if not in_solution[b]]
            
            if not candidates:
                continue
            
            random.shuffle(candidates)
            
            for B_idx in candidates[:5]:
                if swap_block(A_idx, B_idx):
                    improved = True
                    no_improvement_count = 0
                    break
            
            if improved:
                if len(solution_blocks) < best_len:
                    best_len = len(solution_blocks)
                    best_solution = solution_blocks.copy()
                break
        
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > 10:
                break

    final_blocks = [list(blocks[b_idx]) for b_idx in best_solution]
    return final_blocks
