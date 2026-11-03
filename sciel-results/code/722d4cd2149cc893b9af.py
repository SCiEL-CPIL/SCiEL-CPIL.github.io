import itertools
import random
import numpy as np
import time
import math
from collections import defaultdict

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using Efficient Greedy Initialization 
    and Local Search with 'Remove 2, Add 1' moves.
    """
    start_time = time.time()
    time_limit = 115.0

    # 1. Precomputation
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute block -> list of t-subset indices
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
    # Precompute t-subset -> list of block indices
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for t_idx in indices:
            t_to_blocks[t_idx].append(b_idx)

    # Efficient Greedy Initialization
    def greedy_init():
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        uncovered_set = set(range(num_t_subsets))
        current_blocks = []
        
        while uncovered_set:
            if time.time() - start_time > time_limit * 0.3:
                break # Time out greedy, return partial? No, must be valid.
                # If we break, we must ensure validity. 
                # Better to let it finish or use a faster method.
                # With the counting method, it should be fast.
                
            # Calculate gain for all candidate blocks
            gain_counts = defaultdict(int)
            
            # Iterate over uncovered t-subsets
            # To avoid modifying set while iterating, we use a list or just iterate over the set
            for t_idx in uncovered_set:
                for b_idx in t_to_blocks[t_idx]:
                    gain_counts[b_idx] += 1
            
            if not gain_counts:
                break # Should not happen
                
            # Find block with max gain
            best_b_idx = max(gain_counts, key=gain_counts.get)
            best_gain = gain_counts[best_b_idx]
            
            if best_gain == 0:
                break
                
            current_blocks.append(best_b_idx)
            
            # Update coverage and uncovered_set
            for t_idx in block_t_indices[best_b_idx]:
                coverage[t_idx] += 1
                if coverage[t_idx] == 1:
                    uncovered_set.discard(t_idx)
                    
        # Remove redundancies
        current_blocks = remove_redundant(current_blocks, coverage)
        return current_blocks, coverage

    def remove_redundant(blocks, coverage):
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                b_idx = blocks[i]
                is_redundant = True
                for t_idx in block_t_indices[b_idx]:
                    if coverage[t_idx] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    blocks.pop(i)
                    for t_idx in block_t_indices[b_idx]:
                        coverage[t_idx] -= 1
                    changed = True
                    break
        return blocks, coverage

    best_blocks = None
    best_cost = float('inf')
    
    # Run greedy
    current_blocks, coverage = greedy_init()
    
    # Ensure validity (in case of timeout in greedy)
    # If greedy timed out, we might have uncovered sets.
    # We must finish covering them.
    # Let's check if valid.
    if np.sum(coverage == 0) > 0:
        # Continue greedy to finish
        uncovered_set = set(np.where(coverage == 0)[0])
        while uncovered_set:
            if time.time() - start_time > time_limit * 0.5:
                break
            gain_counts = defaultdict(int)
            for t_idx in uncovered_set:
                for b_idx in t_to_blocks[t_idx]:
                    gain_counts[b_idx] += 1
            if not gain_counts: break
            best_b_idx = max(gain_counts, key=gain_counts.get)
            if best_b_idx in current_blocks:
                 # If best is already there, pick next best?
                 # Or just add it? No, distinct blocks.
                 # If best is in current_blocks, it means it covers some uncovered sets?
                 # But if it's in current_blocks, its t-subsets should be covered?
                 # Unless coverage was updated incorrectly.
                 # Let's just pick the max gain, and if it's in blocks, skip it?
                 # Actually, if it's in blocks, it covers its t-subsets.
                 # Why are they uncovered?
                 # It implies the block was added, but coverage wasn't updated?
                 # Or we are in the "finish" loop after a timeout.
                 pass
                 
            current_blocks.append(best_b_idx)
            for t_idx in block_t_indices[best_b_idx]:
                coverage[t_idx] += 1
                if coverage[t_idx] == 1:
                    uncovered_set.discard(t_idx)
                    
        current_blocks = remove_redundant(current_blocks, coverage)

    if len(current_blocks) < best_cost:
        best_cost = len(current_blocks)
        best_blocks = current_blocks[:]
        
    # Local Search
    for iteration in range(50):
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        
        # 1. Greedy Deletion
        for i in range(len(current_blocks)):
            if time.time() - start_time > time_limit:
                break
            b_idx = current_blocks[i]
            redundant = True
            for t_idx in block_t_indices[b_idx]:
                if coverage[t_idx] == 1:
                    redundant = False
                    break
            if redundant:
                for t_idx in block_t_indices[b_idx]:
                    coverage[t_idx] -= 1
                current_blocks.pop(i)
                improved = True
                break
        
        if improved:
            if len(current_blocks) < best_cost:
                best_cost = len(current_blocks)
                best_blocks = current_blocks[:]
            continue
        
        # 2. Remove 2, Add 1
        for _ in range(50):
            if time.time() - start_time > time_limit:
                break
            if len(current_blocks) < 2:
                break
                
            i1 = random.randint(0, len(current_blocks) - 1)
            i2 = random.randint(0, len(current_blocks) - 1)
            if i1 == i2:
                continue
            if i1 > i2:
                i1, i2 = i2, i1
                
            b1 = current_blocks[i1]
            b2 = current_blocks[i2]
            
            set_b2 = set(block_t_indices[b2])
            U = []
            for t_idx in block_t_indices[b1]:
                if t_idx in set_b2:
                    if coverage[t_idx] == 2:
                        U.append(t_idx)
                else:
                    if coverage[t_idx] == 1:
                        U.append(t_idx)
                        
            set_b1 = set(block_t_indices[b1])
            for t_idx in block_t_indices[b2]:
                if t_idx not in set_b1:
                    if coverage[t_idx] == 1:
                        U.append(t_idx)
            
            if not U:
                for t_idx in block_t_indices[b1]:
                    coverage[t_idx] -= 1
                for t_idx in block_t_indices[b2]:
                    coverage[t_idx] -= 1
                current_blocks.pop(i2)
                current_blocks.pop(i1)
                improved = True
                break
            
            # Find B' covering U
            if U:
                candidates = t_to_blocks[U[0]]
                valid_candidate = -1
                for b_prime in candidates:
                    if b_prime == b1 or b_prime == b2:
                        continue
                    if b_prime in current_blocks:
                        continue
                    
                    covers_all = True
                    # Optimization: convert U to set for fast lookup?
                    # U is small.
                    for t_idx in U:
                        if t_idx not in block_t_indices[b_prime]:
                            covers_all = False
                            break
                    if covers_all:
                        valid_candidate = b_prime
                        break
                        
                if valid_candidate != -1:
                    for t_idx in block_t_indices[b1]:
                        coverage[t_idx] -= 1
                    for t_idx in block_t_indices[b2]:
                        coverage[t_idx] -= 1
                    current_blocks.pop(i2)
                    current_blocks.pop(i1)
                    
                    for t_idx in block_t_indices[valid_candidate]:
                        coverage[t_idx] += 1
                    current_blocks.append(valid_candidate)
                    
                    improved = True
                    if len(current_blocks) < best_cost:
                        best_cost = len(current_blocks)
                        best_blocks = current_blocks[:]
                    break
        
        if not improved:
            pass

    if best_blocks is None:
        current_blocks, coverage = greedy_init()
        best_blocks = current_blocks

    final_blocks = [list(all_k_subsets[idx]) for idx in best_blocks]
    return final_blocks
