import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s buffer

    # 1. Enumerate all t-subsets and assign indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    full_mask = (1 << num_t_subsets) - 1

    # 2. Enumerate all k-subsets and compute their coverage bitmask
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    # masks[i] is an integer bitmask where bit j is 1 if k_subset i covers t_subset j
    # Also store the list of t-subset indices covered by each k-subset for local search
    masks = [0] * num_k_subsets
    covered_indices = [[] for _ in range(num_k_subsets)]
    
    for i, ks in enumerate(k_subsets):
        mask = 0
        for ts in itertools.combinations(ks, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
            covered_indices[i].append(idx)
        masks[i] = mask

    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Greedy + Local Search with multiple restarts
    while time.time() < deadline - 5.0:
        uncovered = full_mask
        current_indices = []
        used_blocks = set()
        
        # Greedy Phase
        while uncovered:
            if time.time() > deadline:
                break
                
            max_gain = 0
            candidates = []
            
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                gain = popcount(masks[i] & uncovered)
                if gain > max_gain:
                    max_gain = gain
                    candidates = [i]
                elif gain == max_gain and gain > 0:
                    candidates.append(i)
            
            if max_gain == 0:
                break
                
            chosen_idx = random.choice(candidates)
            current_indices.append(chosen_idx)
            used_blocks.add(chosen_idx)
            uncovered &= ~masks[chosen_idx]
            
        if uncovered != 0:
            continue # Failed to cover all
            
        # Local Search Phase: Remove redundant blocks
        # Maintain coverage counts
        cover_counts = [0] * num_t_subsets
        for idx in current_indices:
            for j in covered_indices[idx]:
                cover_counts[j] += 1
                
        improved_indices = list(current_indices)
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            new_indices = []
            for idx in improved_indices:
                # Check if any t-subset covered by this block is covered ONLY by this block
                is_essential = False
                for j in covered_indices[idx]:
                    if cover_counts[j] == 1:
                        is_essential = True
                        break
                
                if is_essential:
                    new_indices.append(idx)
                else:
                    # Remove this block, update counts
                    for j in covered_indices[idx]:
                        cover_counts[j] -= 1
                    changed = True
                    
            improved_indices = new_indices
            
        if len(improved_indices) < best_count:
            best_count = len(improved_indices)
            best_blocks = [list(k_subsets[i]) for i in improved_indices]
            
    if best_blocks is None:
        # Fallback: Deterministic greedy
        uncovered = full_mask
        current_indices = []
        used_blocks = set()
        while uncovered:
            max_gain = 0
            best_idx = -1
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                gain = popcount(masks[i] & uncovered)
                if gain > max_gain:
                    max_gain = gain
                    best_idx = i
            if max_gain == 0:
                break
            current_indices.append(best_idx)
            used_blocks.add(best_idx)
            uncovered &= ~masks[best_idx]
        best_blocks = [list(k_subsets[i]) for i in current_indices]
        
    return best_blocks
