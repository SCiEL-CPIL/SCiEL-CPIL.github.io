import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # Generate all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each block, precompute the list of triple indices it covers
    block_triple_indices = []
    for b in all_blocks:
        tris = [triple_to_idx[tri] for tri in itertools.combinations(b, t)]
        block_triple_indices.append(tris)
    
    # 2. Fast Greedy Construction using bitsets
    # We use Python's large integers as bitsets for speed
    # bit j is set if triple j is covered
    
    # Precompute block bitsets
    block_bitsets = [0] * num_blocks
    for i, tris in enumerate(block_triple_indices):
        bs = 0
        for ti in tris:
            bs |= (1 << ti)
        block_bitsets[i] = bs
    
    uncovered = (1 << num_triples) - 1
    selected = set()
    selected_blocks = []
    
    while uncovered:
        if time.time() > deadline - 40:
            break
        best_gain = -1
        best_idx = -1
        for i in range(num_blocks):
            if i in selected:
                continue
            gain = (block_bitsets[i] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1:
            break
        selected.add(best_idx)
        selected_blocks.append(best_idx)
        uncovered &= ~block_bitsets[best_idx]
    
    # 3. Simulated Annealing
    # State: set of selected block indices
    # Energy: num_blocks + penalty * uncovered_count
    
    # Maintain counts of how many selected blocks cover each triple
    counts = [0] * num_triples
    for idx in selected_blocks:
        for ti in block_triple_indices[idx]:
            counts[ti] += 1
    
    uncovered_count = sum(1 for c in counts if c == 0)
    penalty = 10
    current_energy = len(selected_blocks) + penalty * uncovered_count
    
    best_energy = current_energy
    best_solution = selected_blocks.copy()
    
    T = 50.0
    cooling_rate = 0.999
    T_min = 0.1
    
    while T > T_min and time.time() < deadline - 5:
        move_type = random.random()
        
        if move_type < 0.3:  # Remove
            if not selected_blocks:
                T *= cooling_rate
                continue
            idx = random.choice(selected_blocks)
            # Check if removal is safe
            can_remove = True
            for ti in block_triple_indices[idx]:
                if counts[ti] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Apply
                selected_blocks.remove(idx)
                selected.remove(idx)
                for ti in block_triple_indices[idx]:
                    counts[ti] -= 1
                
                new_uc = sum(1 for c in counts if c == 0)
                new_energy = len(selected_blocks) + penalty * new_uc
                delta_e = new_energy - current_energy
                
                if delta_e < 0 or random.random() < pow(2.71828, -delta_e / T):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_solution = selected_blocks.copy()
                else:
                    # Revert
                    selected_blocks.append(idx)
                    selected.add(idx)
                    for ti in block_triple_indices[idx]:
                        counts[ti] += 1
        
        elif move_type < 0.6:  # Add
            idx = random.randint(0, num_blocks - 1)
            if idx in selected:
                T *= cooling_rate
                continue
            
            # Apply
            selected_blocks.append(idx)
            selected.add(idx)
            for ti in block_triple_indices[idx]:
                counts[ti] += 1
            
            new_uc = sum(1 for c in counts if c == 0)
            new_energy = len(selected_blocks) + penalty * new_uc
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or random.random() < pow(2.71828, -delta_e / T):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_solution = selected_blocks.copy()
            else:
                # Revert
                selected_blocks.remove(idx)
                selected.remove(idx)
                for ti in block_triple_indices[idx]:
                    counts[ti] -= 1
        
        else:  # Swap
            if not selected_blocks:
                T *= cooling_rate
                continue
            idx_out = random.choice(selected_blocks)
            idx_in = random.randint(0, num_blocks - 1)
            if idx_in in selected or idx_in == idx_out:
                T *= cooling_rate
                continue
            
            # Apply swap
            for ti in block_triple_indices[idx_out]:
                counts[ti] -= 1
            for ti in block_triple_indices[idx_in]:
                counts[ti] += 1
            
            new_uc = sum(1 for c in counts if c == 0)
            new_energy = len(selected_blocks) + penalty * new_uc  # Size unchanged
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or random.random() < pow(2.71828, -delta_e / T):
                current_energy = new_energy
                # Commit swap
                selected_blocks.remove(idx_out)
                selected.add(idx_in)
                selected_blocks.append(idx_in)
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_solution = selected_blocks.copy()
            else:
                # Revert
                for ti in block_triple_indices[idx_in]:
                    counts[ti] -= 1
                for ti in block_triple_indices[idx_out]:
                    counts[ti] += 1
        
        T *= cooling_rate
    
    # 4. Final Local Search on Best Solution
    selected_blocks = best_solution.copy()
    selected = set(selected_blocks)
    
    # Recalculate counts
    counts = [0] * num_triples
    for idx in selected_blocks:
        for ti in block_triple_indices[idx]:
            counts[ti] += 1
    
    # Pass 1: Remove Redundant Blocks
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        to_remove = []
        for idx in selected_blocks:
            redundant = True
            for ti in block_triple_indices[idx]:
                if counts[ti] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(idx)
        
        if to_remove:
            for idx in to_remove:
                selected_blocks.remove(idx)
                selected.remove(idx)
                for ti in block_triple_indices[idx]:
                    counts[ti] -= 1
            improved = True
    
    # Pass 2: Try 1-swap improvements to reduce size
    # If we can swap out a block for one that covers all its unique triples,
    # but that's not possible since size stays same. 
    # Instead, we already removed redundant blocks.
    
    final_blocks = [all_blocks[idx] for idx in selected_blocks]
    
    return final_blocks
