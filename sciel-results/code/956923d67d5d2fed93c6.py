import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precomputation
    block_triple_indices = [[] for _ in range(num_blocks)]
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    for i, b in enumerate(all_blocks):
        tris = []
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            tris.append(idx)
            triple_to_blocks[idx].append(i)
        block_triple_indices[i] = tris
        
    # Convert to numpy arrays for vectorized operations
    max_tris = 35
    block_triple_matrix = np.full((num_blocks, max_tris), -1, dtype=np.int32)
    block_triple_lens = np.zeros(num_blocks, dtype=np.int32)
    
    for i, tris in enumerate(block_triple_indices):
        block_triple_matrix[i, :len(tris)] = tris
        block_triple_lens[i] = len(tris)
    
    # Precompute mask and safe matrix for fast gain calculation
    mask = block_triple_matrix != -1
    safe_matrix = np.where(mask, block_triple_matrix, 0)
    
    # Precompute triple_to_blocks as numpy arrays for faster intersection
    triple_to_blocks_arr = []
    for lst in triple_to_blocks:
        triple_to_blocks_arr.append(np.array(lst, dtype=np.int32))
    
    best_sol = None
    best_energy = float('inf')
    
    seed_counter = 0
    
    while time.time() < deadline - 10:
        seed_counter += 1
        rng = random.Random(seed_counter)
        
        # 1. Vectorized Greedy Construction
        sol = _greedy_cover_vectorized(rng, num_blocks, block_triple_matrix, block_triple_lens, num_triples, safe_matrix, mask, deadline - 5)
        
        # 2. Remove Redundant Blocks
        sol = _remove_redundant(sol, block_triple_matrix, block_triple_lens, num_triples)
        
        # 3. Local Search (SA)
        time_left_for_sa = deadline - 5 - time.time()
        if time_left_for_sa < 1:
            break
            
        sol = _local_search(sol, num_blocks, block_triple_matrix, block_triple_lens, triple_to_blocks_arr, num_triples, time_left_for_sa, rng)
        
        # Evaluate
        if sol:
            # Check validity and count
            covered = np.zeros(num_triples, dtype=bool)
            for b_idx in sol:
                tris = block_triple_matrix[b_idx]
                valid_tris = tris[tris != -1]
                if len(valid_tris) > 0:
                    covered[valid_tris] = True
            
            uncovered_count = np.sum(~covered)
            
            if uncovered_count == 0:
                energy = len(sol)
                if energy < best_energy:
                    best_energy = energy
                    best_sol = list(sol)

    if best_sol is None:
        return []
        
    # Convert indices back to point lists
    return [list(all_blocks[i]) for i in best_sol]

def _greedy_cover_vectorized(rng, num_blocks, block_triple_matrix, block_triple_lens, num_triples, safe_matrix, mask, deadline):
    uncovered = np.ones(num_triples, dtype=bool)
    selected = []
    selected_set = set()
    
    while np.any(uncovered) and time.time() < deadline:
        # Calculate gain for all remaining blocks
        vals = uncovered[safe_matrix]
        vals = np.where(mask, vals, 0)
        gains = vals.sum(axis=1)
        
        # Exclude already selected blocks
        if selected_set:
            gains[list(selected_set)] = -1
            
        best_gain = np.max(gains)
        
        if best_gain <= 0:
            break
            
        candidates = np.where(gains == best_gain)[0]
        chosen = int(rng.choice(candidates))
        
        selected.append(chosen)
        selected_set.add(chosen)
        
        # Update uncovered
        tris = block_triple_matrix[chosen]
        valid_tris = tris[tris != -1]
        uncovered[valid_tris] = False
        
    return selected

def _remove_redundant(sol, block_triple_matrix, block_triple_lens, num_triples):
    if not sol:
        return sol
        
    # Count coverage
    counts = np.zeros(num_triples, dtype=np.int32)
    for idx in sol:
        tris = block_triple_matrix[idx]
        valid_tris = tris[tris != -1]
        if len(valid_tris) > 0:
            counts[valid_tris] += 1
            
    sol_set = set(sol)
    changed = True
    while changed:
        changed = False
        for idx in list(sol):
            # Check if removing idx leaves any triple with count 0
            tris = block_triple_matrix[idx]
            valid_tris = tris[tris != -1]
            if len(valid_tris) == 0:
                continue
            if np.any(counts[valid_tris] == 1):
                continue
            # Redundant
            counts[valid_tris] -= 1
            sol.remove(idx)
            sol_set.discard(idx)
            changed = True
            
    return sol

def _local_search(initial_sol, num_blocks, block_triple_matrix, block_triple_lens, triple_to_blocks, num_triples, time_limit, rng):
    sol = list(initial_sol)
    sol_set = set(sol)
    
    # Coverage counts
    counts = np.zeros(num_triples, dtype=np.int32)
    for idx in sol:
        tris = block_triple_matrix[idx]
        valid_tris = tris[tris != -1]
        if len(valid_tris) > 0:
            counts[valid_tris] += 1
            
    current_uncovered = np.sum(counts == 0)
    current_energy = len(sol) + 100 * current_uncovered
    best_sol = list(sol)
    best_energy = current_energy
    
    T = 10.0
    cooling = 0.995
    
    while time.time() < time_limit and T > 0.01:
        move_type = rng.random()
        
        if move_type < 0.4 and sol:
            # Removal
            to_remove = rng.choice(sol)
            tris = block_triple_matrix[to_remove]
            valid_tris = tris[tris != -1]
            
            if len(valid_tris) == 0:
                continue
                
            # Check if any triple in valid_tris has count 1
            min_count_in_block = np.min(counts[valid_tris])
            
            if min_count_in_block > 1:
                # Redundant, always accept
                counts[valid_tris] -= 1
                sol.remove(to_remove)
                sol_set.discard(to_remove)
                current_uncovered = np.sum(counts == 0)
                current_energy = len(sol) + 100 * current_uncovered
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_sol = list(sol)
            else:
                # Not redundant
                # Count how many become 0
                delta_uncovered = np.sum(counts[valid_tris] == 1)
                delta_e = 1 + 100 * delta_uncovered
                
                if rng.random() < np.exp(-delta_e / T):
                    counts[valid_tris] -= 1
                    sol.remove(to_remove)
                    sol_set.discard(to_remove)
                    current_uncovered += delta_uncovered
                    current_energy += delta_e
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_sol = list(sol)

        elif move_type < 0.7 and len(sol) >= 1:
            # Swap: remove one, add one
            to_remove = rng.choice(sol)
            tris_remove = block_triple_matrix[to_remove]
            valid_tris_remove = tris_remove[tris_remove != -1]
            
            if len(valid_tris_remove) == 0:
                continue
                
            # Find triples that will be uncovered
            uncovered_mask = (counts[valid_tris_remove] == 1)
            uncovered_tris = valid_tris_remove[uncovered_mask]
            
            if len(uncovered_tris) == 0:
                continue
            
            # Find a block that covers all uncovered_tris
            candidates = None
            for t_idx in uncovered_tris:
                cand_list = triple_to_blocks[t_idx]
                if candidates is None:
                    candidates = set(cand_list.tolist())
                else:
                    candidates.intersection_update(cand_list.tolist())
                
                candidates.difference_update(sol_set)
                
                if not candidates:
                    break
            
            if candidates:
                to_add = rng.choice(list(candidates))
                
                # Perform swap
                tris_add = block_triple_matrix[to_add]
                valid_tris_add = tris_add[tris_add != -1]
                
                # Update counts
                counts[valid_tris_remove] -= 1
                counts[valid_tris_add] += 1
                
                sol.remove(to_remove)
                sol_set.discard(to_remove)
                sol.append(to_add)
                sol_set.add(to_add)
                
                new_uncovered = np.sum(counts == 0)
                new_energy = len(sol) + 100 * new_uncovered
                
                delta_e = new_energy - current_energy
                if delta_e <= 0 or rng.random() < np.exp(-delta_e / T):
                    current_uncovered = new_uncovered
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_sol = list(sol)
                else:
                    # Reject: Undo swap
                    counts[valid_tris_add] -= 1
                    counts[valid_tris_remove] += 1
                    
                    sol.remove(to_add)
                    sol_set.discard(to_add)
                    sol.append(to_remove)
                    sol_set.add(to_remove)

        elif move_type < 0.9 and len(sol) >= 2:
            # 2-for-1: remove two blocks, add one block that covers their uncovered triples
            idx1, idx2 = rng.sample(sol, 2)
            tris1 = block_triple_matrix[idx1]
            tris2 = block_triple_matrix[idx2]
            valid_tris1 = tris1[tris1 != -1]
            valid_tris2 = tris2[tris2 != -1]
            
            if len(valid_tris1) == 0 or len(valid_tris2) == 0:
                continue
            
            # Combine all triples from both blocks
            all_tris = np.concatenate([valid_tris1, valid_tris2])
            unique_tris = np.unique(all_tris)
            
            # Find triples that will be uncovered (count == 1)
            # A triple is uncovered if it appears in exactly one of the two blocks and has count 1
            # More precisely: after removing both blocks, a triple is uncovered if its count drops to 0
            # count_after = count_before - (1 if in block1 else 0) - (1 if in block2 else 0)
            # We need count_after == 0, i.e., count_before == number of blocks containing it (1 or 2)
            
            in_block1 = np.zeros(num_triples, dtype=bool)
            in_block2 = np.zeros(num_triples, dtype=bool)
            in_block1[valid_tris1] = True
            in_block2[valid_tris2] = True
            
            # Triples that would become uncovered
            would_be_uncovered = (counts == (in_block1.astype(np.int32) + in_block2.astype(np.int32)))
            would_be_uncovered = would_be_uncovered & (in_block1 | in_block2)
            
            uncovered_tris = np.where(would_be_uncovered)[0]
            
            if len(uncovered_tris) == 0:
                # Both blocks are redundant, just remove them
                counts[valid_tris1] -= 1
                counts[valid_tris2] -= 1
                sol.remove(idx1)
                sol.remove(idx2)
                sol_set.discard(idx1)
                sol_set.discard(idx2)
                current_uncovered = np.sum(counts == 0)
                current_energy = len(sol) + 100 * current_uncovered
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_sol = list(sol)
                continue
            
            # Find a block that covers all uncovered_tris
            candidates = None
            for t_idx in uncovered_tris:
                cand_list = triple_to_blocks[t_idx]
                if candidates is None:
                    candidates = set(cand_list.tolist())
                else:
                    candidates.intersection_update(cand_list.tolist())
                
                candidates.difference_update(sol_set)
                
                if not candidates:
                    break
            
            if candidates:
                to_add = rng.choice(list(candidates))
                
                # Perform 2-for-1 swap
                tris_add = block_triple_matrix[to_add]
                valid_tris_add = tris_add[tris_add != -1]
                
                # Update counts
                counts[valid_tris1] -= 1
                counts[valid_tris2] -= 1
                counts[valid_tris_add] += 1
                
                sol.remove(idx1)
                sol.remove(idx2)
                sol_set.discard(idx1)
                sol_set.discard(idx2)
                sol.append(to_add)
                sol_set.add(to_add)
                
                new_uncovered = np.sum(counts == 0)
                new_energy = len(sol) + 100 * new_uncovered
                
                delta_e = new_energy - current_energy
                if delta_e <= 0 or rng.random() < np.exp(-delta_e / T):
                    current_uncovered = new_uncovered
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_sol = list(sol)
                else:
                    # Reject: Undo 2-for-1 swap
                    counts[valid_tris_add] -= 1
                    counts[valid_tris1] += 1
                    counts[valid_tris2] += 1
                    
                    sol.remove(to_add)
                    sol_set.discard(to_add)
                    sol.append(idx1)
                    sol.append(idx2)
                    sol_set.add(idx1)
                    sol_set.add(idx2)

        T *= cooling
        
        if time.time() > time_limit:
            break
            
    return best_sol
