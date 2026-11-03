import itertools
import numpy as np
import time
import random

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
            
    # 4. Calculate scarcity weights: how many k-blocks cover each t-subset
    t_counts = np.bincount(block_t_indices.ravel(), minlength=n_t)
    t_counts[t_counts == 0] = 1
    # Scarcity weight: rarer t-subsets get higher priority
    t_scarcity = 1.0 / t_counts.astype(np.float64)
    
    # 5. Build sub_to_blocks for local search
    sub_to_blocks = [[] for _ in range(n_t)]
    for b_idx in range(n_k):
        for s_idx in block_t_indices[b_idx]:
            sub_to_blocks[s_idx].append(b_idx)
    
    # 6. Greedy Set Cover with scarcity-aware weighting
    uncovered = np.ones(n_t, dtype=bool)
    selected_blocks = []
    rng = random.Random(42)
    
    while True:
        if time.time() > deadline - 30.0:
            break
        if not uncovered.any():
            break
            
        # Calculate weighted gains: prioritize covering scarce subsets
        gains = (uncovered[block_t_indices] * t_scarcity[block_t_indices]).sum(axis=1)
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        candidates = np.where(gains == max_gain)[0]
        best_idx = rng.choice(candidates)
        selected_blocks.append(best_idx)
        uncovered[block_t_indices[best_idx]] = False
        
    if not selected_blocks:
        return []
        
    best_solution = [k_subsets[i] for i in selected_blocks]
    best_size = len(selected_blocks)
    
    # 7. Simulated Annealing Optimization
    def simulated_annealing(selected_indices, deadline):
        if not selected_indices:
            return [], 0
            
        # Build coverage counts
        coverage_count = np.zeros(n_t, dtype=np.int32)
        for bi in selected_indices:
            coverage_count[block_t_indices[bi]] += 1
            
        current_size = len(selected_indices)
        best_indices = list(selected_indices)
        best_size_sa = current_size
        
        # Temperature schedule
        initial_temp = 1.0
        cooling_rate = 0.98
        min_temp = 0.01
        temp = initial_temp
        
        no_improve_count = 0
        max_no_improve = 5000
        
        while temp > min_temp and time.time() < deadline - 5.0:
            # Choose a random move: add, remove, or swap
            move_type = rng.random()
            
            if move_type < 0.4:
                # Try to remove a block
                if len(selected_indices) <= 1:
                    temp *= cooling_rate
                    continue
                    
                bi = rng.choice(selected_indices)
                ts = block_t_indices[bi]
                
                # Check if removal is valid (all subsets still covered)
                if np.all(coverage_count[ts] > 1):
                    # Valid removal
                    coverage_count[ts] -= 1
                    selected_indices.remove(bi)
                    current_size -= 1
                    temp *= cooling_rate
                    
                    if current_size < best_size_sa:
                        best_size_sa = current_size
                        best_indices = list(selected_indices)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    # Invalid removal: accept with probability exp(-delta/T)
                    # delta = number of subsets that would become uncovered
                    uncovered_count = np.sum(coverage_count[ts] == 1)
                    delta = uncovered_count
                    if delta == 0:
                        # Should have been valid
                        coverage_count[ts] -= 1
                        selected_indices.remove(bi)
                        current_size -= 1
                        temp *= cooling_rate
                        if current_size < best_size_sa:
                            best_size_sa = current_size
                            best_indices = list(selected_indices)
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                    else:
                        # Accept with probability exp(-delta/temp)
                        if rng.random() < np.exp(-delta / max(temp, 1e-10)):
                            coverage_count[ts] -= 1
                            selected_indices.remove(bi)
                            current_size -= 1
                            temp *= cooling_rate
                            no_improve_count += 1
                        else:
                            temp *= cooling_rate
            elif move_type < 0.7:
                # Try to add a block
                if len(selected_indices) >= n_k:
                    temp *= cooling_rate
                    continue
                    
                # Find a random block not in selected
                candidate = rng.randint(0, n_k - 1)
                if candidate in selected_indices:
                    temp *= cooling_rate
                    continue
                    
                ts = block_t_indices[candidate]
                
                # Calculate how many new subsets this block covers
                new_coverage = np.sum(coverage_count[ts] == 0)
                
                # Adding a block increases size by 1, so delta = 1 - new_coverage
                # We accept if it helps (new_coverage > 0) or with some probability
                delta = 1 - new_coverage  # Change in "cost" (size)
                
                if delta <= 0:
                    # Beneficial or neutral
                    coverage_count[ts] += 1
                    selected_indices.append(candidate)
                    current_size += 1
                    temp *= cooling_rate
                    no_improve_count += 1
                else:
                    # Detrimental: accept with probability exp(-delta/T)
                    if rng.random() < np.exp(-delta / max(temp, 1e-10)):
                        coverage_count[ts] += 1
                        selected_indices.append(candidate)
                        current_size += 1
                        temp *= cooling_rate
                        no_improve_count += 1
                    else:
                        temp *= cooling_rate
            else:
                # Try a swap: remove one block, add another
                if len(selected_indices) < 2:
                    temp *= cooling_rate
                    continue
                    
                bi = rng.choice(selected_indices)
                ts_b = block_t_indices[bi]
                
                # Find critical subsets (those with count == 1)
                critical_mask = (coverage_count[ts_b] == 1)
                critical_subsets = ts_b[critical_mask]
                
                if len(critical_subsets) == 0:
                    # Block is redundant, just remove it
                    coverage_count[ts_b] -= 1
                    selected_indices.remove(bi)
                    current_size -= 1
                    temp *= cooling_rate
                    if current_size < best_size_sa:
                        best_size_sa = current_size
                        best_indices = list(selected_indices)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                elif len(critical_subsets) <= 3:
                    # Find candidate blocks that cover all critical subsets
                    cands_lists = [sub_to_blocks[s] for s in critical_subsets]
                    cands_lists.sort(key=len)
                    
                    if cands_lists:
                        candidates = set(cands_lists[0])
                        for lst in cands_lists[1:]:
                            if not candidates:
                                break
                            candidates = candidates.intersection(lst)
                        
                        candidates.discard(bi)
                        
                        if candidates:
                            cand_list = list(candidates)
                            if len(cand_list) > 20:
                                cand_list = rng.sample(cand_list, 20)
                            
                            bc = rng.choice(cand_list)
                            ts_bc = block_t_indices[bc]
                            
                            # Perform swap
                            coverage_count[ts_b] -= 1
                            coverage_count[ts_bc] += 1
                            selected_indices[bi] = bc
                            temp *= cooling_rate
                            no_improve_count += 1
                        else:
                            temp *= cooling_rate
                    else:
                        temp *= cooling_rate
                else:
                    temp *= cooling_rate
            
            # Check for stagnation
            if no_improve_count > max_no_improve:
                # Restart with best solution
                selected_indices = list(best_indices)
                coverage_count = np.zeros(n_t, dtype=np.int32)
                for bi in selected_indices:
                    coverage_count[block_t_indices[bi]] += 1
                current_size = len(selected_indices)
                no_improve_count = 0
                temp = max(initial_temp * 0.5, min_temp)
            
            # Cool down
            temp *= cooling_rate
        
        return best_indices, best_size_sa
    
    # Run simulated annealing
    current_indices = list(selected_blocks)
    current_indices, current_size = simulated_annealing(current_indices, deadline)
    
    if current_size < best_size:
        best_size = current_size
        best_solution = [k_subsets[i] for i in current_indices]
    
    # If we have time, try another round with different initial temperature
    if time.time() < deadline - 15.0:
        current_indices, current_size = simulated_annealing(current_indices, deadline)
        if current_size < best_size:
            best_size = current_size
            best_solution = [k_subsets[i] for i in current_indices]
    
    if best_solution is None:
        return []
        
    return [list(b) for b in best_solution]
