import itertools
import numpy as np
import time
import random
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 105  # 15s margin for safety

    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)

    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)

    # Build data structures for fast access
    subsets_of_block = [[] for _ in range(n_k)]
    blocks_of_subset = [[] for _ in range(n_t)]
    
    rows = []
    cols = []
    data = []

    for j, B in enumerate(k_subsets):
        for S in itertools.combinations(B, t):
            i = t_index[S]
            subsets_of_block[j].append(i)
            blocks_of_subset[i].append(j)
            rows.append(i)
            cols.append(j)
            data.append(1.0)

    M = csr_matrix((data, (rows, cols)), shape=(n_t, n_k))
    M_T = M.T  # CSC matrix, efficient for dot product with dense vector

    # Precompute block sizes (number of t-subsets each block covers)
    block_sizes = np.array([len(subs) for subs in subsets_of_block], dtype=np.int32)

    # 2. Use static duals based on subset frequency (inverse of how many blocks cover each subset)
    subset_freq = np.array([len(blocks) for blocks in blocks_of_subset], dtype=np.float64)
    duals = 1.0 / (subset_freq + 1e-10)
    duals /= np.sum(duals) * n_t  # Normalize so average is ~1

    # Pre-compute base weights for blocks based on duals
    base_block_weights = np.zeros(n_k)
    for j in range(n_k):
        if subsets_of_block[j]:
            base_block_weights[j] = np.sum(duals[subsets_of_block[j]])

    # 3. Multi-start Vectorized Greedy Search with Local Search
    rng = np.random.default_rng(42)
    delta_duals = np.zeros(n_t)

    best_solution = None
    best_size = float('inf')

    # Multiple noise scales for diversity
    noise_scales = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]

    while time.time() < deadline - 30:
        # Cycle through different noise scales
        noise_scale = np.max(base_block_weights) * noise_scales[int(time.time() * 1000) % len(noise_scales)]
        
        gains = base_block_weights + rng.uniform(0, noise_scale, n_k)
        uncovered_mask = np.ones(n_t, dtype=bool)
        solution_blocks = []
        
        while True:
            if time.time() > deadline - 2:
                break

            best_idx = np.argmax(gains)
            
            if gains[best_idx] <= 1e-9:
                if np.any(uncovered_mask):
                    found = False
                    for j in range(n_k):
                        if any(uncovered_mask[s] for s in subsets_of_block[j]):
                            best_idx = j
                            found = True
                            break
                    if not found:
                        break
                else:
                    break

            solution_blocks.append(best_idx)
            
            subs_b = subsets_of_block[best_idx]
            newly_covered_indices = [s for s in subs_b if uncovered_mask[s]]
            
            if not newly_covered_indices:
                continue

            uncovered_mask[newly_covered_indices] = False
            
            delta_duals[:] = 0
            delta_duals[newly_covered_indices] = duals[newly_covered_indices]
            gains -= M_T.dot(delta_duals)

            if not np.any(uncovered_mask):
                break

        if len(solution_blocks) < best_size and not np.any(uncovered_mask):
            best_size = len(solution_blocks)
            best_solution = solution_blocks.copy()
            
            # Local search: 1-for-1 swaps
            if time.time() < deadline - 10:
                best_solution, best_size = _local_search_1for1(
                    best_solution, subsets_of_block, blocks_of_subset, n_t, n_k, 
                    deadline, M_T, duals, base_block_weights
                )
                
                # Local search: 2-for-1 swaps (remove 2 blocks, add 1)
                if time.time() < deadline - 5:
                    best_solution, best_size = _local_search_2for1(
                        best_solution, subsets_of_block, blocks_of_subset, n_t, n_k,
                        deadline, M_T, duals, base_block_weights
                    )
                
                # Deletion pass
                if time.time() < deadline - 3:
                    best_solution, best_size = _remove_redundant_blocks(
                        best_solution, subsets_of_block, n_t
                    )

    # Fallback if no solution found
    if best_solution is None:
        uncovered_mask = np.ones(n_t, dtype=bool)
        solution_blocks = []
        while np.any(uncovered_mask):
            best_idx = -1
            max_cov = -1
            for j in range(n_k):
                cnt = np.sum(uncovered_mask[subsets_of_block[j]])
                if cnt > max_cov:
                    max_cov = cnt
                    best_idx = j
            if best_idx == -1:
                break
            solution_blocks.append(best_idx)
            uncovered_mask[subsets_of_block[best_idx]] = False
        best_solution = solution_blocks
        best_size = len(best_solution)

    return [list(k_subsets[idx]) for idx in best_solution]


def _local_search_1for1(solution, subsets_of_block, blocks_of_subset, n_t, n_k, deadline, M_T, duals, base_weights):
    """Try to replace one block with another to maintain coverage while potentially improving."""
    if time.time() > deadline - 5:
        return solution, len(solution)
    
    # Build cover counts
    cover_count = np.zeros(n_t, dtype=np.int32)
    for b_idx in solution:
        cover_count[subsets_of_block[b_idx]] += 1
    
    solution_set = set(solution)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i, b_idx in enumerate(list(solution)):
            if time.time() > deadline - 5:
                break
            # Try removing b_idx
            subs_b = subsets_of_block[b_idx]
            # Check if any subset is only covered by this block
            if any(cover_count[s] == 1 for s in subs_b):
                continue
            
            # Try to find a replacement block that covers all subsets currently only covered by b_idx
            critical_subs = [s for s in subs_b if cover_count[s] == 1]
            if not critical_subs:
                # Can remove without replacement
                cover_count[subs_b] -= 1
                solution.pop(i)
                improved = True
                continue
            
            # Find blocks that cover all critical subsets
            found_cand = None
            for cand in blocks_of_subset[critical_subs[0]]:
                if cand in solution_set:
                    continue
                # Check if cand covers all critical subsets
                cand_subs = set(subsets_of_block[cand])
                if all(s in cand_subs for s in critical_subs):
                    found_cand = cand
                    break
            
            if found_cand is not None:
                # Replace b_idx with cand
                cover_count[subs_b] -= 1
                cover_count[subsets_of_block[found_cand]] += 1
                solution[i] = found_cand
                solution_set.discard(b_idx)
                solution_set.add(found_cand)
                improved = True
                break  # Restart scan after any change to avoid index issues
    
    return solution, len(solution)


def _local_search_2for1(solution, subsets_of_block, blocks_of_subset, n_t, n_k, deadline, M_T, duals, base_weights):
    """Try to replace two blocks with one block."""
    if time.time() > deadline - 5:
        return solution, len(solution)
    
    cover_count = np.zeros(n_t, dtype=np.int32)
    for b_idx in solution:
        cover_count[subsets_of_block[b_idx]] += 1
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        n = len(solution)
        for i in range(n):
            if time.time() > deadline - 5:
                break
            for j in range(i+1, n):
                if time.time() > deadline - 3:
                    break
                b1, b2 = solution[i], solution[j]
                subs1_set = set(subsets_of_block[b1])
                subs2_set = set(subsets_of_block[b2])
                all_subs = subs1_set | subs2_set
                
                # Check if we can remove both
                can_remove = True
                critical = []
                for s in all_subs:
                    cnt = cover_count[s]
                    in_b1 = s in subs1_set
                    in_b2 = s in subs2_set
                    if in_b1 and in_b2:
                        if cnt <= 2:
                            can_remove = False
                            break
                        if cnt == 2:
                            critical.append(s)
                    elif in_b1 or in_b2:
                        if cnt <= 1:
                            can_remove = False
                            break
                        if cnt == 1:
                            critical.append(s)
                
                if not can_remove:
                    continue
                
                if not critical:
                    # Can remove both without replacement
                    cover_count[subsets_of_block[b1]] -= 1
                    cover_count[subsets_of_block[b2]] -= 1
                    solution.pop(j)
                    solution.pop(i)
                    improved = True
                    break
                else:
                    # Find a block covering all critical subsets
                    found_cand = None
                    for cand in blocks_of_subset[critical[0]]:
                        if cand in solution:
                            continue
                        cand_subs = set(subsets_of_block[cand])
                        if all(s in cand_subs for s in critical):
                            found_cand = cand
                            break
                    
                    if found_cand is not None:
                        cover_count[subsets_of_block[b1]] -= 1
                        cover_count[subsets_of_block[b2]] -= 1
                        cover_count[subsets_of_block[found_cand]] += 1
                        solution.pop(j)
                        solution[i] = found_cand
                        improved = True
                        break
            if improved:
                break
    
    return solution, len(solution)


def _remove_redundant_blocks(solution, subsets_of_block, n_t):
    """Remove blocks that are completely redundant."""
    cover_count = np.zeros(n_t, dtype=np.int32)
    for b_idx in solution:
        cover_count[subsets_of_block[b_idx]] += 1
    
    changed = True
    while changed:
        changed = False
        new_solution = []
        for b_idx in solution:
            subs_b = subsets_of_block[b_idx]
            can_remove = True
            for s_idx in subs_b:
                if cover_count[s_idx] == 1:
                    can_remove = False
                    break
            if can_remove:
                cover_count[subs_b] -= 1
                changed = True
            else:
                new_solution.append(b_idx)
        solution = new_solution
    
    return solution, len(solution)
