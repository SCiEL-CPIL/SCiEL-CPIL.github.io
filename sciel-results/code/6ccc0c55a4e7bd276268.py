import itertools
import numpy as np
import time
import random
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin

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

    best_solution = None
    best_size = float('inf')

    # 2. LP Relaxation to get duals (weights)
    duals = None
    try:
        if time.time() > deadline - 20:
            raise TimeoutError()

        c_lp = np.ones(n_k)
        A_ub = -M.T.tocsr()
        b_ub = -np.ones(n_t)
        bounds_lp = [(0, 1)] * n_k

        res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs', 
                      options={"presolve": True, "time_limit": 30})

        if res.success:
            duals = res.ineqlin.marginals
            duals = np.abs(duals)
            if np.sum(duals) == 0:
                duals = None
    except Exception:
        duals = None

    if duals is None:
        duals = np.ones(n_t)

    # Pre-compute base weights for blocks based on duals
    base_block_weights = np.zeros(n_k)
    for j in range(n_k):
        if subsets_of_block[j]:
            base_block_weights[j] = np.sum(duals[subsets_of_block[j]])

    # 3. Multi-start Vectorized Greedy Search with local refinement
    num_restarts = 500
    rng = np.random.default_rng(42)

    # Precompute: for each block, the list of blocks it conflicts with (shares at least one t-subset)
    # This is expensive, so we'll use it selectively
    # Instead, let's precompute for each block the set of other blocks sharing subsets
    # We'll build this lazily or use the blocks_of_subset structure

    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break

        # Initialize gains with base weights + noise
        if restart == 0:
            gains = base_block_weights.copy()
        else:
            # Adaptive noise: more noise early, less later
            noise_frac = max(0.01, 0.08 * (1 - restart / num_restarts))
            noise_scale = np.max(base_block_weights) * noise_frac
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
            
            for s_idx in newly_covered_indices:
                w_s = duals[s_idx]
                if w_s > 0:
                    for b_prime in blocks_of_subset[s_idx]:
                        gains[b_prime] -= w_s

            if not np.any(uncovered_mask):
                break

        # Check if solution is valid and better
        if len(solution_blocks) < best_size:
            if not np.any(uncovered_mask):
                best_size = len(solution_blocks)
                best_solution = solution_blocks.copy()

    # 4. Fallback if no solution found
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

    # 5. Post-processing: Deletion Pass + Local Search (swap/replace)
    if best_solution is not None and time.time() < deadline - 1:
        # Build cover count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            cover_count[subsets_of_block[b_idx]] += 1

        # Deletion pass
        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            new_solution = []
            for b_idx in best_solution:
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
            best_solution = new_solution
            best_size = len(best_solution)

        # Local search: try to replace one block with another that covers
        # more uncovered subsets (or enables removal of another block)
        # We do multiple rounds of this
        max_local_rounds = 5
        for _ in range(max_local_rounds):
            if time.time() > deadline - 1:
                break
            
            improved = False
            sol_set = set(best_solution)
            
            # For each block in solution, try replacing it with a non-solution block
            # that covers at least one subset not covered by any other block in solution
            # and check if the new solution is still valid
            for pos, b_idx in enumerate(list(best_solution)):
                if time.time() > deadline - 1:
                    break
                
                # Find subsets only covered by b_idx
                unique_subs = [s for s in subsets_of_block[b_idx] if cover_count[s] == 1]
                
                if not unique_subs:
                    # This block is redundant in terms of unique coverage
                    # Try removing it
                    cover_count[subsets_of_block[b_idx]] -= 1
                    best_solution.pop(pos)
                    best_size -= 1
                    improved = True
                    # Restart the loop since indices changed
                    break
                
                # Try to find a replacement block that covers all unique_subs
                # and is not already in the solution
                for cand in range(n_k):
                    if cand in sol_set:
                        continue
                    
                    # Check if cand covers all unique_subs
                    cand_subs_set = set(subsets_of_block[cand])
                    if all(s in cand_subs_set for s in unique_subs):
                        # Try the swap: remove b_idx, add cand
                        # Check validity: after removing b_idx, are all subsets still covered?
                        # Since cand covers unique_subs, and other blocks cover the rest,
                        # we need to verify
                        temp_cover = cover_count.copy()
                        temp_cover[subsets_of_block[b_idx]] -= 1
                        temp_cover[subsets_of_block[cand]] += 1
                        
                        if np.all(temp_cover >= 1):
                            # Valid swap
                            cover_count = temp_cover
                            best_solution[pos] = cand
                            sol_set.discard(b_idx)
                            sol_set.add(cand)
                            improved = True
                            break
                
                if improved:
                    break
            
            if not improved:
                break

    return [list(k_subsets[idx]) for idx in best_solution]
