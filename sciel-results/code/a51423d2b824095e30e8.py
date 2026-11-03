import itertools
import numpy as np
import time
import random
import heapq
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 35s margin out of 150s

    # Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)

    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)

    subsets_of_block = [[] for _ in range(n_k)]
    blocks_of_subset = [[] for _ in range(n_t)]

    rows = []
    cols = []
    data = []

    for j, B in enumerate(k_subsets):
        for S in itertools.combinations(B, t):
            i = t_index[S]
            rows.append(i)
            cols.append(j)
            data.append(1.0)
            subsets_of_block[j].append(i)
            blocks_of_subset[i].append(j)

    # Convert to numpy arrays for faster access in local search
    subsets_of_block_np = [np.array(subs, dtype=np.int32) for subs in subsets_of_block]
    blocks_of_subset_np = [np.array(subs, dtype=np.int32) for subs in blocks_of_subset]

    best_solution = None
    best_size = float('inf')

    # LP Relaxation for duals
    duals = None
    try:
        if time.time() > deadline - 10:
            raise TimeoutError()

        M = csr_matrix((data, (rows, cols)), shape=(n_t, n_k))

        c = np.ones(n_k)
        A_ub = -M.T.tocsr()
        b_ub = -np.ones(n_t)
        bounds = [(0, 1)] * n_k

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if res.success:
            duals = np.abs(res.ineqlin.marginals)
            if np.sum(duals) == 0:
                duals = None
    except Exception:
        duals = None

    if duals is None:
        duals = np.ones(n_t)

    # Multi-restart vectorized greedy with LP duals
    num_restarts = 50  # Increased restarts due to faster greedy

    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break

        # Initialize weights
        block_weights = np.zeros(n_k)
        for j in range(n_k):
            if subsets_of_block[j]:
                block_weights[j] = np.sum(duals[subsets_of_block[j]])
        
        # Add noise for diversity
        if restart > 0:
            noise_scale = 0.01 * (restart + 1)
            block_weights += np.random.uniform(0, noise_scale, n_k)

        covered = np.zeros(n_t, dtype=bool)
        picked = np.zeros(n_k, dtype=bool)
        solution_blocks = []

        while True:
            if time.time() > deadline - 15:
                break

            # Find best block
            max_w = np.max(block_weights)
            if max_w <= 1e-9:
                break

            j = np.argmax(block_weights)
            
            if picked[j]:
                # Should not happen if we set weights to 0 for picked blocks
                continue

            picked[j] = True
            solution_blocks.append(j)

            # Update weights for neighbors
            subs_b = subsets_of_block[j]
            newly_covered = [s for s in subs_b if not covered[s]]
            
            if newly_covered:
                for s_idx in newly_covered:
                    covered[s_idx] = True
                    y_s = duals[s_idx]
                    if y_s > 0:
                        for b_prime in blocks_of_subset[s_idx]:
                            if not picked[b_prime]:
                                block_weights[b_prime] -= y_s

            if np.all(covered):
                break

        if len(solution_blocks) < best_size and np.all(covered):
            best_size = len(solution_blocks)
            best_solution = solution_blocks.copy()

    # Local Search: Deletion and Swap
    if best_solution is not None:
        # Compute cover counts
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            cover_count[subsets_of_block_np[b_idx]] += 1

        # Deletion Pass
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            new_solution = []
            for b_idx in best_solution:
                subs_b = subsets_of_block_np[b_idx]
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

        # Swap Pass
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            new_solution = best_solution.copy()
            solution_set = set(new_solution)

            for i, b_idx in enumerate(best_solution):
                if time.time() > deadline - 2:
                    break

                subs_b = subsets_of_block_np[b_idx]
                unique_subs = [s for s in subs_b if cover_count[s] == 1]

                if not unique_subs:
                    # Can remove
                    cover_count[subs_b] -= 1
                    new_solution.pop(i)
                    improved = True
                    # Update cover count for the rest? No, we'll recompute if needed or continue carefully.
                    # Actually, removing a block changes cover_count for its subsets.
                    # We need to be careful with indices if we pop.
                    # Let's just break and restart the loop with updated state.
                    break
                else:
                    # Find replacement
                    # Intersect blocks_of_subset for unique_subs
                    if len(unique_subs) == 1:
                        candidates = set(blocks_of_subset[unique_subs[0]])
                    else:
                        candidates = set(blocks_of_subset[unique_subs[0]])
                        for s in unique_subs[1:]:
                            candidates &= set(blocks_of_subset[s])
                    
                    # Exclude current solution blocks
                    candidates -= solution_set
                    
                    if candidates:
                        # Pick best candidate
                        # Score: minimize number of subsets in candidate that are already covered by >= 2 blocks
                        # (i.e., minimize redundancy added)
                        best_b_prime = -1
                        best_score = float('inf')
                        
                        for b_prime in candidates:
                            # Optional: limit candidates if too many
                            subs_p = subsets_of_block_np[b_prime]
                            score = np.sum(cover_count[subs_p] >= 2)
                            if score < best_score:
                                best_score = score
                                best_b_prime = b_prime

                        if best_b_prime != -1:
                            # Replace b_idx with best_b_prime
                            cover_count[subs_b] -= 1
                            new_solution[i] = best_b_prime
                            cover_count[subsets_of_block_np[best_b_prime]] += 1
                            solution_set.remove(b_idx)
                            solution_set.add(best_b_prime)
                            improved = True
                            break

            if improved:
                best_solution = new_solution
                best_size = len(best_solution)
                # Recompute cover_count for next iteration of while loop
                cover_count = np.zeros(n_t, dtype=np.int32)
                for b_idx in best_solution:
                    cover_count[subsets_of_block_np[b_idx]] += 1

    # Final deletion pass
    if best_solution is not None and time.time() < deadline - 1:
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            cover_count[subsets_of_block_np[b_idx]] += 1

        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            new_solution = []
            for b_idx in best_solution:
                subs_b = subsets_of_block_np[b_idx]
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

    return [list(k_subsets[idx]) for idx in best_solution]
