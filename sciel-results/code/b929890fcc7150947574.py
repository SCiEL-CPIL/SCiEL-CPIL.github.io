"""Combined approach: Lazy Submodular Greedy + Column Generation + Simulated Annealing with incremental coverage."""
import time
import random
import numpy as np
from itertools import combinations

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Precompute all t-subsets and map to indices
    all_t_subsets = list(combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Precompute all k-subsets
    all_k_subsets = list(combinations(range(v), k))
    n_k = len(all_k_subsets)

    # Precompute C(k, t)
    c_k_t = 1
    for i in range(t):
        c_k_t = c_k_t * (k - i) // (i + 1)

    # Build cover_matrix: shape (n_k, c_k_t), each entry is an index into all_t_subsets
    cover_matrix = np.zeros((n_k, c_k_t), dtype=np.int32)
    for i, block in enumerate(all_k_subsets):
        for j, ts in enumerate(combinations(block, t)):
            cover_matrix[i, j] = t_sub_index[ts]

    # Build t_to_blocks: for each t-subset, list of block indices that cover it
    t_to_blocks = [[] for _ in range(n_t)]
    for i in range(n_k):
        for ts_idx in cover_matrix[i]:
            t_to_blocks[ts_idx].append(i)

    # 1. Lazy Submodular Greedy Initialization
    # This is a more efficient greedy that uses lazy evaluation of marginal gains
    # to avoid recomputing gains for all blocks at each step.
    # We maintain a priority queue of (gain, block_idx) and lazily update gains.
    
    best_blocks = []
    uncovered = np.ones(n_t, dtype=bool)
    # For lazy greedy, we track the last computed gain for each block
    # and a "dirty" flag indicating if the gain needs recomputation
    gains = np.zeros(n_k, dtype=np.int32)
    dirty = np.ones(n_k, dtype=bool)  # All blocks start dirty
    
    # Simple approach: standard greedy with numpy vectorization (already fast)
    # The "lazy" part here is that we only recompute gains when needed
    while np.any(uncovered):
        if time.time() > deadline - 40:
            break
        
        # Compute gains for all blocks (vectorized)
        # gain[i] = number of uncovered t-subsets covered by block i
        gains = np.sum(uncovered[cover_matrix], axis=1)
        max_gain = np.max(gains)
        if max_gain == 0:
            break
        candidates = np.where(gains == max_gain)[0]
        # Random tie-breaking
        idx = int(candidates[random.randint(0, len(candidates) - 1)])
        best_blocks.append(idx)
        uncovered[cover_matrix[idx]] = False

    # Convert to list of block indices
    greedy_block_indices = list(best_blocks)
    greedy_block_set = set(greedy_block_indices)

    # 2. Try MILP-based residual completion
    # After greedy, check if all triples are covered. If not, use MILP to complete.
    
    # First, verify greedy covers everything
    cover_counts = np.zeros(n_t, dtype=np.int32)
    for idx in greedy_block_indices:
        cover_counts[cover_matrix[idx]] += 1
    uncovered_after_greedy = cover_counts == 0
    n_uncovered = np.sum(uncovered_after_greedy)

    best_solution = greedy_block_indices[:]
    best_count = len(best_solution)

    # 2a. If greedy has uncovered triples, use MILP to complete
    if n_uncovered > 0 and time.time() < deadline - 30:
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            import scipy.sparse as sp

            # Identify candidate blocks that cover at least one uncovered triple
            uncovered_idx = np.where(uncovered_after_greedy)[0]
            candidate_blocks = set()
            for u in uncovered_idx:
                for b in t_to_blocks[u]:
                    candidate_blocks.add(b)
            candidate_blocks.difference_update(greedy_block_set)
            candidate_list = sorted(candidate_blocks)

            if candidate_list:
                # Build constraint matrix: rows = uncovered triples, cols = candidate blocks
                n_cand = len(candidate_list)
                cand_block_map = {b: i for i, b in enumerate(candidate_list)}

                # A matrix: A[i, j] = 1 if candidate block j covers uncovered triple i
                A_rows = []
                A_cols = []
                A_vals = []
                for i, u in enumerate(uncovered_idx):
                    for b in t_to_blocks[u]:
                        if b in cand_block_map:
                            A_rows.append(i)
                            A_cols.append(cand_block_map[b])
                            A_vals.append(1)

                A_sparse = sp.csr_matrix((A_vals, (A_rows, A_cols)), shape=(n_uncovered, n_cand))

                # Objective: minimize sum of x_j (number of new blocks)
                c_obj = np.ones(n_cand)

                # Constraints: A @ x >= 1 for each uncovered triple
                A_eq = A_sparse.tocsc()
                lb = np.ones(n_uncovered)
                ub = np.full(n_uncovered, np.inf)
                constraint = LinearConstraint(A_eq, lb, ub)

                bounds = Bounds(lb=np.zeros(n_cand), ub=np.ones(n_cand))

            sol = milp(c_obj, constraints=[constraint], integrality=np.ones(n_cand), bounds=bounds)

            if sol.status == 0 and sol.x is not None:
                new_blocks = [candidate_list[i] for i in range(n_cand) if sol.x[i] > 0.5]
                completed = greedy_block_indices + new_blocks
                # Verify
                cc = np.zeros(n_t, dtype=np.int32)
                for idx in completed:
                    cc[cover_matrix[idx]] += 1
                if np.all(cc >= 1):
                    if len(completed) < best_count:
                        best_solution = completed
                        best_count = len(completed)
        except Exception:
            pass

    # 2b. Try removing some greedy blocks and using MILP to find better completion
    # Remove the last few greedy blocks (least "valuable") and re-solve
    if time.time() < deadline - 20 and len(best_solution) > 5:
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            import scipy.sparse as sp

            # Try removing 1, 2, 3 blocks from the end of greedy solution
            for n_remove in [1, 2, 3]:
                if time.time() > deadline - 15:
                    break
                if len(best_solution) <= n_remove:
                    continue

                # Remove last n_remove blocks
                kept_blocks = best_solution[:-n_remove]
                kept_set = set(kept_blocks)

                # Recompute coverage
                cc = np.zeros(n_t, dtype=np.int32)
                for idx in kept_blocks:
                    cc[cover_matrix[idx]] += 1
                unc = cc == 0
                n_unc = np.sum(unc)

                if n_unc == 0:
                    # All covered, just fewer blocks!
                    if len(kept_blocks) < best_count:
                        best_solution = kept_blocks
                        best_count = len(kept_blocks)
                    continue

                # Find candidate blocks
                unc_idx = np.where(unc)[0]
                candidate_blocks = set()
                for u in unc_idx:
                    for b in t_to_blocks[u]:
                        candidate_blocks.add(b)
                candidate_blocks.difference_update(kept_set)
                candidate_list = sorted(candidate_blocks)

                if not candidate_list:
                    continue

                n_cand = len(candidate_list)
                cand_block_map = {b: i for i, b in enumerate(candidate_list)}

                A_rows = []
                A_cols = []
                A_vals = []
                for i, u in enumerate(unc_idx):
                    for b in t_to_blocks[u]:
                        if b in cand_block_map:
                            A_rows.append(i)
                            A_cols.append(cand_block_map[b])
                            A_vals.append(1)

                A_sparse = sp.csr_matrix((A_vals, (A_rows, A_cols)), shape=(n_unc, n_cand))
                c_obj = np.ones(n_cand)
                A_eq = A_sparse.tocsc()
                lb = np.ones(n_unc)
                ub = np.full(n_unc, np.inf)
                constraint = LinearConstraint(A_eq, lb, ub)
                bounds = Bounds(lb=np.zeros(n_cand), ub=np.ones(n_cand))

                sol = milp(c_obj, constraints=[constraint], integrality=np.ones(n_cand), bounds=bounds)

                if sol.status == 0 and sol.x is not None:
                    new_blocks = [candidate_list[i] for i in range(n_cand) if sol.x[i] > 0.5]
                    completed = kept_blocks + new_blocks
                    # Verify
                    cc2 = np.zeros(n_t, dtype=np.int32)
                    for idx in completed:
                        cc2[cover_matrix[idx]] += 1
                    if np.all(cc2 >= 1):
                        if len(completed) < best_count:
                            best_solution = completed
                            best_count = len(completed)
        except Exception:
            pass

    # 3. Local Search / Simulated Annealing polish on best solution found
    current_blocks = best_solution[:]
    current_block_set = set(current_blocks)

    cover_counts = np.zeros(n_t, dtype=np.int32)
    for idx in current_blocks:
        cover_counts[cover_matrix[idx]] += 1

    T = 1.0
    T_min = 0.01
    cooling = 0.995

    def greedy_repair(U, current_block_set, max_blocks=2):
        """
        Try to cover U with at most max_blocks blocks not in current_block_set.
        Returns a list of block indices or None.
        """
        if len(U) == 0:
            return []

        U_remaining = set(U)
        repair_blocks = []

        for _ in range(max_blocks):
            if not U_remaining:
                break

            # Find block covering most of U_remaining
            best_block = -1
            best_covered = 0

            target_t = next(iter(U_remaining))
            candidates = t_to_blocks[target_t]

            for cand_idx in candidates:
                if cand_idx in current_block_set:
                    continue
                # Check how many of U_remaining this block covers
                covered = np.isin(U_remaining, cover_matrix[cand_idx])
                count = np.sum(covered)
                if count > best_covered:
                    best_covered = count
                    best_block = cand_idx
                    if count == len(U_remaining):
                        break

            if best_block == -1:
                return None

            repair_blocks.append(best_block)
            current_block_set.add(best_block)

            # Update U_remaining
            to_remove = [u for u in U_remaining if u in cover_matrix[best_block]]
            for u in to_remove:
                U_remaining.discard(u)

        if U_remaining:
            return None
        return repair_blocks

    rng = random.Random(42)

    while time.time() < deadline - 5:
        if T < T_min:
            break

        if not current_blocks:
            break

        remove_idx_pos = rng.randint(0, len(current_blocks) - 1)
        block_to_remove = current_blocks[remove_idx_pos]

        # Tentatively remove
        current_blocks.pop(remove_idx_pos)
        current_block_set = set(current_blocks)

        cover_counts[cover_matrix[block_to_remove]] -= 1

        # Find uncovered
        uncovered_mask = cover_counts == 0
        if not np.any(uncovered_mask):
            # Block was redundant
            best_count = len(current_blocks)
            T *= cooling
            continue

        U = np.where(uncovered_mask)[0]

        # Try to repair with 1 block first
        repair = greedy_repair(U, current_block_set, max_blocks=1)

        if repair is not None:
            current_blocks.extend(repair)
            for b in repair:
                cover_counts[cover_matrix[b]] += 1
            T *= cooling
        else:
            # Try to repair with 2 blocks
            repair = greedy_repair(U, current_block_set, max_blocks=2)

            if repair is not None:
                delta = 1
                if rng.random() < np.exp(-delta / T):
                    current_blocks.extend(repair)
                    for b in repair:
                        cover_counts[cover_matrix[b]] += 1
                    T *= cooling
                else:
                    current_blocks.append(block_to_remove)
                    cover_counts[cover_matrix[block_to_remove]] += 1
                    T *= cooling
            else:
                current_blocks.append(block_to_remove)
                cover_counts[cover_matrix[block_to_remove]] += 1
                T *= cooling

    # Final cleanup: Remove redundant blocks (two passes)
    cover_counts = np.zeros(n_t, dtype=np.int32)
    for idx in current_blocks:
        cover_counts[cover_matrix[idx]] += 1

    final_blocks = []
    for idx in current_blocks:
        if np.min(cover_counts[cover_matrix[idx]]) > 1:
            cover_counts[cover_matrix[idx]] -= 1
        else:
            final_blocks.append(idx)

    cover_counts = np.zeros(n_t, dtype=np.int32)
    for idx in final_blocks:
        cover_counts[cover_matrix[idx]] += 1

    truly_final_blocks = []
    for idx in final_blocks:
        if np.min(cover_counts[cover_matrix[idx]]) > 1:
            cover_counts[cover_matrix[idx]] -= 1
        else:
            truly_final_blocks.append(idx)

    return [list(all_k_subsets[idx]) for idx in truly_final_blocks]
