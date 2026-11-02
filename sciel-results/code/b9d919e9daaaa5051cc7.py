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
    # subsets_of_block[j]: indices of t-subsets contained in k-subset j
    # blocks_of_subset[i]: indices of k-subsets containing t-subset i
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

    # Convert to numpy arrays for vectorized operations where possible
    # Note: subsets_of_block is a list of lists of different lengths, so we keep it as list of lists
    # but we can use numpy for the 'gains' and 'uncovered' arrays.
    
    M = csr_matrix((data, (rows, cols)), shape=(n_t, n_k))

    best_solution = None
    best_size = float('inf')

    # 2. LP Relaxation to get duals (weights)
    # This provides a strong initial heuristic for the greedy search.
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
            # Dual variables are associated with the constraints M.T @ x >= 1
            # In linprog, for A_ub @ x <= b_ub, the marginals are for the constraints.
            # Our constraint is -M.T @ x <= -1.
            duals = res.ineqlin.marginals
            duals = np.abs(duals)
            if np.sum(duals) == 0:
                duals = None
    except Exception:
        duals = None

    if duals is None:
        duals = np.ones(n_t)

    # Pre-compute base weights for blocks based on duals
    # Weight of a block is the sum of duals of the subsets it covers.
    # We will use this as the initial 'gain' in the greedy step.
    base_block_weights = np.zeros(n_k)
    for j in range(n_k):
        if subsets_of_block[j]:
            base_block_weights[j] = np.sum(duals[subsets_of_block[j]])

    # 3. Multi-start Vectorized Greedy Search
    # We use the LP duals to guide the greedy selection.
    # As we pick blocks, we reduce the weight of subsets they cover, 
    # effectively lowering the gain of blocks that share those subsets.
    
    num_restarts = 500 # We can afford many restarts now
    rng = np.random.default_rng(42)

    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break

        # Initialize gains with base weights + noise
        # The noise helps escape symmetry and find different local optima
        if restart == 0:
            gains = base_block_weights.copy()
        else:
            # Add noise proportional to the scale of weights
            noise_scale = np.max(base_block_weights) * 0.05
            gains = base_block_weights + rng.uniform(0, noise_scale, n_k)

        uncovered_mask = np.ones(n_t, dtype=bool)
        solution_blocks = []
        
        # Track current coverage count to allow for "deletion" logic during greedy?
        # No, standard greedy is pick-and-go. We'll do deletion at the end.
        
        while True:
            if time.time() > deadline - 2:
                break

            # Find the block with the highest gain
            # Gain is the sum of duals of *uncovered* subsets in the block
            # Wait, the standard greedy for set cover uses "number of uncovered elements covered".
            # Here, we are using a weighted version.
            # We need to calculate the current gain for each block:
            # gain[j] = sum(duals[s] for s in subsets_of_block[j] if uncovered[s])
            
            # Doing this from scratch every step is O(N_k * C(k,t)) which is too slow.
            # We must update gains incrementally.
            
            # Initial gain is base_block_weights (sum of all duals).
            # When we pick a block j, we mark its subsets as covered.
            # For every other block j' that shares a subset s with j, 
            # we should subtract duals[s] from gains[j'].
            
            # Let's implement this incremental update.
            
            best_idx = np.argmax(gains)
            
            if gains[best_idx] <= 1e-9:
                # If max gain is ~0, we are stuck or done.
                # If not all covered, we need a fallback.
                if np.any(uncovered_mask):
                    # Fallback: pick any block that covers at least one uncovered subset
                    # This can happen if weights are very small or due to floating point issues
                    # Or if we ran out of "good" blocks.
                    # In a valid set cover, there is always a block covering an uncovered set.
                    # We just pick the first one found.
                    found = False
                    for j in range(n_k):
                        # Check if block j covers any uncovered subset
                        # This is slow, so only do it if necessary
                        # Actually, if gains[best_idx] <= 0, it means no block has positive weight.
                        # This implies all subsets are covered (if weights > 0) or weights are 0.
                        # If duals are positive, and uncovered exists, some block MUST have positive gain.
                        # Unless we subtracted too much? No, we only subtract when covering.
                        # If we covered all subsets, gains should be 0.
                        # If not all covered, and gains are 0, it's a bug or duals were 0.
                        # Let's force a pick of a random block that covers something.
                        for j in range(n_k):
                            if any(uncovered_mask[s] for s in subsets_of_block[j]):
                                best_idx = j
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        break
                else:
                    break

            # Pick best_idx
            solution_blocks.append(best_idx)
            
            # Update uncovered_mask
            subs_b = subsets_of_block[best_idx]
            newly_covered_indices = [s for s in subs_b if uncovered_mask[s]]
            
            if not newly_covered_indices:
                # Should not happen if gain > 0
                continue

            uncovered_mask[newly_covered_indices] = False
            
            # Update gains for all blocks sharing these subsets
            # For each newly covered subset s, subtract duals[s] from all blocks containing s
            for s_idx in newly_covered_indices:
                w_s = duals[s_idx]
                if w_s > 0:
                    for b_prime in blocks_of_subset[s_idx]:
                        gains[b_prime] -= w_s

            # Optimization: If all covered, break
            if not np.any(uncovered_mask):
                break

        # Check if solution is valid and better
        if len(solution_blocks) < best_size:
            # Verify coverage just in case
            # We can check if uncovered_mask is all False
            if not np.any(uncovered_mask):
                best_size = len(solution_blocks)
                best_solution = solution_blocks.copy()
                
                # Early exit if we hit a very good score?
                # 151 is the target. 150 would be a breakthrough.
                # If we find 151, we might continue to try for 150.
                # If we find 145, we can stop.
                if best_size <= 151:
                    # Keep searching for a while
                    pass

    # 4. Fallback if no solution found (should not happen)
    if best_solution is None:
        # Simple greedy without weights
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

    # 5. Post-processing: Deletion Pass
    # Try to remove blocks that are redundant.
    if best_solution is not None and time.time() < deadline - 1:
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            cover_count[subsets_of_block[b_idx]] += 1

        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            new_solution = []
            for b_idx in best_solution:
                subs_b = subsets_of_block[b_idx]
                can_remove = True
                # If any subset in this block is covered ONLY by this block, we can't remove it
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
