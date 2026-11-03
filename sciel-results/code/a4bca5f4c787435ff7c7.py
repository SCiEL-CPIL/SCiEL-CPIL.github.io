import itertools
import numpy as np
import time
import random
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin

    # Precompute t-subsets and k-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}

    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)

    # For each k-subset (block), compute the bitmask of t-subsets it covers
    n_words = (n_t + 63) // 64

    block_masks = [np.zeros(n_words, dtype=np.int64) for _ in range(n_k)]
    block_tlist = [[] for _ in range(n_k)]

    for j, B in enumerate(k_subsets):
        for S in itertools.combinations(B, t):
            i = t_index[S]
            word = i // 64
            bit = i % 64
            block_masks[j][word] |= (np.int64(1) << np.int64(bit))
            block_tlist[j].append(i)

    # Precompute for each t-subset, which blocks cover it
    t_to_blocks = [[] for _ in range(n_t)]
    for j in range(n_k):
        for i in block_tlist[j]:
            t_to_blocks[i].append(j)

    # Incidence matrix as sparse-like structure: for each block, list of t-subsets
    # And for each t-subset, list of blocks

    def popcount_mask(m):
        """Count bits set in a multi-word bitmask."""
        total = 0
        for w in range(n_words):
            total += np.bit_count(np.uint64(m[w]))
        return total

    def compute_cover_counts(sol):
        """Compute cover counts for each t-subset given a solution (list of block indices)."""
        counts = np.zeros(n_t, dtype=np.int32)
        for b in sol:
            for i in block_tlist[b]:
                counts[i] += 1
        return counts

    # Phase 1: Frank-Wolfe (Mirror Descent) on the LP relaxation of set cover
    # Minimize sum_j x_j subject to sum_{j: i in block_j} x_j >= 1 for all i, x_j >= 0
    # FW: at each step, find s = argmin_j c_j where c_j = sum_i w_i * A_{ij}
    #     where w_i = max(0, 1 - sum_j x_j * A_{ij}) are dual variables (uncoveredness)
    #     Then x_new = (1-alpha)*x + alpha*e_s, alpha = 2/(t+2) for step t

    def frank_wolfe_round(n_fw_iters=50):
        """Run Frank-Wolfe to get a fractional solution, then threshold."""
        x = np.zeros(n_k, dtype=np.float64)
        
        # Weights w[i] = how much t-subset i is still uncovered (dual variable)
        w = np.ones(n_t, dtype=np.float64)
        
        # Precompute: for each block j, the list of t-subset indices (already have block_tlist)
        # c_j = sum_i w[i] * A[i,j] = sum of w[i] over i in block_tlist[j]
        
        for fw_step in range(n_fw_iters):
            if time.time() > deadline - 45:
                break
            
            # Linear oracle: find j minimizing c_j = sum_i w[i] * A[i,j]
            # c_j = sum of w[i] for i in block_tlist[j]
            # Compute all c_j efficiently
            # c_j = sum_{i in block_tlist[j]} w[i]
            # We can compute this by: for each i, add w[i] to all j in t_to_blocks[i]
            c = np.zeros(n_k, dtype=np.float64)
            for i in range(n_t):
                wi = w[i]
                if wi > 0:
                    for j in t_to_blocks[i]:
                        c[j] += wi
            
            s = int(np.argmin(c))
            
            # Frank-Wolfe step size
            alpha = 2.0 / (fw_step + 2)
            
            # Update x: x = (1-alpha)*x + alpha*e_s
            x *= (1 - alpha)
            x[s] += alpha
            
            # Update dual variables w[i] = max(0, 1 - sum_j x[j]*A[i,j])
            # sum_j x[j]*A[i,j] = sum of x[j] for j in t_to_blocks[i]
            for i in range(n_t):
                cov = 0.0
                for j in t_to_blocks[i]:
                    cov += x[j]
                w[i] = max(0.0, 1.0 - cov)
            
            # Check convergence: if all w[i] == 0, we have a feasible fractional solution
            if np.max(w) < 1e-10:
                break
        
        return x

    def threshold_fractional(x, n_thresholds=20):
        """Convert fractional solution to integer via randomized thresholding."""
        best_sol = None
        best_size = float('inf')
        
        for trial in range(n_thresholds):
            if time.time() > deadline - 40:
                break
            
            # Random threshold: include block j with probability x[j]
            # Or: use a random threshold tau, include j if x[j] >= tau
            # Better: randomized rounding - include j with prob x[j], but ensure coverage
            sol = []
            in_sol = np.zeros(n_k, dtype=bool)
            
            # First pass: include blocks with x[j] >= 0.5
            for j in range(n_k):
                if x[j] >= 0.5:
                    sol.append(j)
                    in_sol[j] = True
            
            # Check coverage
            cover_counts = np.zeros(n_t, dtype=np.int32)
            for j in sol:
                for i in block_tlist[j]:
                    cover_counts[i] += 1
            
            # For uncovered t-subsets, add blocks with highest x[j] that cover them
            for i in range(n_t):
                if cover_counts[i] == 0:
                    # Find best block covering i not already in solution
                    best_j = -1
                    best_x = -1
                    for j in t_to_blocks[i]:
                        if not in_sol[j] and x[j] > best_x:
                            best_x = x[j]
                            best_j = j
                    if best_j != -1:
                        sol.append(best_j)
                        in_sol[best_j] = True
                        for ii in block_tlist[best_j]:
                            cover_counts[ii] += 1
            
            # Verify full coverage
            if np.all(cover_counts >= 1):
                if len(sol) < best_size:
                    best_size = len(sol)
                    best_sol = sol.copy()
        
        if best_sol is None:
            # Fallback: greedy from scratch
            best_sol = greedy_fallback()
        
        return best_sol

    def greedy_fallback():
        """Simple greedy as fallback."""
        covered = np.zeros(n_words, dtype=np.int64)
        solution = []
        remaining = n_t
        picked = np.zeros(n_k, dtype=bool)
        gain = np.zeros(n_k, dtype=np.int32)
        for j in range(n_k):
            gain[j] = len(block_tlist[j])

        while remaining > 0:
            if time.time() > deadline - 35:
                break
            best_j = -1
            best_gain = -1
            for j in range(n_k):
                if not picked[j] and gain[j] > best_gain:
                    best_gain = gain[j]
                    best_j = j
            if best_j == -1 or best_gain <= 0:
                break
            picked[best_j] = True
            solution.append(best_j)
            for i in block_tlist[best_j]:
                if not (covered[i // 64] >> (i % 64)) & 1:
                    covered[i // 64] |= (np.int64(1) << np.int64(i % 64))
                    remaining -= 1
                    for b in t_to_blocks[i]:
                        if not picked[b]:
                            gain[b] -= 1
        return solution

    # Run Frank-Wolfe with different numbers of iterations and threshold trials
    best_solution = None
    best_size = float('inf')

    random.seed(42)
    
    # Try FW with varying iteration counts
    fw_configs = [(30, 15), (50, 20), (80, 10)]
    
    for n_fw, n_thresh in fw_configs:
        if time.time() > deadline - 30:
            break
        
        x = frank_wolfe_round(n_fw)
        sol = threshold_fractional(x, n_thresh)
        
        if sol is not None and len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol.copy()
    
    # Also try a plain greedy as baseline
    if time.time() < deadline - 25:
        sol_greedy = greedy_fallback()
        if len(sol_greedy) < best_size:
            best_size = len(sol_greedy)
            best_solution = sol_greedy.copy()

    if best_solution is None:
        return [list(k_subsets[i]) for i in range(min(151, n_k))]

    # Phase 2: Local search with bitset-accelerated swaps and deletions
    current_solution = best_solution.copy()
    current_size = len(current_solution)
    best_sa_solution = current_solution.copy()
    best_sa_size = current_size

    cover_counts = compute_cover_counts(current_solution)
    
    # Build a set for fast membership check
    in_solution = set(current_solution)
    
    # SA parameters
    T_start = 3.0
    T_end = 0.01
    cooling = 0.998
    T = T_start
    iters_done = 0
    max_iters = 8000

    while T > T_end and iters_done < max_iters:
        if time.time() > deadline - 8:
            break

        iters_done += 1

        # Try deletion first
        if len(current_solution) > 1 and random.random() < 0.2:
            idx = random.randrange(len(current_solution))
            b_remove = current_solution[idx]
            can_remove = True
            for i in block_tlist[b_remove]:
                if cover_counts[i] == 1:
                    can_remove = False
                    break
            if can_remove:
                for i in block_tlist[b_remove]:
                    cover_counts[i] -= 1
                current_solution.pop(idx)
                in_solution.discard(b_remove)
                current_size -= 1
                if current_size < best_sa_size:
                    best_sa_size = current_size
                    best_sa_solution = current_solution.copy()
                continue

        # Swap move
        if len(current_solution) > 0:
            idx = random.randrange(len(current_solution))
            b_old = current_solution[idx]
            critical = [i for i in block_tlist[b_old] if cover_counts[i] == 1]

            if critical:
                # Find blocks covering all critical t-subsets
                candidates = set(t_to_blocks[critical[0]])
                candidates.discard(b_old)
                for ci in critical[1:]:
                    candidates.intersection_update(set(t_to_blocks[ci]))
                    if not candidates:
                        break

                if candidates:
                    # Score: minimize increase in singly-covered t-subsets
                    best_cand = -1
                    best_delta = float('inf')
                    cand_list = list(candidates)
                    random.shuffle(cand_list)
                    for cand in cand_list[:30]:
                        # Compute delta in critical count
                        delta = 0
                        # Remove b_old: t-subsets in b_old with count 1 become 0 (bad)
                        # Add cand: t-subsets in cand with count 0 become 1 (good)
                        # Net change in number of uncovered t-subsets
                        # Actually track: how many t-subsets go from count 1 to 0, 
                        # and how many go from count 0 to 1
                        uncov_increase = 0  # t-subsets going from 1 to 0
                        cov_decrease = 0    # t-subsets going from 0 to 1
                        for i in block_tlist[b_old]:
                            if cover_counts[i] == 1:
                                uncov_increase += 1
                        for i in block_tlist[cand]:
                            if cover_counts[i] == 0:
                                cov_decrease += 1
                        # But some overlap: t-subsets in both b_old and cand
                        # If in both: count goes 1->0->1, net 0
                        # Need to be more careful
                        delta = uncov_increase - cov_decrease
                        if delta < best_delta:
                            best_delta = delta
                            best_cand = cand

                    if best_cand != -1:
                        # Apply swap
                        for i in block_tlist[b_old]:
                            cover_counts[i] -= 1
                        for i in block_tlist[best_cand]:
                            cover_counts[i] += 1
                        current_solution[idx] = best_cand
                        in_solution.discard(b_old)
                        in_solution.add(best_cand)
                    else:
                        # No valid swap; try random exploration
                        pass
            else:
                # No critical t-subsets: can swap freely
                # Pick a random block not in solution
                cand = random.randrange(n_k)
                while cand in in_solution:
                    cand = random.randrange(n_k)
                for i in block_tlist[b_old]:
                    cover_counts[i] -= 1
                for i in block_tlist[cand]:
                    cover_counts[i] += 1
                current_solution[idx] = cand
                in_solution.discard(b_old)
                in_solution.add(cand)

        # Cooling
        T *= cooling

        # Periodic deletion pass
        if iters_done % 100 == 0:
            changed = True
            while changed and time.time() < deadline - 8:
                changed = False
                new_sol = []
                for b in current_solution:
                    can_remove = True
                    for i in block_tlist[b]:
                        if cover_counts[i] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for i in block_tlist[b]:
                            cover_counts[i] -= 1
                        changed = True
                    else:
                        new_sol.append(b)
                current_solution = new_sol
                in_solution = set(current_solution)
                current_size = len(current_solution)
                if current_size < best_sa_size:
                    best_sa_size = current_size
                    best_sa_solution = current_solution.copy()

    # Final deletion pass
    cover_counts = compute_cover_counts(best_sa_solution)
    changed = True
    while changed and time.time() < deadline - 3:
        changed = False
        new_sol = []
        for b in best_sa_solution:
            can_remove = True
            for i in block_tlist[b]:
                if cover_counts[i] == 1:
                    can_remove = False
                    break
            if can_remove:
                for i in block_tlist[b]:
                    cover_counts[i] -= 1
                changed = True
            else:
                new_sol.append(b)
        best_sa_solution = new_sol

    # Verify and return
    final_solution = best_sa_solution
    if len(final_solution) > best_size:
        final_solution = best_solution

    return [list(k_subsets[idx]) for idx in final_solution]
