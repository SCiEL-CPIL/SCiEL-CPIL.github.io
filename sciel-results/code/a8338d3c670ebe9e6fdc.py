# EVOLVE-BLOCK-START
import numpy as np
import time
import collections


def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115.0

    # Step 1: Initial assignment using rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central

    # Compute initial cost
    def compute_cost(perm):
        b_perm = b[np.ix_(perm, perm)]
        return np.sum(a * b_perm)

    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()

    # Precompute for fast delta evaluation
    a_rows = a
    a_cols = a.T
    b_matrix = b

    def compute_delta(i, j, perm):
        li = perm[i]
        lj = perm[j]
        if li == lj:
            return 0

        diff_row = b_matrix[lj, :] - b_matrix[li, :]
        diff_col = b_matrix[:, lj] - b_matrix[:, li]

        s1 = np.dot(a_rows[i, :], diff_row[perm])
        s1 -= a_rows[i, i] * diff_row[li]
        s1 -= a_rows[i, j] * diff_row[lj]

        s2 = -np.dot(a_rows[j, :], diff_row[perm])
        s2 += a_rows[j, i] * diff_row[li]
        s2 += a_rows[j, j] * diff_row[lj]

        s3 = np.dot(a_cols[i, :], diff_col[perm])
        s3 -= a_cols[i, i] * diff_col[li]
        s3 -= a_cols[i, j] * diff_col[lj]

        s4 = -np.dot(a_cols[j, :], diff_col[perm])
        s4 += a_cols[j, i] * diff_col[li]
        s4 += a_cols[j, j] * diff_col[lj]

        cross = (a_rows[j, i] - a_rows[i, j]) * (b_matrix[li, lj] - b_matrix[lj, li])

        return s1 + s2 + s3 + s4 + cross

    # Tabu Search Parameters
    tabu_tenure = n // 5  # Shorter tenure for faster exploration
    tabu_queue = collections.deque()
    tabu_set = set()
    
    # Helper to apply a move
    def apply_move(i, j):
        nonlocal p, current_cost
        p[i], p[j] = p[j], p[i]
        # Update tabu
        move = (i, j)
        tabu_queue.append(move)
        tabu_set.add(move)
        if len(tabu_queue) > tabu_tenure:
            old_move = tabu_queue.popleft()
            tabu_set.discard(old_move)

    # Main Loop: Iterated Local Search with First-Improvement Tabu
    no_improvement_count = 0
    max_no_improvement = 500 # Threshold for perturbation

    while time.time() - start_time < time_budget - 2.0:
        # Phase 1: Local Search (First Improvement Tabu)
        improved = False
        iterations_in_phase = 0
        max_iter_phase = 5000 # Limit iterations per phase to avoid getting stuck in slow convergence
        
        while iterations_in_phase < max_iter_phase and time.time() - start_time < time_budget - 2.0:
            move_found = False
            # Shuffle order to avoid bias? Or just iterate.
            # For first improvement, order matters. Randomizing can help.
            # But shuffling n=100 is cheap.
            indices = np.arange(n)
            np.random.shuffle(indices)
            
            for i_idx in indices:
                i = i_idx
                for j_idx in indices:
                    if i == j_idx:
                        continue
                    j = j_idx
                    
                    # Check time
                    if time.time() - start_time > time_budget - 2.0:
                        break
                    
                    # Skip if tabu
                    move = (i, j)
                    if move in tabu_set:
                        continue
                        
                    delta = compute_delta(i, j, p)
                    
                    # Aspiration: if it improves global best, accept even if tabu
                    # But we already skipped tabu above. 
                    # Standard First Improvement: accept if delta < 0.
                    if delta < 0:
                        # Apply move
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_p = p.copy()
                            no_improvement_count = 0
                        else:
                            no_improvement_count += 1
                            
                        # Update Tabu
                        tabu_queue.append(move)
                        tabu_set.add(move)
                        if len(tabu_queue) > tabu_tenure:
                            old_move = tabu_queue.popleft()
                            tabu_set.discard(old_move)
                        
                        improved = True
                        move_found = True
                        break # Break inner loop, restart outer with new state
                
                if move_found:
                    break # Break outer loop, restart scan
            
            if not move_found:
                break # Local optimum reached
            iterations_in_phase += 1

        # Phase 2: Perturbation if stuck or after a phase
        if not improved:
            # Perform a random perturbation
            # Swap k random pairs
            k = 3
            for _ in range(k):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i != j:
                    delta = compute_delta(i, j, p)
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    # Update tabu? Clear tabu after perturbation
                    tabu_queue.clear()
                    tabu_set.clear()
            
            # If perturbation worsened cost significantly, it's okay, we are exploring.
            # Reset no_improvement counter to allow new search
            no_improvement_count = 0

    return best_p.tolist()
# EVOLVE-BLOCK-END
