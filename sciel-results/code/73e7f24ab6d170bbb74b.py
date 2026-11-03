# EVOLVE-BLOCK-START
import numpy as np
import time
import random


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

    # Simulated Annealing Parameters
    # Initial temperature should be high enough to accept worse moves
    # Estimate average delta magnitude to set T0
    # For now, a heuristic T0 based on current cost magnitude or just a large number
    # Cost is ~273k. Deltas can be in thousands.
    T0 = 1000.0
    T_end = 0.1
    cooling_rate = 0.999
    
    # We will run until time runs out
    # To ensure we don't get stuck, we can restart if no improvement for a long time, 
    # but simple SA with good T usually works.
    
    # Random seed for reproducibility in testing, but evaluator might not care.
    # random.seed(42) 

    while time.time() - start_time < time_budget - 2.0:
        # Generate random move
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        
        if i == j:
            continue

        delta = compute_delta(i, j, p)
        
        if delta < 0:
            # Accept improvement
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # Accept worse move with probability exp(-delta / T)
            # Avoid division by zero if T is too small
            if T_end > 0:
                prob = np.exp(-delta / T0)
                if random.random() < prob:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
            
        # Cool down
        T0 *= cooling_rate
        if T0 < T_end:
            T0 = T_end

    return best_p.tolist()
# EVOLVE-BLOCK-END
