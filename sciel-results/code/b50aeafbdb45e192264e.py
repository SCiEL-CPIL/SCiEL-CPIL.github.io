# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Precompute row sums for initial heuristic
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)

    # Initial rank-matching: heavy flows to central locations
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()

    def compute_cost(p):
        """Compute exact QAP cost."""
        return int(np.sum(a * b[np.ix_(p, p)]))

    def two_opt_delta(p, i, j):
        """
        Compute delta cost for swapping facilities i and j in assignment p.
        Assumes b is symmetric.
        """
        li = p[i]
        lj = p[j]
        
        d = b[lj, p] - b[li, p]
        d[i] = 0
        d[j] = 0
        
        a_diff = a[i, :] - a[j, :] + a[:, i] - a[:, j]
        a_diff[i] = 0
        a_diff[j] = 0
        
        return int(np.dot(a_diff, d))

    def hungarian_reassign(p):
        """
        Use Hungarian algorithm to globally reassign facilities based on linearized costs.
        """
        # Linearized cost matrix: C[i, l] = sum_j a[i,j] * b[l, p[j]] + sum_j a[j,i] * b[p[j], l]
        # For symmetric b: C[i, l] = sum_j (a[i,j] + a[j,i]) * b[l, p[j]]
        # Let S = a + a.T (symmetric part of flow matrix)
        # Then C[i, l] = sum_j S[i,j] * b[l, p[j]]
        # In matrix form: C = S @ b[:, p]  (if b is symmetric, b[l, p[j]] = b[p[j], l])
        
        # Actually, let's be precise.
        # Cost(i -> l) = sum_j a[i,j] * b[l, p[j]] + sum_j a[j,i] * b[p[j], l]
        # If b is symmetric, b[l, p[j]] = b[p[j], l].
        # So Cost(i -> l) = sum_j (a[i,j] + a[j,i]) * b[l, p[j]]
        
        S = a + a.T
        # b_l_p[j] = b[l, p[j]] for all l, j
        # This is b[:, p]
        bp = b[:, p]  # shape (n, n), bp[l, j] = b[l, p[j]]
        
        # C[i, l] = sum_j S[i, j] * bp[l, j]
        # C = S @ bp.T  -> C[i, l] = sum_j S[i, j] * bp[l, j]
        # Wait, (S @ bp.T)[i, l] = sum_j S[i, j] * bp.T[j, l] = sum_j S[i, j] * bp[l, j]. Yes.
        
        C = S @ bp.T
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(C)
        
        # new_p[facility] = location
        # row_ind is facilities, col_ind is locations
        # So new_p[row_ind[k]] = col_ind[k]
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        
        return new_p

    best_p = p.copy()
    best_cost = compute_cost(best_p)
    
    # Phase 1: 2-opt local search
    improved = True
    while improved and time.time() < deadline - 30.0:
        improved = False
        for i in range(n):
            if time.time() > deadline - 30.0:
                break
            for j in range(i+1, n):
                delta = two_opt_delta(p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    best_cost += delta
                    best_p = p.copy()
                    best_cost = compute_cost(best_p)  # Re-verify to be safe
                    improved = True
                    break
            if improved:
                pass
    
    # Phase 2: Hungarian reassignment iterations
    # We alternate between Hungarian reassignment and 2-opt polishing
    for _ in range(10):  # Max 10 iterations, but limited by time
        if time.time() > deadline - 15.0:
            break
        
        # Hungarian reassign
        new_p = hungarian_reassign(p)
        new_cost = compute_cost(new_p)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_p = new_p.copy()
            p = new_p.copy()
        else:
            # If Hungarian didn't improve, try a few 2-opt passes from current p
            # Then break if no improvement after a few tries
            # Or just break to save time for final polishing
            break
        
        # 2-opt polish after Hungarian
        if time.time() > deadline - 10.0:
            break
        
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            for i in range(n):
                if time.time() > deadline - 10.0:
                    break
                for j in range(i+1, n):
                    delta = two_opt_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        best_cost += delta
                        best_p = p.copy()
                        best_cost = compute_cost(best_p)
                        improved = True
                        break
                if improved:
                    pass
    
    # Final check: ensure best_p is the best found
    final_cost = compute_cost(best_p)
    if final_cost < best_cost:
        best_cost = final_cost
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
