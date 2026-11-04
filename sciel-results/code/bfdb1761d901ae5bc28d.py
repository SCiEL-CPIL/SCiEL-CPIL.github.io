"""QAP solver: Sinkhorn-Knopp Continuous Relaxation + Discrete Refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    
    start_time = time.time()
    deadline = 115.0

    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))

    def get_swap_delta(p, i, j):
        u = p[i]
        v = p[j]
        
        # Direct pair term
        delta = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])
        
        # Diagonal terms
        delta += a[i, i] * (b[v, v] - b[u, u])
        delta += a[j, j] * (b[u, u] - b[v, v])
        
        # Cross terms with all other k
        diff_a_row = a[i] - a[j]
        diff_a_col = a[:, i] - a[:, j]
        
        diff_b_row = b[v, p] - b[u, p]
        diff_b_col = b[p, v] - b[p, u]
        
        sum_T1 = np.dot(diff_a_row, diff_b_row)
        sum_T2 = np.dot(diff_a_col, diff_b_col)
        
        # Subtract k=i terms
        sum_T1 -= (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        sum_T2 -= (a[i, i] - a[i, j]) * (b[u, v] - b[u, u])
        
        # Subtract k=j terms
        sum_T1 -= (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        sum_T2 -= (a[j, i] - a[j, j]) * (b[v, v] - b[v, u])
        
        delta += sum_T1 + sum_T2
        return int(delta)

    def local_search_swaps(p, current_cost, max_iters=50):
        """
        First-improvement local search using swaps for speed.
        """
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() - start_time > deadline - 2.0:
                break
            
            # Use a randomized scan order to avoid getting stuck in local minima quickly
            # But for determinism in this block, we do a full scan for first improvement
            for i in range(n):
                if time.time() - start_time > deadline - 2.0:
                    return p, current_cost
                for j in range(i + 1, n):
                    delta = get_swap_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
                        break
                if improved:
                    break
        return p, current_cost

    def sinkhorn_relaxation(epsilon=0.1, max_iter=100):
        """
        Solve the QAP relaxation using Sinkhorn-Knopp scaling.
        The cost in the relaxed space is approx by the interaction of row/col sums 
        or by directly optimizing the doubly stochastic matrix.
        
        For QAP, the objective is Tr(A^T X B X^T).
        We can use a gradient-based approach on the log-domain of the doubly stochastic matrix,
        or simpler: use the surrogate cost matrix C = - (A @ 1) (B @ 1)^T 
        and solve the assignment problem, then use the resulting permutation to seed a 
        more sophisticated continuous refinement if needed.
        
        However, to be "structurally different", let's implement a simplified 
        "Entropy Regularized" assignment using Sinkhorn on a cost matrix derived 
        from the QAP structure, then round.
        
        Actually, a better structural change is to use the **Continuous QAP** 
        approximation where we optimize X in the Birkhoff polytope.
        Since exact gradient descent on Birkhoff is complex, we will use 
        **Sinkhorn Iterations** on a surrogate matrix that captures the QAP 
        structure, followed by **Rounding** via Hungarian on the scaled matrix.
        
        Let's try a multi-scale approach:
        1. Compute surrogate cost C = - (A.sum(axis=1) * B.sum(axis=1)).
        2. Apply Sinkhorn to exp(-C/epsilon) to get a smooth assignment.
        3. Round to permutation.
        """
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        
        # Surrogate cost matrix
        C = -np.outer(a_row_sums, b_row_sums)
        
        # Normalize C to prevent overflow in exp
        C_min = np.min(C)
        C_max = np.max(C)
        if C_max > C_min:
            C_norm = (C - C_min) / (C_max - C_min)
        else:
            C_norm = C - C_min
            
        # Sinkhorn-Knopp to find doubly stochastic matrix close to exp(-C_norm / epsilon)
        # We want to maximize alignment, so we minimize cost.
        # Let M = exp(-C_norm / epsilon).
        # We iterate: M <- D1 M D2 such that row/col sums are 1.
        
        eps = 0.5 # Temperature
        M = np.exp(-C_norm / eps)
        
        for _ in range(max_iter):
            # Row scaling
            row_sums = M.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            M = M / row_sums
            # Col scaling
            col_sums = M.sum(axis=0, keepdims=True)
            col_sums = np.where(col_sums == 0, 1, col_sums)
            M = M / col_sums
            
            if time.time() - start_time > deadline - 2.0:
                break
        
        # Round to permutation using Hungarian on the negative of the scaled matrix
        # The higher the value in M, the better the assignment.
        # So we minimize -M.
        row_ind, col_ind = linear_sum_assignment(-M)
        
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    # 1. Initialization using Sinkhorn Relaxation
    p = sinkhorn_relaxation()
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost

    # 2. Discrete Refinement (Local Search)
    p, current_cost = local_search_swaps(p, current_cost, max_iters=100)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()

    # 3. Multi-start with different Sinkhorn temperatures and random restarts
    # The continuous relaxation provides diverse initial points.
    
    while time.time() - start_time < deadline - 2.0:
        # Generate a new initial point by perturbing the surrogate or using different eps
        eps_val = np.random.uniform(0.1, 2.0)
        
        # Recompute surrogate with slight noise to escape basins?
        # Or just re-run Sinkhorn with different eps.
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        C = -np.outer(a_row_sums, b_row_sums)
        
        C_min = np.min(C)
        C_max = np.max(C)
        if C_max > C_min:
            C_norm = (C - C_min) / (C_max - C_min)
        else:
            C_norm = C - C_min
            
        M = np.exp(-C_norm / eps_val)
        
        for _ in range(50):
            row_sums = M.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            M = M / row_sums
            col_sums = M.sum(axis=0, keepdims=True)
            col_sums = np.where(col_sums == 0, 1, col_sums)
            M = M / col_sums
            
            if time.time() - start_time > deadline - 2.0:
                break
        
        row_ind, col_ind = linear_sum_assignment(-M)
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        
        # Refine
        p_new, cost_new = local_search_swaps(p_new, calc_cost(p_new), max_iters=50)
        
        if cost_new < best_cost:
            best_cost = cost_new
            best_p = p_new.copy()
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
