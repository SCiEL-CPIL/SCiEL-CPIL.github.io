    import numpy as np
    import time
    import random

    def solve(a, b):
        n = len(a)
        start_time = time.time()
        deadline = start_time + 145.0 

        def compute_cost(p):
            b_sub = b[np.ix_(p, p)]
            return int(np.sum(a * b_sub))

        def steepest_descent_ls(p, current_cost, time_limit):
            # p is numpy array
            while time.time() < time_limit:
                B_P = b[p, p]
                X = a @ B_P.T
                Z = a.T @ B_P
                
                # Compute Delta Matrix
                # Term1[i,j] = X[i,j] - X[i,i] - X[j,j] + X[j,i]
                # We need vectors of diagonals
                diag_X = np.diag(X)
                # Term1 = X - diag_X[:, None] - diag_X[None, :] + X.T
                # Wait, X[j,i] is X.T[i,j].
                term1 = X - diag_X[:, None] - diag_X[None, :] + X.T
                
                # Term2[i,j] = Z[i, p[j]] - Z[i, p[i]] - Z[j, p[j]] + Z[j, p[i]]
                # Z is n x n.
                # Z[i, p[j]]: For each i, we want Z[i, p[j]] for all j.
                # This is Z[:, p] ? No.
                # Z is indexed [row, col].
                # We want a matrix M2 where M2[i, j] = Z[i, p[j]].
                # This is Z[:, p] ?
                # Z[:, p] gives a matrix where row i is Z[i, p[0]], Z[i, p[1]], ...
                # Yes, Z[:, p] works.
                Zp = Z[:, p]
                
                # Z[i, p[i]] is Z[np.arange(n), p]
                Z_ii = Z[np.arange(n), p]
                
                # Term2[i,j] = Zp[i, j] - Z_ii[i] - Zp[j, i] + Z_ii[j]
                # Zp[j, i] is Zp.T[i, j]
                term2 = Zp - Z_ii[:, None] - Zp.T + Z_ii[None, :]
                
                delta_matrix = term1 + term2
                
                # Mask out i >= j
                mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                delta_matrix[~mask] = np.inf
                
                min_idx = np.unravel_index(np.argmin(delta_matrix), delta_matrix.shape)
                min_delta = delta_matrix[min_idx]
                
                if min_delta >= 0:
                    break
                
                i, j = min_idx
                p[i], p[j] = p[j], p[i]
                current_cost += int(min_delta)
                
            return p, current_cost

        # Initial Solution
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        heavy_order = np.argsort(-a_row_sums, kind="stable")
        central_order = np.argsort(b_row_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy_order] = central_order
        
        current_cost = compute_cost(p)
        best_p = p.copy()
        best_cost = current_cost
        
        # Initial LS
        initial_limit = start_time + 5.0
        p, current_cost = steepest_descent_ls(p, current_cost, initial_limit)
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
        # ILS
        while time.time() < deadline - 2.0:
            # Perturb
            num_swaps = random.randint(2, 10)
            p_curr = best_p.copy()
            for _ in range(num_swaps):
                i, j = np.random.choice(n, 2, replace=False)
                p_curr[i], p_curr[j] = p_curr[j], p_curr[i]
            
            curr_cost = compute_cost(p_curr)
            
            # LS
            ls_limit = deadline - 1.0
            p_ls, ls_cost = steepest_descent_ls(p_curr, curr_cost, ls_limit)
            
            if ls_cost < best_cost:
                best_cost = ls_cost
                best_p = p_ls.copy()
                
            if time.time() > deadline - 5.0:
                break
                
        return best_p.tolist()
    