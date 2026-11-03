"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def solve(a, b):
    n = len(a)
    t0 = time.time()
    time_limit = 115.0  # leave 5s buffer for overhead

    # Convert to float for optimization
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)

    def perm_cost(p):
        """Compute cost of permutation p."""
        return int(np.sum(a_f * b_f[np.ix_(p, p)]))

    def sinkhorn_project(X, max_iter=100, tol=1e-12):
        """Project X to doubly stochastic matrix using Sinkhorn-Knopp."""
        X = np.maximum(X, 1e-15)
        
        total = X.sum()
        if total < 1e-15:
            return np.eye(n) / n
        X = X / total
        
        for _ in range(max_iter):
            row_sums = X.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-15)
            X = X / row_sums
            
            col_sums = X.sum(axis=0, keepdims=True)
            col_sums = np.maximum(col_sums, 1e-15)
            X = X / col_sums
            
            row_diff = np.max(np.abs(X.sum(axis=1) - 1.0))
            col_diff = np.max(np.abs(X.sum(axis=0) - 1.0))
            if row_diff < tol and col_diff < tol:
                break
        return X

    def relaxed_cost(X):
        """Compute tr(A X B X^T) - the relaxed QAP cost."""
        M = X @ b_f @ X.T
        return np.sum(a_f * M)

    def admm_solve(X0, max_iter=200, rho=1.0, time_budget=10.0):
        """
        Solve the convexified QAP using ADMM-style alternating minimization.
        Minimize tr(A X B X^T) subject to X being doubly stochastic.
        
        Uses a decomposition:
        - X-subproblem: minimize tr(A X B X^T) + (rho/2)||X - Z + U||^2 s.t. X doubly stochastic
        - Z-subproblem: minimize (rho/2)||X - Z + U||^2 s.t. Z doubly stochastic
        - U-update: U += X - Z
        """
        X = X0.copy()
        Z = X0.copy()
        U = np.zeros((n, n), dtype=np.float64)
        
        best_cost = np.inf
        best_X = X0.copy()
        
        for it in range(max_iter):
            if time.time() - t0 > time_limit - 8:
                break
            
            # X-subproblem: gradient step on tr(A X B X^T) + (rho/2)||X - Z + U||^2
            # Gradient of tr(A X B X^T) w.r.t. X: A X B^T + B X^T A^T
            # Gradient of (rho/2)||X - Z + U||^2 w.r.t. X: rho*(X - Z + U)
            
            AX = a_f @ X
            AXB_T = AX @ b_f.T
            BX_T = b_f @ X.T
            BX_T_A_T = BX_T @ a_f.T
            grad_obj = AXB_T + BX_T_A_T
            
            # Add ADMM quadratic term gradient
            grad_total = grad_obj + rho * (X - Z + U)
            
            # Adaptive step size
            # Estimate Lipschitz constant
            step_size = 1.0 / (2.0 * (np.max(np.abs(grad_obj)) + rho))
            step_size = min(step_size, 0.5)
            
            X_new = X - step_size * grad_total
            X_new = sinkhorn_project(X_new, max_iter=50)
            
            # Z-subproblem: projection of (X + U) onto doubly stochastic set
            Z_new = sinkhorn_project(X + U, max_iter=50)
            
            # U-update
            U = U + X_new - Z_new
            
            # Check convergence and track best
            X = X_new
            Z = Z_new
            
            cost = relaxed_cost(X)
            if cost < best_cost:
                best_cost = cost
                best_X = X.copy()
            
            # Check convergence
            primal_residual = np.max(np.abs(X - Z))
            dual_residual = rho * np.max(np.abs(U))
            if primal_residual < 1e-8 and dual_residual < 1e-8:
                break
        
        return best_X, best_cost

    def swap_delta(p, i, j):
        """Compute cost change from swapping p[i] and p[j]."""
        pi, pj = p[i], p[j]
        
        b_pi_p = b_f[pi, p]
        b_pj_p = b_f[pj, p]
        b_p_pi = b_f[p, pi]
        b_p_pj = b_f[p, pj]
        
        delta = 0.0
        delta += np.dot(a_f[i, :], b_pj_p - b_pi_p)
        delta += np.dot(a_f[j, :], b_pi_p - b_pj_p)
        delta += np.dot(a_f[:, i], b_p_pj - b_p_pi)
        delta += np.dot(a_f[:, j], b_p_pi - b_p_pj)
        
        correction = (a_f[i, i] * (b_f[pj, pj] - b_f[pi, pi]) + 
                      a_f[j, j] * (b_f[pi, pi] - b_f[pj, pj]) + 
                      a_f[i, j] * (b_f[pj, pj] + b_f[pi, pi] - 2 * b_f[pj, pi]) + 
                      a_f[j, i] * (b_f[pi, pi] + b_f[pj, pj] - 2 * b_f[pi, pj]))
                      
        return delta - correction

    def local_search_2opt(p, max_iter=5000, time_budget=5.0):
        """Improve permutation via pairwise swaps with adaptive sampling."""
        p = p.copy()
        current_cost = perm_cost(p)
        t_start = time.time()
        
        no_improve_count = 0
        best_cost_in_ls = current_cost
        best_p_in_ls = p.copy()
        
        for it in range(max_iter):
            if time.time() - t_start > time_budget:
                break
            
            improved = False
            # Adaptive number of random pairs to try per iteration
            num_trials = min(300, max(50, int(20000 / (it + 1))))
            
            for _ in range(num_trials):
                i, j = np.random.choice(n, size=2, replace=False)
                if i == j:
                    continue
                
                delta = swap_delta(p, i, j)
                
                if delta < -1e-10:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    no_improve_count = 0
                    if current_cost < best_cost_in_ls:
                        best_cost_in_ls = current_cost
                        best_p_in_ls = p.copy()
                    break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > 150:
                    break
            else:
                no_improve_count = 0
        
        return best_p_in_ls, best_cost_in_ls

    def local_search_systematic(p, time_budget=3.0):
        """Systematic local search: try all swaps in order."""
        p = p.copy()
        current_cost = perm_cost(p)
        t_start = time.time()
        best_cost = current_cost
        best_p = p.copy()
        
        for i in range(n):
            if time.time() - t_start > time_budget:
                break
            for j in range(i + 1, n):
                if time.time() - t_start > time_budget:
                    break
                
                delta = swap_delta(p, i, j)
                if delta < -1e-10:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
        
        return best_p, best_cost

    # Phase 1: Generate initial solutions and run ADMM on continuous relaxation
    # Initialize with rank matching
    heavy = np.argsort(-a_f.sum(axis=1), kind="stable")
    central = np.argsort(b_f.sum(axis=1), kind="stable")
    p_init = np.empty(n, dtype=np.int64)
    p_init[heavy] = central

    best_p = None
    best_cost = np.inf

    # Starting points for ADMM
    start_perms = [p_init]
    for _ in range(3):
        start_perms.append(np.random.permutation(n))

    # Run ADMM from each starting point
    for start_p in start_perms:
        if time.time() - t0 > time_limit - 15:
            break
        
        # Initialize X from permutation (relax to doubly stochastic)
        X0 = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            X0[i, start_p[i]] = 1.0
        X0 = sinkhorn_project(X0, max_iter=30)

        # Run ADMM
        X_opt, relaxed_cost_val = admm_solve(X0, max_iter=150, rho=1.0, time_budget=min(10.0, time_limit - 20 - (time.time() - t0)))

        # Recover permutation from relaxed solution using Hungarian
        row_ind, col_ind = linear_sum_assignment(-X_opt)
        p_relaxed = np.empty(n, dtype=np.int64)
        for i, j in zip(row_ind, col_ind):
            p_relaxed[i] = j
        
        cost_relaxed = perm_cost(p_relaxed)
        if cost_relaxed < best_cost:
            best_cost = cost_relaxed
            best_p = p_relaxed

    # Also keep the initial rank-matching solution as a candidate
    cost_init = perm_cost(p_init)
    if cost_init < best_cost:
        best_cost = cost_init
        best_p = p_init

    # Phase 2: Local search refinement from best solutions
    candidates = [best_p.copy(), p_init.copy(), np.random.permutation(n)]

    ls_time_remaining = time_limit - 5 - (time.time() - t0)
    if ls_time_remaining <= 0:
        return best_p.tolist() if best_p is not None else p_init.tolist()

    # Run local search on each candidate
    num_candidates = len(candidates)
    time_per_candidate = max(1.0, ls_time_remaining / num_candidates)

    for cand in candidates:
        if time.time() - t0 > time_limit - 3:
            break
        
        # First do systematic local search
        p_sys, cost_sys = local_search_systematic(cand, time_budget=min(2.0, time_per_candidate * 0.4))
        if cost_sys < best_cost:
            best_cost = cost_sys
            best_p = p_sys
        
        # Then do random local search from the best
        remaining_for_this = time_per_candidate - (time.time() - t0) - (time_limit - 3)
        if remaining_for_this > 0.5:
            p_rand, cost_rand = local_search_2opt(p_sys, max_iter=5000, time_budget=min(4.0, remaining_for_this))
            if cost_rand < best_cost:
                best_cost = cost_rand
                best_p = p_rand

    # Phase 3: If time remains, do more intensive local search from best
    time_remaining = time_limit - 3 - (time.time() - t0)
    if time_remaining > 2 and best_p is not None:
        p_final, cost_final = local_search_2opt(best_p, max_iter=10000, time_budget=time_remaining)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final

    if best_p is None:
        best_p = p_init

    return best_p.tolist()
# EVOLVE-BLOCK-END
