"""QAP solver using Sinkhorn-Knopp continuous relaxation + gradient descent + local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _compute_cost(a, b, p):
    """Compute total QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _sinkhorn_init(a, b, n, rng, time_budget, start_time):
    """
    Initialize using Sinkhorn-Knopp on the Birkhoff polytope.
    We minimize sum_{i,j} a[i,j] * (X^T B X)[i,j] where X is doubly stochastic.
    Use gradient descent on log-sum-exp smoothed assignment.
    """
    elapsed = time.time() - start_time
    if elapsed > time_budget * 0.3:
        return None
    
    # Cost matrix for Sinkhorn: we want to align high-flow with low-distance
    # The quadratic form suggests using a weighted cost
    # For initialization, use a linear approximation: minimize sum_i a_row_sum[i] * b_row_sum[perm[i]]
    # But let's do proper Sinkhorn on a modified cost
    
    # Create a cost matrix that captures the QAP structure
    # C[i,j] = sum_k a[i,k] * b[j,k] (linearization)
    C = a @ b  # shape (n, n)
    
    # Make C symmetric for better behavior
    C = 0.5 * (C + C.T)
    
    # Normalize C for Sinkhorn
    C_min = C.min()
    C_max = C.max()
    if C_max - C_min > 1e-10:
        C_norm = (C - C_min) / (C_max - C_min)
    else:
        C_norm = np.zeros_like(C)
    
    # Sinkhorn-Knopp with temperature annealing
    # Start with high temperature (uniform), anneal to low temperature (permutation-like)
    n_iters = 200
    X = np.ones((n, n), dtype=np.float64) / n
    
    for it in range(n_iters):
        if time.time() - start_time > time_budget * 0.35:
            break
        
        # Temperature schedule: exponential decay
        t_frac = it / n_iters
        temp = 1.0 / (1.0 + 10.0 * t_frac**2)
        
        # Sinkhorn iterations
        for _ in range(10):
            # Scale rows
            row_sums = X.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-12)
            X = X / row_sums
            # Scale by exp(-C/temp)
            X = X * np.exp(-C_norm / temp)
            
            # Scale columns
            col_sums = X.sum(axis=0, keepdims=True)
            col_sums = np.maximum(col_sums, 1e-12)
            X = X / col_sums
            # Scale by exp(-C.T/temp)
            X = X * np.exp(-C_norm.T / temp)
        
        # Normalize to ensure doubly stochastic
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-12)
        X = X / row_sums
    
    # Extract permutation from X
    # Use Hungarian algorithm on -X (maximize X) or on C_norm
    row_ind, col_ind = linear_sum_assignment(C_norm)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p

def _gradient_descent_init(a, b, n, rng, time_budget, start_time):
    """
    Gradient descent on the smoothed assignment energy.
    f(X) = sum_{i,j} a[i,j] * (X^T B X)[i,j] + entropy regularization
    """
    elapsed = time.time() - start_time
    if elapsed > time_budget * 0.2:
        return None
    
    # Start from a random doubly stochastic matrix
    X = rng.random((n, n))
    # Make it doubly stochastic via Sinkhorn
    for _ in range(20):
        X = X / np.maximum(X.sum(axis=1, keepdims=True), 1e-12)
        X = X / np.maximum(X.sum(axis=0, keepdims=True), 1e-12)
    
    # Gradient descent with temperature
    lr = 0.01
    temp = 1.0
    n_iters = 100
    
    B = b  # distance matrix
    
    for it in range(n_iters):
        if time.time() - start_time > time_budget * 0.25:
            break
        
        # Compute X^T B X
        XTX = X.T @ B @ X  # shape (n, n)
        
        # Objective gradient: d/dX sum_{i,j} a[i,j] * XTX[i,j]
        # = 2 * A @ (B @ X) @ X.T ... let's derive carefully
        # f(X) = trace(A^T * X^T B X) = trace(B X A^T X^T)
        # df/dX = 2 * B X A^T X^T + 2 * B^T X A X^T (if symmetric, just 2*B*X*A*X^T)
        # But we also need the entropy term: -sum X log X, gradient = -(log X + 1)
        
        A_sym = 0.5 * (a + a.T)
        B_sym = 0.5 * (b + b.T)
        
        # Gradient of quadratic form
        grad_quad = 2.0 * (B_sym @ X @ A_sym.T @ X.T + B_sym.T @ X @ A_sym @ X.T)
        
        # Gradient of entropy regularization (we want to encourage sparsity/permutation)
        # Use -temp * sum(X * log(X)) as regularization, gradient = -temp * (log(X) + 1)
        X_safe = np.maximum(X, 1e-12)
        grad_ent = -temp * (np.log(X_safe) + 1.0)
        
        grad = grad_quad + grad_ent
        
        # Project gradient onto the tangent space of doubly stochastic matrices
        # Subtract row and column means, then add back the grand mean
        grad_row_mean = grad.mean(axis=1, keepdims=True)
        grad_col_mean = grad.mean(axis=0, keepdims=True)
        grad_grand_mean = grad.mean()
        grad = grad - grad_row_mean - grad_col_mean + grad_grand_mean
        
        # Update
        X = X - lr * grad
        
        # Project back to doubly stochastic via Sinkhorn
        X = np.maximum(X, 1e-12)
        for _ in range(5):
            X = X / np.maximum(X.sum(axis=1, keepdims=True), 1e-12)
            X = X / np.maximum(X.sum(axis=0, keepdims=True), 1e-12)
        
        # Anneal temperature
        temp *= 0.95
    
    # Extract permutation
    row_ind, col_ind = linear_sum_assignment(-X)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p

def _local_search(a, b, p, time_limit, start_time, max_passes=3):
    """
    Fast local search with 2-opt swaps using vectorized delta computation.
    Returns (p, cost).
    """
    n = len(p)
    p = np.array(p, dtype=np.int64).copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = float(np.sum(a * D))
    a_sym = a + a.T  # symmetric part for delta computation
    
    pass_count = 0
    improved = True
    while improved and pass_count < max_passes and (time.time() - start_time) < time_limit:
        improved = False
        pass_count += 1
        
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                return p, current_cost
            
            js = np.arange(i+1, n)
            if len(js) == 0:
                continue
            
            a_i = a[i]
            a_js = a[js]
            D_i = D[i]
            D_js = D[js]
            
            diff_D = D_js - D_i[None, :]
            sum_a = a_i[None, :] + a_js
            main_term = np.sum(sum_a * diff_D, axis=1)
            
            a_ii = a[i, i]
            a_ij = a[i, js]
            Dj_i = D[js, i]
            Di_i = D[i, i]
            corr1 = -(a_ii + a_ij) * (Dj_i - Di_i)
            
            a_ji = a[js, i]
            a_jj = np.diag(a[js])
            Dj_j = np.diag(D_js)
            Di_j = D_i[js]
            corr2 = -(a_ji + a_jj) * (Dj_j - Di_j)
            
            corr3 = (a_ii - a_jj) * (Dj_j - Di_i)
            corr4 = (a_ij - a_ji) * (Dj_i - Di_j)
            
            deltas = main_term + corr1 + corr2 + corr3 + corr4
            
            min_idx = np.argmin(deltas)
            if deltas[min_idx] < -1e-9:
                j = js[min_idx]
                delta = deltas[min_idx]
                
                p[i], p[j] = p[j], p[i]
                D[i, :], D[j, :] = D[j, :].copy(), D[i, :].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                current_cost += delta
                improved = True
    
    return p, current_cost

def _perturb(p, n, k, rng):
    """Perturb permutation by k random swaps."""
    p = p.copy()
    for _ in range(k):
        i, j = rng.choice(n, size=2, replace=False)
        p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 145.0  # leave 5s buffer for safety
    
    rng = np.random.default_rng(42)
    
    best_p = None
    best_cost = np.inf
    
    # Generate multiple initial solutions
    initial_solutions = []
    
    # 1. Sinkhorn-based initialization
    p_sinkhorn = _sinkhorn_init(a, b, n, rng, time_budget, start_time)
    if p_sinkhorn is not None:
        initial_solutions.append(p_sinkhorn)
    
    # 2. Gradient descent initialization
    p_grad = _gradient_descent_init(a, b, n, rng, time_budget, start_time)
    if p_grad is not None:
        initial_solutions.append(p_grad)
    
    # 3. Random initialization
    initial_solutions.append(rng.permutation(n))
    
    # 4. Row-sum matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    a_order = np.argsort(a_row_sums)[::-1]
    b_order = np.argsort(b_row_sums)
    p2 = np.empty(n, dtype=np.int64)
    p2[a_order] = b_order
    initial_solutions.append(p2)
    
    # 5. Another random
    initial_solutions.append(rng.permutation(n))
    
    # Local search on each initial solution
    for idx, p_init in enumerate(initial_solutions):
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 5:
            break
        
        p_ls, cost_ls = _local_search(a, b, p_init, remaining - 3, start_time, max_passes=2)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Iterated Local Search
    stagnation_count = 0
    best_cost_ils = best_cost
    
    while (time.time() - start_time) < time_budget - 8:
        elapsed = time.time() - start_time
        progress = elapsed / time_budget
        k_perturb = max(2, int(6 * (1 - progress)))
        
        p_pert = _perturb(best_p, n, k_perturb, rng)
        
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5:
            break
        
        p_ls, cost_ls = _local_search(a, b, p_pert, remaining - 3, start_time, max_passes=2)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            stagnation_count = 0
        else:
            stagnation_count += 1
        
        # If stuck, try a larger perturbation or restart
        if stagnation_count > 5:
            # Try a more aggressive perturbation
            k_large = max(5, int(15 * (1 - progress)))
            p_pert = _perturb(best_p, n, k_large, rng)
            remaining = time_budget - (time.time() - start_time)
            if remaining < 5:
                break
            p_ls, cost_ls = _local_search(a, b, p_pert, remaining - 3, start_time, max_passes=2)
            if cost_ls < best_cost:
                best_cost = cost_ls
                best_p = p_ls.copy()
            stagnation_count = 0
    
    # Final verification
    if best_p is not None:
        final_cost = _compute_cost(a, b, best_p)
        if final_cost < best_cost:
            best_cost = final_cost
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
