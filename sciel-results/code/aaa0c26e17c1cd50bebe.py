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
    
    # Precompute some useful quantities
    a_max = np.max(np.abs(a_f))
    b_max = np.max(np.abs(b_f))
    scale = max(a_max, b_max, 1.0)
    
    # Initialize X as a doubly stochastic matrix from a good initial permutation
    # Use rank-matching: assign facilities with high total flow to locations with low average distance
    a_row_sums = a_f.sum(axis=1)
    b_row_sums = b_f.sum(axis=1)
    
    # Sort facilities by decreasing flow, locations by increasing distance
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    
    p0 = np.empty(n, dtype=np.int64)
    p0[heavy] = central
    
    # Create initial doubly stochastic matrix from permutation
    X = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        X[i, p0[i]] = 1.0
    
    # Sinkhorn-Knopp projection to doubly stochastic
    def sinkhorn_project(X, max_iter=100, tol=1e-8):
        """Project X to doubly stochastic matrix using Sinkhorn-Knopp."""
        X = np.maximum(X, 1e-12)
        for it in range(max_iter):
            row_sums = X.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-12)
            X = X / row_sums
            col_sums = X.sum(axis=0, keepdims=True)
            col_sums = np.maximum(col_sums, 1e-12)
            X = X / col_sums
            # Check convergence
            if it % 10 == 0:
                err = max(np.max(np.abs(X.sum(axis=1) - 1.0)), 
                         np.max(np.abs(X.sum(axis=0) - 1.0)))
                if err < tol:
                    break
        return X
    
    # Compute the objective: f(X) = sum_{i,j,k,l} a[i,j] * b[k,l] * X[i,k] * X[j,l]
    # = trace(X^T @ A @ X @ B^T)
    # Gradient: grad[i,k] = sum_{j,l} A[i,j] * B[k,l] * X[j,l] + sum_{j,l} A[k,j] * B[i,l] * X[j,l]
    #           = (A @ X @ B.T)[i,k] + (B @ X.T @ A.T)[i,k]
    
    def compute_cost(X):
        """Compute the relaxed QAP cost."""
        # f(X) = sum_{k,l} b[k,l] * (X.T @ A @ X)[k,l]
        C = X.T @ a_f @ X
        return np.sum(b_f * C)
    
    def compute_grad(X):
        """Compute gradient of the relaxed QAP objective."""
        # grad = A @ X @ B.T + B @ X.T @ A.T
        return a_f @ X @ b_f.T + b_f @ X.T @ a_f.T
    
    # Projected gradient descent with backtracking line search
    best_cost = compute_cost(X)
    best_X = X.copy()
    
    # Initial step size
    step_size = 1.0 / (n * scale)
    
    for iteration in range(2000):
        if time.time() - t0 > time_limit:
            break
        
        # Compute current cost and gradient
        cost = compute_cost(X)
        
        if cost < best_cost:
            best_cost = cost
            best_X = X.copy()
        
        grad = compute_grad(X)
        
        # Backtracking line search
        alpha = step_size
        X_new = X
        cost_new = cost
        
        for _ in range(15):
            X_candidate = X_new - alpha * grad
            X_candidate = sinkhorn_project(X_candidate, max_iter=30)
            cost_candidate = compute_cost(X_candidate)
            
            if cost_candidate <= cost * (1.0 + 1e-6):
                X_new = X_candidate
                cost_new = cost_candidate
                break
            alpha *= 0.5
        
        X = X_new
        cost = cost_new
        
        # Adaptive step size
        if cost < best_cost * 0.999:
            step_size = min(step_size * 1.02, 1.0 / (scale * 0.1))
        elif cost > best_cost * 1.001:
            step_size *= 0.9
    
    # Recover permutation from best_X using Hungarian algorithm
    # We want to find the permutation that maximizes sum X[i,p[i]]
    # This is equivalent to minimizing sum (-X[i,p[i]])
    row_ind, col_ind = linear_sum_assignment(-best_X)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    # Verify it's a valid permutation
    assert len(set(p.tolist())) == n, "Invalid permutation"
    
    return p.tolist()
# EVOLVE-BLOCK-END
