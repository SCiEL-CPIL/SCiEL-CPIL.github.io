"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 145.0  # Leave 5s safety margin for 150s limit

    # Ensure arrays are int64 for exact integer arithmetic in final cost
    # But use float64 for optimization
    a_int = a.astype(np.int64)
    b_int = b.astype(np.int64)
    
    A = a.astype(np.float64)
    B = b.astype(np.float64)

    # 1. Spectral Initialization for the Discrete Problem
    # This provides a good starting point for the discrete search and a check for the continuous method
    try:
        A_sym = (A + A.T) / 2.0
        B_sym = (B + B.T) / 2.0
        
        # Get top eigenvectors
        eigvals_A, eigvecs_A = np.linalg.eigh(A_sym)
        eigvals_B, eigvecs_B = np.linalg.eigh(B_sym)
        
        idx_A = np.argsort(eigvals_A)[::-1]
        idx_B = np.argsort(eigvals_B)[::-1]
        
        v_A = eigvecs_A[:, idx_A[0]]
        v_B = eigvecs_B[:, idx_B[0]]
        
        # Rank matching based on spectral coordinates
        rank_A = np.argsort(v_A)
        rank_B = np.argsort(v_B)
        
        p_init = np.empty(n, dtype=np.int64)
        # Facility i (ranked by v_A) goes to location j (ranked by v_B)
        # We want to align the "structure". 
        # Usually, high flow connects to high distance? No, we minimize sum(flow * dist).
        # So we want high flow pairs to have low distance.
        # This implies matching similar "positions" in the embedding if the matrices have similar structure.
        # Let's try both alignments and pick the better one.
        
        p1 = np.empty(n, dtype=np.int64)
        p1[rank_A] = rank_B
        
        # Try inverse alignment
        p2 = np.empty(n, dtype=np.int64)
        p2[rank_A] = rank_B[::-1]
        
        cost1 = int(np.sum(a_int * b_int[p1]))
        cost2 = int(np.sum(a_int * b_int[p2]))
        
        if cost1 < cost2:
            p = p1
            best_cost = cost1
        else:
            p = p2
            best_cost = cost2
            
    except Exception:
        # Fallback to rank matching
        a_sums = A.sum(axis=1)
        b_sums = B.sum(axis=1)
        heavy_idx = np.argsort(-a_sums, kind="stable")
        central_idx = np.argsort(b_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy_idx] = central_idx
        best_cost = int(np.sum(a_int * b_int[p]))

    best_p = p.copy()

    # 2. Continuous Relaxation: Projected Gradient Descent on Birkhoff Polytope
    # We minimize f(X) = <A, B^T X B> subject to X in Birkhoff.
    # Gradient G = B B^T A^T ? 
    # Let's derive: f(X) = sum_{i,j,k,l} A[i,j] B[k,l] X[i,k] X[j,l]
    # df/dX[i,k] = sum_{j,l} A[i,j] B[k,l] X[j,l] + sum_{j,l} A[j,i] B[l,k] X[j,l]
    # If A, B symmetric: df/dX[i,k] = 2 * (A @ X @ B)[i,k]
    # So Gradient G = 2 * (A @ X @ B)
    
    # Initialize X with the discrete solution (one-hot)
    X = np.zeros((n, n), dtype=np.float64)
    X[np.arange(n), p] = 1.0
    
    # Project onto Birkhoff polytope (Doubly Stochastic Matrix)
    # Use Sinkhorn-Knopp algorithm
    def project_birkhoff(X, max_iter=100, tol=1e-5):
        # Normalize to ensure no zeros (add small epsilon)
        X = X + 1e-10
        u = np.ones(n)
        v = np.ones(n)
        for _ in range(max_iter):
            # Row scaling
            u = 1.0 / (X @ v)
            # Column scaling
            v = 1.0 / (X.T @ u)
            # Check convergence
            if np.max(np.abs(u - 1.0)) < tol and np.max(np.abs(v - 1.0)) < tol:
                break
        return X * np.outer(u, v)

    # Gradient Descent
    lr = 0.01
    step = 0
    max_steps = 5000
    
    # Precompute for speed
    # We need to update X, project, compute grad.
    
    while time.time() - start_time < time_limit and step < max_steps:
        # Compute Gradient: G = 2 * (A @ X @ B)
        # A and B are 150x150, X is 150x150.
        # A @ X is 150x150. (150x150) @ (150x150) ~ 3.3M ops.
        # (A @ X) @ B is another 3.3M ops.
        # Total ~ 6.6M ops per step.
        # 5000 steps * 6.6M = 33 Billion ops. Too slow in Python/NumPy for 150s?
        # NumPy matmul is optimized C. 150x150 matmul is very fast.
        # Let's estimate: 150^3 * 2 * 5000 = 3.3 * 10^9 flops. 
        # Modern CPU can do ~10-50 GFLOPS. So 0.1-0.3 seconds? 
        # Wait, 150^3 = 3,375,000.
        # 2 * 3,375,000 = 6,750,000 flops per matmul pair.
        # 5000 steps * 6.75M = 33.75 Gflops.
        # At 10 GFLOPS, this is 3.3 seconds. Feasible.
        
        G = 2.0 * (A @ X @ B)
        
        # Update X
        X_new = X - lr * G
        
        # Project to Birkhoff
        X_new = project_birkhoff(X_new)
        
        # Check for convergence (optional, skip to save time if budget allows)
        # if np.linalg.norm(X_new - X) < 1e-6:
        #     break
        
        X = X_new
        step += 1
        
        # Periodically recover discrete solution from X
        # Use Hungarian algorithm on -X (max weight matching)
        # Or just take the row maxes? No, that's not a permutation.
        # Hungarian is O(n^3) ~ 3.3M ops. Very fast.
        if step % 10 == 0:
            # Cost matrix for Hungarian: we want to maximize sum X[i, pi(i)]
            # So minimize sum -X[i, pi(i)]
            row_ind, col_ind = linear_sum_assignment(-X)
            p_cont = col_ind
            cost_cont = int(np.sum(a_int * b_int[p_cont]))
            
            if cost_cont < best_cost:
                best_cost = cost_cont
                best_p = p_cont.copy()
                # Restart continuous optimization from this discrete point?
                # Or just continue?
                # Let's restart X with this new discrete solution to exploit the basin
                X = np.zeros((n, n), dtype=np.float64)
                X[np.arange(n), best_p] = 1.0
                lr = max(lr * 0.5, 0.001) # Decay learning rate

    # 3. Discrete Local Search Polish
    # Use the best discrete solution found so far (from spectral or continuous recovery)
    p = best_p.copy()
    cost = best_cost
    
    # Fast 2-opt local search
    # Precompute row sums of B for potential speedups? 
    # Let's use a simple random 2-opt search with time limit.
    
    def get_delta_ij(i, j, p, A, B, c):
        u = p[i]
        v = p[j]
        
        # Diagonal terms
        delta_diag_i = A[i, i] * (B[v, v] - B[u, u])
        delta_diag_j = A[j, j] * (B[u, u] - B[v, v])
        
        # Sum over k != i, j
        diff_a = A[:, i] - A[:, j]
        diff_b = B[v, p] - B[u, p]
        
        total_dot = np.dot(diff_a, diff_b)
        
        term_i = (A[i, i] - A[i, j]) * (B[v, u] - B[u, u])
        term_j = (A[j, i] - A[j, j]) * (B[v, v] - B[u, v])
        
        sum_k = total_dot - term_i - term_j
        
        delta = delta_diag_i + delta_diag_j + sum_k
        return delta

    # Initialize c
    B_p = B_int[np.ix_(p, p)]
    c = (a_int * B_p).sum(axis=1).astype(np.int64)

    time_left = time_limit - (time.time() - start_time)
    if time_left > 5.0:
        # Perform local search
        # Use a larger batch size and more iterations
        while time.time() - start_time < time_limit:
            improved = False
            batch_size = 2000
            i_idx = np.random.randint(0, n, size=batch_size)
            j_idx = np.random.randint(0, n, size=batch_size)
            mask = i_idx == j_idx
            i_idx[mask] = (i_idx[mask] + 1) % n
            
            best_delta_in_batch = np.inf
            best_i = -1
            best_j = -1
            
            # Vectorized delta calculation is hard due to p dependency in diff_b
            # Looping 2000 times with O(n) vectorized ops is acceptable.
            # 2000 * 150 = 300k ops. Fast.
            
            for k in range(batch_size):
                i, j = i_idx[k], j_idx[k]
                if i == j:
                    continue
                d = get_delta_ij(i, j, p, A, B, c)
                if d < best_delta_in_batch:
                    best_delta_in_batch = d
                    best_i = i
                    best_j = j
                    
            if best_delta_in_batch < 0 and best_i != -1:
                i, j = best_i, best_j
                u, v = p[i], p[j]
                p[i], p[j] = v, u
                cost += int(best_delta_in_batch)
                improved = True
                
                # Update c
                B_p = B_int[np.ix_(p, p)]
                c = (a_int * B_p).sum(axis=1).astype(np.int64)
                
                if cost < best_cost:
                    best_cost = cost
                    best_p = p.copy()
                    
            # If no improvement in a batch, maybe restart with random perturbation?
            # For now, just continue.
            
    return best_p.tolist()

# EVOLVE-BLOCK-END
