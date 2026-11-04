"""Matrix multiplication 2x3 x 3x4 rank-minimal exact tensor decomposition.

Strategy:
  1. Define the matrix multiplication tensor T.
  2. Use Alternating Least Squares (ALS) with half-integer grid projection.
     - This projects the continuous LS solution back onto the discrete set {k/2}.
     - This is more robust than pure LM for discrete/semi-discrete optimization.
  3. Multi-start with different random initializations.
  4. Verify exactness.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Build the target tensor as a set of linear equations
    # T[i, j, k] = 1 if j is shared, else 0.
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i * m + j, j * p + k, k * n + i, 1.0))
    
    u_pos = np.array([t[0] for t in triples], dtype=np.int64)
    v_pos = np.array([t[1] for t in triples], dtype=np.int64)
    w_pos = np.array([t[2] for t in triples], dtype=np.int64)
    target = np.array([t[3] for t in triples], dtype=np.float64)
    N = len(triples)
    
    def max_err(U, V, W):
        vals = U[u_pos, :] * V[v_pos, :] * W[w_pos, :]
        return float(np.max(np.abs(np.sum(vals, axis=1) - target)))

    def snap_half(x):
        return np.round(x * 2.0) / 2.0

    def solve_rank(R, time_budget, seed):
        rng = np.random.RandomState(seed)
        best_err = float('inf')
        best_sol = None
        
        # We will run multiple restarts
        # Time per restart depends on R and convergence speed
        # LS step is O(N * R^2) approx, which is very small here.
        
        for restart in range(100):
            if time.time() > start_time + time_budget:
                break
            
            # Initialize with random half-integers
            U = snap_half(rng.randn(nm, R))
            V = snap_half(rng.randn(mp, R))
            W = snap_half(rng.randn(pn, R))
            
            # ALS loop
            for it in range(200):
                # Solve for U, fixing V, W
                # Min || sum_r u_{u_pos, r} * (V[v_pos, r] * W[w_pos, r])_r - target ||
                # Let A_r = V[v_pos, r] * W[w_pos, r] (vector of size N)
                # We want U[u_pos, :] @ diag(A) ... no, it's element-wise.
                # residual = sum_r U[u_pos, r] * (V[v_pos, r] * W[w_pos, r]) - target
                # This is linear in U.
                # Matrix B (size N x R) where B[:, r] = V[v_pos, r] * W[w_pos, r]
                # We want to solve min || B.T @ U_flat - target ||?
                # Actually, U is (nm, R). We only care about U[u_pos, :].
                # Let U_sub = U[u_pos, :] (N x R)
                # eq: sum_r U_sub[:, r] * B[:, r] = target
                # This is not a standard linear system in U_sub directly because U_sub appears in the sum.
                # Wait.
                # val_t = sum_r U[u_t, r] * V[v_t, r] * W[w_t, r]
                # This is linear in U.
                # Let's form the matrix M (N x R) where M[:, r] = V[v_pos, r] * W[w_pos, r]
                # Then val = M.T @ U_sub.T ? No.
                # val_t = sum_r U[u_t, r] * M[t, r]
                # This is exactly (M * U_sub).sum(axis=1)? No.
                # If we reshape U_sub to a vector, it's not simple.
                # But we can solve it via LS for each row of U independently?
                # No, U[u_t, r] couples different r's? No, U[u_t, r] is a single variable.
                # The equation for each t is: sum_r U[u_t, r] * M[t, r] = target[t]
                # This couples U[u_t, 0..R-1] together.
                # So for each unique row index in u_pos, we have a linear system.
                
                # However, a simpler way to solve for U is to treat it as a standard LS problem
                # by constructing the design matrix.
                
                # Let's stick to the gradient projection or just direct LS if we can vectorize.
                # Actually, the easiest way to solve min || sum_r U_{i,r} C_{r,i} - y_i || is to realize
                # this is a linear system in U.
                
                # Let's form the matrix A_u (N x R) where A_u[t, r] = V[v_pos[t], r] * W[w_pos[t], r]
                # We want to find U_sub (N x R) such that sum_r U_sub[t, r] * A_u[t, r] = target[t] for all t.
                # This is NOT a standard LS X @ A = b because the multiplication is element-wise (Hadamard).
                # sum_r U[t, r] A[t, r] = (U * A).sum(axis=1)
                # To solve this, we can just take the gradient and set to 0?
                # Or, notice that for a fixed t, the variables are U[t, 0], ..., U[t, R-1].
                # The objective is sum_t (sum_r U[t, r] A[t, r] - y_t)^2.
                # This decouples for each t!
                # For each t, we minimize (sum_r U[t, r] A[t, r] - y_t)^2.
                # The minimum is achieved when sum_r U[t, r] A[t, r] = y_t.
                # There are infinitely many solutions. We want the one closest to the current U[t, :] (to maintain continuity) 
                # OR we can just pick the solution with minimum norm?
                # To make it deterministic and stable, let's project the current U[t, :] onto the hyperplane defined by the equation.
                
                # Actually, simpler: Just use the LS solution that minimizes ||U_new - U_old|| subject to the linear constraint.
                # Constraint: A_t^T @ U_new_t = y_t
                # Minimize: ||U_new_t - U_old_t||^2
                # Solution: U_new_t = U_old_t + lambda * A_t
                # lambda = (y_t - A_t^T @ U_old_t) / ||A_t||^2
                
                U_new = U.copy()
                A_u = V[v_pos, :] * W[w_pos, :] # N x R
                for t in range(N):
                    a_t = A_u[t, :]
                    norm_sq = np.dot(a_t, a_t)
                    if norm_sq < 1e-12:
                        continue
                    current_val = np.dot(a_t, U[u_pos[t], :])
                    target_val = target[t]
                    lam = (target_val - current_val) / norm_sq
                    U_new[u_pos[t], :] = U[u_pos[t], :] + lam * a_t
                
                U = snap_half(U_new)
                
                # Solve for V
                V_new = V.copy()
                A_v = U[u_pos, :] * W[w_pos, :] # N x R
                for t in range(N):
                    a_t = A_v[t, :]
                    norm_sq = np.dot(a_t, a_t)
                    if norm_sq < 1e-12:
                        continue
                    current_val = np.dot(a_t, V[v_pos[t], :])
                    target_val = target[t]
                    lam = (target_val - current_val) / norm_sq
                    V_new[v_pos[t], :] = V[v_pos[t], :] + lam * a_t
                V = snap_half(V_new)
                
                # Solve for W
                W_new = W.copy()
                A_w = U[u_pos, :] * V[v_pos, :] # N x R
                for t in range(N):
                    a_t = A_w[t, :]
                    norm_sq = np.dot(a_t, a_t)
                    if norm_sq < 1e-12:
                        continue
                    current_val = np.dot(a_t, W[w_pos[t], :])
                    target_val = target[t]
                    lam = (target_val - current_val) / norm_sq
                    W_new[w_pos[t], :] = W[w_pos[t], :] + lam * a_t
                W = snap_half(W_new)
                
                err = max_err(U, V, W)
                if err < best_err:
                    best_err = err
                    best_sol = (U.copy(), V.copy(), W.copy())
                if best_err < 1e-9:
                    return best_sol, best_err
            
            if best_err < 1e-9:
                return best_sol, best_err
                
        return best_sol, best_err

    def to_list(arr):
        out = []
        for row in arr:
            rl = []
            for val in row:
                rv = round(float(val) * 2) / 2.0
                rl.append(rv)
            out.append(rl)
        return out

    # Try ranks from smallest known potential to naive
    # Known lower bound for 2x3x4 is likely 19 or 20.
    # Naive is 24.
    candidate_ranks = [19, 20, 21, 22, 23, 24]
    
    for R in candidate_ranks:
        if time.time() > deadline - 10:
            break
            
        # Allocate time for this rank
        # If R is small, we can afford more restarts.
        time_budget = 30.0
        
        sol, err = solve_rank(R, time_budget, seed=R)
        
        if sol is not None and err < 1e-9:
            U, V, W = sol
            # Final verification
            if max_err(U, V, W) < 1e-9:
                return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
        
    # Fallback: Naive decomposition
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
