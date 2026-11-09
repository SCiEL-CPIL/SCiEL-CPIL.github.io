import numpy as np
import time
import random

def construct():
    n, m, p = 3, 4, 4
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Trivial solution rank 48
    best_rank = n * m * p
    best_solution = None
    
    def make_trivial_solution(rank):
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
        return U, V, W
    
    U_triv, V_triv, W_triv = make_trivial_solution(best_rank)
    best_solution = (U_triv, V_triv, W_triv)
    
    # Precompute indices
    # Equations: (i, j, k)
    # i in 0..n-1, j in 0..m-1, k in 0..p-1
    eqs = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eqs.append((i, j, k))
                
    def solve_for_W(U, V, r):
        # A shape: (48, 12*r)
        A = np.zeros((n * m * p, p * n * r))
        for e_idx, (i, j, k) in enumerate(eqs):
            u_idx = i * m + j
            v_idx = j * p + k
            # For each c, the variable is W[k*n+i, c]
            # Its coefficient is U[u_idx, c] * V[v_idx, c]
            for c in range(r):
                w_var_idx = (k * n + i) * r + c
                A[e_idx, w_var_idx] = U[u_idx, c] * V[v_idx, c]
        
        b = np.ones(n * m * p)
        # Solve least squares
        try:
            W_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            W = W_flat.reshape((p * n, r))
            return W
        except Exception:
            return np.zeros((p * n, r))
            
    def solve_for_V(U, W, r):
        # Variable V[j*p+k, c]
        # Coefficient in eq (i,j,k) is U[i*m+j, c] * W[k*n+i, c]
        A = np.zeros((n * m * p, m * p * r))
        for e_idx, (i, j, k) in enumerate(eqs):
            u_idx = i * m + j
            w_idx = k * n + i
            for c in range(r):
                v_var_idx = (j * p + k) * r + c
                A[e_idx, v_var_idx] = U[u_idx, c] * W[w_idx, c]
        
        b = np.ones(n * m * p)
        try:
            V_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            V = V_flat.reshape((m * p, r))
            return V
        except Exception:
            return np.zeros((m * p, r))
            
    def solve_for_U(V, W, r):
        # Variable U[i*m+j, c]
        # Coefficient in eq (i,j,k) is V[j*p+k, c] * W[k*n+i, c]
        A = np.zeros((n * m * p, n * m * r))
        for e_idx, (i, j, k) in enumerate(eqs):
            v_idx = j * p + k
            w_idx = k * n + i
            for c in range(r):
                u_var_idx = (i * m + j) * r + c
                A[e_idx, u_var_idx] = V[v_idx, c] * W[w_idx, c]
        
        b = np.ones(n * m * p)
        try:
            U_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            U = U_flat.reshape((n * m, r))
            return U
        except Exception:
            return np.zeros((n * m, r))

    def verify_solution(U, V, W, r):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for c in range(r):
                        s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                    max_err = max(max_err, abs(s - 1.0))
        return max_err

    # Search for lower ranks
    # Lower bound is roughly 16, but let's try from 20 to 47
    # We start from 47 down to 16 to find the minimal quickly?
    # Or 16 up to 47?
    # If we find a solution at 47, we break.
    # It is more likely to find a solution near 48 than 16.
    # So iterate downwards from 47 to 16.
    
    for rank in range(47, 15, -1):
        if time.time() > deadline - 5:
            break
            
        # Try a few random initializations
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
                
            random.seed(rank * 100 + attempt)
            np.random.seed(rank * 100 + attempt)
            
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # ALS iterations
            converged = False
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                    
                W_new = solve_for_W(U, V, rank)
                V_new = solve_for_V(U, W_new, rank)
                U_new = solve_for_U(V_new, W_new, rank)
                
                # Check change
                diff = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
                U, V, W = U_new, V_new, W_new
                
                if diff < 1e-8:
                    converged = True
                    break
            
            if converged:
                err = verify_solution(U, V, W, rank)
                if err < 1e-8:
                    # Try to round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    err_r = verify_solution(U_r, V_r, W_r, rank)
                    if err_r < 1e-8:
                        best_rank = rank
                        best_solution = (U_r, V_r, W_r)
                        break # Found valid solution at this rank, try lower
        
        if best_solution and best_rank < 48:
             # We found a solution, but we want the MINIMAL rank.
             # So we continue searching lower ranks.
             pass

    # If we found a solution better than 48, return it.
    # Otherwise return the trivial one.
    if best_solution:
        U, V, W = best_solution
    else:
        U, V, W = make_trivial_solution(48)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
