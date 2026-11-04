"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly reconstructs the matrix multiplication tensor.
    The tensor T[i, j, k] = 1 if and only if the (i,j) entry of A and (j,k) entry of B
    contribute to (i,k) entry of C. Actually, the multiplication tensor has T[i,j,k] = 1
    for all i in [0,n), j in [0,m), k in [0,p).
    
    The reconstruction is: for each term c, U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    summed over c should equal 1 for all (i,j,k).
    
    We check this by computing the full tensor and comparing to the all-ones tensor.
    """
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the reconstructed tensor
    # T_recon[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    T_recon = np.zeros((n, m, p), dtype=float)
    
    for c in range(rank):
        uc = U[:, c]  # shape (n*m,)
        vc = V[:, c]  # shape (m*p,)
        wc = W[:, c]  # shape (p*n,)
        
        for i in range(n):
            for j in range(m):
                u_val = uc[i * m + j]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = vc[j * p + k]
                    w_val = wc[k * n + i]
                    T_recon[i, j, k] += u_val * v_val * w_val
    
    # The true multiplication tensor is all ones
    T_true = np.ones((n, m, p), dtype=float)
    
    return np.allclose(T_recon, T_true, atol=1e-10)


def _try_rank(n, m, p, rank, max_iter=500, tol=1e-10, seed=0):
    """
    Try to find an exact rank-r decomposition using HOOI-style iteration.
    Returns (U, V, W) as lists if successful, else None.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize with random factors
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # The target tensor T[i, j, k] = 1 for all i, j, k
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = float('inf')
    
    for it in range(max_iter):
        # Update U: for each (i,j), U[i*m+j, :] should satisfy
        # sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1 for all k
        # This is a least squares problem: U[i*m+j, :] @ (V_j * W_i)^T = 1^T
        # where V_j is the row of V for index j, and W_i is the row of W for index i
        # Actually, we need to be more careful.
        
        # Let's define for each (i, j):
        # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k in [0,p)
        
        # This is: U[i*m+j, :] @ M_{i,j} = ones(p)
        # where M_{i,j}[k, c] = V[j*p+k, c] * W[k*n+i, c]
        
        for i in range(n):
            for j in range(m):
                # Build M_{i,j} of shape (p, rank)
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # Solve U[i*m+j, :] @ M = ones(p)
                # i.e., M^T @ U[i*m+j, :]^T = ones(p)
                # Use least squares
                try:
                    sol, residuals, rank_ls, sv = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                    U[i * m + j, :] = sol
                except:
                    pass
        
        # Update V: for each j, we need for all (i, k):
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # For fixed j, this is: V[j*p+k, :] @ N_{j,k} = ... hmm, this is coupled.
        # Actually, for fixed j and k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i
        # This is: V[j*p+k, :] @ (U_j * W_k)^T = ones(n)
        # where U_j[i, c] = U[i*m+j, c] and W_k[i, c] = W[k*n+i, c]
        
        for j in range(m):
            for k in range(p):
                N = np.zeros((n, rank))
                for i in range(n):
                    N[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    sol, residuals, rank_ls, sv = np.linalg.lstsq(N.T, np.ones(n), rcond=None)
                    V[j * p + k, :] = sol
                except:
                    pass
        
        # Update W: for each k, we need for all (i, j):
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # For fixed k and i: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
        # This is: W[k*n+i, :] @ (U_i * V_k)^T = ones(m)
        # where U_i[j, c] = U[i*m+j, c] and V_k[j, c] = V[j*p+k, c]
        
        for k in range(p):
            for i in range(n):
                P = np.zeros((m, rank))
                for j in range(m):
                    P[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    sol, residuals, rank_ls, sv = np.linalg.lstsq(P.T, np.ones(m), rcond=None)
                    W[k * n + i, :] = sol
                except:
                    pass
        
        # Check error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (val - 1.0) ** 2
        err = np.sqrt(err)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
    
    if best_err < tol:
        # Convert to lists
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        # Verify
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # The schoolbook rank is n*m*p = 50
    schoolbook_rank = n * m * p
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # But we can try lower
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower_bound to schoolbook_rank
    for rank in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random seeds
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            
            result = _try_rank(n, m, p, rank, max_iter=200, tol=1e-10, seed=seed)
            if result is not None:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
