"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import random

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    total_eq = n * m * p
    schoolbook_rank = total_eq

    # Function to check if a decomposition is valid
    def is_valid(U, V, W, tol=1e-5):
        # Check if sum_c U[i,j,c] V[j,k,c] W[k,i,c] == 1
        # U shape: (nm, rank) -> U[i*m+j, c]
        # V shape: (mp, rank) -> V[j*p+k, c]
        # W shape: (pn, rank) -> W[k*n+i, c]
        
        # Reshape for easier indexing
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        
        # Compute tensor product
        # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        # We can do this with einsum
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        
        # Check if T is all 1s
        return np.allclose(T, 1.0, atol=tol)

    # Try to find a lower rank decomposition using ALS
    # We iterate from schoolbook_rank down to a limit
    # To save time, we only try a few ranks and attempts
    min_rank_to_try = max(28, schoolbook_rank - 20)
    
    for rank in range(schoolbook_rank, min_rank_to_try - 1, -1):
        if rank == schoolbook_rank:
            continue # Skip schoolbook, we have a fallback
        
        for attempt in range(10):
            # Random initialization
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # ALS Iterations
            converged = False
            for it in range(50):
                # Solve for U
                # For each (i,j), we have a linear system for U[i*m+j, :]
                # Actually, the system is coupled.
                # Let's construct the linear system A_U * vec(U) = b
                # b is all 1s of size n*m*p
                # A_U has rows for each (i,j,k) and columns for each (i',j',c)
                # A_U[(i,j,k), (i',j',c)] = V[j*p+k, c] * W[k*n+i, c] if i==i' and j==j' else 0
                # This is a block diagonal structure if we group by (i,j)?
                # No, W depends on i. So U[i,j,c] couples with W[k,i,c].
                # The system is not block diagonal in (i,j).
                
                # Efficient way:
                # Fix V, W.
                # For each c, U[:, c] is a column.
                # The equation is sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1
                # Let M_c be a matrix of size nm x np? No.
                # Let's just use np.linalg.lstsq on the full system.
                
                # Construct A_U
                # Size: (n*m*p) x (n*m*rank)
                A_U = np.zeros((n * m * p, n * m * rank))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row_idx = i * m * p + j * p + k
                            for c in range(rank):
                                col_idx = (i * m + j) * rank + c
                                A_U[row_idx, col_idx] = V[j * p + k, c] * W[k * n + i, c]
                
                b = np.ones(n * m * p)
                sol, _, _, _ = np.linalg.lstsq(A_U, b, rcond=None)
                U = sol.reshape(nm, rank)
                
                # Solve for V
                A_V = np.zeros((n * m * p, m * p * rank))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row_idx = i * m * p + j * p + k
                            for c in range(rank):
                                col_idx = (j * p + k) * rank + c
                                A_V[row_idx, col_idx] = U[i * m + j, c] * W[k * n + i, c]
                
                sol, _, _, _ = np.linalg.lstsq(A_V, b, rcond=None)
                V = sol.reshape(mp, rank)
                
                # Solve for W
                A_W = np.zeros((n * m * p, p * n * rank))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row_idx = i * m * p + j * p + k
                            for c in range(rank):
                                col_idx = (k * n + i) * rank + c
                                A_W[row_idx, col_idx] = U[i * m + j, c] * V[j * p + k, c]
                
                sol, _, _, _ = np.linalg.lstsq(A_W, b, rcond=None)
                W = sol.reshape(pn, rank)
                
                if is_valid(U, V, W, tol=1e-4):
                    converged = True
                    break
            
            if converged:
                # Try to round to nearest 0.5
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                if is_valid(U_rounded, V_rounded, W_rounded, tol=1e-5):
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                else:
                    # Return unrounded if it's close enough? 
                    # The task requires integer/half-integer entries.
                    # So we must return rounded. If it fails, we discard this solution.
                    pass

    # Fall back to schoolbook
    rank = schoolbook_rank
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
