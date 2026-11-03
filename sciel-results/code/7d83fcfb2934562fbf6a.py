import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor T_ijk = 1 for all i,j,k
    # We want U, V, W such that sum_c U_ij^c V_jk^c W_ki^c = 1
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Lower bound for rank is max(n*m, m*p, p*n) = 15
    # Known upper bound is 45
    # We search from 44 down to 15
    
    best_rank = 45
    best_U, best_V, best_W = None, None, None
    
    # Initialize with schoolbook
    U0 = np.zeros((n*m, n*m*p))
    V0 = np.zeros((m*p, n*m*p))
    W0 = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U0, V0, W0
    best_rank = n*m*p

    def check_decomposition(U, V, W):
        """Check if U, V, W exactly decompose the tensor."""
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        R = U.shape[1]
        # Compute tensor
        # T_ijk = sum_c U_ij^c V_jk^c W_ki^c
        # We can compute this efficiently using einsum
        # U_ij^c -> U[i*m+j, c]
        # V_jk^c -> V[j*p+k, c]
        # W_ki^c -> W[k*n+i, c]
        
        # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U_reshaped = U.reshape(n, m, R)
        V_reshaped = V.reshape(m, p, R)
        W_reshaped = W.reshape(p, n, R)
        
        # Compute sum over c
        # T_ijk = sum_c U_reshaped[i,j,c] * V_reshaped[j,k,c] * W_reshaped[k,i,c]
        # We can do this in steps
        # M_jk = sum_c V_reshaped[j,k,c] * W_reshaped[k,i,c] ? No, W depends on i.
        
        # Let's use einsum
        # T_ijk = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Check if T is all 1s
        if np.allclose(T, 1.0, atol=1e-9):
            return True
        return False

    def round_to_half_int(M):
        """Round matrix to nearest half-integer."""
        return np.round(M * 2) / 2.0

    def solve_linear_block(U, V, W, R):
        """
        Update U, V, W using block coordinate descent.
        Returns updated U, V, W.
        """
        U_new = U.copy()
        V_new = V.copy()
        W_new = W.copy()
        
        # Solve for U
        # For each (i,j), we have system: sum_c U_ij^c * (V_jk^c * W_ki^c) = 1 for all k
        # Let A_ij be a matrix of size (p, R) where A_ij[k, c] = V_jk^c * W_ki^c
        # We want U_ij such that A_ij @ U_ij = 1 (vector of 1s of size p)
        
        for i in range(n):
            for j in range(m):
                # Construct A_ij
                # V_jk^c: V[j*p+k, c]
                # W_ki^c: W[k*n+i, c]
                V_block = V[j*p:(j+1)*p, :R]  # (p, R)
                W_block = W[i:(i+n)*n:n, :R]  # Wait, W index is k*n+i
                # W is (p*n, R). Index k*n+i.
                # For fixed i, k varies 0..p-1.
                # W[k*n+i, c]
                W_block = np.array([W[k*n+i, :R] for k in range(p)])  # (p, R)
                
                A_ij = V_block * W_block  # (p, R)
                
                # Solve A_ij @ u = 1
                # Least squares
                try:
                    u_sol, residuals, rank_A, s = np.linalg.lstsq(A_ij, np.ones(p), rcond=None)
                    U_new[i*m+j, :R] = u_sol
                except:
                    pass
        
        U = U_new
        
        # Solve for V
        # For each (j,k), sum_c V_jk^c * (U_ij^c * W_ki^c) = 1 for all i
        # A_jk[i, c] = U_ij^c * W_ki^c
        for j in range(m):
            for k in range(p):
                V_block = V[j*p+k, :R]
                # We need U_ij^c for all i, and W_ki^c for all i
                # U is (n*m, R). Index i*m+j.
                # W is (p*n, R). Index k*n+i.
                
                A_jk = np.zeros((n, R))
                for i in range(n):
                    U_block = U[i*m+j, :R]
                    W_block = W[k*n+i, :R]
                    A_jk[i, :] = U_block * W_block
                
                try:
                    v_sol, residuals, rank_A, s = np.linalg.lstsq(A_jk, np.ones(n), rcond=None)
                    V_new[j*p+k, :R] = v_sol
                except:
                    pass
        
        V = V_new
        
        # Solve for W
        # For each (k,i), sum_c W_ki^c * (U_ij^c * V_jk^c) = 1 for all j
        # A_ki[j, c] = U_ij^c * V_jk^c
        for k in range(p):
            for i in range(n):
                W_block = W[k*n+i, :R]
                
                A_ki = np.zeros((m, R))
                for j in range(m):
                    U_block = U[i*m+j, :R]
                    V_block = V[j*p+k, :R]
                    A_ki[j, :] = U_block * V_block
                
                try:
                    w_sol, residuals, rank_A, s = np.linalg.lstsq(A_ki, np.ones(m), rcond=None)
                    W_new[k*n+i, :R] = w_sol
                except:
                    pass
        
        W = W_new
        
        return U, V, W

    # Search for lower rank
    for R in range(44, 14, -1):
        if time.time() > deadline:
            break
            
        # Try several random restarts for this rank
        for restart in range(10):
            if time.time() > deadline:
                break
                
            # Initialize with random small integers
            U = np.random.randint(-2, 3, size=(n*m, R))
            V = np.random.randint(-2, 3, size=(m*p, R))
            W = np.random.randint(-2, 3, size=(p*n, R))
            
            # Run block coordinate descent
            for iter in range(50):
                U, V, W = solve_linear_block(U, V, W, R)
                
                # Round to half-integers
                U_r = round_to_half_int(U)
                V_r = round_to_half_int(V)
                W_r = round_to_half_int(W)
                
                # Check exactness
                if check_decomposition(U_r, V_r, W_r):
                    best_rank = R
                    best_U, best_V, best_W = U_r, V_r, W_r
                    break
            
            if best_rank == R:
                break  # Found a decomposition for this rank, try lower
        
        if best_rank == R:
            break  # Found a decomposition for this rank, try lower

    # Convert to lists of integers/half-integers
    # Ensure they are floats or ints as required by the interface
    # The interface expects lists of lists
    
    def to_list(M):
        return M.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
