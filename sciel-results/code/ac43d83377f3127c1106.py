import numpy as np
import time

DIMENSIONS = [2, 2, 7]

def construct():
    """
    Attempt to find a rank-24 decomposition. If failed, return rank-25.
    """
    n, m, p = 2, 2, 7
    
    # Build the target tensor T (4, 14, 14)
    T = np.zeros((4, 14, 14))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # Try Rank 24
    rank_target = 24
    start_time = time.time()
    time_limit = 10.0 # seconds
    
    found_solution = None
    for restart in range(20): # 20 restarts
        if time.time() - start_time > time_limit:
            break
            
        # Initialize U, V, W with random values
        U = np.random.randn(4, rank_target) * 0.1
        V = np.random.randn(14, rank_target) * 0.1
        W = np.random.randn(14, rank_target) * 0.1
        
        # ALS Loop
        for it in range(200):
            # Update U
            A = np.zeros((196, rank_target))
            for r in range(rank_target):
                M_r = np.outer(V[:, r], W[:, r])
                A[:, r] = M_r.flatten()
                
            for a in range(4):
                t_a = T[a, :, :].flatten()
                U[a, :], residuals, rank, sv = np.linalg.lstsq(A, t_a, rcond=None)
                
            # Update V
            A_V = np.zeros((56, rank_target))
            for r in range(rank_target):
                N_r = np.outer(U[:, r], W[:, r])
                A_V[:, r] = N_r.flatten()
                
            for b in range(14):
                t_b = T[:, b, :].flatten()
                V[b, :], residuals, rank, sv = np.linalg.lstsq(A_V, t_b, rcond=None)
                
            # Update W
            A_W = np.zeros((56, rank_target))
            for r in range(rank_target):
                N_r = np.outer(U[:, r], V[:, r])
                A_W[:, r] = N_r.flatten()
                
            for c in range(14):
                t_c = T[:, :, c].flatten()
                W[c, :], residuals, rank, sv = np.linalg.lstsq(A_W, t_c, rcond=None)
                
            # Check residual
            T_est = np.zeros((4, 14, 14))
            for r in range(rank_target):
                T_est += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                
            err = np.linalg.norm(T - T_est)
            if err < 1e-9:
                found_solution = (U, V, W)
                break
                
        if found_solution is not None:
            break
            
    if found_solution is not None:
        U, V, W = found_solution
        # Round to half-integers
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Verify exactness with rounded values
        T_est = np.zeros((4, 14, 14))
        for r in range(rank_target):
            T_est += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
        if np.allclose(T, T_est, atol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to Rank 25
    return _construct_rank_25()

def _construct_rank_25():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    Strategy: Strassen rank-7 for columns (0,1), (2,3), (4,5); schoolbook rank-4 for column 6.
    Total rank: 7 + 7 + 7 + 4 = 25
    """
    n, m, p = 2, 2, 7
    rank = 25
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Indexing:
    # A[i,j] -> U[2*i+j]
    # B[j,k] -> V[7*j+k]
    # C[i,k] -> W[2*k+i]
    
    def add_strassen(col_start, k_start):
        """Add Strassen rank-7 terms for 2x2 block at columns k_start, k_start+1."""
        
        # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
        # C0,ks += M1, C1,ks+1 += M1
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1

        # M2 = (A10 + A11) * B0,ks
        # C1,ks += M2, C1,ks+1 -= M2
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1

        # M3 = A00 * (B0,ks+1 - B1,ks+1)
        # C0,ks+1 += M3, C1,ks+1 += M3
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1

        # M4 = A11 * (B1,ks - B0,ks)
        # C0,ks += M4, C1,ks += M4
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1

        # M5 = (A00 + A01) * B1,ks+1
        # C0,ks -= M5, C0,ks+1 += M5
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1

        # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
        # C1,ks+1 += M6
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1

        # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
        # C0,ks += M7
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1
        
    # Add Strassen for columns 0,1
    add_strassen(0, 0)
    
    # Add Strassen for columns 2,3
    add_strassen(7, 2)
    
    # Add Strassen for columns 4,5
    add_strassen(14, 4)
    
    # Add schoolbook for column 6 (4 terms)
    col_start = 21
    # A00 * B0,6 -> C0,6
    U[0, col_start+0] += 1
    V[7*0 + 6, col_start+0] += 1
    W[2*6 + 0, col_start+0] += 1
    
    # A01 * B1,6 -> C0,6
    U[1, col_start+1] += 1
    V[7*1 + 6, col_start+1] += 1
    W[2*6 + 0, col_start+1] += 1
    
    # A10 * B0,6 -> C1,6
    U[2, col_start+2] += 1
    V[7*0 + 6, col_start+2] += 1
    W[2*6 + 1, col_start+2] += 1
    
    # A11 * B1,6 -> C1,6
    U[3, col_start+3] += 1
    V[7*1 + 6, col_start+3] += 1
    W[2*6 + 1, col_start+3] += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
