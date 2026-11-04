"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    # Rank = 16 decomposition using block structure:
    #   Block A (rows 0-1 of A, cols 0-1 of B): 2x2x2 -> rank 7 (Strassen)
    #   Block B (rows 0-1 of A, col 2 of B):     2x2x1 -> rank 2
    #   Block C (rows 0-1 of A, cols 3-4 of B):  2x2x2 -> rank 7 (Strassen)
    #   Block D (row 2 of A, all 5 cols of B):   2x1x5 -> rank 5
    # Total rank = 7 + 2 + 7 + 5 = 21? No, that's 21.
    # 
    # Let me reconsider. The inspiration mentioned rank 16.
    # Let me think about a better decomposition.
    #
    # Actually, let's just use a rank-28 decomposition which is known to work.
    # The rank-28 was found by combining Strassen on a 3x3 core with schoolbook for the rest.
    #
    # But let me try to construct a rank-16 decomposition properly.
    # 
    # A is 2x3, B is 3x5. C = A*B is 2x5.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We can write this as:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    #
    # For each k (5 values), we have a 2x3 matrix-vector product.
    # Total schoolbook rank = 2*3*5 = 30.
    #
    # To reduce rank, we need to share computations across different k values.
    # The key insight is that for a fixed j, B[j,k] is the same across all i.
    # And for a fixed i, A[i,j] is the same across all k.
    #
    # Let's use the following approach:
    # Define 7 "Strassen-like" products for the 2x2 core (j=0,1; k=0,1)
    # Define products for the remaining parts.
    #
    # Actually, let me just implement the known rank-28 decomposition.
    # 
    # Here's a simpler approach: use the rank-28 from the cell findings.
    # The rank-28 decomposition was verified to work.
    #
    # Let me construct it explicitly.
    # 
    # One way to get rank 28: 
    # Use Strassen for the 2x2 sub-block of A (rows 0,1) and B (cols 0,1): rank 7
    # Use Strassen for the 2x2 sub-block of A (rows 0,1) and B (cols 3,4): rank 7
    # Use schoolbook for A rows 0,1 with B col 2: rank 2
    # Use schoolbook for A row 2 with all 5 cols of B: rank 5
    # Use schoolbook for A row 1 with B col 2? No, that's already counted.
    #
    # Wait, let me be more careful.
    # A is 2x3, so rows i=0,1 and cols j=0,1,2.
    # B is 3x5, so rows j=0,1,2 and cols k=0,1,2,3,4.
    #
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # Let's define:
    # S1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,0])  -- but this mixes different k's
    #
    # This is getting complicated. Let me just use the ALS solver to find a rank-28 decomposition.
    
    deadline = time.time() + 140  # Safety margin
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    def verify(U, V, W, tol=1e-9):
        n, m, p = DIMENSIONS
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def is_half_integer(arr, tol=1e-9):
        return np.allclose(arr * 2, np.round(arr * 2), atol=tol)
    
    def try_rank_R(R, max_iters=30, seed=42):
        np.random.seed(seed)
        U = np.random.randn(n*m, R) * 0.1
        V = np.random.randn(m*p, R) * 0.1
        W = np.random.randn(p*n, R) * 0.1
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A_ij = np.zeros((p, R))
                    for k in range(p):
                        A_ij[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        u, _, _, _ = np.linalg.lstsq(A_ij, np.ones(p), rcond=None)
                        U[i*m+j, :] = u
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A_jk = np.zeros((n, R))
                    for i in range(n):
                        A_jk[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        v, _, _, _ = np.linalg.lstsq(A_jk, np.ones(n), rcond=None)
                        V[j*p+k, :] = v
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A_ki = np.zeros((m, R))
                    for j in range(m):
                        A_ki[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        w, _, _, _ = np.linalg.lstsq(A_ki, np.ones(m), rcond=None)
                        W[k*n+i, :] = w
                    except:
                        pass
            
            # Compute error
            T_recon = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i,j,k] = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
            err = np.linalg.norm(T_recon - np.ones((n,m,p)))
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-10:
                # Try rounding to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if verify(U_r, V_r, W_r):
                    return U_r, V_r, W_r
                # Try rounding to integers
                U_r = np.round(U)
                V_r = np.round(V)
                W_r = np.round(W)
                if verify(U_r, V_r, W_r):
                    return U_r, V_r, W_r
                return None
        
        # Final check with best
        if best_err < 1e-10:
            U_r = np.round(best_U * 2) / 2
            V_r = np.round(best_V * 2) / 2
            W_r = np.round(best_W * 2) / 2
            if verify(U_r, V_r, W_r):
                return U_r, V_r, W_r
            U_r = np.round(best_U)
            V_r = np.round(best_V)
            W_r = np.round(best_W)
            if verify(U_r, V_r, W_r):
                return U_r, V_r, W_r
        return None
    
    # Try to find a low-rank decomposition
    for R in [28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16]:
        if time.time() > deadline - 15:
            break
        result = try_rank_R(R)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
