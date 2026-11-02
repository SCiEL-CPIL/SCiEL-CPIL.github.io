import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Build the tensor T
    # T[a,b,c] where a in [0, n*m), b in [0, m*p), c in [0, p*n)
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # Target rank
    target_rank = 29
    
    # Pre-compute indices for verification
    # We need to check sum_r U[a,r] V[b,r] W[c,r] == T[a,b,c]
    
    def verify_decomposition(U, V, W):
        """
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        Returns True if decomposition is exact (within float precision, assuming half-integers)
        """
        # Use integer arithmetic for exactness
        # Multiply by 8 to clear halves (assuming inputs are multiples of 0.5)
        # Actually, let's just check with high precision float
        # If we rounded to 0.5, the values are exact in binary?
        # 0.5 is exact. 1.5 is exact.
        # So we can use float64.
        
        # Compute T_hat
        # T_hat[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Efficient computation:
        # For each r, add outer product U[:,r] * V[:,r] * W[:,r]
        # Or use einsum
        T_hat = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check if T_hat matches T
        # Since T is 0 or 1, and T_hat should be 0 or 1
        # Check max difference
        diff = np.abs(T_hat - T)
        if np.max(diff) < 1e-9:
            return True
        return False

    def round_to_half_int(M):
        """Round matrix entries to nearest 0.5"""
        return np.round(M * 2) / 2

    def als_decompose(T, rank, max_iters=1000, tol=1e-10):
        """
        Alternating Least Squares for tensor decomposition.
        T: (I, J, K)
        Returns U, V, W
        """
        I, J, K = T.shape
        # Initialize randomly
        # Use small random values to avoid division by zero issues in early steps
        U = np.random.randn(I, rank) * 0.1
        V = np.random.randn(J, rank) * 0.1
        W = np.random.randn(K, rank) * 0.1
        
        prev_err = float('inf')
        
        for it in range(max_iters):
            # Update U
            for a in range(I):
                # T[a, :, :] is (J, K)
                # We want to solve min_Ua || T[a,:,:] - sum_r Ua[r] * V[:,r] * W[:,r] ||_F
                # Vectorize T[a,:,:] to (J*K,)
                T_a = T[a, :, :].flatten()
                # Construct matrix M_a of size (J*K, rank)
                # M_a[idx, r] = V[idx//K, r] * W[idx%K, r]
                # This is slow in Python loop. Use vectorization.
                # V_flat: (J, rank) -> (J*K, rank) by repeating each row K times
                # W_flat: (K, rank) -> (J*K, rank) by tiling
                V_rep = np.repeat(V, K, axis=0)
                W_tile = np.tile(W, (J, 1))
                M_a = V_rep * W_tile
                # Solve M_a @ Ua = T_a
                U[a, :], residuals, rank_solve, sv = np.linalg.lstsq(M_a, T_a, rcond=None)
                
            # Update V
            for b in range(J):
                T_b = T[:, b, :].flatten() # Shape (I*K)
                # M_b of size (I*K, rank)
                # M_b[idx, r] = U[idx//K, r] * W[idx%K, r]
                U_rep = np.repeat(U, K, axis=0)
                W_tile = np.tile(W, (I, 1))
                M_b = U_rep * W_tile
                V[b, :], residuals, rank_solve, sv = np.linalg.lstsq(M_b, T_b, rcond=None)
                
            # Update W
            for c in range(K):
                T_c = T[:, :, c].flatten() # Shape (I*J)
                # M_c of size (I*J, rank)
                # M_c[idx, r] = U[idx//J, r] * V[idx%J, r]
                U_rep = np.repeat(U, J, axis=0)
                V_tile = np.tile(V, (I, 1))
                M_c = U_rep * V_tile
                W[c, :], residuals, rank_solve, sv = np.linalg.lstsq(M_c, T_c, rcond=None)
                
            # Check convergence
            # Compute error
            # T_hat = np.einsum('ar,br,cr->abc', U, V, W)
            # err = np.linalg.norm(T_hat - T)
            # If err < tol: break
            
            # Optimization: Check time
            if time.time() > deadline:
                break
                
        return U, V, W

    best_result = None
    
    # Try to find rank 29 solution
    np.random.seed(42)
    
    max_restarts = 100
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        U, V, W = als_decompose(T, target_rank, max_iters=200)
        
        # Round to half-integers
        U_rounded = round_to_half_int(U)
        V_rounded = round_to_half_int(V)
        W_rounded = round_to_half_int(W)
        
        # Verify
        if verify_decomposition(U_rounded, V_rounded, W_rounded):
            best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
            break # Found rank 29, stop
            
        # Also check if unrounded is very close, maybe rounding was too aggressive?
        # But the problem requires integer/half-integer.
        # If unrounded is close, maybe we can try snapping to integers or halves more carefully?
        # For now, if rounded fails, we assume this restart failed.
        
    if best_result is not None:
        return best_result
        
    # Fall back to schoolbook rank-30
    rank = n * m * p
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
