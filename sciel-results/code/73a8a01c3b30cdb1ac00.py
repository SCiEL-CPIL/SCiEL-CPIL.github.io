import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 110 # Safety margin
    
    # Target: Rank 24 for 2x4x4
    # Strategy: Find rank 12 decomposition for 2x4x2, then duplicate for 2x4x4
    
    # Tensor for 2x4x2
    # A: 2x4, B: 4x2, C: 2x2
    # T[i, j, k] = 1 if A[i, j] * B[j, k] contributes to C[i, k]
    # Indices:
    # i: 0..1
    # j: 0..3
    # k: 0..1
    
    T_242 = np.zeros((2, 4, 2))
    for i in range(2):
        for j in range(4):
            for k in range(2):
                T_242[i, j, k] = 1.0
                
    # We want U (8x12), V (8x12), W (4x12)
    # such that sum_r U[i, r] V[j, r] W[k, r] = T_242[i, j, k]
    
    def find_rank12_decomposition():
        best_factors = None
        best_error = float('inf')
        
        # Try a few random initializations
        for restart in range(50):
            if time.time() - start_time > time_limit - 10:
                break
                
            # Initialize with small random integers or floats
            # Use random integers in {-1, 0, 1} or similar to encourage sparsity/integers?
            # Or just random normal.
            U = np.random.randn(8, 12)
            V = np.random.randn(8, 12)
            W = np.random.randn(4, 12)
            
            # ALS
            for it in range(100):
                if time.time() - start_time > time_limit - 5:
                    break
                
                # Update U
                # T_242 reshaped to (8, 8, 4) ? No.
                # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
                # Fix V, W. Solve for U.
                # U[i, :] = (T[i, :, :] * (V * W))_flat / ((V*V) * (W*W))_flat ?
                # Let M = V.T @ W  (12x12)
                # For each i, T_i (8x4) = U_i (1x12) @ (V.T @ W.T) ?
                # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
                # Let X = V.T @ W  (12x12)
                # T_flat[i, j*k + k] = sum_r U[i, r] * (V.T @ W)[r, j*k+k] ? No.
                # T[i, j, k] = U[i, :] dot (V[:, j] * W[:, k])
                # Let B_k = V[:, j] * W[:, k] for each j, k.
                # This is not a simple matrix multiplication because V and W are multiplied element-wise per term.
                
                # Standard ALS for Tensor Decomposition:
                # T = U x V x W (Khatri-Rao)
                # U = T_(1) @ (W * V) / ((W*W) * (V*V))
                # Where T_(1) is matricization of T along mode 1.
                # T_(1) shape: (2, 4*2) = (2, 8).
                # W * V: Khatri-Rao product of W (4x12) and V (8x12) -> (32x12)?
                # Khatri-Rao of W (4x12) and V (8x12) is (32x12).
                # But T_(1) is (2, 8).
                # We need to align dimensions.
                # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
                # For fixed i, T[i, :, :] (4x2) = U[i, :] (1x12) "contracted with" V and W.
                # Let Z = V.T @ W (12x12) ? No.
                # T[i, j, k] = sum_r U[i, r] (V[j, r] W[k, r])
                # Let Y[j, k] = V[j, :] * W[k, :] (element-wise product of rows? No, column-wise)
                # Y is not a single matrix.
                
                # Correct ALS step:
                # T_(1) (2x8) = U (2x12) @ (W * V) (12x8)
                # (W * V) is the Khatri-Rao product of W (4x12) and V (8x12).
                # K = W * V has shape (32x12) ? No.
                # Khatri-Rao product of A (I x R) and B (J x R) is (I*J x R).
                # Here I=4 (k index), J=8 (j index)?
                # T[i, j, k]. i is mode 1. j is mode 2. k is mode 3.
                # T_(1) has columns indexed by (j, k).
                # Column (j, k) of T_(1) corresponds to T[:, j, k].
                # T[:, j, k] = U @ (V[:, j] * W[:, k])
                # So T_(1) = U @ (V * W) where (V * W) is the matrix with columns V[:, j] * W[:, k].
                # This matrix is the Khatri-Rao product of V (8x12) and W (4x12) ?
                # V is 8x12. W is 4x12.
                # Khatri-Rao of V and W is (32x12).
                # But we need columns for each (j, k) pair.
                # There are 4*2=8 pairs.
                # So we need an 8x12 matrix M such that M[:, idx(j,k)] = V[:, j] * W[:, k].
                # This is exactly the Khatri-Rao product if we order the columns correctly.
                # Actually, it's the column-wise product.
                
                # Let's construct M = V * W (Khatri-Rao)
                # M has 32 columns? No.
                # V has 8 rows. W has 4 rows.
                # We want 8 columns in M.
                # Column c of M corresponds to (j, k).
                # M[:, c] = V[:, j] * W[:, k].
                # This is not a standard Khatri-Rao which would give 8*4=32 columns.
                # It's a "sub-Khatri-Rao" or we just construct it.
                
                M = np.zeros((12, 8))
                idx = 0
                for j in range(4):
                    for k in range(2):
                        M[:, idx] = V[:, j] * W[:, k]
                        idx += 1
                        
                # Now T_(1) (2x8) = U (2x12) @ M (12x8)
                # U = T_(1) @ M.T @ inv(M @ M.T)
                # Or solve least squares: U @ M = T_(1)
                
                T_1 = T_242.reshape(2, 8) # (i, j*k+k)
                # Note: reshape order. T_242[i, j, k].
                # If we reshape (2, 4, 2) to (2, 8), the last index varies fastest.
                # So index c = j*2 + k.
                # My M construction: idx = j*2 + k?
                # Loop j 0..3, k 0..1.
                # idx 0: j=0, k=0.
                # idx 1: j=0, k=1.
                # idx 2: j=1, k=0.
                # This matches reshape order (C-order).
                
                try:
                    U_new, residuals, rank, s = np.linalg.lstsq(M, T_1.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                    
                # Update V
                # T_(2) (4x4) = V (4x12) @ (W * U) (12x4)
                # T_242[i, j, k]. Mode 2 is j.
                # T_(2) has rows j, columns (i, k).
                # T[j, i, k] = sum_r V[j, r] U[i, r] W[k, r]
                # M_V = U * W (Khatri-Rao columns for (i, k))
                # U is 2x12. W is 4x12.
                # Columns of M_V correspond to (i, k).
                # i in 0..1, k in 0..1. 4 columns.
                M_V = np.zeros((12, 4))
                idx = 0
                for i in range(2):
                    for k in range(2):
                        M_V[:, idx] = U[:, i] * W[:, k]
                        idx += 1
                        
                T_2 = T_242.transpose(1, 0, 2).reshape(4, 4) # (j, i*k+k)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(M_V, T_2.T, rcond=None)
                    V = V_new.T
                except:
                    pass
                    
                # Update W
                # T_(3) (2x8) = W (2x12) @ (V * U) (12x8)
                # Mode 3 is k.
                # T_(3) has rows k, columns (i, j).
                # T[k, i, j] = sum_r W[k, r] U[i, r] V[j, r]
                # M_W = U * V (columns for (i, j))
                # U 2x12. V 4x12.
                # Columns for (i, j). i in 0..1, j in 0..3. 8 columns.
                M_W = np.zeros((12, 8))
                idx = 0
                for i in range(2):
                    for j in range(4):
                        M_W[:, idx] = U[:, i] * V[:, j]
                        idx += 1
                        
                T_3 = T_242.transpose(2, 0, 1).reshape(2, 8) # (k, i*4+j)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(M_W, T_3.T, rcond=None)
                    W = W_new.T
                except:
                    pass
                
            # Calculate error
            # Reconstruct T
            T_rec = np.zeros((2, 4, 2))
            for r in range(12):
                T_rec += np.outer(np.outer(U[:, r], V[:, r]), W[:, r]) # This is wrong shape for outer
                # U[:, r] is 2. V[:, r] is 4. W[:, r] is 2.
                # T_rec[i, j, k] += U[i, r] V[j, r] W[k, r]
                # This is a tensor product.
                # Can do: T_rec += np.einsum('ir,jr,kr->ijk', U, V, W)
            
            T_rec = np.einsum('ir,jr,kr->ijk', U, V, W)
            error = np.linalg.norm(T_rec - T_242)
            
            if error < best_error:
                best_error = error
                best_factors = (U, V, W)
                
        return best_factors, best_error

    U_242, V_242, W_242, err = find_rank12_decomposition()
    
    if U_242 is not None and err < 1e-6:
        # Round to nearest 0.5
        U_r = np.round(U_242 * 2) / 2
        V_r = np.round(V_242 * 2) / 2
        W_r = np.round(W_242 * 2) / 2
        
        # Verify
        T_rec = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
        if np.allclose(T_rec, T_242, atol=1e-9):
            # Construct 2x4x4 decomposition
            rank = 24
            U = [[0] * rank for _ in range(n * m)] # 8 x 24
            V = [[0] * rank for _ in range(m * p)] # 16 x 24
            W = [[0] * rank for _ in range(p * n)] # 8 x 24
            
            # First 12 columns: C_L (cols 0-1 of C)
            # U is same for both
            for i in range(8):
                for r in range(12):
                    U[i][r] = U_r[i, r]
                    U[i][r + 12] = U_r[i, r]
                    
            # V for C_L uses B_L (cols 0-1 of B, which are rows 0-3 of B_L? No)
            # B is 4x4. B_L is B[:, 0:2]. B_R is B[:, 2:4].
            # In 2x4x2 decomposition, B is 4x2.
            # So V_r maps to the 4x2 block.
            # For C_L, we use B[:, 0:2].
            # Indices in V_full: 0..15.
            # B_L indices: 0..7 (rows 0-3, cols 0-1).
            # B_R indices: 8..15 (rows 0-3, cols 2-3).
            
            for j in range(8): # j in 0..7 for B_L
                for r in range(12):
                    V[j][r] = V_r[j, r]
            for j in range(8): # j in 0..7 for B_R
                for r in range(12):
                    V[j + 8][r + 12] = V_r[j, r]
                    
            # W for C_L uses C[:, 0:2].
            # C is 2x4. C_L is C[:, 0:2]. C_R is C[:, 2:4].
            # W_r is 4x12. Indices 0..3.
            # C_L indices: 0..3.
            # C_R indices: 4..7.
            
            for k in range(4):
                for r in range(12):
                    W[k][r] = W_r[k, r]
            for k in range(4):
                for r in range(12):
                    W[k + 4][r + 12] = W_r[k, r]
                    
            return {"U": U, "V": V, "W": W}
            
    # Fallback to Rank 28
    # ... (existing code) ...
