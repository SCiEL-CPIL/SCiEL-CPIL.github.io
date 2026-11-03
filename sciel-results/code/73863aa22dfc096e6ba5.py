"""Matrix multiplication tensor decomposition for 3x4 * 4x7."""
import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, tol=1e-8):
    n, m, p = DIMENSIONS
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    R = U.shape[1]
    C_comp = np.zeros((n, p))
    
    U_reshaped = U.reshape(n, m, R)
    S_A = np.einsum('ijr,ij->ir', U_reshaped, A)
    
    V_reshaped = V.reshape(m, p, R)
    S_B = np.einsum('jkr,jk->kr', V_reshaped, B)
    
    W_reshaped = W.reshape(p, n, R)
    C_comp = np.einsum('kir,ir,kr->ik', W_reshaped, S_A, S_B)
    
    return np.allclose(C_true, C_comp, atol=tol)

def _solve_block(n, m, p, target_rank, max_iter=5000, tol=1e-10, seed=0):
    """
    Finds a rank-target_rank decomposition for M_{n,m,p} using vectorized ALS.
    Returns U, V, W and error.
    """
    np.random.seed(seed)
    U = np.random.randn(n, m, target_rank) * 0.1
    V = np.random.randn(m, p, target_rank) * 0.1
    W = np.random.randn(p, n, target_rank) * 0.1
    
    prev_err = float('inf')
    stagnant_count = 0
    
    for it in range(max_iter):
        # Update U
        for i in range(n):
            for j in range(m):
                # M[k, r] = V[j, k, r] * W[k, i, r]
                M = V[j, :, :] * W[:, i, :]
                b = np.ones(p)
                try:
                    x, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                    U[i, j, :] = x
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                # M[i, r] = U[i, j, r] * W[k, i, r]
                M = U[:, j, :] * W[k, :, :]
                b = np.ones(n)
                try:
                    x, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                    V[j, k, :] = x
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                # M[j, r] = U[i, j, r] * V[j, k, r]
                M = U[i, :, :] * V[:, k, :]
                b = np.ones(m)
                try:
                    x, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                    W[k, i, :] = x
                except:
                    pass
        
        # Calculate Error
        try:
            T_comp = np.einsum('ijr,jkr,kir->ijk', U, V, W)
            err = np.max(np.abs(T_comp - 1.0))
        except:
            err = prev_err
            
        if err < prev_err - 1e-12:
            prev_err = err
            stagnant_count = 0
        else:
            stagnant_count += 1
            if stagnant_count > 100:
                break
                
        if prev_err < tol:
            break
            
    return U, V, W, prev_err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Try to find a rank 72 decomposition using block structure
    # M_{3,4,7} = M_{3,2,7} + M_{3,2,7}
    # M_{3,2,7} = M_{3,2,3} + M_{3,2,4}
    # M_{3,2,4} = M_{3,2,2} + M_{3,2,2}
    
    # We need:
    # 2x M_{3,2,3} (Rank 16)
    # 4x M_{3,2,2} (Rank 10)
    
    # Solve for blocks
    try:
        # M_{3,2,2}
        U_322, V_322, W_322, err_322 = _solve_block(3, 2, 2, 10, max_iter=2000, seed=1)
        if err_322 > 1e-6:
            raise ValueError(f"Block 322 error too high: {err_322}")
            
        # M_{3,2,3}
        U_323, V_323, W_323, err_323 = _solve_block(3, 2, 3, 16, max_iter=2000, seed=2)
        if err_323 > 1e-6:
            raise ValueError(f"Block 323 error too high: {err_323}")
            
        # Assembly
        # Total Rank = 2 * 16 + 4 * 10 = 32 + 40 = 72
        
        R_total = 72
        U_full = np.zeros((n, m, R_total))
        V_full = np.zeros((m, p, R_total))
        W_full = np.zeros((p, n, R_total))
        
        col_offset = 0
        
        # Block 1: A1 B1
        # A1 is cols 0,1 of A. B1 is rows 0,1 of B.
        # B1 splits into B11 (cols 0,1,2 of B1 -> rows 0,1 of B, cols 0,1,2 of B) 
        # Wait, B is 4x7.
        # B1 is top 2 rows (2x7).
        # B11 is first 3 cols of B1 (2x3).
        # B12 is last 4 cols of B1 (2x4).
        # B12 splits into B121 (2x2) and B122 (2x2).
        
        # 1. A1 B11 (3x2 * 2x3) -> Rank 16
        # Indices:
        # U: rows 0,1 (A1), cols 0,1 (B1 part 1)
        # V: rows 0,1 (B1), cols 0,1,2 (B11)
        # W: rows 0,1,2 (C cols 0,1,2), cols 0,1 (C rows 0,1)
        
        U_full[0:3, 0:2, col_offset:col_offset+16] = U_323
        V_full[0:2, 0:3, col_offset:col_offset+16] = V_323
        W_full[0:3, 0:3, col_offset:col_offset+16] = W_323
        col_offset += 16
        
        # 2. A1 B121 (3x2 * 2x2) -> Rank 10
        # B121 is cols 3,4 of B1 (which are cols 3,4 of B)
        U_full[0:3, 0:2, col_offset:col_offset+10] = U_322
        V_full[0:2, 3:5, col_offset:col_offset+10] = V_322
        W_full[3:5, 0:3, col_offset:col_offset+10] = W_322
        col_offset += 10
        
        # 3. A1 B122 (3x2 * 2x2) -> Rank 10
        # B122 is cols 5,6 of B1 (which are cols 5,6 of B)
        U_full[0:3, 0:2, col_offset:col_offset+10] = U_322
        V_full[0:2, 5:7, col_offset:col_offset+10] = V_322
        W_full[5:7, 0:3, col_offset:col_offset+10] = W_322
        col_offset += 10
        
        # Block 2: A2 B2
        # A2 is cols 2,3 of A. B2 is rows 2,3 of B.
        
        # 4. A2 B21 (3x2 * 2x3) -> Rank 16
        # B21 is rows 2,3 of B, cols 0,1,2
        U_full[0:3, 2:4, col_offset:col_offset+16] = U_323
        V_full[2:4, 0:3, col_offset:col_offset+16] = V_323
        W_full[0:3, 0:3, col_offset:col_offset+16] = W_323
        col_offset += 16
        
        # 5. A2 B221 (3x2 * 2x2) -> Rank 10
        # B221 is rows 2,3 of B, cols 3,4
        U_full[0:3, 2:4, col_offset:col_offset+10] = U_322
        V_full[2:4, 3:5, col_offset:col_offset+10] = V_322
        W_full[3:5, 0:3, col_offset:col_offset+10] = W_322
        col_offset += 10
        
        # 6. A2 B222 (3x2 * 2x2) -> Rank 10
        # B222 is rows 2,3 of B, cols 5,6
        U_full[0:3, 2:4, col_offset:col_offset+10] = U_322
        V_full[2:4, 5:7, col_offset:col_offset+10] = V_322
        W_full[5:7, 0:3, col_offset:col_offset+10] = W_322
        col_offset += 10
        
        # Verify
        if col_offset != R_total:
            raise ValueError("Assembly error")
            
        U_flat = U_full.reshape(n*m, R_total)
        V_flat = V_full.reshape(m*p, R_total)
        W_flat = W_full.reshape(p*n, R_total)
        
        if _verify(U_flat, V_flat, W_flat):
            return {
                "U": U_flat.tolist(),
                "V": V_flat.tolist(),
                "W": W_flat.tolist()
            }
            
    except Exception as e:
        pass
        
    # Fallback to schoolbook
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
