import numpy as np
import time

DIMENSIONS = [2, 2, 11]

def _verify(U, V, W, n, m, p, tol=1e-12):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    rank = U.shape[1]
    np.random.seed(42)
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        
        C_dec = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for j in range(m):
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                C_dec[i, k] = val
        
        if not np.allclose(C, C_dec, atol=tol):
            return False
    return True

def _build_strassen_2x2x2():
    """
    Build a rank-7 decomposition for 2x2 by 2x2 matrix multiplication.
    Returns U, V, W of shapes (4, 7), (4, 7), (4, 7).
    Convention:
      U indexed by (i, j) with row index i*m + j, m=2
      V indexed by (j, k) with row index j*p + k, p=2
      W indexed by (k, i) with row index k*n + i, n=2
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (a00 + a11)(b00 + b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (a10 + a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3: a00(b01 - b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4: a11(b10 - b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5: (a00 + a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: (a10 - a00)(b00 + b10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7: (a01 - a11)(b11 + b10)
    U[1, 6] = 1; U[3, 6] = -1
    V[3, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1; W[2, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 11
    deadline = time.time() + 115  # safety margin
    
    # Strategy: Block Strassen
    # Split p=11 into 5 blocks of 2 and 1 block of 1.
    # Rank = 5 * 7 + 4 = 39.
    
    total_rank = 39
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    U_base, V_base, W_base = _build_strassen_2x2x2()
    
    current_col = 0
    current_k = 0
    
    # 5 blocks of 2x2
    for block_idx in range(5):
        # Copy U base to U[:, current_col : current_col+7]
        U[:, current_col : current_col+7] = U_base
        
        # Map V base (p_local=2) to V global (p=11) for columns current_k, current_k+1
        for j in range(m):
            for k_local in range(2):
                v_row_global = j * p + (current_k + k_local)
                v_row_base = j * 2 + k_local
                V[v_row_global, current_col : current_col+7] = V_base[v_row_base, :]
                
        # Map W base (p_local=2) to W global (p=11) for columns current_k, current_k+1
        for k_local in range(2):
            for i in range(n):
                w_row_global = (current_k + k_local) * n + i
                w_row_base = k_local * 2 + i
                W[w_row_global, current_col : current_col+7] = W_base[w_row_base, :]
                
        current_col += 7
        current_k += 2
        
    # 1 block of 2x1 (Schoolbook)
    # Rank 4 for 2x2x1
    # Terms: a00 b00, a00 b10? No, B is 2x1.
    # B has indices (j, k) with k=0 (local). j=0,1.
    # A has indices (i, j).
    # C has indices (k, i) with k=0 (local).
    # C_0i = A_00 B_00 + A_10 B_10 (for i=0? No)
    # C_{k,i} = sum_j A_{i,j} B_{j,k}
    # For k_local=0:
    # C_{0,0} = A_{0,0} B_{0,0} + A_{0,1} B_{1,0}
    # C_{0,1} = A_{1,0} B_{0,0} + A_{1,1} B_{1,0}
    
    # Term 1: A_00 B_00 -> C_00
    U[0, current_col] = 1
    V[0 * p + current_k, current_col] = 1  # B_00
    W[0 * n + 0, current_col] = 1          # C_00
    
    # Term 2: A_01 B_10 -> C_00
    U[1, current_col+1] = 1
    V[1 * p + current_k, current_col+1] = 1 # B_10
    W[0 * n + 0, current_col+1] = 1         # C_00
    
    # Term 3: A_10 B_00 -> C_01
    U[2, current_col+2] = 1
    V[0 * p + current_k, current_col+2] = 1 # B_00
    W[0 * n + 1, current_col+2] = 1         # C_01
    
    # Term 4: A_11 B_10 -> C_01
    U[3, current_col+3] = 1
    V[1 * p + current_k, current_col+3] = 1 # B_10
    W[0 * n + 1, current_col+3] = 1         # C_01
    
    current_col += 4
    # current_k += 1 (end)
    
    # Verify
    if time.time() < deadline:
        if not _verify(U, V, W, n, m, p):
            # Fall back to schoolbook
            rank = n * m * p
            U_fb = np.zeros((n*m, rank))
            V_fb = np.zeros((m*p, rank))
            W_fb = np.zeros((p*n, rank))
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i*m+j, column] = 1
                        V_fb[j*p+k, column] = 1
                        W_fb[k*n+i, column] = 1
                        column += 1
            U, V, W = U_fb, V_fb, W_fb
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
