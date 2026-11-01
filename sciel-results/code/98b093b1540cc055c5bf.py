import numpy as np
import time

DIMENSIONS = [2, 2, 11]

def _verify(U, V, W, n, m, p, tol=1e-12):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    rank = U.shape[1]
    # Build the tensor from decomposition: T[i*m+j, k*n+i, j*p+k]
    # Actually, the tensor is T_{(i,j),(k,l)}? No.
    # Matrix mult: C = A * B. A is n x m, B is m x p. C is n x p.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Tensor T is a 3-way tensor? No, matrix mult is a 3-way tensor T_{i,j,k} such that
    # C_{i,k} = sum_j sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}
    # Wait, the output factor is W[k*n+i, c].
    # So C_{i,k} = sum_j sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}
    
    # Let's check a few random instances
    np.random.seed(42)
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        
        # Compute C from decomposition
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
    The convention is:
      U indexed by (i, j) with row index i*m + j, m=2
      V indexed by (j, k) with row index j*p + k, p=2
      W indexed by (k, i) with row index k*n + i, n=2
    """
    # Strassen's 7 products for 2x2:
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b10)
    # M7 = (a01 - a11) * (b11 + b10)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    #
    # U rows: (0,0), (0,1), (1,0), (1,1)
    # V rows: (0,0), (0,1), (1,0), (1,1)
    # W rows: (0,0), (1,0), (0,1), (1,1)  -- W[k*n+i] so W[0]=W[0*2+0], W[1]=W[1*2+0], W[2]=W[0*2+1], W[3]=W[1*2+1]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (a00 + a11)(b00 + b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C00 and C11
    
    # M2: (a10 + a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1  # C10 = +M2, C11 = -M2
    
    # M3: a00(b01 - b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1  # C01 = +M3, C11 = +M3
    
    # M4: a11(b10 - b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1  # C00 = +M4, C10 = +M4
    
    # M5: (a00 + a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1  # C00 = -M5, C01 = +M5
    
    # M6: (a10 - a00)(b00 + b10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1  # C11 = +M6
    
    # M7: (a01 - a11)(b11 + b10)
    U[1, 6] = 1; U[3, 6] = -1
    V[3, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1; W[2, 6] = 1  # C00 = +M7, C01 = +M7
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 11
    deadline = time.time() + 120
    
    # Build the base 2x2x2 Strassen decomposition
    U_base, V_base, W_base = _build_strassen_2x2x2()
    
    # We have p=11 columns in B. We group them into blocks.
    # For 2x2 by 2x11:
    # We can split the 11 columns into 5 blocks of 2 and 1 block of 1.
    # Each 2-block uses the 2x2x2 Strassen (rank 7)
    # Each 1-block uses schoolbook 2x2x2x1 which is rank 4 (just 4 multiplies)
    
    # Actually, let's think about this more carefully.
    # For a 2x2 by 2x1 multiplication:
    # C0 = a00*b0 + a01*b1
    # C1 = a10*b0 + a11*b1
    # This is 4 multiplications, rank 4.
    
    # For 2x2 by 2x2: rank 7 (Strassen)
    
    # So for 2x2 by 2x11:
    # Split 11 columns into: 5 groups of 2, 1 group of 1
    # Total rank: 5*7 + 1*4 = 35 + 4 = 39
    
    # Let's build the full decomposition
    blocks = []
    col_offset = 0
    for _ in range(5):
        blocks.append((2, col_offset))
        col_offset += 2
    blocks.append((1, col_offset))  # last block of size 1
    
    total_rank = 5 * 7 + 1 * 4
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    for block_size, start_col in blocks:
        if block_size == 2:
            # Use Strassen 2x2x2
            # U_base: (4, 7)
            # V_base: (4, 7)
            # W_base: (4, 7)
            
            # U: copy U_base
            U[:, col:col+7] = U_base
            
            # V: V_base indexed by (j, k_local). Global k = start_col + k_local
            # V_base rows: (0,0), (0,1), (1,0), (1,1)
            # Global V rows: j*p + k where k = start_col + k_local
            for j in range(2):
                for k_local in range(2):
                    global_k = start_col + k_local
                    v_row_global = j * p + global_k
                    v_row_base = j * 2 + k_local
                    V[v_row_global, col:col+7] = V_base[v_row_base, :]
            
            # W: W_base indexed by (k_local, i). Global k = start_col + k_local
            # W_base rows: (0,0), (1,0), (0,1), (1,1)
            # Global W rows: k*n + i where k = start_col + k_local
            for k_local in range(2):
                for i in range(2):
                    global_k = start_col + k_local
                    w_row_global = global_k * n + i
                    w_row_base = k_local * 2 + i
                    W[w_row_global, col:col+7] = W_base[w_row_base, :]
            
            col += 7
        elif block_size == 1:
            # Schoolbook 2x2x2x1
            # C_i = sum_j A[i,j] * B[j, start_col]
            # 4 products: (i=0,j=0), (i=0,j=1), (i=1,j=0), (i=1,j=1)
            # U: (4, 4), V: (2, 4), W: (2, 4)
            
            U_sub = np.zeros((4, 4))
            V_sub = np.zeros((2, 4))
            W_sub = np.zeros((2, 4))
            
            # (i=0, j=0): product a00 * b0
            U_sub[0, 0] = 1  # i=0, j=0
            V_sub[0, 0] = 1  # j=0, k=0 (local)
            W_sub[0, 0] = 1  # k=0, i=0
            
            # (i=0, j=1): product a01 * b1
            U_sub[1, 1] = 1  # i=0, j=1
            V_sub[1, 1] = 1  # j=1, k=0 (local)
            W_sub[0, 1] = 1  # k=0, i=0
            
            # (i=1, j=0): product a10 * b0
            U_sub[2, 2] = 1  # i=1, j=0
            V_sub[0, 2] = 1  # j=0, k=0 (local)
            W_sub[1, 2] = 1  # k=0, i=1
            
            # (i=1, j=1): product a11 * b1
            U_sub[3, 3] = 1  # i=1, j=1
            V_sub[1, 3] = 1  # j=1, k=0 (local)
            W_sub[1, 3] = 1  # k=0, i=1
            
            U[:, col:col+4] = U_sub
            
            # V: j * p + (start_col + 0)
            for j in range(2):
                V[j * p + start_col, col:col+4] = V_sub[j, :]
            
            # W: (start_col + 0) * n + i
            for i in range(2):
                W[start_col * n + i, col:col+4] = W_sub[i, :]
            
            col += 4
    
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
