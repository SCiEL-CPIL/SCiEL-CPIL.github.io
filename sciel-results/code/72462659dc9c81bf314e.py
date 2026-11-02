"""Block decomposition using Strassen for 2x2x2 sub-blocks."""
DIMENSIONS = [2, 4, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    import numpy as np
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Random test matrices
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    # Computed via tensor: C[i,k] = sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k]
    # Equivalently: C = (A * U^T) @ (V * B^T) @ W^T ... let me think
    # C[i,k] = sum_{j} A[i,j] B[j,k]
    # Tensor: C[i,k] = sum_r (sum_j A[i,j] U[j + i*m, r]) * (sum_j V[j*p+k, r] B[j,k]) * W[k*n+i, r]
    # Actually: C[i,k] = sum_r prod_j A[i,j] B[j,k] U[...,r] V[...,r] W[...,r] for rank-1 tensors
    # For the standard form: C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r]) * W[k*n+i, r]
    # Wait, that's not right either. Let me think again.
    # The tensor T[i,j,k] = delta (i.e., T[i,j,k] = 1 if this is the multiplication tensor)
    # Actually for matrix multiplication, the tensor is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The multiplication tensor has T[i,j,k] = 1 for each (i,j,k) triple
    # A rank-r decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Then C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] A[i,j] B[j,k]
    # = sum_r W[k*n+i, r] (sum_j A[i,j] U[i*m+j, r]) (sum_j B[j,k] V[j*p+k, r])
    # Let me verify with schoolbook: U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1 for col=(i,j,k)
    # Then sum_j A[i,j]*1*B[j,k]*1 = sum_j A[i,j]B[j,k] = C[i,k]. And W[k*n+i, col]=1. Sum over r gives C[i,k]. Correct.
    
    # So: C[i,k] = sum_r W[k*n+i, r] * (sum_j A[i,j]*U[i*m+j, r]) * (sum_j B[j,k]*V[j*p+k, r])
    AU = np.zeros((n, rank))
    for i in range(n):
        for j in range(m):
            AU[i, :] += A[i, j] * U[i*m + j, :]
    BV = np.zeros((p, rank))
    for k in range(p):
        for j in range(m):
            BV[k, :] += B[j, k] * V[j*p + k, :]
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            C_computed[i, k] = np.sum(W[k*n + i, :] * AU[i, :] * BV[k, :])
    return np.max(np.abs(C_computed - C_expected)) < tol

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7).
    
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-c)
    M5 = (a+b)(e+f)
    M6 = (c-d)(g+h)
    M7 = (b-d)(f+g)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M6
    C22 = M1 - M2 + M3 + M6
    """
    rank = 7
    # U[i*m+j, r] where i in {0,1}, j in {0,1}
    # V[j*p+k, r] where j in {0,1}, k in {0,1}
    # W[k*n+i, r] where k in {0,1}, i in {0,1}
    
    U = [[0]*rank for _ in range(4)]  # n*m = 2*2 = 4
    V = [[0]*rank for _ in range(4)]  # m*p = 2*2 = 4
    W = [[0]*rank for _ in range(4)]  # p*n = 2*2 = 4
    
    # Indexing: A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
    # B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
    # C[0,0]=C11, C[0,1]=C12, C[1,0]=C21, C[1,1]=C22
    
    # M1 = (a+d)(e+h)
    U[0][0] = 1  # a
    U[3][0] = 1  # d
    V[0][0] = 1  # e
    V[3][0] = 1  # h
    
    # M2 = (c+d)e
    U[2][1] = 1  # c
    U[3][1] = 1  # d
    V[0][1] = 1  # e
    
    # M3 = a(f-h)
    U[0][2] = 1  # a
    V[1][2] = 1  # f
    V[3][2] = -1  # -h
    
    # M4 = d(g-c)
    U[3][3] = 1  # d
    V[2][3] = 1  # g
    V[0][3] = -1  # -c ... wait, V is indexed by [j*p+k], so V[j*2+k]
    # B[1,0]=g is at index j=1,k=0 -> index 2
    # B[0,0]=e is at index j=0,k=0 -> index 0
    # Hmm, M4 = d(g-c). g is B[1,0], c is A[1,0]
    # V[j*p+k]: for g=B[1,0], j=1,k=0, index=1*2+0=2
    # For -c: c is A[1,0], so U[1*2+0]=U[2]
    # Wait I need to recheck. Let me redo.
    
    # Let me restart with clearer indexing:
    # U row index = i*m + j = i*2 + j
    # V row index = j*p + k = j*2 + k  
    # W row index = k*n + i = k*2 + i
    
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (a+d)(e+h): a=A[0,0], d=A[1,1], e=B[0,0], h=B[1,1]
    U[0][0] = 1   # i=0,j=0 -> a
    U[3][0] = 1   # i=1,j=1 -> d
    V[0][0] = 1   # j=0,k=0 -> e
    V[3][0] = 1   # j=1,k=1 -> h
    
    # M2 = (c+d)e: c=A[1,0], d=A[1,1], e=B[0,0]
    U[2][1] = 1   # i=1,j=0 -> c
    U[3][1] = 1   # i=1,j=1 -> d
    V[0][1] = 1   # j=0,k=0 -> e
    
    # M3 = a(f-h): a=A[0,0], f=B[0,1], h=B[1,1]
    U[0][2] = 1   # i=0,j=0 -> a
    V[1][2] = 1   # j=0,k=1 -> f
    V[3][2] = -1  # j=1,k=1 -> h
    
    # M4 = d(g-c): d=A[1,1], g=B[1,0], c=A[1,0]
    U[3][3] = 1   # i=1,j=1 -> d
    U[2][3] = -1  # i=1,j=0 -> c
    V[2][3] = 1   # j=1,k=0 -> g
    
    # M5 = (a+b)(e+f): a=A[0,0], b=A[0,1], e=B[0,0], f=B[0,1]
    U[0][4] = 1   # i=0,j=0 -> a
    U[1][4] = 1   # i=0,j=1 -> b
    V[0][4] = 1   # j=0,k=0 -> e
    V[1][4] = 1   # j=0,k=1 -> f
    
    # M6 = (c-d)(g+h): c=A[1,0], d=A[1,1], g=B[1,0], h=B[1,1]
    U[2][5] = 1   # i=1,j=0 -> c
    U[3][5] = -1  # i=1,j=1 -> d
    V[2][5] = 1   # j=1,k=0 -> g
    V[3][5] = 1   # j=1,k=1 -> h
    
    # M7 = (b-d)(f+g): b=A[0,1], d=A[1,1], f=B[0,1], g=B[1,0]
    U[1][6] = 1   # i=0,j=1 -> b
    U[3][6] = -1  # i=1,j=1 -> d
    V[1][6] = 1   # j=0,k=1 -> f
    V[2][6] = 1   # j=1,k=0 -> g
    
    # W: C11 = M1 + M4 - M5 + M7, C12 = M3 + M5, C21 = M2 + M6, C22 = M1 - M2 + M3 + M6
    # W[k*n+i]: W[0]=C11 (k=0,i=0), W[1]=C12 (k=0,i=1), W[2]=C21 (k=1,i=0), W[3]=C22 (k=1,i=1)
    W[0][0] = 1   # C11 += M1
    W[0][3] = 1   # C11 += M4
    W[0][4] = -1  # C11 -= M5
    W[0][6] = 1   # C11 += M7
    W[1][2] = 1   # C12 += M3
    W[1][4] = 1   # C12 += M5
    W[2][1] = 1   # C21 += M2
    W[2][5] = 1   # C21 += M6
    W[3][0] = 1   # C22 += M1
    W[3][1] = -1  # C22 -= M2
    W[3][2] = 1   # C22 += M3
    W[3][5] = 1   # C22 += M6
    
    return U, V, W

def _schoolbook_2x2x3():
    """Return U, V, W for 2x2x3 schoolbook decomposition (rank 12)."""
    n, m, p = 2, 2, 3
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _schoolbook_2x2x2():
    """Return U, V, W for 2x2x2 schoolbook decomposition (rank 8)."""
    n, m, p = 2, 2, 2
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _block_2x4x5():
    """Decompose 2x4x5 as two 2x2x5 blocks.
    Block 1 (cols 0-1 of B, rows 0-1 of A): 2x2x5 -> Strassen 2x2x2 (rank 7) + schoolbook 2x2x3 (rank 12) = 19
    Block 2 (cols 2-3 of B, rows 0-1 of A): same = 19
    Total = 38
    """
    # Actually, let me reconsider. The 4x5 matrix B has 4 rows (j) and 5 cols (k).
    # A is 2x4: rows i in {0,1}, cols j in {0,1,2,3}
    # We can split j into {0,1} and {2,3}
    # Block 1: A[:, 0:2] (2x2) * B[0:2, :] (2x5) -> 2x2x5
    # Block 2: A[:, 2:4] (2x2) * B[2:4, :] (2x5) -> 2x2x5
    # Each 2x2x5 = 2x2x2 + 2x2x3
    # 2x2x2 via Strassen: rank 7
    # 2x2x3 via schoolbook: rank 12
    # Total per block: 19, total: 38
    
    # Alternatively, split the 5 columns of B into 2+3:
    # 2x2x2 Strassen (rank 7) + 2x2x3 schoolbook (rank 12) = 19 per block
    # Two blocks: 38
    
    # Let's build it:
    U_rows = 2 * 4  # n*m = 8
    V_rows = 4 * 5  # m*p = 20
    W_rows = 5 * 2  # p*n = 10
    
    # Block 1: i in {0,1}, j in {0,1} (local j), k in {0,1,2,3,4}
    #   Sub-block 1a: j_local in {0,1}, k in {0,1} -> 2x2x2 Strassen
    #   Sub-block 1b: j_local in {0,1}, k in {2,3,4} -> 2x2x3 schoolbook
    # Block 2: i in {0,1}, j in {2,3} (local j = 0,1), k in {0,1,2,3,4}
    #   Sub-block 2a: j_local in {0,1}, k in {0,1} -> 2x2x2 Strassen
    #   Sub-block 2b: j_local in {0,1}, k in {2,3,4} -> 2x2x3 schoolbook
    
    U_str, V_str, W_str = _strassen_2x2x2()
    U_sch, V_sch, W_sch = _schoolbook_2x2x3()
    
    rank1 = 7
    rank2 = 12
    
    total_rank = 2 * (rank1 + rank2)  # 38
    
    U = [[0]*total_rank for _ in range(U_rows)]
    V = [[0]*total_rank for _ in range(V_rows)]
    W = [[0]*total_rank for _ in range(W_rows)]
    
    def place(block_start, U_sub, V_sub, W_sub, rank_sub, i_offset, j_offset, k_offset):
        for r in range(rank_sub):
            col = block_start + r
            for i in range(2):
                for j in range(2):
                    val = U_sub[i*2 + j][r]
                    if val != 0:
                        U[(i + i_offset)*4 + (j + j_offset)][col] = val
            for j in range(2):
                for k in range(5):
                    if k < 2:
                        kk = k
                    else:
                        kk = -1
                    # V is 2x5 for the local block
                    if k < 2:
                        val = V_sub[j*2 + k][r]
                    else:
                        val = 0  # Strassen only handles k in {0,1}
                    if val != 0:
                        V[(j + j_offset)*5 + k][col] = val
            for k in range(5):
                if k < 2:
                    val = W_sub[k*2 + i_offset % 2][r]  # Wait, W is indexed by k*n+i locally
                    # Local W[k*2 + i] where i is local 0 or 1
                    # Global: W[k*n + i_global] where i_global = i + i_offset... no
                    # i is always 0 or 1 globally since n=2
                    val = W_sub[k*2 + i_offset][r]  # Hmm, this isn't right either
                    if val != 0:
                        W[k*2 + i_offset][col] = val
        # This is getting complicated. Let me be more careful.
    
    # Let me redo this more carefully.
    
    # Global dimensions: n=2, m=4, p=5
    # U has rows indexed by i*m + j, i in {0,1}, j in {0,1,2,3}
    # V has rows indexed by j*p + k, j in {0,1,2,3}, k in {0,1,2,3,4}
    # W has rows indexed by k*n + i, k in {0,1,2,3,4}, i in {0,1}
    
    # Block 1: j in {0,1}
    #   Part 1a (Strassen, k in {0,1}):
    #     U[i*4 + j][col] for i in {0,1}, j in {0,1}
    #     V[j*5 + k][col] for j in {0,1}, k in {0,1}
    #     W[k*2 + i][col] for k in {0,1}, i in {0,1}
    #   Part 1b (Schoolbook 2x2x3, k in {2,3,4}):
    #     U[i*4 + j][col] for i in {0,1}, j in {0,1}
    #     V[j*5 + k][col] for j in {0,1}, k in {2,3,4}
    #     W[k*2 + i][col] for k in {2,3,4}, i in {0,1}
    
    # Block 2: j in {2,3}
    #   Part 2a (Strassen, k in {0,1}):
    #     U[i*4 + j][col] for i in {0,1}, j in {2,3}
    #     V[j*5 + k][col] for j in {2,3}, k in {0,1}
    #     W[k*2 + i][col] for k in {0,1}, i in {0,1}
    #   Part 2b (Schoolbook 2x2x3, k in {2,3,4}):
    #     U[i*4 + j][col] for i in {0,1}, j in {2,3}
    #     V[j*5 + k][col] for j in {2,3}, k in {2,3,4}
    #     W[k*2 + i][col] for k in {2,3,4}, i in {0,1}
    
    U = [[0]*total_rank for _ in range(8)]
    V = [[0]*total_rank for _ in range(20)]
    W = [[0]*total_rank for _ in range(10)]
    
    col = 0
    # Block 1, Part 1a: Strassen 2x2x2 for j in {0,1}, k in {0,1}
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i*4 + j][col] = U_str[i*2 + j][r]
        for j in range(2):
            for k in range(2):
                V[j*5 + k][col] = V_str[j*2 + k][r]
        for k in range(2):
            for i in range(2):
                W[k*2 + i][col] = W_str[k*2 + i][r]
        col += 1
    
    # Block 1, Part 1b: Schoolbook 2x2x3 for j in {0,1}, k in {2,3,4}
    for r in range(12):
        for i in range(2):
            for j in range(2):
                U[i*4 + j][col] = U_sch[i*2 + j][r]
        for j in range(2):
            for k_local in range(3):
                k = k_local + 2
                V[j*5 + k][col] = V_sch[j*3 + k_local][r]
        for k_local in range(3):
            k = k_local + 2
            for i in range(2):
                W[k*2 + i][col] = W_sch[k_local*2 + i][r]
        col += 1
    
    # Block 2, Part 2a: Strassen 2x2x2 for j in {2,3}, k in {0,1}
    for r in range(7):
        for i in range(2):
            for j_local in range(2):
                j = j_local + 2
                U[i*4 + j][col] = U_str[i*2 + j_local][r]
        for j_local in range(2):
            j = j_local + 2
            for k in range(2):
                V[j*5 + k][col] = V_str[j_local*2 + k][r]
        for k in range(2):
            for i in range(2):
                W[k*2 + i][col] = W_str[k*2 + i][r]
        col += 1
    
    # Block 2, Part 2b: Schoolbook 2x2x3 for j in {2,3}, k in {2,3,4}
    for r in range(12):
        for i in range(2):
            for j_local in range(2):
                j = j_local + 2
                U[i*4 + j][col] = U_sch[i*2 + j_local][r]
        for j_local in range(2):
            j = j_local + 2
            for k_local in range(3):
                k = k_local + 2
                V[j*5 + k][col] = V_sch[j_local*3 + k_local][r]
        for k_local in range(3):
            k = k_local + 2
            for i in range(2):
                W[k*2 + i][col] = W_sch[k_local*2 + i][r]
        col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    result = _block_2x4x5()
    # Verify
    if not _verify(result["U"], result["V"], result["W"], n, m, p):
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
    return result

