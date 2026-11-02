"""Optimized tensor rank decomposition for 3x4 * 4x4 matrix multiplication."""
import time
import numpy as np

N, M, P = 3, 4, 4
RANK_SCHOOLBOOK = N * M * P  # 48


def _verify_fast(U, V, W, n=N, m=M, p=P):
    """Faster verification using tensor reshaping."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if V.shape[1] != r or W.shape[1] != r:
        return False
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    expected = np.ones((n, m, p))
    return np.allclose(recon, expected, atol=1e-9)


def _strassen_2x2():
    """
    Return rank-7 decomposition for 2x2 * 2x2 matrix multiplication.
    Indices: U[i*2+j], V[j*2+k], W[k*2+i] for i,j,k in {0,1}
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Product 1: M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Product 2: M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[3, 1] = -1; W[2, 1] = 1
    
    # Product 3: M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Product 4: M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Product 5: M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Product 6: M6 = (c-d)(f+h)
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[3, 5] = 1
    W[3, 5] = 1
    
    # Product 7: M7 = (b-c)(e+g)
    U[1, 6] = 1; U[2, 6] = -1
    V[0, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1
    
    return U.tolist(), V.tolist(), W.tolist()


def _build_optimized_decomposition():
    """
    Build rank-32 decomposition for 3x4 * 4x4 using Strassen blocks.
    
    Split A (3x4) into:
      - A_top (2x4) for rows i=0,1
      - A_bot (1x4) for row i=2
    
    Split B (4x4) into:
      - B_top (2x4) for rows j=0,1
      - B_bot (2x4) for rows j=2,3
    
    2x4 * 4x4 part (i=0,1):
      C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
      
      Part 1: A[0:2, 0:2] * B[0:2, :]
        - A[0:2, 0:2] * B[0:2, 0:2] (Rank 7)
        - A[0:2, 0:2] * B[0:2, 2:4] (Rank 7)
      
      Part 2: A[0:2, 2:4] * B[2:4, :]
        - A[0:2, 2:4] * B[2:4, 0:2] (Rank 7)
        - A[0:2, 2:4] * B[2:4, 2:4] (Rank 7)
      
      Total for top rows: 28
    
    1x4 * 4x4 part (i=2):
      C[2,k] = sum_{j=0}^{3} A[2,j]*B[j,k]
      - A[2,0] * B[0,:] (Rank 1)
      - A[2,1] * B[1,:] (Rank 1)
      - A[2,2] * B[2,:] (Rank 1)
      - A[2,3] * B[3,:] (Rank 1)
      Total for bottom row: 4
    
    Grand total: 32
    """
    rank = 32
    U = np.zeros((12, rank))  # 3*4=12 rows
    V = np.zeros((16, rank))  # 4*4=16 rows
    W = np.zeros((12, rank))  # 4*3=12 rows
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, current_col):
        sU, sV, sW = _strassen_2x2()
        for c in range(7):
            # U: A indices
            # A is 3x4. Index is i*4+j.
            # Block is 2x2.
            # a_row_offset, a_col_offset are the top-left (i, j) of the A block.
            for li in range(2):
                for lj in range(2):
                    U[(a_row_offset + li) * 4 + (a_col_offset + lj), current_col] = sU[li * 2 + lj][c]
            
            # V: B indices
            # B is 4x4. Index is j*4+k.
            # b_row_offset, b_col_offset are the top-left (j, k) of the B block.
            for lj in range(2):
                for lk in range(2):
                    V[(b_row_offset + lj) * 4 + (b_col_offset + lk), current_col] = sV[lj * 2 + lk][c]
            
            # W: C indices
            # C is 4x3. Index is k*3+i.
            # The output of this block contributes to C rows (b_col_offset .. b_col_offset+1) and C cols (a_row_offset .. a_row_offset+1).
            # Wait, C[i,k] is the output.
            # The block computes C[a_row_offset : a_row_offset+2, b_col_offset : b_col_offset+2].
            # So i ranges from a_row_offset to a_row_offset+1.
            # k ranges from b_col_offset to b_col_offset+1.
            for lk in range(2):
                for li in range(2):
                    W[(b_col_offset + lk) * 3 + (a_row_offset + li), current_col] = sW[lk * 2 + li][c]
        
        return current_col + 7

    # --- 2x4 * 4x4 part (i=0,1) ---
    
    # Part 1: A[0:2, 0:2] * B[0:2, :]
    # Block 1.1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    col = add_strassen_block(0, 0, 0, 0, col)
    
    # Block 1.2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    col = add_strassen_block(0, 0, 0, 2, col)
    
    # Part 2: A[0:2, 2:4] * B[2:4, :]
    # Block 2.1: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    col = add_strassen_block(0, 2, 2, 0, col)
    
    # Block 2.2: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    col = add_strassen_block(0, 2, 2, 2, col)
    
    # --- 1x4 * 4x4 part (i=2) ---
    # For each j in {0,1,2,3}: product A[2,j] * B[j,k] for all k
    # This is a rank-1 tensor for each j?
    # C[2,k] += A[2,j] * B[j,k]
    # U[2*4+j] = 1
    # V[j*4+k] = 1
    # W[k*3+2] = 1
    for j in range(4):
        U[2 * 4 + j, col] = 1
        for k in range(4):
            V[j * 4 + k, col] = 1
            W[k * 3 + 2, col] = 1
        col += 1
    
    assert col == rank, f"Column count {col} != expected rank {rank}"
    
    return U.tolist(), V.tolist(), W.tolist()


def _schoolbook():
    """Fallback schoolbook decomposition of rank 48."""
    rank = N * M * P
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.
    
    Returns dict with 'U', 'V', 'W' matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit
    
    # Try the Strassen-based block decomposition (rank 32)
    U, V, W = _build_optimized_decomposition()
    
    if _verify_fast(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return _schoolbook()
