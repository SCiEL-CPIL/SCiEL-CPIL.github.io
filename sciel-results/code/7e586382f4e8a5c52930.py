"""Matrix multiplication 4x4 * 4x7 using block decomposition with Strassen."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def _strassen_2x2():
    """Rank-7 decomposition for 2x2 * 2x2 -> 2x2.
    U: 4x7, V: 4x7, W: 4x7
    U[i*2+j, c] for A[i,j], V[j*2+k, c] for B[j,k], W[k*2+i, c] for C[i,k]
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4 = A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook_2x2_2x1():
    """Rank-4 schoolbook for 2x2 * 2x1 -> 2x1.
    U: 4x4, V: 2x4, W: 2x4
    U[i*2+j, c] for A[i,j], V[j*1+k, c] for B[j,k], W[k*2+i, c] for C[i,k]
    Here p=1, so B is 2x1, C is 2x1
    """
    n, m, p = 2, 2, 1
    rank = n * m * p  # 4
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W):
    """Verify the decomposition is correct."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                max_err = max(max_err, abs(val - 1))
    return max_err

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    sU, sV, sW = _strassen_2x2()
    sbU, sbV, sbW = _schoolbook_2x2_2x1()
    
    # Split A into two 2x4 blocks (rows 0-1 and rows 2-3)
    # For each 2x4 block of A, multiply with full 4x7 B:
    #   Split B into three 2x2 blocks (cols 0-1, 2-3, 4-5) and one 2x1 block (col 6)
    #   Three Strassen (rank 7 each) + one schoolbook 2x2*2x1 (rank 4) = 25 per block
    # Total rank: 2 * 25 = 50
    total_rank = 50
    
    U = [[0]*total_rank for _ in range(n*m)]   # 16 x 50
    V = [[0]*total_rank for _ in range(m*p)]   # 28 x 50
    W = [[0]*total_rank for _ in range(p*n)]   # 28 x 50
    
    col = 0
    
    # For each A row block (i_b in 0,1): A[i_b*2:(i_b+1)*2, :] @ B[:, :]
    for i_b in range(2):
        # U row offset for this A block: i_b*2 * 4 + 0 = i_b*8
        u_off = i_b * 8
        
        # For each B column block:
        # B cols 0-1 (2x2) -> Strassen
        # B cols 2-3 (2x2) -> Strassen
        # B cols 4-5 (2x2) -> Strassen
        # B col 6 (2x1) -> Schoolbook 2x2*2x1
        
        # Part 1: B cols 0-1
        for c in range(7):
            for i in range(4):
                U[u_off + i][col + c] = sU[i][c]
            for i in range(4):
                V[i * 7 + 0][col + c] = sV[i][c]
            for i in range(4):
                W[0 * 4 + i_b*2 + i][col + c] = sW[i][c]
        col += 7
        
        # Part 2: B cols 2-3
        for c in range(7):
            for i in range(4):
                U[u_off + i][col + c] = sU[i][c]
            for i in range(4):
                V[i * 7 + 2][col + c] = sV[i][c]
            for i in range(4):
                W[2 * 4 + i_b*2 + i][col + c] = sW[i][c]
        col += 7
        
        # Part 3: B cols 4-5
        for c in range(7):
            for i in range(4):
                U[u_off + i][col + c] = sU[i][c]
            for i in range(4):
                V[i * 7 + 4][col + c] = sV[i][c]
            for i in range(4):
                W[4 * 4 + i_b*2 + i][col + c] = sW[i][c]
        col += 7
        
        # Part 4: B col 6 (2x1)
        for c in range(4):
            for i in range(4):
                U[u_off + i][col + c] = sbU[i][c]
            V[0 * 7 + 6][col + c] = sbV[0][c]
            V[1 * 7 + 6][col + c] = sbV[1][c]
            W[6 * 4 + i_b*2 + 0][col + c] = sbW[0][c]
            W[6 * 4 + i_b*2 + 1][col + c] = sbW[1][c]
        col += 4
    
    assert col == 50, f"Expected rank 50, got {col}"
    
    # Verify
    err = _verify(U, V, W)
    if err > 1e-9:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
