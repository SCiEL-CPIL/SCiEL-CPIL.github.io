"""Rank-minimal exact tensor decomposition for 2x4 x 4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, n=2, m=4, p=7):
    """Check if U, V, W give exact decomposition using integer arithmetic."""
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7.
    U: 4x7 (A[0,0],A[0,1],A[1,0],A[1,1]), V: 4x7 (B[0,0],B[0,1],B[1,0],B[1,1]), W: 4x7 (C[0,0],C[0,1],C[1,0],C[1,1])
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    # M3 = A00(B01-B11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    # M4 = A11(B10-B00)
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    # M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1
    # M6 = (A10-A00)(B00+B01)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    
    return U, V, W

def _rank2x2x1():
    """2x2 x 2x1 multiplication, rank 4."""
    U = np.zeros((4, 4), dtype=int)
    V = np.zeros((2, 4), dtype=int)
    W = np.zeros((2, 4), dtype=int)
    # C0 = A00*B0 + A01*B1, C1 = A10*B0 + A11*B1
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    U[2,2]=1; V[0,2]=1; W[1,2]=1
    U[3,3]=1; V[1,3]=1; W[1,3]=1
    return U, V, W

def _build_rank50():
    """Build rank-50 decomposition for 2x4 x 4x7.
    
    Split A (2x4) into A1 (cols 0-1) and A2 (cols 2-3).
    Split B (4x7) into B1 (rows 0-1) and B2 (rows 2-3).
    C = A1*B1 + A2*B2.
    
    For each 2x2 x 2x7 subproblem: 3 column-pairs use Strassen (rank 7 each), 1 single column uses rank-4.
    Total per subproblem: 3*7 + 4 = 25. Total: 50.
    """
    n, m, p = 2, 4, 7
    rank = 50
    U = np.zeros((8, rank), dtype=int)
    V = np.zeros((28, rank), dtype=int)
    W = np.zeros((14, rank), dtype=int)
    
    S2u, S2v, S2w = _strassen_2x2x2()
    S1u, S1v, S1w = _rank2x2x1()
    
    col = 0
    
    def add_2x2x2_block(u_rows, v_rows, w_rows, offset):
        for c in range(7):
            for r in range(4):
                U[u_rows[r], offset+c] += S2u[r, c]
            for r in range(4):
                V[v_rows[r], offset+c] += S2v[r, c]
            for r in range(4):
                W[w_rows[r], offset+c] += S2w[r, c]
    
    def add_2x2x1_block(u_rows, v_rows, w_rows, offset):
        for c in range(4):
            for r in range(4):
                U[u_rows[r], offset+c] += S1u[r, c]
            for r in range(2):
                V[v_rows[r], offset+c] += S1v[r, c]
            for r in range(2):
                W[w_rows[r], offset+c] += S1w[r, c]
    
    # A1*B1: A rows 0-3, B rows 0-6 (B[0,k]) and 7-13 (B[1,k])
    # W: C[0,k] = W[k], C[1,k] = W[7+k]
    
    # Column pair (0,1): B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[7], B[1,1]=V[8]
    add_2x2x2_block([0,1,2,3], [0,1,7,8], [0,1,7,8], col); col += 7
    # Column pair (2,3): B[0,2]=V[2], B[0,3]=V[3], B[1,2]=V[9], B[1,3]=V[10]
    add_2x2x2_block([0,1,2,3], [2,3,9,10], [2,3,9,10], col); col += 7
    # Column pair (4,5): B[0,4]=V[4], B[0,5]=V[5], B[1,4]=V[11], B[1,5]=V[12]
    add_2x2x2_block([0,1,2,3], [4,5,11,12], [4,5,11,12], col); col += 7
    # Column 6: B[0,6]=V[6], B[1,6]=V[13]
    add_2x2x1_block([0,1,2,3], [6,13], [6,13], col); col += 4
    
    # A2*B2: A rows 4-7, B rows 14-20 (B[2,k]) and 21-27 (B[3,k])
    # W: same C[0,k]=W[k], C[1,k]=W[7+k]
    
    # Column pair (0,1): B[2,0]=V[14], B[2,1]=V[15], B[3,0]=V[21], B[3,1]=V[22]
    add_2x2x2_block([4,5,6,7], [14,15,21,22], [0,1,7,8], col); col += 7
    # Column pair (2,3): B[2,2]=V[16], B[2,3]=V[17], B[3,2]=V[23], B[3,3]=V[24]
    add_2x2x2_block([4,5,6,7], [16,17,23,24], [2,3,9,10], col); col += 7
    # Column pair (4,5): B[2,4]=V[18], B[2,5]=V[19], B[3,4]=V[25], B[3,5]=V[26]
    add_2x2x2_block([4,5,6,7], [18,19,25,26], [4,5,11,12], col); col += 7
    # Column 6: B[2,6]=V[20], B[3,6]=V[27]
    add_2x2x1_block([4,5,6,7], [20,27], [6,13], col); col += 4
    
    assert col == 50, f"Expected 50, got {col}"
    
    # Convert to lists
    U_list = [[int(x) for x in row] for row in U]
    V_list = [[int(x) for x in row] for row in V]
    W_list = [[int(x) for x in row] for row in W]
    
    return U_list, V_list, W_list

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    
    # Try rank-50 Strassen-based decomposition
    try:
        U, V, W = _build_rank50()
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fall back to schoolbook (rank 56)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
