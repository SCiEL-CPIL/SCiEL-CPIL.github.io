"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """Return U, V, W for 2x2x2 matrix multiplication, rank 7.
    U, V, W are lists of shape (4, 7).
    U[i*2+j, c] corresponds to A[i,j]
    V[i*2+j, c] corresponds to B[i,j]
    W[i*2+j, c] corresponds to C[i,j]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 0: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 1: (A10+A11)B00 -> C10 += 1, C11 -= 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # Term 2: A00(B01-B11) -> C01 += 1, C11 += 1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # Term 3: A11(B10-B00) -> C10 += 1, C00 += 1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = 1
    
    # Term 4: (A00+A01)B11 -> C01 -= 1, C11 += 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = -1; W[3][4] = 1
    
    # Term 5: (A10-A00)(B00+B01) -> C11 += 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 6: (A01-A11)(B10+B11) -> C00 += 1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _strassen_4x4():
    """Return U, V, W for 4x4x4 matrix multiplication, rank 49.
    U, V, W are lists of shape (16, 49).
    """
    U_inner, V_inner, W_inner = _strassen_2x2()
    
    # Outer Strassen A-block coefficients
    outer_A = [
        {(0,0):1, (1,1):1},   # t=0
        {(1,0):1, (1,1):1},   # t=1
        {(0,0):1},             # t=2
        {(1,1):1},             # t=3
        {(0,0):1, (0,1):1},   # t=4
        {(1,0):1, (0,0):-1},  # t=5
        {(0,1):1, (1,1):-1},  # t=6
    ]
    outer_B = [
        {(0,0):1, (1,1):1},   # t=0
        {(0,0):1},             # t=1
        {(0,1):1, (1,1):-1},  # t=2
        {(1,0):1, (0,0):-1},  # t=3
        {(1,1):1},             # t=4
        {(0,0):1, (0,1):1},   # t=5
        {(1,0):1, (1,1):1},   # t=6
    ]
    outer_C = [
        {(0,0):1, (1,1):1},   # t=0
        {(1,0):1, (1,1):-1},  # t=1
        {(0,1):1, (1,1):1},   # t=2
        {(1,0):1, (0,0):1},   # t=3
        {(0,1):-1, (1,1):1},  # t=4
        {(1,1):1},             # t=5
        {(0,0):1},             # t=6
    ]
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for t in range(7):
        for s in range(7):
            col = t*7 + s
            # U
            for i in range(4):
                for j in range(4):
                    bi, bj = i//2, j//2
                    li, lj = i%2, j%2
                    coeff = outer_A[t].get((bi,bj), 0)
                    if coeff != 0:
                        U[i*4+j][col] = coeff * U_inner[li*2+lj][s]
            # V
            for i in range(4):
                for j in range(4):
                    bi, bj = i//2, j//2
                    li, lj = i%2, j%2
                    coeff = outer_B[t].get((bi,bj), 0)
                    if coeff != 0:
                        V[i*4+j][col] = coeff * V_inner[li*2+lj][s]
            # W
            for i in range(4):
                for j in range(4):
                    bi, bj = i//2, j//2
                    li, lj = i%2, j%2
                    coeff = outer_C[t].get((bi,bj), 0)
                    if coeff != 0:
                        W[i*4+j][col] = coeff * W_inner[li*2+lj][s]
    
    return U, V, W

def _schoolbook_4x4x1():
    """Return U, V, W for 4x4x1 matrix multiplication, rank 16.
    U shape (16, 16), V shape (4, 16), W shape (4, 16).
    """
    U = [[0]*16 for _ in range(16)]
    V = [[0]*16 for _ in range(4)]
    W = [[0]*16 for _ in range(4)]
    col = 0
    for i in range(4):
        for j in range(4):
            U[i*4+j][col] = 1
            V[j][col] = 1
            W[i][col] = 1
            col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    # Build numpy arrays
    Um = np.array(U, dtype=np.float64)
    Vm = np.array(V, dtype=np.float64)
    Wm = np.array(W, dtype=np.float64)
    
    # Check: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Get 4x4x4 Strassen decomposition (rank 49)
    U_444, V_444, W_444 = _strassen_4x4()
    
    # Get 4x4x1 schoolbook decomposition (rank 16)
    U_441, V_441, W_441 = _schoolbook_4x4x1()
    
    rank = 49 + 16  # 65
    
    # U_full: shape (16, 65)
    U_full = [[0]*rank for _ in range(16)]
    for i in range(16):
        for c in range(49):
            U_full[i][c] = U_444[i][c]
        for c in range(16):
            U_full[i][49+c] = U_441[i][c]
    
    # V_full: shape (20, 65)
    V_full = [[0]*rank for _ in range(20)]
    # For 4x4x4 part: V_444[j*4+k, c] for j=0..3, k=0..3
    # V_full index is j*5+k
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V_full[j*5+k][c] = V_444[j*4+k][c]
    # For 4x4x1 part: V_441[j, c] for j=0..3
    # V_full index is j*5+4
    for j in range(4):
        for c in range(16):
            V_full[j*5+4][49+c] = V_441[j][c]
    
    # W_full: shape (20, 65)
    W_full = [[0]*rank for _ in range(20)]
    # For 4x4x4 part: W_444[k*4+i, c] for k=0..3, i=0..3
    # W_full index is k*4+i
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W_full[k*4+i][c] = W_444[k*4+i][c]
    # For 4x4x1 part: W_441[i, c] for i=0..3
    # W_full index is 4*4+i = 16+i
    for i in range(4):
        for c in range(16):
            W_full[16+i][49+c] = W_441[i][c]
    
    result = {"U": U_full, "V": V_full, "W": W_full}
    
    # Verify
    if time.time() < deadline:
        if not _verify_decomposition(U_full, V_full, W_full, n, m, p):
            # Fallback to schoolbook
            rank_sb = n * m * p
            U_sb = [[0] * rank_sb for _ in range(n * m)]
            V_sb = [[0] * rank_sb for _ in range(m * p)]
            W_sb = [[0] * rank_sb for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i * m + j][col] = 1
                        V_sb[j * p + k][col] = 1
                        W_sb[k * n + i][col] = 1
                        col += 1
            result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return result

# EVOLVE-BLOCK-END
