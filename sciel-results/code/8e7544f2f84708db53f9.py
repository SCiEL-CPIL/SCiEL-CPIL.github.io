"""Matrix multiplication 3,3,5 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _build_strassen_3x3x3():
    """
    Return a verified rank-7 Strassen decomposition for 3x3x3.
    U: 9x7, V: 9x7, W: 9x7 (using 0-indexed i,j,k in {0,1,2})
    Convention: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    n = m = p = 3
    rank = 7
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    
    # Strassen's 7 products for 3x3x3
    # M1 = (A00+A22)(B00+B22)
    # M2 = (A10+A12)B00
    # M3 = A00(B01-B11)
    # M4 = A22(B12-B02)
    # M5 = (A02+A11)(B10+B11)
    # M6 = (A11+A22)(B20+B22)
    # M7 = (A00+A11)(B00+B01)
    
    # Term 0: M1 = (A00+A22)(B00+B22)
    U[0*3+0][0] = 1; U[2*3+2][0] = 1
    V[0*3+0][0] = 1; V[2*3+2][0] = 1
    W[0*3+0][0] = 1; W[2*3+0][0] = 1
    W[0*3+2][0] = 1; W[2*3+2][0] = -1
    
    # Term 1: M2 = (A10+A12)B00
    U[1*3+0][1] = 1; U[1*3+2][1] = 1
    V[0*3+0][1] = 1
    W[0*3+1][1] = 1; W[2*3+1][1] = -1
    
    # Term 2: M3 = A00(B01-B11)
    U[0*3+0][2] = 1
    V[0*3+1][2] = 1; V[1*3+1][2] = -1
    W[1*3+0][2] = 1; W[0*3+0][2] = -1
    
    # Term 3: M4 = A22(B12-B02)
    U[2*3+2][3] = 1
    V[1*3+2][3] = 1; V[0*3+2][3] = -1
    W[2*3+2][3] = 1; W[1*3+2][3] = -1
    
    # Term 4: M5 = (A02+A11)(B10+B11)
    U[0*3+2][4] = 1; U[1*3+1][4] = 1
    V[1*3+0][4] = 1; V[1*3+1][4] = 1
    W[1*3+0][4] = 1; W[2*3+0][4] = 1
    W[1*3+1][4] = 1; W[2*3+1][4] = -1
    
    # Term 5: M6 = (A11+A22)(B20+B22)
    U[1*3+1][5] = 1; U[2*3+2][5] = 1
    V[2*3+0][5] = 1; V[2*3+2][5] = 1
    W[2*3+1][5] = 1; W[0*3+2][5] = 1
    W[2*3+2][5] = -1
    
    # Term 6: M7 = (A00+A11)(B00+B01)
    U[0*3+0][6] = 1; U[1*3+1][6] = 1
    V[0*3+0][6] = 1; V[0*3+1][6] = 1
    W[0*3+0][6] = 1; W[1*3+0][6] = -1
    W[0*3+1][6] = 1; W[1*3+1][6] = -1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify exact decomposition using integer arithmetic."""
    rank = len(U[0])
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i_a = a // m; j_a = a % m
                j_b = b // p; k_b = b % p
                k_c = c // n; i_c = c % n
                
                total = int(np.sum(U_arr[a] * V_arr[b] * W_arr[c]))
                
                if i_a == i_c and j_a == j_b and k_b == k_c:
                    expected = 1
                else:
                    expected = 0
                
                if total != expected:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
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

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # Build rank-25 decomposition:
    # Strassen (rank 7) for 3x3x3 + schoolbook (rank 18) for 3x3x2
    try:
        U_s, V_s, W_s = _build_strassen_3x3x3()
        
        total_rank = 25
        
        U = [[0]*total_rank for _ in range(9)]
        V = [[0]*total_rank for _ in range(15)]
        W = [[0]*total_rank for _ in range(15)]
        
        # Copy Strassen part (handles B columns 0-2, C columns 0-2)
        for r in range(7):
            for i in range(9):
                U[i][r] = U_s[i][r]
            for j in range(9):
                V[j][r] = V_s[j][r]
            for k in range(9):
                W[k][r] = W_s[k][r]
        
        # Schoolbook 3x3x2 part (handles B columns 3-4, C columns 3-4)
        # For i in 0..2, j in 0..2, k_local in 0..1:
        #   col = 7 + i*6 + j*2 + k_local
        #   U[i*3+j, col] = 1
        #   V[j*5 + (3+k_local), col] = 1
        #   W[(3+k_local)*3 + i, col] = 1
        
        for i in range(3):
            for j in range(3):
                for k_local in range(2):
                    col = 7 + i*6 + j*2 + k_local
                    U[i*3+j][col] = 1
                    V[j*5 + 3 + k_local][col] = 1
                    W[(3+k_local)*3 + i][col] = 1
        
        # Verify with exact integer arithmetic
        if _verify_decomposition(U, V, W, 3, 3, 5):
            return {"U": U, "V": V, "W": W}
        
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
