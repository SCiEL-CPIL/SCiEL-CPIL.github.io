import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition."""
    if not U or not V or not W:
        return False
    r = len(U[0])
    # Check dimensions
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for row in U:
        if len(row) != r: return False
    for row in V:
        if len(row) != r: return False
    for row in W:
        if len(row) != r: return False
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(r))
                if val != 1:
                    return False
    return True

def _strassen_2x2():
    """
    Return the Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    Returns U, V, W as lists of lists.
    U: 4 x 7, V: 4 x 7, W: 4 x 7
    Indices:
    U: (i,j) -> i*2+j
    V: (j,k) -> j*2+k
    W: (k,i) -> k*2+i
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a+d)*(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (c+d)*e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = a*(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = d*(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (a+b)*h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (c-a)*(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (b-d)*(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _construct_rank34(n, m, p):
    """
    Construct a rank-34 decomposition for 3x3 * 3x4.
    Uses Strassen for the 2x2 blocks in the top-left 2x2 of A and 
    the corresponding 2x2 blocks of B.
    """
    # Dimensions
    # A is 3x3, B is 3x4, C is 3x4
    # U: 9 x r, V: 12 x r, W: 12 x r
    
    r = 34
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    col = 0
    
    # 1. Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    sU, sV, sW = _strassen_2x2()
    for c in range(7):
        for a_idx in range(4):
            i, j = divmod(a_idx, 2)
            U[i*m+j][col+c] = sU[a_idx][c]
        for b_idx in range(4):
            j, k = divmod(b_idx, 2)
            V[j*p+k][col+c] = sV[b_idx][c]
        for c_idx in range(4):
            k, i = divmod(c_idx, 2)
            W[k*n+i][col+c] = sW[c_idx][c]
    col += 7
    
    # 2. Strassen for A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # Note: B[0:2, 2:4] corresponds to B indices (j=0,1; k=2,3)
    # We reuse the same Strassen structure but map B and W indices differently.
    for c in range(7):
        for a_idx in range(4):
            i, j = divmod(a_idx, 2)
            U[i*m+j][col+c] = sU[a_idx][c]
        for b_idx in range(4):
            j, k_local = divmod(b_idx, 2)
            k = k_local + 2 # Offset k by 2
            V[j*p+k][col+c] = sV[b_idx][c]
        for c_idx in range(4):
            k_local, i = divmod(c_idx, 2)
            k = k_local + 2 # Offset k by 2
            W[k*n+i][col+c] = sW[c_idx][c]
    col += 7
    
    # 3. A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    # A[0,2], A[1,2] with B[2,0], B[2,1]
    for i in range(2):
        for k in range(2):
            U[i*m+2][col] = 1
            V[2*p+k][col] = 1
            W[k*n+i][col] = 1
            col += 1
            
    # 4. A[0:2, 2] * B[2, 2:4] -> C[0:2, 2:4]
    for i in range(2):
        for k in range(2, 4):
            U[i*m+2][col] = 1
            V[2*p+k][col] = 1
            W[k*n+i][col] = 1
            col += 1
            
    # 5. A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
    for j in range(2):
        for k in range(2):
            U[2*m+j][col] = 1
            V[j*p+k][col] = 1
            W[k*n+2][col] = 1
            col += 1
            
    # 6. A[2, 2] * B[2, 0:2] -> C[2, 0:2]
    for k in range(2):
        U[2*m+2][col] = 1
        V[2*p+k][col] = 1
        W[k*n+2][col] = 1
        col += 1
        
    # 7. A[2, 0:2] * B[0:2, 2:4] -> C[2, 2:4]
    for j in range(2):
        for k in range(2, 4):
            U[2*m+j][col] = 1
            V[j*p+k][col] = 1
            W[k*n+2][col] = 1
            col += 1
            
    # 8. A[2, 2] * B[2, 2:4] -> C[2, 2:4]
    for k in range(2, 4):
        U[2*m+2][col] = 1
        V[2*p+k][col] = 1
        W[k*n+2][col] = 1
        col += 1
        
    if col != r:
        raise ValueError(f"Rank mismatch: expected {r}, got {col}")
        
    return U, V, W

def _schoolbook(n, m, p):
    rank = n * m * p
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try rank 34
    try:
        U, V, W = _construct_rank34(n, m, p)
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
