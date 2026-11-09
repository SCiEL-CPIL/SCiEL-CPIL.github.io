"""Rewritten program: construct a rank-98 exact tensor decomposition for 4x4 * 4x8."""
import time

DIMENSIONS = [4, 4, 8]

# Deadline guard
T0 = time.time()
TIME_LIMIT = 145.0  # safety margin before 150s

def _strassen_2x2_2():
    """Return (U, V, W) as lists of lists for 2x2x2 Strassen, rank 7.
    
    Conventions:
      U: shape (2*2, 7) = (4, 7); U[i*2+j, t]
      V: shape (2*2, 7) = (4, 7); V[j*2+k, t]
      W: shape (2*2, 7) = (4, 7); W[k*2+i, t]
    
    The 7 rank-1 terms (M1..M7) with standard Strassen:
      M1 = (A11+A22)(B11+B22)
      M2 = (A21+A22)B11
      M3 = A11(B12-B22)
      M4 = A22(B21-B11)
      M5 = (A11+A12)B22
      M6 = (A21-A11)(B11+B12)
      M7 = (A12-A22)(B21+B22)
    
    C11 = M1 - M4 + M5 - M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M3 + M6 + M7
    """
    # U rows indexed by (i,j) -> i*2+j: A11=0, A12=1, A21=2, A22=3
    # V rows indexed by (j,k) -> j*2+k: B11=0, B12=1, B21=2, B22=3
    # W rows indexed by (k,i) -> k*2+i: C11=0, C12=1, C21=2, C22=3
    
    # Each term t: U[:,t], V[:,t], W[:,t]
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 0: M1 = (A11+A22)(B11+B22) -> C11 + C22
    # U: A11=1, A22=1; V: B11=1, B22=1; W: C11=1, C22=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 1: M2 = (A21+A22)B11 -> C21
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # Term 2: M3 = A11(B12-B22) -> C12 - C22
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = -1
    
    # Term 3: M4 = A22(B21-B11) -> -C11 + C21
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[2][3] = 1
    
    # Term 4: M5 = (A11+A12)B22 -> C11 + C12
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # Term 5: M6 = (A21-A11)(B11+B12) -> C22
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 6: M7 = (A12-A22)(B21+B22) -> -C11 + C22
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[3][6] = 1
    
    return U, V, W

def _kronecker_product_4x4x8():
    """Construct rank-98 decomposition for 4x4 * 4x8 using Strassen tensor square.
    
    We view 4x4x8 as two copies of 4x4x4:
      B = [B1; B2] where B1, B2 are 4x4
      C = [C1; C2] where C1, C2 are 4x4
      C1 = A * B1, C2 = A * B2
    
    For each 4x4x4, we use Strassen^2 (tensor product of two 2x2x2 Strassen).
    Strassen^2 has rank 7*7 = 49.
    Total rank = 49 + 49 = 98.
    
    Indexing for 4x4x4:
      A is 4x4: row index = a1*2 + a2 where a1,a2 in {0,1}
      B is 4x4: col index = b1*2 + b2
      C is 4x4: index = c1*2 + c2 (c1 is row of C, c2 is col of C)
      W index = k*4 + i where k is col of C, i is row of C
      So W[k*4+i, t]
    
    For the 2x2x2 sub-problems:
      A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
      Similarly for B and C.
    
    Kronecker: for 4x4x4, we have outer (2x2) and inner (2x2).
    A[i,j] = A_outer[i//2, j//2] * A_inner[i%2, j%2]
    The rank-1 terms factorize: term (t1, t2) where t1 is outer Strassen term, t2 is inner.
    U[i*4+j, t1*7+t2] = U_outer[i//2*2+j//2, t1] * U_inner[i%2*2+j%2, t2]
    V[j*4+k, t1*7+t2] = V_outer[j//2*2+k//2, t1] * V_inner[j%2*2+k%2, t2]
    W[k*4+i, t1*7+t2] = W_outer[k//2*2+i//2, t1] * W_inner[k%2*2+i%2, t2]
    """
    U2, V2, W2 = _strassen_2x2_2()
    
    rank = 49
    # For 4x4x4: U shape (16, 49), V shape (16, 49), W shape (16, 49)
    U4 = [[0]*rank for _ in range(16)]
    V4 = [[0]*rank for _ in range(16)]
    W4 = [[0]*rank for _ in range(16)]
    
    for t1 in range(7):
        for t2 in range(7):
            t = t1 * 7 + t2
            for i in range(4):
                for j in range(4):
                    u_val = U2[i//2 * 2 + j//2][t1] * U2[i%2 * 2 + j%2][t2]
                    if u_val != 0:
                        U4[i * 4 + j][t] = u_val
            for j in range(4):
                for k in range(4):
                    v_val = V2[j//2 * 2 + k//2][t1] * V2[j%2 * 2 + k%2][t2]
                    if v_val != 0:
                        V4[j * 4 + k][t] = v_val
            for k in range(4):
                for i in range(4):
                    w_val = W2[k//2 * 2 + i//2][t1] * W2[k%2 * 2 + i%2][t2]
                    if w_val != 0:
                        W4[k * 4 + i][t] = w_val
    
    # Now build 4x4x8: rank 98
    # C1 = A * B1 (first 4 cols of B, first 4 rows of C)
    # C2 = A * B2 (last 4 cols of B, last 4 rows of C)
    #
    # B is 4x8: B[j, k] where j in 0..3, k in 0..7
    #   B1: k in 0..3, B2: k in 4..7
    # C is 8x4: C[k, i] where k in 0..7, i in 0..3
    #   C1: k in 0..3, C2: k in 4..7
    #
    # U shape: (4*4, 98) = (16, 98)
    # V shape: (4*8, 98) = (32, 98)
    # W shape: (8*4, 98) = (32, 98)
    
    total_rank = 98
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(32)]
    W = [[0]*total_rank for _ in range(32)]
    
    # First 49 columns: C1 = A * B1
    # U for C1: same as U4 (A is the same)
    # V for C1: B1 rows 0..3 of B -> V4 rows 0..3 mapped to V rows 0..3
    #   V4[j*4+k, t] for k in 0..3 -> V[j*8+k, t] for k in 0..3
    # W for C1: C1 rows 0..3 of C -> W4 rows 0..3 mapped to W rows 0..3
    #   W4[k*4+i, t] for k in 0..3 -> W[k*4+i, t] for k in 0..3
    
    for t in range(49):
        for i in range(4):
            for j in range(4):
                U[i*4+j][t] = U4[i*4+j][t]
        for j in range(4):
            for k in range(4):
                V[j*8+k][t] = V4[j*4+k][t]
        for k in range(4):
            for i in range(4):
                W[k*4+i][t] = W4[k*4+i][t]
    
    # Last 49 columns (offset 49): C2 = A * B2
    # U for C2: same as U4
    # V for C2: B2 rows 4..7 of B -> V4 rows 0..3 mapped to V rows 4..7
    #   V4[j*4+k, t] for k in 0..3 -> V[j*8+(k+4), t]
    # W for C2: C2 rows 4..7 of C -> W4 rows 0..3 mapped to W rows 16..31
    #   W4[k*4+i, t] for k in 0..3 -> W[(k+4)*4+i, t]
    
    for t in range(49):
        t_off = t + 49
        for i in range(4):
            for j in range(4):
                U[i*4+j][t_off] = U4[i*4+j][t]
        for j in range(4):
            for k in range(4):
                V[j*8 + (k+4)][t_off] = V4[j*4+k][t]
        for k in range(4):
            for i in range(4):
                W[(k+4)*4 + i][t_off] = W4[k*4+i][t]
    
    return U, V, W

def construct():
    if time.time() - T0 > TIME_LIMIT:
        # Fallback: return schoolbook if somehow we're out of time (shouldn't happen)
        n, m, p = DIMENSIONS
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
    
    U, V, W = _kronecker_product_4x4x8()
    return {"U": U, "V": V, "W": W}
