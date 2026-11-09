"""Find a rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using einsum.
    
    T[i,j,jp,k,kp,ip] = sum_r U[i*m+j,r] * V[jp*p+k,r] * W[kp*n+ip,r]
    must equal delta_{i,ip} * delta_{j,jp} * delta_{k,kp}
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    R = U.shape[1]
    
    U3 = U.reshape(n, m, R)   # U3[i,j,r]
    V3 = V.reshape(m, p, R)   # V3[jp,k,r]
    W3 = W.reshape(p, n, R)   # W3[kp,ip,r]
    
    # T[i,j,jp,k,kp,ip] = sum_r U3[i,j,r] * V3[jp,k,r] * W3[kp,ip,r]
    T = np.einsum('ijr,jkr,pir->ijjkpi', U3, V3, W3, optimize=True)
    
    # Build target: delta_{i,ip} * delta_{j,jp} * delta_{k,kp}
    # Target shape: (n, m, m, p, p, n)
    # T[i,j,jp,k,kp,ip] should be 1 if i==ip and j==jp and k==kp
    I, J, JP, K, KP, IP = np.meshgrid(
        range(n), range(m), range(m), range(p), range(p), range(n),
        indexing='ij'
    )
    target = ((I == IP) & (J == JP) & (K == KP)).astype(np.float64)
    
    return np.allclose(T, target, atol=1e-9, rtol=1e-9)

def _strassen_2x2x2():
    """Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    Shapes: (4, 7), (4, 7), (4, 7)
    Indexing: i*2+j
    """
    # Strassen's 7 products
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = -1
    
    return U, V, W

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    n, m, p = DIMENSIONS
    
    # 1. Construct Strassen-based decomposition for rank 42
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Build 2x2x6 decomposition (Rank 21)
    # A is 2x2 (4 entries), B is 2x6 (12 entries), C is 2x6 (12 entries)
    # B = [B1, B2, B3] where Bk is 2x2
    # C = C1 + C2 + C3 where Ck = A Bk
    # We use 3 independent Strassen decompositions.
    
    R_226 = 21
    U_226 = np.zeros((4, R_226))
    V_226 = np.zeros((12, R_226))
    W_226 = np.zeros((12, R_226))
    
    for k in range(3):
        for r in range(7):
            col = k * 7 + r
            U_226[:, col] = U_s[:, r]
            # V_226 indices: 0-3 for B1, 4-7 for B2, 8-11 for B3
            # Bk is 2x2, so indices are k*4 to k*4+3
            V_226[k*4:(k+1)*4, col] = V_s[:, r]
            # W_226 indices: same as V_226
            W_226[k*4:(k+1)*4, col] = W_s[:, r]
            
    # Build 2x4x6 decomposition (Rank 42)
    # A = [A1, A2] (2x2 blocks)
    # B = [B1; B2] (2x6 blocks)
    # C = A1 B1 + A2 B2
    
    R_total = 42
    U = np.zeros((8, R_total))
    V = np.zeros((24, R_total))
    W = np.zeros((12, R_total))
    
    # Part 1: A1 B1 (uses first 21 cols)
    # A1 is rows 0-3 of A
    # B1 is rows 0-11 of B
    U[0:4, 0:21] = U_226
    V[0:12, 0:21] = V_226
    W[:, 0:21] = W_226
    
    # Part 2: A2 B2 (uses next 21 cols)
    # A2 is rows 4-7 of A
    # B2 is rows 12-23 of B
    U[4:8, 21:42] = U_226
    V[12:24, 21:42] = V_226
    W[:, 21:42] = W_226
    
    # Verify
    if _verify_fast(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank_school = n * m * p
        U_sb = np.zeros((n * m, rank_school))
        V_sb = np.zeros((m * p, rank_school))
        W_sb = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
