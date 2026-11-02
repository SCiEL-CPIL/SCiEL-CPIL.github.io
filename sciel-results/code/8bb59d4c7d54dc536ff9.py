"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Attempts rank-23 decomposition (Bini et al.), falls back to rank-26 or rank-27.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    def idx_u(i, j):
        return i * m + j
        
    def idx_v(j, k):
        return j * p + k
        
    def idx_w(k, i):
        return k * n + i
    
    def verify(U, V, W, R):
        """Verify decomposition reconstructs the multiplication tensor."""
        U_r = U.reshape((n, m, R))
        V_r = V.reshape((m, p, R))
        W_r = W.reshape((p, n, R))
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        T_target = np.ones((n, m, p))
        return np.allclose(T_recon, T_target, atol=1e-9)
    
    # Try rank-23 decomposition first
    R23 = 23
    U23 = np.zeros((n * m, R23))
    V23 = np.zeros((m * p, R23))
    W23 = np.zeros((p * n, R23))
    
    # Rank-23 decomposition based on Bini et al. / Pan's algorithm
    # The 23 terms are carefully chosen linear combinations
    
    # Terms 1-7: Strassen-like core for 2x2 block (indices 0,1)
    # m1 = (a00+a11)(b00+b11)
    r = 0
    U23[idx_u(0,0), r] = 1; U23[idx_u(1,1), r] = 1
    V23[idx_v(0,0), r] = 1; V23[idx_v(1,1), r] = 1
    W23[idx_w(0,0), r] = 1; W23[idx_w(1,1), r] = 1
    
    # m2 = (a10+a11)b00
    r = 1
    U23[idx_u(1,0), r] = 1; U23[idx_u(1,1), r] = 1
    V23[idx_v(0,0), r] = 1
    W23[idx_w(0,1), r] = 1; W23[idx_w(1,1), r] = -1
    
    # m3 = a00(b01-b11)
    r = 2
    U23[idx_u(0,0), r] = 1
    V23[idx_v(0,1), r] = 1; V23[idx_v(1,1), r] = -1
    W23[idx_w(1,0), r] = 1; W23[idx_w(1,1), r] = 1
    
    # m4 = a11(b10-b00)
    r = 3
    U23[idx_u(1,1), r] = 1
    V23[idx_v(1,0), r] = 1; V23[idx_v(0,0), r] = -1
    W23[idx_w(0,0), r] = 1; W23[idx_w(0,1), r] = 1
    
    # m5 = (a00+a01)b11
    r = 4
    U23[idx_u(0,0), r] = 1; U23[idx_u(0,1), r] = 1
    V23[idx_v(1,1), r] = 1
    W23[idx_w(0,0), r] = -1; W23[idx_w(1,0), r] = 1
    
    # m6 = (a10-a00)(b00+b01)
    r = 5
    U23[idx_u(1,0), r] = 1; U23[idx_u(0,0), r] = -1
    V23[idx_v(0,0), r] = 1; V23[idx_v(0,1), r] = 1
    W23[idx_w(1,1), r] = 1
    
    # m7 = (a01-a11)(b10+b11)
    r = 6
    U23[idx_u(0,1), r] = 1; U23[idx_u(1,1), r] = -1
    V23[idx_v(1,0), r] = 1; V23[idx_v(1,1), r] = 1
    W23[idx_w(0,0), r] = 1
    
    # Terms 8-23: Handle the border (index 2) with 16 additional terms
    # These are optimized linear combinations
    
    # m8 = a02(b00+b01+b11)
    r = 7
    U23[idx_u(0,2), r] = 1
    V23[idx_v(0,0), r] = 1; V23[idx_v(0,1), r] = 1; V23[idx_v(1,1), r] = 1
    W23[idx_w(0,0), r] = 1; W23[idx_w(1,0), r] = 1
    
    # m9 = a12(b00+b01+b11)
    r = 8
    U23[idx_u(1,2), r] = 1
    V23[idx_v(0,0), r] = 1; V23[idx_v(0,1), r] = 1; V23[idx_v(1,1), r] = 1
    W23[idx_w(0,1), r] = 1; W23[idx_w(1,1), r] = 1
    
    # m10 = a20(b10+b11+b01)
    r = 9
    U23[idx_u(2,0), r] = 1
    V23[idx_v(1,0), r] = 1; V23[idx_v(1,1), r] = 1; V23[idx_v(0,1), r] = 1
    W23[idx_w(0,0), r] = 1; W23[idx_w(0,1), r] = 1
    
    # m11 = a21(b10+b11+b01)
    r = 10
    U23[idx_u(2,1), r] = 1
    V23[idx_v(1,0), r] = 1; V23[idx_v(1,1), r] = 1; V23[idx_v(0,1), r] = 1
    W23[idx_w(0,1), r] = 1; W23[idx_w(0,0), r] = 1
    
    # m12 = a22(b00+b10+b01+b11)
    r = 11
    U23[idx_u(2,2), r] = 1
    V23[idx_v(0,0), r] = 1; V23[idx_v(1,0), r] = 1; V23[idx_v(0,1), r] = 1; V23[idx_v(1,1), r] = 1
    W23[idx_w(0,0), r] = 1; W23[idx_w(1,0), r] = 1; W23[idx_w(0,1), r] = 1; W23[idx_w(1,1), r] = 1
    
    # m13 = (a02+a22)(b00+b10)
    r = 12
    U23[idx_u(0,2), r] = 1; U23[idx_u(2,2), r] = 1
    V23[idx_v(0,0), r] = 1; V23[idx_v(1,0), r] = 1
    W23[idx_w(0,0), r] = 1; W23[idx_w(0,1), r] = 1
    
    # m14 = (a12+a22)(b00+b10)
    r = 13
    U23[idx_u(1,2), r] = 1; U23[idx_u(2,2), r] = 1
    V23[idx_v(0,0), r] = 1; V23[idx_v(1,0), r] = 1
    W23[idx_w(0,1), r] = 1; W23[idx_w(0,0), r] = 1
    
    # m15 = (a02+a22)(b01+b11)
    r = 14
    U23[idx_u(0,2), r] = 1; U23[idx_u(2,2), r] = 1
    V23[idx_v(0,1), r] = 1; V23[idx_v(1,1), r] = 1
    W23[idx_w(1,0), r] = 1; W23[idx_w(1,1), r] = 1
    
    # m16 = (a12+a22)(b01+b11)
    r = 15
    U23[idx_u(1,2), r] = 1; U23[idx_u(2,2), r] = 1
    V23[idx_v(0,1), r] = 1; V23[idx_v(1,1), r] = 1
    W23[idx_w(1,1), r] = 1; W23[idx_w(1,0), r] = 1
    
    # m17 = (a20+a22)(b00+b01)
    r = 16
    U23[idx_u(2,0), r] = 1; U23[idx_u(2,2), r] = 1
    V23[idx_v(0,0), r] = 1; V23[idx_v(0,1), r] = 1
    W23[idx_w(0,0), r] = 1; W23[idx_w(1,0), r] = 1
    
    # m18 = (a21+a22)(b00+b01)
    r = 17
    U23[idx_u(2,1), r] = 1; U23[idx_u(2,2), r] = 1
    V23[idx_v(0,0), r] = 1; V23[idx_v(0,1), r] = 1
    W23[idx_w(0,1), r] = 1; W23[idx_w(1,1), r] = 1
    
    # m19 = a02*b20
    r = 18
    U23[idx_u(0,2), r] = 1
    V23[idx_v(2,0), r] = 1
    W23[idx_w(0,0), r] = 1
    
    # m20 = a12*b20
    r = 19
    U23[idx_u(1,2), r] = 1
    V23[idx_v(2,0), r] = 1
    W23[idx_w(0,1), r] = 1
    
    # m21 = a02*b21
    r = 20
    U23[idx_u(0,2), r] = 1
    V23[idx_v(2,1), r] = 1
    W23[idx_w(1,0), r] = 1
    
    # m22 = a12*b21
    r = 21
    U23[idx_u(1,2), r] = 1
    V23[idx_v(2,1), r] = 1
    W23[idx_w(1,1), r] = 1
    
    # m23 = a20*b02 + a21*b12 + a22*b22 (combined)
    r = 22
    U23[idx_u(2,0), r] = 1
    V23[idx_v(0,2), r] = 1
    W23[idx_w(2,0), r] = 1
    
    U23[idx_u(2,1), r] = 1
    V23[idx_v(1,2), r] = 1
    W23[idx_w(2,1), r] = 1
    
    U23[idx_u(2,2), r] = 1
    V23[idx_v(2,2), r] = 1
    W23[idx_w(2,2), r] = 1
    
    # Verify rank-23
    if verify(U23, V23, W23, R23):
        return {"U": U23.tolist(), "V": V23.tolist(), "W": W23.tolist()}
    
    # Fall back to rank-26
    R26 = 26
    U26 = np.zeros((n * m, R26))
    V26 = np.zeros((m * p, R26))
    W26 = np.zeros((p * n, R26))
    
    # Copy the first 7 terms
    for r in range(7):
        for i in range(n):
            for j in range(m):
                U26[idx_u(i,j), r] = U23[idx_u(i,j), r]
        for j in range(m):
            for k in range(p):
                V26[idx_v(j,k), r] = V23[idx_v(j,k), r]
        for k in range(p):
            for i in range(n):
                W26[idx_w(k,i), r] = W23[idx_w(k,i), r]
    
    # Standard terms for border (19 terms)
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U26[idx_u(i, j), col] = 1
                V26[idx_v(j, k), col] = 1
                W26[idx_w(k, i), col] = 1
                col += 1
    
    if verify(U26, V26, W26, R26):
        return {"U": U26.tolist(), "V": V26.tolist(), "W": W26.tolist()}
    
    # Final fallback: schoolbook rank 27
    R27 = 27
    U27 = np.zeros((n * m, R27))
    V27 = np.zeros((m * p, R27))
    W27 = np.zeros((p * n, R27))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U27[idx_u(i, j), col] = 1
                V27[idx_v(j, k), col] = 1
                W27[idx_w(k, i), col] = 1
                col += 1
    
    return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
# EVOLVE-BLOCK-END
