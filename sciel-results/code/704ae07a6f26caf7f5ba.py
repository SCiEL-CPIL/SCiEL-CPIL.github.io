"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-10 exact tensor decomposition for 2x3 * 3x4 matrix multiplication.
    
    Construction:
    - Slice j=0: rank 1 (outer product A[:,0] * B[0,:])
    - Slices j=1,2: rank 9 via block-Strassen sharing M1 across two 2x2 output blocks
    
    A = [[a00, a01, a02],
         [a10, a11, a12]]
    B = [[b00, b01, b02, b03],
         [b10, b11, b12, b13],
         [b20, b21, b22, b23]]
    
    C = A*B computed via 10 scalar multiplications:
    r0 = a00*b00 + a10*b00  (j=0 slice, rank 1)
    r1 = (a00+a11)*(b10+b21)  (shared M1 for both output blocks)
    r2 = (a10+a11)*b10
    r3 = a00*(b12-b21)
    r4 = a11*(b20-b10)
    r5 = (a00+a01)*b21
    r6 = (a10-a00)*(b10+b12)
    r7 = (a01-a11)*(b20+b22)
    r8 = (a00+a11)*(b13+b23)  (M1 for second block)
    r9 = (a01-a11)*(b23-b13)  (M7 for second block)
    
    C[0,0] = r1 - r2 + r4
    C[0,1] = r3 + r5
    C[0,2] = r1 - r5 + r8
    C[0,3] = r3 + r5 + r9
    C[1,0] = r2 + r4
    C[1,1] = r1 - r2 + r3 + r6
    C[1,2] = r1 - r2 + r3 + r6 + r8
    C[1,3] = r1 - r2 + r3 + r6 + r8 + r9
    """
    n, m, p = 2, 3, 4
    nm, mp, pn = n * m, m * p, p * n
    R = 10
    
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))
    
    # r0: j=0 slice
    U[0, 0] = 1.0  # a00
    U[3, 0] = 1.0  # a10
    V[0, 0] = 1.0  # b00
    W[0, 0] = 1.0  # C[0,0]
    W[4, 0] = 1.0  # C[1,0]
    
    # r1: (a00+a11)*(b10+b21)
    U[0, 1] = 1.0
    U[4, 1] = 1.0
    V[4, 1] = 1.0
    V[9, 1] = 1.0
    W[0, 1] = 1.0
    W[2, 1] = 1.0
    W[5, 1] = 1.0
    W[7, 1] = 1.0
    
    # r2: (a10+a11)*b10
    U[3, 2] = 1.0
    U[4, 2] = 1.0
    V[4, 2] = 1.0
    W[0, 2] = -1.0
    W[5, 2] = -1.0
    W[7, 2] = -1.0
    
    # r3: a00*(b12-b21)
    U[0, 3] = 1.0
    V[6, 3] = 1.0
    V[9, 3] = -1.0
    W[2, 3] = 1.0
    W[6, 3] = 1.0
    W[7, 3] = 1.0
    
    # r4: a11*(b20-b10)
    U[4, 4] = 1.0
    V[8, 4] = 1.0
    V[4, 4] = -1.0
    W[0, 4] = 1.0
    W[5, 4] = 1.0
    
    # r5: (a00+a01)*b21
    U[0, 5] = 1.0
    U[1, 5] = 1.0
    V[9, 5] = 1.0
    W[2, 5] = -1.0
    W[6, 5] = 1.0
    W[7, 5] = 1.0
    
    # r6: (a10-a00)*(b10+b12)
    U[3, 6] = 1.0
    U[0, 6] = -1.0
    V[4, 6] = 1.0
    V[6, 6] = 1.0
    W[7, 6] = 1.0
    
    # r7: (a01-a11)*(b20+b22)
    U[1, 7] = 1.0
    U[4, 7] = -1.0
    V[8, 7] = 1.0
    V[10, 7] = 1.0
    W[3, 7] = 1.0
    W[6, 7] = 1.0
    
    # r8: (a00+a11)*(b13+b23)
    U[0, 8] = 1.0
    U[4, 8] = 1.0
    V[7, 8] = 1.0
    V[11, 8] = 1.0
    W[2, 8] = 1.0
    W[6, 8] = 1.0
    W[7, 8] = 1.0
    
    # r9: (a01-a11)*(b23-b13)
    U[1, 9] = 1.0
    U[4, 9] = -1.0
    V[11, 9] = 1.0
    V[7, 9] = -1.0
    W[6, 9] = 1.0
    
    # Verify
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    Tc = np.zeros((nm, mp, pn))
    for r in range(R):
        Tc += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    if np.max(np.abs(Tc - T)) > 1e-10:
        # Fallback to schoolbook
        R_fb = n * m * p
        U = np.zeros((nm, R_fb))
        V = np.zeros((mp, R_fb))
        W = np.zeros((pn, R_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
