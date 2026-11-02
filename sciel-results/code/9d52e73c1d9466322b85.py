"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-26 decomposition for 3x3 matrix multiplication.
    This is based on Strassen's algorithm for the 2x2 core (rank 7)
    and standard multiplication for the border (rank 19).
    Total rank = 26.
    """
    n, m, p = 3, 3, 3
    R = 26
    
    U = np.zeros((9, R))
    V = np.zeros((9, R))
    W = np.zeros((9, R))
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        pass # We will fill U, V, W directly

    # Indices: i,j for U (0-8), j,k for V (0-8), k,i for W (0-8)
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # Strassen Terms for 2x2 Core (i,j,k in 0,1)
    # m1 = (a00 + a11)(b00 + b11) -> c00 += m1, c11 += m1
    r = 0
    U[0*3+0, r] = 1; U[1*3+1, r] = 1
    V[0*3+0, r] = 1; V[1*3+1, r] = 1
    W[0*3+0, r] = 1; W[1*3+1, r] = 1
    
    # m2 = (a10 + a11)b00 -> c10 += m2, c11 -= m2
    r = 1
    U[1*3+0, r] = 1; U[1*3+1, r] = 1
    V[0*3+0, r] = 1
    W[0*3+1, r] = 1; W[1*3+1, r] = -1
    
    # m3 = a00(b01 - b11) -> c01 += m3, c11 += m3
    r = 2
    U[0*3+0, r] = 1
    V[0*3+1, r] = 1; V[1*3+1, r] = -1
    W[1*3+0, r] = 1; W[1*3+1, r] = 1
    
    # m4 = a11(b10 - b00) -> c00 += m4, c10 += m4
    r = 3
    U[1*3+1, r] = 1
    V[1*3+0, r] = 1; V[0*3+0, r] = -1
    W[0*3+0, r] = 1; W[0*3+1, r] = 1
    
    # m5 = (a00 + a01)b11 -> c00 -= m5, c01 += m5
    r = 4
    U[0*3+0, r] = 1; U[0*3+1, r] = 1
    V[1*3+1, r] = 1
    W[0*3+0, r] = -1; W[1*3+0, r] = 1
    
    # m6 = (a10 - a00)(b00 + b01) -> c11 += m6
    r = 5
    U[1*3+0, r] = 1; U[0*3+0, r] = -1
    V[0*3+0, r] = 1; V[0*3+1, r] = 1
    W[1*3+1, r] = 1
    
    # m7 = (a01 - a11)(b10 + b11) -> c00 += m7
    r = 6
    U[0*3+1, r] = 1; U[1*3+1, r] = -1
    V[1*3+0, r] = 1; V[1*3+1, r] = 1
    W[0*3+0, r] = 1
    
    # Border Terms (19 terms)
    # i=2, j=0, k=0
    r = 7
    U[2*3+0, r] = 1
    V[0*3+0, r] = 1
    W[0*3+2, r] = 1
    
    # i=2, j=0, k=1
    r = 8
    U[2*3+0, r] = 1
    V[0*3+1, r] = 1
    W[1*3+2, r] = 1
    
    # i=2, j=0, k=2
    r = 9
    U[2*3+0, r] = 1
    V[0*3+2, r] = 1
    W[2*3+2, r] = 1
    
    # i=2, j=1, k=0
    r = 10
    U[2*3+1, r] = 1
    V[1*3+0, r] = 1
    W[0*3+2, r] = 1
    
    # i=2, j=1, k=1
    r = 11
    U[2*3+1, r] = 1
    V[1*3+1, r] = 1
    W[1*3+2, r] = 1
    
    # i=2, j=1, k=2
    r = 12
    U[2*3+1, r] = 1
    V[1*3+2, r] = 1
    W[2*3+2, r] = 1
    
    # i=2, j=2, k=0
    r = 13
    U[2*3+2, r] = 1
    V[2*3+0, r] = 1
    W[0*3+2, r] = 1
    
    # i=2, j=2, k=1
    r = 14
    U[2*3+2, r] = 1
    V[2*3+1, r] = 1
    W[1*3+2, r] = 1
    
    # i=2, j=2, k=2
    r = 15
    U[2*3+2, r] = 1
    V[2*3+2, r] = 1
    W[2*3+2, r] = 1
    
    # i=0, j=2, k=0
    r = 16
    U[0*3+2, r] = 1
    V[2*3+0, r] = 1
    W[0*3+0, r] = 1
    
    # i=0, j=2, k=1
    r = 17
    U[0*3+2, r] = 1
    V[2*3+1, r] = 1
    W[1*3+0, r] = 1
    
    # i=0, j=2, k=2
    r = 18
    U[0*3+2, r] = 1
    V[2*3+2, r] = 1
    W[2*3+0, r] = 1
    
    # i=1, j=2, k=0
    r = 19
    U[1*3+2, r] = 1
    V[2*3+0, r] = 1
    W[0*3+1, r] = 1
    
    # i=1, j=2, k=1
    r = 20
    U[1*3+2, r] = 1
    V[2*3+1, r] = 1
    W[1*3+1, r] = 1
    
    # i=1, j=2, k=2
    r = 21
    U[1*3+2, r] = 1
    V[2*3+2, r] = 1
    W[2*3+1, r] = 1
    
    # i=0, j=0, k=2
    r = 22
    U[0*3+0, r] = 1
    V[0*3+2, r] = 1
    W[2*3+0, r] = 1
    
    # i=0, j=1, k=2
    r = 23
    U[0*3+1, r] = 1
    V[1*3+2, r] = 1
    W[2*3+0, r] = 1
    
    # i=1, j=0, k=2
    r = 24
    U[1*3+0, r] = 1
    V[0*3+2, r] = 1
    W[2*3+1, r] = 1
    
    # i=1, j=1, k=2
    r = 25
    U[1*3+1, r] = 1
    V[1*3+2, r] = 1
    W[2*3+1, r] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
