"""Matrix multiplication 3,3,5 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Returns U, V, W for the 2x2x2 Strassen decomposition.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    Indices:
    U: [i*2 + j]
    V: [j*2 + k]
    W: [k*2 + i]
    """
    n = 2
    m = 2
    p = 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
    W[0*2+0, 0] = 1; W[1*2+1, 0] = 1
    
    # M2 = (A10 + A11)B00
    U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[1*2+0, 1] = 1
    
    # M3 = A00(B01 - B11)
    U[0*2+0, 2] = 1
    V[0*2+1, 2] = 1; V[1*2+1, 2] = -1
    W[0*2+1, 2] = 1
    
    # M4 = A11(B10 - B00)
    U[1*2+1, 3] = 1
    V[1*2+0, 3] = 1; V[0*2+0, 3] = -1
    W[1*2+0, 3] = 1
    
    # M5 = (A00 + A01)B11
    U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[0*2+1, 4] = 1; W[1*2+1, 4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1; V[0*2+1, 5] = 1
    W[1*2+1, 5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
    V[1*2+0, 6] = 1; V[1*2+1, 6] = 1
    W[0*2+0, 6] = 1; W[1*2+1, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # We construct a decomposition of rank 44
    # 7 Strassen terms for the 2x2x2 core (i in {0,1}, j in {0,1}, k in {0,1})
    # 37 Schoolbook terms for the rest
    
    rank = 44
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # 1. Strassen terms for core
    U_s, V_s, W_s = _strassen_2x2x2()
    # Map 2x2 indices to 3x3/3x5 indices
    # U_s indices: i*2+j -> i*3+j
    for c in range(7):
        for u_idx in range(4):
            i = u_idx // 2
            j = u_idx % 2
            U[i*3+j, c] = U_s[u_idx, c]
            
        for v_idx in range(4):
            j = v_idx // 2
            k = v_idx % 2
            V[j*5+k, c] = V_s[v_idx, c]
            
        for w_idx in range(4):
            k = w_idx // 2
            i = w_idx % 2
            W[k*3+i, c] = W_s[w_idx, c]
            
    # 2. Schoolbook terms for the rest
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip the core
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
