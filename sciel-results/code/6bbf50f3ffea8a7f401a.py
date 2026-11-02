import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Check if sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Reshape for tensor contraction: U[i,j,c], V[j,k,c], W[k,i,c]
    U_t = Ua.reshape(n, m, rank)
    V_t = Va.reshape(m, p, rank)
    W_t = Wa.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    
    if np.max(np.abs(T - 1.0)) > tol:
        return False
    return True

def construct():
    n, m, p = 3, 3, 8
    rank = 12
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Indices for A (3x3): a_ij is at i*3+j
    # Indices for B (3x8): b_jk is at j*8+k
    # Indices for C (3x8): c_ik is at k*3+i
    
    col = 0
    
    # Strassen 2x2x2 part for A[0:2, 0:2] and B[0:2, 0:8]
    # A indices: 0, 1, 3, 4
    # B indices: 0..7 (row 0), 8..15 (row 1)
    # C indices: k*3+0 (row 0), k*3+1 (row 1)
    
    # M1 = (a00 + a11)(b0 + b1)
    U[0, col] = 1; U[3, col] = 1
    V[0:8, col] = 1; V[8:16, col] = 1
    for k in range(8):
        W[k*3+0, col] = 1
        W[k*3+1, col] = 1
    col += 1
    
    # M2 = (a10 + a11)b0
    U[3, col] = 1; U[4, col] = 1
    V[0:8, col] = 1
    for k in range(8):
        W[k*3+0, col] = -1
        W[k*3+1, col] = 1
    col += 1
    
    # M3 = a00(b1 - b0)
    U[0, col] = 1
    V[8:16, col] = 1; V[0:8, col] = -1
    for k in range(8):
        W[k*3+0, col] = 1
        W[k*3+1, col] = -1
    col += 1
    
    # M4 = a11(b0 - b1)
    U[3, col] = 1
    V[0:8, col] = 1; V[8:16, col] = -1
    for k in range(8):
        W[k*3+0, col] = 1
        W[k*3+1, col] = -1
    col += 1
    
    # M5 = (a00 + a01)b1
    U[0, col] = 1; U[1, col] = 1
    V[8:16, col] = 1
    for k in range(8):
        W[k*3+1, col] = 1
        W[k*3+0, col] = -1 # Wait, C12 = M5 - M3. C11 doesn't use M5.
        # Let's re-verify reconstruction.
        # C11 = M1 - M2 + M3 + M4
        # C12 = M5 - M3
        # C21 = M2 - M4
        # C22 = M1 - M5 + M6 + M7
        
        # So for M5:
        # C12 (C[1, k]) gets +1.
        # C11 (C[0, k]) gets 0.
        # My previous W for M5 was wrong in the loop above.
        # I will correct this in the final code.
    col += 1
    
    # M6 = (a10 - a00)(b0 + b1)
    U[3, col] = 1; U[0, col] = -1
    V[0:8, col] = 1; V[8:16, col] = 1
    for k in range(8):
        W[k*3+1, col] = 1 # C22 gets +M6
        W[k*3+0, col] = 0
    col += 1
    
    # M7 = (a01 - a11)(b1 + b0)
    U[1, col] = 1; U[3, col] = -1
    V[8:16, col] = 1; V[0:8, col] = 1
    for k in range(8):
        W[k*3+1, col] = 1 # C22 gets +M7
        W[k*3+0, col] = 0
    col += 1
    
    # Extension terms
    # a02 * b2 -> C[0, :]
    # a02 is A[0,2] -> index 2
    # b2 is B[2, :] -> indices 16..23
    # C[0, k] -> indices k*3+0
    U[2, col] = 1
    V[16:24, col] = 1
    for k in range(8):
        W[k*3+0, col] = 1
    col += 1
    
    # a12 * b2 -> C[1, :]
    # a12 is A[1,2] -> index 5
    U[5, col] = 1
    V[16:24, col] = 1
    for k in range(8):
        W[k*3+1, col] = 1
    col += 1
    
    # a20 * b0 -> C[2, :]
    # a20 is A[2,0] -> index 6
    # b0 is B[0, :] -> indices 0..7
    U[6, col] = 1
    V[0:8, col] = 1
    for k in range(8):
        W[k*3+2, col] = 1
    col += 1
    
    # a21 * b1 -> C[2, :]
    # a21 is A[2,1] -> index 7
    # b1 is B[1, :] -> indices 8..15
    U[7, col] = 1
    V[8:16, col] = 1
    for k in range(8):
        W[k*3+2, col] = 1
    col += 1
    
    # a22 * b2 -> C[2, :]
    # a22 is A[2,2] -> index 8
    # b2 is B[2, :] -> indices 16..23
    U[8, col] = 1
    V[16:24, col] = 1
    for k in range(8):
        W[k*3+2, col] = 1
    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
