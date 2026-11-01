"""Optimized 4x4x4 matrix multiplication tensor decomposition."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2_decomposition():
    """
    Return U, V, W for 2x2 matrix multiplication using Strassen's 7-multiplication algorithm.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] with n=m=p=2.
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen multiplications and their contributions:
    # M1 = (A00+A11) * (B00+B11) -> C00 += 1, C11 += 1
    # M2 = (A10+A11) * B00       -> C10 += 1
    # M3 = A00 * (B01-B11)       -> C01 += 1, C11 -= 1
    # M4 = A11 * (B10-B00)       -> C00 += 1, C10 += 1
    # M5 = (A00+A01) * B11       -> C00 -= 1, C01 += 1
    # M6 = (A10-A00) * (B00+B01) -> C11 += 1
    # M7 = (A01-A11) * (B10+B11) -> C00 += 1, C11 -= 1
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = -1
    
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = -1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct using numpy."""
    rank = U.shape[1]
    # Reshape for efficient computation
    U_r = U.reshape(n, m, rank)  # U[i, j, r]
    V_r = V.reshape(m, p, rank)  # V[j, k, r]
    W_r = W.reshape(p, n, rank)  # W[k, i, r]
    
    # For each (i, j, k), check sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1
    # This is equivalent to checking that the tensor T[i,j,k] = 1 for all i,j,k
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Build the tensor T
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We can compute this as: for each r, outer product of U[:,:,r], V[:,:,r], W[:,:,r]
    # but with appropriate index alignment.
    
    # Actually, let's just do it directly with numpy broadcasting
    # U_r[i,j,r], V_r[j,k,r], W_r[k,i,r]
    # We need: sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    
    # This is a 4D contraction. Let's do it step by step.
    # First, for fixed i, j: sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    # = sum_k (sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r])
    
    # Let's compute M[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    # Using einsum: M[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    # = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    return np.allclose(T, 1.0, atol=1e-9)

def _strassen_4x4_decomposition():
    """
    Construct rank-49 decomposition for 4x4 matrix multiplication using two levels of Strassen.
    """
    n = m = p = 4
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Get the 2x2 Strassen decomposition
    U2, V2, W2 = _strassen_2x2_decomposition()
    
    # Block-level Strassen terms for 4x4:
    # Each term has P (coefficients for A blocks), Q (coefficients for B blocks), C (coefficients for C blocks)
    block_terms = [
        # M1: P = A00+A11, Q = B00+B11, C: C00+=1, C11+=1
        {'P': {(0,0): 1, (1,1): 1}, 'Q': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
        # M2: P = A10+A11, Q = B00, C: C10+=1
        {'P': {(1,0): 1, (1,1): 1}, 'Q': {(0,0): 1}, 'C': {(1,0): 1}},
        # M3: P = A00, Q = B01-B11, C: C01+=1, C11+=-1
        {'P': {(0,0): 1}, 'Q': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): -1}},
        # M4: P = A11, Q = B10-B00, C: C00+=1, C10+=1
        {'P': {(1,1): 1}, 'Q': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
        # M5: P = A00+A01, Q = B11, C: C00+=-1, C01+=1
        {'P': {(0,0): 1, (0,1): 1}, 'Q': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
        # M6: P = A10-A00, Q = B00+B01, C: C11+=1
        {'P': {(1,0): 1, (0,0): -1}, 'Q': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
        # M7: P = A01-A11, Q = B10+B11, C: C00+=1, C11+=-1
        {'P': {(0,1): 1, (1,1): -1}, 'Q': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): -1}},
    ]
    
    for rb, term in enumerate(block_terms):
        P_coeffs = term['P']
        Q_coeffs = term['Q']
        C_coeffs = term['C']
        
        for rs in range(7):
            r = rb * 7 + rs
            
            # For U: P[a,b] = sum over (bi,bj) in P_coeffs: pc * A[2*bi+a, 2*bj+b]
            # U2[a*2+b][rs] is the coefficient for P[a,b] in the rs-th Strassen term
            for (bi, bj), pc in P_coeffs.items():
                for a in range(2):
                    for b in range(2):
                        u2_coeff = U2[a * 2 + b][rs]
                        if u2_coeff != 0:
                            ai = 2 * bi + a
                            aj = 2 * bj + b
                            U[ai * 4 + aj][r] += pc * u2_coeff
            
            # For V: Q[a,b] = sum over (bi,bj) in Q_coeffs: qc * B[2*bi+a, 2*bj+b]
            for (bi, bj), qc in Q_coeffs.items():
                for a in range(2):
                    for b in range(2):
                        v2_coeff = V2[a * 2 + b][rs]
                        if v2_coeff != 0:
                            bi_idx = 2 * bi + a
                            bj_idx = 2 * bj + b
                            V[bi_idx * 4 + bj_idx][r] += qc * v2_coeff
            
            # For W: R[a,b] = C[2*ci+a, 2*cj+b] for C block (ci,cj)
            # W2[a*2+b][rs] is the coefficient for R[a,b] in the rs-th Strassen term
            # W index: k*n+i = col*4+row
            for (ci, cj), cc in C_coeffs.items():
                for a in range(2):
                    for b in range(2):
                        w2_coeff = W2[a * 2 + b][rs]
                        if w2_coeff != 0:
                            crow = 2 * ci + a
                            ccol = 2 * cj + b
                            W[ccol * 4 + crow][r] += cc * w2_coeff
    
    return U, V, W

def _schoolbook_decomposition(n, m, p):
    """Construct the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s deadline to be safe within 120s
    
    n, m, p = DIMENSIONS
    
    # Try rank-49 Strassen decomposition
    U, V, W = _strassen_4x4_decomposition()
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
