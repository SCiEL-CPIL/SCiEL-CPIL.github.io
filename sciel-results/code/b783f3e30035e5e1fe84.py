import time
import numpy as np
from fractions import Fraction

_DEADLINE = 150.0 - 10.0
_start = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _start)

def _schoolbook(n, m, p):
    """Schoolbook decomposition: rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W, rank

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    r = U.shape[1]
    U_reshaped = U.reshape(n, m, r)
    V_reshaped = V.reshape(m, p, r)
    W_reshaped = W.reshape(p, n, r)
    T_rec = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    T_expected = np.ones((n, m, p))
    return np.allclose(T_rec, T_expected, atol=1e-9)

def _decomp_2x2x3():
    """
    Decomposition for 2x2 * 2x3 -> 2x3.
    n=2, m=2, p=3.
    Rank 11 (Strassen 7 for 2x2x2 + 4 for 2x2x1).
    """
    n, m, p = 2, 2, 3
    # We split the output k into k=0,1 (2x2x2) and k=2 (2x2x1).
    # Or split the input j? No, j is 2.
    # The product is C[i,k] = sum_j A[i,j] B[j,k].
    # k=0,1: C[i,0:2] = A * B[:, 0:2]. This is 2x2 * 2x2.
    # k=2: C[i,2] = A * B[:, 2]. This is 2x2 * 2x1.
    
    # Part 1: 2x2 * 2x2 (Rank 7)
    U1, V1, W1, r1 = _strassen_2x2()
    
    # Part 2: 2x2 * 2x1 (Rank 4)
    # C[i,2] = sum_j A[i,j] B[j,2]
    # Terms: A00 B02, A01 B12, A10 B02, A11 B12
    U2 = np.zeros((4, 4))
    V2 = np.zeros((4, 4))
    W2 = np.zeros((3, 4)) # p*n = 3*2 = 6? No, W is (p*n, rank). p=2, n=2 -> 4.
    # Wait, for 2x2x1, n=2, m=2, p=1.
    # U: (2*2, 4) = (4,4)
    # V: (2*1, 4) = (2,4)
    # W: (1*2, 4) = (2,4)
    # But we are embedding this into 2x2x3.
    # The k index for this part is 2.
    # So W should have non-zeros only at k=2.
    # W[k*n + i, r]. k=2, n=2. Index = 2*2 + i = 4+i.
    # So W2 should be (6, 4) to match the full W shape of 2x2x3?
    # Or we can just build the full U,V,W for 2x2x3 directly.
    
    U = np.zeros((4, 11))
    V = np.zeros((4, 11))
    W = np.zeros((6, 11)) # p*n = 3*2 = 6
    
    # Copy Strassen part (k=0,1)
    # Strassen U,V are (4,7). W is (4,7) for 2x2x2.
    # In 2x2x3, W is (6,11).
    # The k=0,1 part of W corresponds to indices 0..3.
    # W_strassen[k*n+i, r] where k in 0,1.
    # So we copy W1 to W[0:4, 0:7].
    U[:, 0:7] = U1
    V[:, 0:7] = V1
    W[0:4, 0:7] = W1
    
    # Add 2x2x1 part (k=2)
    # C[i,2] = A00 B02 + A01 B12 + A10 B02 + A11 B12
    # r=7: A00 B02
    U[0, 7] = 1 # i=0,j=0
    V[0, 7] = 1 # j=0,k=2 -> index 0*2+2=2? No.
    # V index is j*p + k. p=3.
    # j=0, k=2 -> 0*3+2 = 2.
    # Wait, V is (m*p, rank) = (2*3, 11) = (6, 11).
    # My previous V1 was (4,7) for 2x2x2 (p=2).
    # Here p=3.
    # So V1 indices were j*2+k.
    # Now V indices are j*3+k.
    # This mismatch means we can't just copy V1 directly if the dimension p changed.
    # We must rebuild V1 for p=2 part embedded in p=3.
    
    # Let's restart the construction of _decomp_2x2x3 properly.
    
    U = np.zeros((4, 11))
    V = np.zeros((6, 11)) # m*p = 2*3
    W = np.zeros((6, 11)) # p*n = 3*2
    
    # Part 1: 2x2 * 2x2 (k=0,1)
    # Strassen factors for 2x2x2.
    # U_s (4,7), V_s (4,7), W_s (4,7).
    # V_s index j*2+k. We need V index j*3+k for k=0,1.
    # So V[j*3+k, r] = V_s[j*2+k, r].
    # W_s index k*2+i. We need W index k*3+i? No, W index is k*n+i = k*2+i.
    # Wait, W dimension is p*n. p=3, n=2. Index k*2+i.
    # Strassen W_s is for p=2. Index k*2+i.
    # So W_s can be copied directly to W[0:4, 0:7].
    # V_s needs remapping.
    
    U_s, V_s, W_s, _ = _strassen_2x2()
    
    U[:, 0:7] = U_s
    W[0:4, 0:7] = W_s
    
    # Remap V_s to V
    for j in range(2):
        for k in range(2):
            V[j*3 + k, 0:7] = V_s[j*2 + k, 0:7]
            
    # Part 2: 2x2 * 2x1 (k=2)
    # C[i,2] = sum_j A[i,j] B[j,2]
    # V index for B[j,2] is j*3 + 2.
    # W index for C[i,2] is 2*2 + i = 4 + i.
    
    # r=7: A00 B02
    U[0, 7] = 1
    V[0*3+2, 7] = 1
    W[4+0, 7] = 1
    
    # r=8: A01 B12
    U[0*2+1, 8] = 1
    V[1*3+2, 8] = 1
    W[4+0, 8] = 1
    
    # r=9: A10 B02
    U[1*2+0, 9] = 1
    V[0*3+2, 9] = 1
    W[4+1, 9] = 1
    
    # r=10: A11 B12
    U[1*2+1, 10] = 1
    V[1*3+2, 10] = 1
    W[4+1, 10] = 1
    
    return U, V, W, 11

def _decomp_2x3x1():
    """
    Decomposition for 2x3 * 3x1 -> 2x1.
    n=2, m=3, p=1.
    Rank 6.
    """
    n, m, p = 2, 3, 1
    U = np.zeros((6, 6)) # n*m
    V = np.zeros((3, 6)) # m*p
    W = np.zeros((2, 6)) # p*n
    
    # C[i,0] = sum_j A[i,j] B[j,0]
    for i in range(2):
        for j in range(3):
            r = i*3 + j
            U[i*m+j, r] = 1
            V[j*p+0, r] = 1
            W[0*n+i, r] = 1
            
    return U, V, W, 6

def _decomp_2x3x2():
    """
    Decomposition for 2x3 * 3x2 -> 2x2.
    n=2, m=3, p=2.
    Isomorphic to 2x2x3.
    Rank 11.
    """
    # We can construct it by permuting indices of _decomp_2x2x3?
    # T_2,2,3: i in 0,1; j in 0,1; k in 0,1,2.
    # T_2,3,2: i in 0,1; j in 0,1,2; k in 0,1.
    # The roles of j and k are swapped in the "3" dimension.
    # But the tensor structure is symmetric in some ways?
    # Let's just build it using the same logic:
    # Split k=0,1 (2x3x1) and k=? No, k is 2.
    # Split j=0,1 (2x2x2) and j=2 (2x1x2).
    
    n, m, p = 2, 3, 2
    U = np.zeros((6, 11))
    V = np.zeros((6, 11))
    W = np.zeros((4, 11))
    
    # Part 1: 2x2 * 2x2 (j=0,1)
    # C[i,k] = sum_{j=0,1} A[i,j] B[j,k]
    # This is 2x2x2.
    U_s, V_s, W_s, _ = _strassen_2x2()
    
    # U_s index i*2+j. We need i*3+j.
    # V_s index j*2+k. We need j*2+k. (m=2 in this subproblem, p=2).
    # W_s index k*2+i. We need k*2+i.
    
    for i in range(2):
        for j in range(2):
            U[i*3+j, 0:7] = U_s[i*2+j, 0:7]
            
    V[0:4, 0:7] = V_s # j*2+k, j in 0,1. Indices 0..3.
    # V is (6,11). j=0,1 use indices 0..3. j=2 uses 4..5.
    
    W[0:4, 0:7] = W_s
    
    # Part 2: 2x1 * 1x2 (j=2)
    # C[i,k] = A[i,2] B[2,k]
    # Rank 4.
    # r=7: A02 B20
    U[0*3+2, 7] = 1
    V[2*2+0, 7] = 1
    W[0*2+0, 7] = 1
    
    # r=8: A02 B21
    U[0*3+2, 8] = 1
    V[2*2+1, 8] = 1
    W[1*2+0, 8] = 1
    
    # r=9: A12 B20
    U[1*3+2, 9] = 1
    V[2*2+0, 9] = 1
    W[0*2+1, 9] = 1
    
    # r=10: A12 B21
    U[1*3+2, 10] = 1
    V[2*2+1, 10] = 1
    W[1*2+1, 10] = 1
    
    return U, V, W, 11

def _decomp_2x3x3():
    """
    Decomposition for 2x3 * 3x3 -> 2x3.
    n=2, m=3, p=3.
    Split k=0,1 (2x3x2) and k=2 (2x3x1).
    Rank 11 + 6 = 17.
    """
    n, m, p = 2, 3, 3
    U1, V1, W1, r1 = _decomp_2x3x2()
    U2, V2, W2, r2 = _decomp_2x3x1()
    
    U = np.zeros((6, r1+r2))
    V = np.zeros((9, r1+r2))
    W = np.zeros((6, r1+r2))
    
    U[:, 0:r1] = U1
    V[:, 0:r1] = V1
    # W1 is for p=2. W is for p=3.
    # W1 index k*2+i. k in 0,1.
    # W index k*2+i. k in 0,1,2.
    # So W1 maps to W[0:4, 0:r1].
    W[0:4, 0:r1] = W1
    
    U[:, r1:r1+r2] = U2
    V[:, r1:r1+r2] = V2
    # W2 is for p=1. W2 index 0*2+i = i.
    # W index k*2+i. k=2. Index 4+i.
    W[4:6, r1:r1+r2] = W2
    
    return U, V, W, r1+r2

def _decomp_2x5x3():
    """
    Decomposition for 2x5 * 5x3 -> 2x3.
    n=2, m=5, p=3.
    Split j=0,1 (2x2x3) and j=2,3,4 (2x3x3).
    Rank 11 + 17 = 28.
    """
    n, m, p = 2, 5, 3
    U1, V1, W1, r1 = _decomp_2x2x3()
    U2, V2, W2, r2 = _decomp_2x3x3()
    
    U = np.zeros((10, r1+r2))
    V = np.zeros((15, r1+r2))
    W = np.zeros((6, r1+r2))
    
    U[:, 0:r1] = U1
    # V1 is for m=2. V is for m=5.
    # V1 index j*3+k. j in 0,1.
    # V index j*3+k. j in 0,1.
    V[0:6, 0:r1] = V1
    
    W[0:6, 0:r1] = W1
    
    U[:, r1:r1+r2] = U2
    # V2 is for m=3. V is for m=5.
    # V2 index j*3+k. j in 0,1,2.
    # We need j=2,3,4 in V.
    # So V2[j*3+k] maps to V[(j+2)*3+k].
    for j in range(3):
        for k in range(3):
            V[(j+2)*3+k, r1:r1+r2] = V2[j*3+k, r1:r1+r2]
            
    W[0:6, r1:r1+r2] = W2
    
    return U, V, W, r1+r2

def _combine_decompositions(n, m, p, U1, V1, W1, r1, U2, V2, W2, r2):
    """
    Combine two decompositions for the same n,m,p.
    """
    U = np.concatenate([U1, U2], axis=1)
    V = np.concatenate([V1, V2], axis=1)
    W = np.concatenate([W1, W2], axis=1)
    return U, V, W, r1+r2

def construct():
    n, m, p = 4, 5, 6
    
    # Split A (4x5) into A1 (2x5) and A2 (2x5).
    # Split B (5x6) into B1 (5x3) and B2 (5x3).
    # C = A1 B1 + A1 B2 + A2 B1 + A2 B2.
    # Each term is 2x5 * 5x3 -> 2x3.
    
    U_blk, V_blk, W_blk, r_blk = _decomp_2x5x3()
    
    # We need to place these 4 blocks into the full 4x5x6 tensor.
    # U is (4*5, R) = (20, R).
    # V is (5*6, R) = (30, R).
    # W is (6*4, R) = (24, R).
    
    R = 4 * r_blk
    U = np.zeros((20, R))
    V = np.zeros((30, R))
    W = np.zeros((24, R))
    
    # Block 1: A1 B1 (i=0,1; j=0..4; k=0,1,2)
    # U indices: i*5+j. i in 0,1.
    # V indices: j*6+k. j in 0..4, k in 0,1,2.
    # W indices: k*4+i. k in 0,1,2, i in 0,1.
    
    # U_blk is (10, r_blk).
    # V_blk is (15, r_blk).
    # W_blk is (6, r_blk).
    
    # Place Block 1
    U[0:10, 0:r_blk] = U_blk
    V[0:15, 0:r_blk] = V_blk
    W[0:6, 0:r_blk] = W_blk
    
    # Place Block 2: A1 B2 (i=0,1; j=0..4; k=3,4,5)
    # U is same.
    # V: k shifted by 3.
    # W: k shifted by 3.
    
    U[0:10, r_blk:2*r_blk] = U_blk
    # V_blk index j*3+k. We need j*6+(k+3).
    for j in range(5):
        for k in range(3):
            V[j*6 + (k+3), r_blk:2*r_blk] = V_blk[j*3+k, 0:r_blk]
            
    # W_blk index k*2+i. We need (k+3)*4+i.
    for k in range(3):
        for i in range(2):
            W[(k+3)*4 + i, r_blk:2*r_blk] = W_blk[k*2+i, 0:r_blk]
            
    # Place Block 3: A2 B1 (i=2,3; j=0..4; k=0,1,2)
    # U: i shifted by 2.
    # V: same.
    # W: i shifted by 2.
    
    # U_blk index i*5+j. We need (i+2)*5+j.
    for i in range(2):
        for j in range(5):
            U[(i+2)*5 + j, 2*r_blk:3*r_blk] = U_blk[i*5+j, 0:r_blk]
            
    V[0:15, 2*r_blk:3*r_blk] = V_blk
    # W_blk index k*2+i. We need k*4+(i+2).
    for k in range(3):
        for i in range(2):
            W[k*4 + (i+2), 2*r_blk:3*r_blk] = W_blk[k*2+i, 0:r_blk]
            
    # Place Block 4: A2 B2 (i=2,3; j=0..4; k=3,4,5)
    
    for i in range(2):
        for j in range(5):
            U[(i+2)*5 + j, 3*r_blk:4*r_blk] = U_blk[i*5+j, 0:r_blk]
            
    for j in range(5):
        for k in range(3):
            V[j*6 + (k+3), 3*r_blk:4*r_blk] = V_blk[j*3+k, 0:r_blk]
            
    for k in range(3):
        for i in range(2):
            W[(k+3)*4 + (i+2), 3*r_blk:4*r_blk] = W_blk[k*2+i, 0:r_blk]
            
    if not _verify(U, V, W, n, m, p):
        # Fallback
        U, V, W, rank = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
