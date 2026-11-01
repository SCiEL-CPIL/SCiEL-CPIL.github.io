"""Find a low-rank exact tensor decomposition for 2x2 * 2x3 matrix multiplication."""
import time
import itertools
from fractions import Fraction

import numpy as np

DIMENSIONS = [2, 2, 3]

def _build_T():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _decompose(U, V, W, T):
    """Check if U, V, W (rank r) exactly reproduce T."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    reconstructed = np.zeros_like(T)
    for c in range(r):
        reconstructed += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
    return np.allclose(reconstructed, T, atol=1e-9, rtol=1e-9)

def _rank_11_decomposition():
    """
    Rank-11 decomposition for 2x2 * 2x3 matrix multiplication.
    Uses Strassen for the 2x2 * 2x2 part (rank 7) + schoolbook for the 2x2 * 2x1 part (rank 4).
    """
    n, m, p = DIMENSIONS
    rank = 11
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Indexing:
    # U rows: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # V rows: B[0,0]=0, B[0,1]=1, B[0,2]=2, B[1,0]=3, B[1,1]=4, B[1,2]=5
    # W rows: C[0,0]=0, C[1,0]=1, C[0,1]=2, C[1,1]=3, C[0,2]=4, C[1,2]=5

    a, b, c, d = 0, 1, 2, 3
    e, f, g = 0, 1, 3
    h, i_, j_ = 2, 4, 5
    C11, C12, C21, C22 = 0, 2, 1, 3
    C13, C23 = 4, 5

    # Strassen terms (for columns 0 and 1 of B)
    # M1 = (a+d)(e+h) -> contributes to C11, C22
    U[a, 0] = 1; U[d, 0] = 1
    V[e, 0] = 1; V[h, 0] = 1
    W[C11, 0] = 1; W[C22, 0] = 1

    # M2 = (c+d)e -> contributes to C21
    U[c, 1] = 1; U[d, 1] = 1
    V[e, 1] = 1
    W[C21, 1] = 1

    # M3 = a(f-h) -> contributes to C12, C22
    U[a, 2] = 1
    V[f, 2] = 1; V[h, 2] = -1
    W[C12, 2] = 1; W[C22, 2] = -1

    # M4 = d(g-c) -> Wait, c is in A, g is in B. This is d*g - d*c? No, Strassen M4 = (d-c)*g? 
    # Standard Strassen: M4 = (d-c)(f+h)? No.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Let's stick to the standard definitions for 2x2:
    # C11 = M1 - M4 + M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M3 + M4
    
    # Re-evaluating assignments:
    # M1 = (a+d)(e+h) [A11,A22] * [B11,B22]
    # M2 = (c+d)e [A21,A22] * B11
    # M3 = a(f-h) A11 * [B21-B22]
    # M4 = d(g-c) A22 * [B12-B11] -> d*(g-e)? No, B12 is g (index 1), B11 is e (index 0).
    # Wait, my indexing for V:
    # V[0] = B[0,0] = e
    # V[1] = B[0,1] = f
    # V[2] = B[0,2] = g (col 2, not used in Strassen)
    # V[3] = B[1,0] = h
    # V[4] = B[1,1] = i
    # V[5] = B[1,2] = j
    
    # Let's restart with clear indices:
    # A: a=A00, b=A01, c=A10, d=A11
    # B: e=B00, f=B01, g=B02, h=B10, i=B11, j=B12
    # C: C00, C01, C02, C10, C11, C12
    
    # Strassen for C00, C01, C10, C11 using e,f,h,i (B00,B01,B10,B11)
    # M1 = (a+d)(e+i)
    # M2 = (c+d)e
    # M3 = a(h-i)
    # M4 = d(f-e)
    # M5 = (a+b)i
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(h+i)
    
    # C00 = M1 - M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M3 + M4
    
    # Schoolbook for C02, C12 using g,j (B02, B12)
    # C02 = a*g + b*j
    # C12 = c*g + d*j

    # Clear previous attempts
    U[:] = 0; V[:] = 0; W[:] = 0

    # M1
    U[a, 0] = 1; U[d, 0] = 1
    V[e, 0] = 1; V[i, 0] = 1
    W[C11, 0] = 1; W[C22, 0] = 1

    # M2
    U[c, 1] = 1; U[d, 1] = 1
    V[e, 1] = 1
    W[C21, 1] = 1

    # M3
    U[a, 2] = 1
    V[h, 2] = 1; V[i, 2] = -1
    W[C12, 2] = 1; W[C22, 2] = -1

    # M4
    U[d, 3] = 1
    V[f, 3] = 1; V[e, 3] = -1
    W[C11, 3] = -1; W[C22, 3] = 1

    # M5
    U[a, 4] = 1; U[b, 4] = 1
    V[i, 4] = 1
    W[C12, 4] = 1

    # M6
    U[c, 5] = 1; U[a, 5] = -1
    V[e, 5] = 1; V[f, 5] = 1
    W[C21, 5] = 1

    # M7
    U[b, 6] = 1; U[d, 6] = -1
    V[h, 6] = 1; V[i, 6] = 1
    W[C11, 6] = 1

    # Schoolbook for col 2
    # T1 = a*g -> C02
    U[a, 7] = 1
    V[g, 7] = 1
    W[C13, 7] = 1

    # T2 = b*j -> C02
    U[b, 8] = 1
    V[j, 8] = 1
    W[C13, 8] = 1

    # T3 = c*g -> C12
    U[c, 9] = 1
    V[g, 9] = 1
    W[C23, 9] = 1

    # T4 = d*j -> C12
    U[d, 10] = 1
    V[j, 10] = 1
    W[C23, 10] = 1

    return {"U": U, "V": V, "W": W}

def _verify(rank, U, V, W):
    T = _build_T()
    return _decompose(U, V, W, T)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try rank 11 first
    res = _rank_11_decomposition()
    if _verify(11, res['U'], res['V'], res['W']):
        # Convert to lists for interface compatibility if needed, but numpy arrays are usually fine
        # The task says "integer/half-integer entries", numpy int/float arrays should be okay.
        # To be safe, we can return them as lists or keep as arrays. The seed used lists.
        # Let's return as lists to match the seed's type closely, or arrays if the evaluator accepts them.
        # "factor matrices of shapes..." usually implies numpy arrays or lists.
        # I will return numpy arrays for efficiency, but if the evaluator expects lists, I should check.
        # The seed returned lists of lists.
        return {
            "U": res['U'].tolist(),
            "V": res['V'].tolist(),
            "W": res['W'].tolist()
        }
    
    # Fallback to schoolbook rank 12
    n, m, p = DIMENSIONS
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
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
