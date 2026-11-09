"""Matrix multiplication 2x3x3: rank-17 exact tensor decomposition using Strassen core + schoolbook."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """Return an exact rank-17 tensor decomposition for 2x3x3 matrix multiplication.
    
    Uses Strassen's 7-multiplication algorithm for the 2x2x2 core (A[0:2,0:2] * B[0:2,0:2])
    and 10 schoolbook terms for the remaining products.
    
    U: shape (6, 17) - A indices (i*m+j) where i in {0,1}, j in {0,1,2}
    V: shape (9, 17) - B indices (j*p+k) where j in {0,1,2}, k in {0,1,2}  
    W: shape (6, 17) - C indices (k*n+i) where k in {0,1,2}, i in {0,1}
    
    Trace convention: W[k*n + i, r]
    """
    n, m, p = 2, 3, 3
    rank = 17
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper: A index for (i, j) -> i*m + j
    def a_idx(i, j):
        return i * m + j
    
    # Helper: B index for (j, k) -> j*p + k
    def b_idx(j, k):
        return j * p + k
    
    # Helper: C index for (k, i) -> k*n + i (trace convention)
    def c_idx(k, i):
        return k * n + i
    
    col = 0
    
    # === Strassen's 7 terms for 2x2x2 core (i in {0,1}, j in {0,1}, k in {0,1}) ===
    # Strassen products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # (U_coeffs for A, V_coeffs for B, W_coeffs for C, description)
        # M1 = (A00+A11)*(B00+B11)
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((0,1), 1), ((1,0), 1), ((1,1), -1)]),
        # M2 = (A10+A11)*B00
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)],
         [((1,0), 1), ((1,1), -1)]),
        # M3 = A00*(B01-B11)
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)],
         [((0,1), 1), ((1,1), 1)]),
        # M4 = A11*(B10-B00)
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((1,0), 1)]),
        # M5 = (A00+A01)*B11
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)],
         [((0,0), -1), ((0,1), 1)]),
        # M6 = (A10-A00)*(B00+B01)
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)],
         [((1,1), 1)]),
        # M7 = (A01-A11)*(B10+B11)
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), -1)]),
    ]
    
    for (u_coeffs, v_coeffs, w_coeffs) in strassen_terms:
        for (idx, coeff) in u_coeffs:
            i, j = idx
            U[a_idx(i, j), col] += coeff
        for (idx, coeff) in v_coeffs:
            j, k = idx
            V[b_idx(j, k), col] += coeff
        for (idx, coeff) in w_coeffs:
            k, i = idx
            W[c_idx(k, i), col] += coeff
        col += 1
    
    # === Schoolbook terms for remaining 10 products ===
    # Terms with j=2: A[i,2] * B[2,k] for i in {0,1}, k in {0,1,2} -> 6 terms
    # Terms with k=2, j in {0,1}: A[i,j] * B[j,2] for i in {0,1}, j in {0,1} -> 4 terms
    
    # j=2 terms
    for i in range(2):
        for k in range(3):
            U[a_idx(i, 2), col] = 1
            V[b_idx(2, k), col] = 1
            W[c_idx(k, i), col] = 1
            col += 1
    
    # k=2, j in {0,1} terms
    for i in range(2):
        for j in range(2):
            U[a_idx(i, j), col] = 1
            V[b_idx(j, 2), col] = 1
            W[c_idx(2, i), col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Convert to lists of Python floats (exact integers/halves)
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
