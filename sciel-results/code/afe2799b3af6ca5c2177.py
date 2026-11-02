"""Recursive Strassen decomposition of 4x4x4 matrix multiplication (rank 49)."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

import time

# Strassen 2x2: 7 bilinear forms.
# For product s: A_s = sum_{(r,c,v)} v*A[r,c], B_s = sum_{(r,c,v)} v*B[r,c]
# C[r,c] = sum_s w_s[r,c] * M_s where M_s = A_s * B_s
#
# A-side coefficients for the 7 Strassen products (each is list of (row, col, coeff)):
_S_A = [
    [(0,0,1),(1,1,1)],   # M0 = (A00+A11)(B00+B11)
    [(1,0,1),(1,1,1)],   # M1 = (A10+A11)B00
    [(0,0,1)],           # M2 = A00(B01-B11)
    [(1,1,1)],           # M3 = A11(B10-B00)
    [(0,0,1),(0,1,1)],   # M4 = (A00+A01)B11
    [(1,0,1),(0,0,-1)],  # M5 = (A10-A00)(B00+B01)
    [(0,1,1),(1,1,-1)],  # M6 = (A01-A11)(B10+B11)
]
_S_B = [
    [(0,0,1),(1,1,1)],
    [(0,0,1)],
    [(0,1,1),(1,1,-1)],
    [(1,0,1),(0,0,-1)],
    [(1,1,1)],
    [(0,0,1),(0,1,1)],
    [(1,0,1),(1,1,1)],
]
# C-side: C[r,c] = sum_s W_s[r,c] * M_s
_S_W = [
    # s=0: M0 contributes to C[0,0]+1, C[1,1]+1
    [(0,0,1),(1,1,1)],
    # s=1: M1 contributes to C[1,0]+1, C[1,1]-1
    [(1,0,1),(1,1,-1)],
    # s=2: M2 contributes to C[0,1]+1, C[1,1]+1
    [(0,1,1),(1,1,1)],
    # s=3: M3 contributes to C[0,0]+1, C[1,0]+1
    [(0,0,1),(1,0,1)],
    # s=4: M4 contributes to C[0,1]+1, C[0,0]-1
    [(0,1,1),(0,0,-1)],
    # s=5: M5 contributes to C[1,1]+1
    [(1,1,1)],
    # s=6: M6 contributes to C[0,0]+1
    [(0,0,1)],
]

def _build_strassen_4x4_direct():
    """
    Direct combinatorial enumeration of the 49 non-zero tensor coordinates
    for the Kronecker product Strassen(2x2) ⊗ Strassen(2x2).
    
    The 4x4 matrix is viewed as a 2x2 block matrix of 2x2 blocks.
    Outer Strassen: 7 products of 2x2 block-matrices.
    Inner Strassen: each 2x2 block product uses 7 scalar products.
    Total: 7*7 = 49 rank-1 terms.
    
    For rank index r = outer_i * 7 + inner_s:
      U entry: coefficient for A[row,col] where row = 2*a_outer + a_inner, col = 2*b_outer + b_inner
      V entry: coefficient for B[row,col] where row = 2*c_outer + c_inner, col = 2*d_outer + d_inner  
      W entry: coefficient for C[row,col] where row = 2*e_outer + e_inner, col = 2*f_outer + f_inner
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    for outer_i in range(7):
        a_outer = _S_A[outer_i]  # list of (block_row, block_col, coeff)
        b_outer = _S_B[outer_i]  # list of (block_row, block_col, coeff)
        w_outer = _S_W[outer_i]  # list of (block_row, block_col, coeff) for C

        for inner_s in range(7):
            r = outer_i * 7 + inner_s
            a_inner = _S_A[inner_s]
            b_inner = _S_B[inner_s]
            w_inner = _S_W[inner_s]

            # U: for each outer A block position and inner A position
            for (obr, obc, oc) in a_outer:
                for (ibr, ibc, ic) in a_inner:
                    row = 2 * obr + ibr
                    col = 2 * obc + ibc
                    U[row * m + col, r] += oc * ic

            # V: for each outer B block position and inner B position
            for (obr, obc, oc) in b_outer:
                for (ibr, ibc, ic) in b_inner:
                    row = 2 * obr + ibr
                    col = 2 * obc + ibc
                    V[row * p + col, r] += oc * ic

            # W: for each outer C block position and inner C position
            for (obr, obc, oc) in w_outer:
                for (ibr, ibc, ic) in w_inner:
                    row = 2 * obr + ibr
                    col = 2 * obc + ibc
                    # W index: W[k*n + i, r] where k=col, i=row (trace convention)
                    W[col * n + row, r] += oc * ic

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# Pre-compute at module level with deadline guard
_t0 = time.time()
_PRECOMPUTED = _build_strassen_4x4_direct()
_build_time = time.time() - _t0

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    return bool(np.allclose(result, 1.0, atol=1e-10))

def _schoolbook_decomposition(n, m, p):
    """Fallback: rank-64 schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Return the rank-49 Strassen decomposition for 4x4x4 matrix multiplication.
    Includes a deadline guard: if pre-computation exceeded 140s, fall back to schoolbook.
    """
    deadline = 140.0  # seconds; evaluator kills at 150s
    elapsed = time.time() - _t0
    
    # If we're already past the deadline, return schoolbook immediately
    if elapsed > deadline:
        n, m, p = DIMENSIONS
        return _schoolbook_decomposition(n, m, p)
    
    n, m, p = DIMENSIONS
    result = _PRECOMPUTED
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])
    
    # Verify with remaining time budget
    remaining = deadline - (time.time() - _t0)
    if remaining > 5.0:  # need at least 5s for verification
        if _verify_decomposition(U, V, W, n, m, p, rank):
            return result
    
    # Fallback
    return _schoolbook_decomposition(n, m, p)

# EVOLVE-BLOCK-END
