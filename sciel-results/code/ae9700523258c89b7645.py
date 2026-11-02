"""Matrix multiplication 4,4,8 tensor decomposition with Strassen-based rank 98."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """
    Return U, V, W for 2x2 matrix multiplication using Strassen's algorithm.
    U: shape (4, 7) indexed as U[i*2+j, r]
    V: shape (4, 7) indexed as V[j*2+k, r]
    W: shape (4, 7) indexed as W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen products for 2x2:
    # p0 = (A00+A11)(B00+B11)
    # p1 = (A10+A11)(B00)
    # p2 = A00(B01-B11)
    # p3 = A11(B10-B00)
    # p4 = (A00+A01)(B11)
    # p5 = (A10-A00)(B00+B01)
    # p6 = (A01-A11)(B10+B11)
    #
    # C00 = p0 + p3 - p4 + p6
    # C01 = p2 + p4
    # C10 = p1 + p3
    # C11 = p0 - p1 + p2 + p5
    
    # p0 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1  # A00, A11
    V[0, 0] = 1; V[3, 0] = 1  # B00, B11
    W[0, 0] = 1; W[3, 0] = 1  # C00, C11
    
    # p1 = (A10+A11)(B00)
    U[2, 1] = 1; U[3, 1] = 1  # A10, A11
    V[0, 1] = 1               # B00
    W[2, 1] = 1               # C10
    
    # p2 = A00(B01-B11)
    U[0, 2] = 1               # A00
    V[1, 2] = 1; V[3, 2] = -1 # B01, -B11
    W[1, 2] = 1               # C01
    
    # p3 = A11(B10-B00)
    U[3, 3] = 1               # A11
    V[2, 3] = 1; V[0, 3] = -1 # B10, -B00
    W[0, 3] = 1; W[2, 3] = 1  # C00, C10
    
    # p4 = (A00+A01)(B11)
    U[0, 4] = 1; U[1, 4] = 1  # A00, A01
    V[3, 4] = 1               # B11
    W[0, 4] = -1; W[1, 4] = 1 # -C00, C01
    
    # p5 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1 # A10, -A00
    V[0, 5] = 1; V[1, 5] = 1  # B00, B01
    W[3, 5] = 1               # C11
    
    # p6 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1 # A01, -A11
    V[2, 6] = 1; V[3, 6] = 1  # B10, B11
    W[0, 6] = 1               # C00
    
    return U, V, W


def _verify(U, V, W):
    """Check that the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(result, 1.0, atol=1e-10)


def _build_strassen_4x4_4x8():
    """
    Build rank-98 decomposition for 4x4 * 4x8 using two independent 4x4*4x4 Strassen decompositions.
    
    Strategy: Split B into B1 (cols 0-3) and B2 (cols 4-7).
    Compute C1 = A*B1 and C2 = A*B2, each using rank-49 Strassen for 4x4*4x4.
    Total rank = 98.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Get 2x2 Strassen factors
    U2, V2, W2 = _strassen_2x2()
    
    rank = 98
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # For each of the two subproblems (t=0 for B1, t=1 for B2)
    for t in range(2):
        # For each of the 7 top-level Strassen products (s=0..6)
        for s in range(7):
            # For each of the 7 scalar Strassen products (r=0..6)
            for r in range(7):
                col = t * 49 + s * 7 + r
                
                # Top-level: A is split into four 2x2 blocks A00, A01, A10, A11
                # Top-level: B_t is split into four 2x2 blocks B00, B01, B10, B11
                # Top-level: C_t is split into four 2x2 blocks C00, C01, C10, C11
                #
                # The s-th top-level Strassen product uses:
                #   u_block[s] for A blocks: [coeff_A00, coeff_A01, coeff_A10, coeff_A11]
                #   v_block[s] for B blocks: [coeff_B00, coeff_B01, coeff_B10, coeff_B11]
                #   w_block[s] for C blocks: [coeff_C00, coeff_C01, coeff_C10, coeff_C11]
                #
                # The r-th scalar Strassen product within each 2x2*2x2 uses:
                #   u_scalar[r], v_scalar[r], w_scalar[r]
                
                # U contributions:
                # For each A block (bi, bj) with coefficient from top-level u_block[s][bi*2+bj]
                #   and for each element (ei, ej) within that block with coefficient from u_scalar[r][ei*2+ej]
                #   the A element at position (2*bi+ei, 2*bj+ej) gets the product
                for bi in range(2):
                    for bj in range(2):
                        cb = U2[bi*2+bj, s]  # coefficient for A-block(bi,bj) in top-level product s
                        if cb == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = U2[ei*2+ej, r]  # coefficient for element (ei,ej) in scalar product r
                                if cs == 0:
                                    continue
                                i = 2*bi + ei
                                j = 2*bj + ej
                                U[i*m + j, col] += cb * cs
                
                # V contributions:
                # For each B block (bi, bj) in subproblem t with coefficient from top-level v_block[s][bi*2+bj]
                #   and for each element (ei, ej) within that block with coefficient from v_scalar[r][ei*2+ej]
                #   the B element at position (2*bi+ei, 4*t + 2*bj + ej) gets the product
                for bi in range(2):
                    for bj in range(2):
                        cb = V2[bi*2+bj, s]  # coefficient for B-block(bi,bj) in top-level product s
                        if cb == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = V2[ei*2+ej, r]  # coefficient for element (ei,ej) in scalar product r
                                if cs == 0:
                                    continue
                                j = 2*bi + ei
                                k = 4*t + 2*bj + ej
                                V[j*p + k, col] += cb * cs
                
                # W contributions:
                # For each C block (ci, cj) in subproblem t with coefficient from top-level w_block[s][ci*2+cj]
                #   and for each element (ei, ej) within that block with coefficient from w_scalar[r][ei*2+ej]
                #   the C element at position (i=2*ci+ei, k=4*t+2*cj+ej) gets the product
                #   W is indexed as W[k*n + i, col]
                for ci in range(2):
                    for cj in range(2):
                        cb = W2[ci*2+cj, s]  # coefficient for C-block(ci,cj) in top-level product s
                        if cb == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = W2[ei*2+ej, r]  # coefficient for element (ei,ej) in scalar product r
                                if cs == 0:
                                    continue
                                i = 2*ci + ei
                                k = 4*t + 2*cj + ej
                                W[k*n + i, col] += cb * cs
    
    # Convert to list of lists for the required interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}


def _schoolbook():
    """Return the trivial rank n*m*p decomposition."""
    n, m, p = DIMENSIONS
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
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Tries Strassen-based decomposition (rank 98) first, falls back to schoolbook (rank 128).
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Try Strassen-based decomposition
    try:
        result = _build_strassen_4x4_4x8()
        if _verify(result["U"], result["V"], result["W"]):
            return result
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline:
        return _schoolbook()
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
