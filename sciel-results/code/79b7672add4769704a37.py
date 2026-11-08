import numpy as np
import time

# EVOLVE-BLOCK-START
_n, _m, _p = 4, 4, 4
_rank = 49

def _build_strassen_decomposition():
    """
    Build the rank-49 decomposition for 4x4 matrix multiplication.
    
    Uses recursive Strassen: split 4x4 into 2x2 blocks of 2x2 sub-blocks.
    Each 2x2 block multiplication uses Strassen's 7 multiplications.
    Total: 7 (top-level) * 7 (sub-level) = 49 rank-1 terms.
    
    The tensor T_{i,j,k} for C_{i,k} = sum_j A_{i,j} * B_{j,k}
    is decomposed as sum_r U_{i,j}^{(r)} * V_{j,k}^{(r)} * W_{k,i}^{(r)}
    following the trace convention W[k*n+i, r].
    """
    U = np.zeros((_n * _m, _rank), dtype=np.float64)
    V = np.zeros((_m * _p, _rank), dtype=np.float64)
    W = np.zeros((_p * _n, _rank), dtype=np.float64)
    
    def u_idx(i, j):
        return i * _m + j
    
    def v_idx(j, k):
        return j * _p + k
    
    def w_idx(k, i):
        return k * _n + i
    
    # Strassen 2x2 decomposition
    # For matrices A, B (2x2), C = A*B:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 - M3 + M5 + M7
    # C12 = M4 + M5
    # C21 = M2 + M3
    # C22 = M1 - M2 + M4 + M6
    
    # Strassen factors for 2x2 matrices
    # U2[r][row, col] = coefficient of A[row,col] in the r-th product
    # V2[r][row, col] = coefficient of B[row,col] in the r-th product
    # W2[r][out_row, out_col] = coefficient contribution to C[out_row, out_col]
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    U2 = np.zeros((7, 2, 2), dtype=np.float64)
    V2 = np.zeros((7, 2, 2), dtype=np.float64)
    W2 = np.zeros((7, 2, 2), dtype=np.float64)
    
    # M1
    U2[0, 0, 0] = 1; U2[0, 1, 1] = 1
    V2[0, 0, 0] = 1; V2[0, 1, 1] = 1
    W2[0, 0, 0] = 1; W2[0, 1, 1] = 1
    
    # M2
    U2[1, 1, 0] = 1; U2[1, 1, 1] = 1
    V2[1, 0, 0] = 1
    W2[1, 1, 0] = 1; W2[1, 1, 1] = -1
    
    # M3
    U2[2, 0, 0] = 1
    V2[2, 0, 1] = 1; V2[2, 1, 1] = -1
    W2[2, 0, 1] = 1; W2[2, 1, 1] = 1
    
    # M4
    U2[3, 1, 1] = 1
    V2[3, 1, 0] = 1; V2[3, 0, 0] = -1
    W2[3, 0, 0] = 1; W2[3, 1, 0] = 1
    
    # M5
    U2[4, 0, 0] = 1; U2[4, 0, 1] = 1
    V2[4, 1, 1] = 1
    W2[4, 0, 0] = -1; W2[4, 0, 1] = 1
    
    # M6
    U2[5, 1, 0] = 1; U2[5, 0, 0] = -1
    V2[5, 0, 0] = 1; V2[5, 0, 1] = 1
    W2[5, 1, 1] = 1
    
    # M7
    U2[6, 0, 1] = 1; U2[6, 1, 1] = -1
    V2[6, 1, 0] = 1; V2[6, 1, 1] = 1
    W2[6, 0, 0] = 1
    
    # For 4x4, we have 2x2 blocks. Block (bi, bj) is at rows 2*bi:2*bi+2, cols 2*bj:2*bj+2
    # Top-level Strassen: 7 products of 2x2 blocks
    # Each 2x2 block product uses 7 sub-products of 2x2 elements
    
    # Top-level Strassen terms for 2x2 blocks:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = A00 * (B01 - B11)
    # P4 = A11 * (B10 - B00)
    # P5 = (A00 + A01) * B11
    # P6 = (A10 - A00) * (B00 + B01)
    # P7 = (A01 - A11) * (B10 + B11)
    
    # Where A00, A01, A10, A11 are 2x2 blocks
    # C00 = P1 - P3 + P5 + P7
    # C01 = P4 + P5
    # C10 = P2 + P3
    # C11 = P1 - P2 + P4 + P6
    
    # Top-level block coefficients
    # topU[t][bi, bj] = coeff of block (bi, bj) of A in term t
    # topV[t][bi, bj] = coeff of block (bi, bj) of B in term t
    # topW[t][ci, cj] = coeff of block (ci, cj) of C in term t
    
    topU = np.zeros((7, 2, 2), dtype=np.float64)
    topV = np.zeros((7, 2, 2), dtype=np.float64)
    topW = np.zeros((7, 2, 2), dtype=np.float64)
    
    # P1 = (A00 + A11)(B00 + B11)
    topU[0, 0, 0] = 1; topU[0, 1, 1] = 1
    topV[0, 0, 0] = 1; topV[0, 1, 1] = 1
    topW[0, 0, 0] = 1; topW[0, 1, 1] = 1
    
    # P2 = (A10 + A11) * B00
    topU[1, 1, 0] = 1; topU[1, 1, 1] = 1
    topV[1, 0, 0] = 1
    topW[1, 1, 0] = 1; topW[1, 1, 1] = -1
    
    # P3 = A00 * (B01 - B11)
    topU[2, 0, 0] = 1
    topV[2, 0, 1] = 1; topV[2, 1, 1] = -1
    topW[2, 0, 1] = 1; topW[2, 1, 1] = 1
    
    # P4 = A11 * (B10 - B00)
    topU[3, 1, 1] = 1
    topV[3, 1, 0] = 1; topV[3, 0, 0] = -1
    topW[3, 0, 0] = 1; topW[3, 1, 0] = 1
    
    # P5 = (A00 + A01) * B11
    topU[4, 0, 0] = 1; topU[4, 0, 1] = 1
    topV[4, 1, 1] = 1
    topW[4, 0, 0] = -1; topW[4, 0, 1] = 1
    
    # P6 = (A10 - A00) * (B00 + B01)
    topU[5, 1, 0] = 1; topU[5, 0, 0] = -1
    topV[5, 0, 0] = 1; topV[5, 0, 1] = 1
    topW[5, 1, 1] = 1
    
    # P7 = (A01 - A11) * (B10 + B11)
    topU[6, 0, 1] = 1; topU[6, 1, 1] = -1
    topV[6, 1, 0] = 1; topV[6, 1, 1] = 1
    topW[6, 0, 0] = 1
    
    # Now build the rank-49 decomposition
    # For each top-level term t (0-6) and sub-term s (0-6):
    # rank_idx = t * 7 + s
    # U contribution: sum over blocks (bi, bj) where topU[t, bi, bj] != 0
    #   and sub-elements (ri, cj) where U2[s, ri, cj] != 0
    #   U[i, j, rank_idx] += topU[t, bi, bj] * U2[s, ri, cj]
    #   where i = 2*bi + ri, j = 2*bj + cj
    # Similarly for V and W
    
    for t in range(7):
        for s in range(7):
            rank_idx = t * 7 + s
            
            # U part
            for bi in range(2):
                for bj in range(2):
                    coeff_top = topU[t, bi, bj]
                    if coeff_top == 0:
                        continue
                    for ri in range(2):
                        for cj in range(2):
                            coeff_sub = U2[s, ri, cj]
                            if coeff_sub == 0:
                                continue
                            i = 2 * bi + ri
                            j = 2 * bj + cj
                            U[u_idx(i, j), rank_idx] += coeff_top * coeff_sub
            
            # V part
            for bi in range(2):
                for bj in range(2):
                    coeff_top = topV[t, bi, bj]
                    if coeff_top == 0:
                        continue
                    for ri in range(2):
                        for cj in range(2):
                            coeff_sub = V2[s, ri, cj]
                            if coeff_sub == 0:
                                continue
                            j = 2 * bi + ri
                            k = 2 * bj + cj
                            V[v_idx(j, k), rank_idx] += coeff_top * coeff_sub
            
            # W part
            for ci in range(2):
                for cj in range(2):
                    coeff_top = topW[t, ci, cj]
                    if coeff_top == 0:
                        continue
                    for ri in range(2):
                        for cj2 in range(2):
                            coeff_sub = W2[s, ri, cj2]
                            if coeff_sub == 0:
                                continue
                            # W index: w_idx(k, i) where k = 2*cj + cj2, i = 2*ci + ri
                            # But wait, W2[s, ri, cj2] corresponds to C[ri, cj2] in the 2x2 result
                            # In the 4x4 context, C[2*ci+ri, 2*cj+cj2]
                            # W[k*n+i, r] where k is col index, i is row index
                            # So for C[row, col], W[col*n + row, r]
                            k = 2 * cj + cj2
                            i = 2 * ci + ri
                            W[w_idx(k, i), rank_idx] += coeff_top * coeff_sub
    
    return U, V, W

# Precompute at module level
_start_time = time.time()
_U, _V, _W = _build_strassen_decomposition()

def construct():
    """
    Returns rank-49 decomposition for 4x4 matrix multiplication.
    U: shape (16, 49), V: shape (16, 49), W: shape (16, 49)
    """
    return {"U": _U, "V": _V, "W": _W}
# EVOLVE-BLOCK-END
