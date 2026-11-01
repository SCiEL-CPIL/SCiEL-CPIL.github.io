import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _verify_factors(U, V, W, n, m, p):
    """Verify that the factors correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 1.0
    C_expected = A @ B
    
    C_reconstructed = np.zeros((n, p))
    for r in range(rank):
        for i in range(n):
            for k in range(p):
                val = 0.0
                for j in range(m):
                    val += U[i * m + j, r] * V[j * p + k, r]
                C_reconstructed[i, k] += W[k * n + i, r] * val
    
    if not np.allclose(C_expected, C_reconstructed):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Build rank-28 decomposition via 2x2 Strassen blocks
    rank = 28
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Strassen 2x2: P00 = M1 + M4 - M5 + M7
    #                  P01 = M3 + M5
    #                  P10 = M2 + M4
    #                  P11 = M1 - M2 + M3 + M6
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    
    strassen_defs = [
        # (M_idx, a_coeffs [(row,col),coeff], b_coeffs [(row,col),coeff], p_coeffs [(row,col),coeff])
        (0, [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        (1, [(1,0,1),(1,1,1)], [(0,0,1)],           [(1,0,1),(1,1,-1)]),
        (2, [(0,0,1)],           [(0,1,1),(1,1,-1)],  [(0,1,1),(1,1,1)]),
        (3, [(1,1,1)],           [(1,0,1),(0,0,-1)],  [(0,0,1),(1,0,1)]),
        (4, [(0,0,1),(0,1,1)],   [(1,1,1)],           [(0,0,-1),(0,1,1)]),
        (5, [(1,0,1),(0,0,-1)],  [(0,0,1),(0,1,1)],   [(1,1,1)]),
        (6, [(0,1,1),(1,1,-1)],  [(1,0,1),(1,1,1)],   [(0,0,1)]),
    ]
    
    # Four 2x2 block products needed:
    # C[:,0:2] = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
    # C[:,2:4] = A[:,0:2] @ B[0:2,2:4] + A[:,2:4] @ B[2:4,2:4]
    
    # Block (A_row_slice, A_col_slice, B_row_slice, B_col_slice, C_col_slice, offset)
    blocks = [
        (slice(0,2), slice(0,2), slice(0,2), slice(0,2), slice(0,2), 0),
        (slice(0,2), slice(2,4), slice(2,4), slice(0,2), slice(0,2), 7),
        (slice(0,2), slice(0,2), slice(0,2), slice(2,4), slice(2,4), 14),
        (slice(0,2), slice(2,4), slice(2,4), slice(2,4), slice(2,4), 21),
    ]
    
    for a_rows, a_cols, b_rows, b_cols, c_cols, offset in blocks:
        a_r0 = list(a_rows)[0]
        a_c0 = list(a_cols)[0]
        b_r0 = list(b_rows)[0]
        b_c0 = list(b_cols)[0]
        c_c0 = list(c_cols)[0]
        
        for m_idx, a_coeffs, b_coeffs, p_coeffs in strassen_defs:
            col = offset + m_idx
            
            for (ar, ac, coeff) in a_coeffs:
                gi = a_r0 + ar
                gj = a_c0 + ac
                U[gi * m + gj, col] += coeff
            
            for (br, bc, coeff) in b_coeffs:
                gi = b_r0 + br
                gj = b_c0 + bc
                V[gi * p + gj, col] += coeff
            
            for (pr, pc, coeff) in p_coeffs:
                gi = pr  # row of C
                gj = c_c0 + pc  # col of C
                W[gj * n + gi, col] += coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    if not _verify_factors(res["U"], res["V"], res["W"], n, m, p):
        print("Verification FAILED")
    else:
        print("OK rank", np.array(res["U"]).shape[1])
