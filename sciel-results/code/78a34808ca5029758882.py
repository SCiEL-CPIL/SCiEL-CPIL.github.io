import time
import numpy as np

_DEADLINE = 120.0


def _strassen_2x2_2x2(k0, k1):
    """
    Return the 7 (U_offsets, V_offsets, W_offsets) triples for a Strassen
    decomposition of a 2x2 * 2x2 sub-block.
    U offsets are (i, j) with j in {1,2}, i in {0,1}
    V offsets are (j, k) with j in {1,2}, k in {k0, k1}
    W offsets are (k, i) with k in {k0, k1}, i in {0,1}
    """
    return [
        # M1 = (A01+A12) * B1k0
        ([ (0,1), (1,2) ], [ (1,k0) ], [ (k0,0), (k1,1) ], 1),
        # M2 = A11 * (B1k0 + B1k1) ... actually let me be more careful
        # Standard Strassen for C = A*B where A,B are 2x2:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
    ]


def _build_strassen_block(U, V, W, col, i_off, j_off, k_off):
    """
    Build 7 Strassen terms for the sub-block A[i_off:i_off+2, j_off:j_off+2] * B[j_off:j_off+2, k_off:k_off+2].
    i_off: offset for rows of A (and C)
    j_off: offset for cols of A / rows of B
    k_off: offset for cols of B (and C)
    """
    a00 = (i_off + 0, j_off + 0)
    a01 = (i_off + 0, j_off + 1)
    a10 = (i_off + 1, j_off + 0)
    a11 = (i_off + 1, j_off + 1)
    
    b00 = (j_off + 0, k_off + 0)
    b01 = (j_off + 0, k_off + 1)
    b10 = (j_off + 1, k_off + 0)
    b11 = (j_off + 1, k_off + 1)
    
    c00 = (k_off + 0, i_off + 0)
    c01 = (k_off + 0, i_off + 1)
    c10 = (k_off + 1, i_off + 0)
    c11 = (k_off + 1, i_off + 1)
    
    # M1 = (A00+A11)*(B00+B11)
    U[i_off * 3 + 0][col] += 1
    U[i_off * 3 + 1][col] += 1
    V[j_off * 8 + 0][col] += 1
    V[j_off * 8 + 1][col] += 1
    W[c00[0] * 2 + c00[1]][col] += 1
    W[c11[0] * 2 + c11[1]][col] += 1
    col += 1
    
    # M2 = (A10+A11)*B00
    U[i_off * 3 + 1][col] += 1
    U[i_off * 3 + 2][col] += 1
    V[j_off * 8 + 0][col] += 1
    W[c10[0] * 2 + c10[1]][col] += 1
    col += 1
    
    # M3 = A00*(B01-B11)
    U[i_off * 3 + 0][col] += 1
    V[j_off * 8 + 1][col] += 1
    V[j_off * 8 + 2][col] -= 1
    W[c01[0] * 2 + c01[1]][col] += 1
    col += 1
    
    # M4 = A11*(B10-B00)
    U[i_off * 3 + 1][col] += 1
    V[j_off * 8 + 2][col] += 1
    V[j_off * 8 + 0][col] -= 1
    W[c00[0] * 2 + c00[1]][col] += 1
    W[c10[0] * 2 + c10[1]][col] += 1
    col += 1
    
    # M5 = (A00+A01)*B11
    U[i_off * 3 + 0][col] += 1
    U[i_off * 3 + 1][col] += 1
    V[j_off * 8 + 3][col] += 1
    W[c00[0] * 2 + c00[1]][col] -= 1
    W[c01[0] * 2 + c01[1]][col] += 1
    col += 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[i_off * 3 + 1][col] += 1
    U[i_off * 3 + 0][col] -= 1
    V[j_off * 8 + 0][col] += 1
    V[j_off * 8 + 1][col] += 1
    W[c11[0] * 2 + c11[1]][col] += 1
    col += 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[i_off * 3 + 2][col] += 1
    U[i_off * 3 + 1][col] -= 1
    V[j_off * 8 + 2][col] += 1
    V[j_off * 8 + 3][col] += 1
    W[c00[0] * 2 + c00[1]][col] += 1
    W[c10[0] * 2 + c10[1]][col] += 1
    col += 1
    
    # Wait, I need to double-check the Strassen formulas. Let me use a standard reference.
    # For 2x2 matrices A and B:
    # M1 = (A11 + A22)(B11 + B22)  [using 1-based indexing]
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B11)
    # M4 = A22(B22 - B12)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # With 0-based indexing:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B00)
    # M4 = A11(B11 - B01)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    return col


def construct():
    t_start = time.time()
    
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    # Part 1: j=0, scalar terms: A[i,0] * B[0,k] -> C[i,k]
    # This is a 2x1 * 1x8 = 2x8 block, which needs 2*8 = 16 scalar terms
    for k in range(p):
        for i in range(n):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # col is now 16
    
    # Part 2: For j=1 and j=2, we have 2x2 * 2x8 sub-blocks
    # Split each 2x2*2x8 into four 2x2*2x2 blocks
    for j_off in [1, 2]:
        for block_k in range(4):
            k_off = block_k * 2
            col = _build_strassen_block(U, V, W, col, 0, j_off, k_off)
    
    # Wait, that gives 16 + 2 * 4 * 7 = 16 + 56 = 72, which is too many.
    # I need to be smarter. Let me reconsider.
    
    # Actually, for a 2x2 * 2x8 block, I can use 7 terms per 2x2*2x2 sub-block.
    # 2x2*2x8 = 2x2*2x2 + 2x2*2x2 + 2x2*2x2 + 2x2*2x2
    # Each 2x2*2x2 needs 7 terms, so 4 * 7 = 28 terms for one 2x2*2x8.
    # For two such blocks (j=1 and j=2), that's 56 terms.
    # Plus 16 for j=0, total = 72. That's worse than schoolbook 48.
    
    # I need a better approach. Let me go back to the original 44-rank construction.
    
    # Let me reset and use the working construction from the seed.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j=0, scalar terms
    for k in range(p):
        for i in range(n):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # col is now 16
    
    # Part 2: 4 blocks of 2x2x2 using 7-term Strassen decomposition
    # Each block covers A[0:2, j] * B[j, k0:k0+2] for j in {1,2} and 4 k-blocks
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # We're computing C[0:2, k0:k1] from A[0:2, 1:3] * B[1:3, k0:k1]
        # This is a 2x2 * 2x2 problem (rows 0,1 of A; cols 1,2 of A; rows 1,2 of B; cols k0,k1 of B)
        
        # M1 = (A01 + A12) * (B1k0 + B2k1)
        U[0 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M2 = (A11 + A12) * B1k0
        U[1 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        # M3 = A01 * (B1k1 - B2k1)
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        V[2 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M4 = A12 * (B2k0 - B1k0)
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        V[1 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # M5 = (A01 + A02) * B2k1
        U[0 * m + 1][col] = 1
        U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # M6 = (A11 - A01) * (B1k0 + B1k1)
        U[1 * m + 1][col] = 1
        U[0 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M7 = (A02 - A12) * (B2k0 + B2k1)
        U[0 * m + 2][col] = 1
        U[1 * m + 2][col] = -1
        V[2 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W):
    n, m, p = 2, 3, 8
    R = len(U[0]) if U else 0
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                expected = np.zeros((n, p))
                expected[i0, k0] = 1.0
                actual = np.zeros((n, p))
                for r in range(R):
                    u_col = U_np[:, r]
                    v_col = V_np[:, r]
                    w_col = W_np[:, r]
                    u_vals = [u_col[i * m + j0] for i in range(n)]
                    v_vals = [v_col[j0 * p + k] for k in range(p)]
                    for i in range(n):
                        for k in range(p):
                            actual[i, k] += u_vals[i] * v_vals[k] * w_col[k * n + i]
                if not np.allclose(actual, expected, atol=1e-9):
                    return False
    return True


if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    print("Rank:", len(res["U"][0]))
    if _verify(res["U"], res["V"], res["W"]):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
    print(f"Time: {time.time() - t0:.2f}s")
