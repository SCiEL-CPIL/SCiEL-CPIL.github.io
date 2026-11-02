"""Matrix multiplication 2x2 x 2x3 - hybrid Strassen + schoolbook decomposition."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=3
    # A is n x m (2x2), B is m x p (2x3), C is n x p (2x3)
    # Split B into B1 (2x2, cols 0,1) and B2 (2x1, col 2)
    # C1 = A*B1 uses Strassen rank 7, C2 = A*B2 uses schoolbook rank 4

    # U: (n*m, rank) = (4, 11)
    # V: (m*p, rank) = (6, 11)
    # W: (p*n, rank) = (6, 11)
    # Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]

    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen for A (2x2) * B1 (2x2) = C1 (2x2)
    # A = [[a00, a01], [a10, a11]]
    # B1 = [[b00, b01], [b10, b11]]
    # C1 = [[c00, c01], [c10, c11]]
    #
    # Strassen products:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    # c00 = M1+M4-M5+M7
    # c01 = M3+M5
    # c10 = M2+M4
    # c11 = M1-M2+M3+M6
    #
    # For each Strassen term t, the contribution to C[r,c1] is U[i*m+j]*V[j*p+k]*W[k*n+i]
    # where c1 is the column in C (0 or 1 for B1 part)

    strassen_terms = [
        # (M_index, U_coeffs [(i,j,coeff)], V_coeffs [(j,k,coeff)], W_coeffs [(k,i,coeff)])
        # M1 = (a00+a11)*(b00+b11)
        (0, [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]),
        # M2 = (a10+a11)*b00
        (1, [(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1)]),
        # M3 = a00*(b01-b11)
        (2, [(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1)]),
        # M4 = a11*(b10-b00)
        (3, [(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,-1),(1,0,1)]),
        # M5 = (a00+a01)*b11
        (4, [(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1)]),
        # M6 = (a10-a00)*(b00+b01)
        (5, [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7 = (a01-a11)*(b10+b11)
        (6, [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    # c00 = M1+M4-M5+M7 : W coeff for (k=0,i=0): M1:+1, M4:-1, M5:-(-1)=+1, M7:+1
    # c01 = M3+M5 : W coeff for (k=1,i=0): M3:+1, M5:+1
    # c10 = M2+M4 : W coeff for (k=0,i=1): M2:+1, M4:+1
    # c11 = M1-M2+M3+M6 : W coeff for (k=1,i=1): M1:+1, M2:-1, M3:+1, M6:+1

    # W contributions per (k, i) for B1 part (k in 0,1; i in 0,1)
    # W[k*n+i, col] where col is 0..6
    w_b1 = {}  # (k,i) -> [coeffs for cols 0..6]
    for (k, i) in [(0,0), (0,1), (1,0), (1,1)]:
        w_b1[(k, i)] = [0] * 7

    # c00 = M1 + M4 - M5 + M7
    w_b1[(0,0)][0] += 1   # M1
    w_b1[(0,0)][3] += 1   # M4
    w_b1[(0,0)][4] += -1  # M5 (with -1)
    w_b1[(0,0)][6] += 1   # M7
    # c01 = M3 + M5
    w_b1[(1,0)][2] += 1   # M3
    w_b1[(1,0)][4] += 1   # M5
    # c10 = M2 + M4
    w_b1[(0,1)][1] += 1   # M2
    w_b1[(0,1)][3] += 1   # M4
    # c11 = M1 - M2 + M3 + M6
    w_b1[(1,1)][0] += 1   # M1
    w_b1[(1,1)][1] += -1  # M2
    w_b1[(1,1)][2] += 1   # M3
    w_b1[(1,1)][5] += 1   # M6

    # Now fill U, V, W for Strassen part (cols 0..6)
    for t, (col, u_coeffs, v_coeffs, w_coeffs_unused) in enumerate(strassen_terms):
        for (i, j, coeff) in u_coeffs:
            U[i * m + j][col] += coeff
        for (j, k, coeff) in v_coeffs:
            V[j * p + k][col] += coeff
        # W for B1 part: k in {0,1}, i in {0,1}
        for (k, i) in [(0,0), (0,1), (1,0), (1,1)]:
            W[k * n + i][col] += w_b1[(k, i)][t]

    # Schoolbook for A (2x2) * B2 (2x1) = C2 (2x1), column 2 of C
    # C2[i] = sum_j A[i,j] * B2[j]
    # B2[j] = B[j, 2]
    # C2[0] = a00*b02 + a01*b12
    # C2[1] = a10*b02 + a11*b12
    # 4 terms: (i,j) pairs
    schoolbook_terms = [
        # (i, j)
        (0, 0),  # a00 * b02
        (0, 1),  # a01 * b12
        (1, 0),  # a10 * b02
        (1, 1),  # a11 * b12
    ]

    for s, (i, j) in enumerate(schoolbook_terms):
        col = 7 + s
        U[i * m + j][col] = 1
        V[j * p + 2][col] = 1
        W[2 * n + i][col] = 1  # k=2, so W[2*n+i]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

# Verification (not called by evaluator, just for debugging)
def _verify():
    import numpy as np
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Tensor T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # T[i,j,k] = 1 for all i,j,k (matrix mult tensor)
    T = np.zeros((n, m, p))
    for c in range(rank):
        T += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
    # T[i,j,k] should be 1
    expected = np.ones((n, m, p))
    diff = np.abs(T - expected)
    return np.max(diff)

if __name__ == "__main__":
    err = _verify()
    print("Max error:", err)
