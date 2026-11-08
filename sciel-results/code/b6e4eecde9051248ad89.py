import numpy as np

def construct():
    n, m, p = 3, 3, 4
    
    # Build the target tensor T[i,j,k] for verification
    # T[i,j,k] = 1 if i = i_row*3+i_col, j = i_col*4+k_row, k = k_row*3+i_row
    T_expected = np.zeros((9, 12, 12), dtype=int)
    for i_row in range(3):
        for i_col in range(3):
            for k_row in range(4):
                i = i_row * 3 + i_col
                j = i_col * 4 + k_row
                k = k_row * 3 + i_row
                T_expected[i, j, k] = 1

    def verify(U, V, W):
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
        return np.array_equal(T_recon, T_expected)

    # Known rank-34 Strassen-based construction
    def build_rank34():
        rank = 34
        U = np.zeros((9, rank), dtype=int)
        V = np.zeros((12, rank), dtype=int)
        W = np.zeros((12, rank), dtype=int)
        col = 0

        # Strassen Left (A11 * B1L) - 7 terms
        # P1
        U[0, col] = 1; U[4, col] = 1
        V[0, col] = 1; V[5, col] = 1
        W[0, col] = 1; W[4, col] = 1
        col += 1
        # P2
        U[3, col] = 1; U[4, col] = 1
        V[0, col] = 1
        W[1, col] = 1; W[4, col] = -1
        col += 1
        # P3
        U[0, col] = 1
        V[1, col] = 1; V[5, col] = -1
        W[3, col] = 1; W[4, col] = 1
        col += 1
        # P4
        U[4, col] = 1
        V[4, col] = 1; V[0, col] = -1
        W[0, col] = 1; W[1, col] = 1
        col += 1
        # P5
        U[0, col] = 1; U[1, col] = 1
        V[5, col] = 1
        W[0, col] = -1; W[3, col] = 1
        col += 1
        # P6
        U[3, col] = 1; U[0, col] = -1
        V[0, col] = 1; V[1, col] = 1
        W[4, col] = 1
        col += 1
        # P7
        U[1, col] = 1; U[4, col] = -1
        V[4, col] = 1; V[5, col] = 1
        W[0, col] = 1
        col += 1

        # A12 * B2L - 4 terms
        U[2, col] = 1; V[8, col] = 1; W[0, col] = 1; col += 1
        U[2, col] = 1; V[9, col] = 1; W[3, col] = 1; col += 1
        U[5, col] = 1; V[8, col] = 1; W[1, col] = 1; col += 1
        U[5, col] = 1; V[9, col] = 1; W[4, col] = 1; col += 1

        # A21 * B1L - 4 terms
        U[6, col] = 1; V[0, col] = 1; W[2, col] = 1; col += 1
        U[6, col] = 1; V[1, col] = 1; W[5, col] = 1; col += 1
        U[7, col] = 1; V[4, col] = 1; W[2, col] = 1; col += 1
        U[7, col] = 1; V[5, col] = 1; W[5, col] = 1; col += 1

        # A22 * B2L - 2 terms
        U[8, col] = 1; V[8, col] = 1; W[2, col] = 1; col += 1
        U[8, col] = 1; V[9, col] = 1; W[5, col] = 1; col += 1

        # Strassen Right (A11 * B1R) - 7 terms
        # P1
        U[0, col] = 1; U[4, col] = 1
        V[2, col] = 1; V[7, col] = 1
        W[6, col] = 1; W[10, col] = 1
        col += 1
        # P2
        U[3, col] = 1; U[4, col] = 1
        V[2, col] = 1
        W[7, col] = 1; W[10, col] = -1
        col += 1
        # P3
        U[0, col] = 1
        V[3, col] = 1; V[7, col] = -1
        W[9, col] = 1; W[10, col] = 1
        col += 1
        # P4
        U[4, col] = 1
        V[6, col] = 1; V[2, col] = -1
        W[6, col] = 1; W[7, col] = 1
        col += 1
        # P5
        U[0, col] = 1; U[1, col] = 1
        V[7, col] = 1
        W[6, col] = -1; W[9, col] = 1
        col += 1
        # P6
        U[3, col] = 1; U[0, col] = -1
        V[2, col] = 1; V[3, col] = 1
        W[10, col] = 1
        col += 1
        # P7
        U[1, col] = 1; U[4, col] = -1
        V[6, col] = 1; V[7, col] = 1
        W[6, col] = 1
        col += 1

        # A12 * B2R - 4 terms
        U[2, col] = 1; V[10, col] = 1; W[6, col] = 1; col += 1
        U[2, col] = 1; V[11, col] = 1; W[9, col] = 1; col += 1
        U[5, col] = 1; V[10, col] = 1; W[7, col] = 1; col += 1
        U[5, col] = 1; V[11, col] = 1; W[10, col] = 1; col += 1

        # A21 * B1R - 4 terms
        U[6, col] = 1; V[2, col] = 1; W[8, col] = 1; col += 1
        U[6, col] = 1; V[3, col] = 1; W[11, col] = 1; col += 1
        U[7, col] = 1; V[6, col] = 1; W[8, col] = 1; col += 1
        U[7, col] = 1; V[7, col] = 1; W[11, col] = 1; col += 1

        # A22 * B2R - 2 terms
        U[8, col] = 1; V[10, col] = 1; W[8, col] = 1; col += 1
        U[8, col] = 1; V[11, col] = 1; W[11, col] = 1; col += 1

        return U, V, W, rank

    U, V, W, rank = build_rank34()

    if not verify(U, V, W):
        # Fallback to trivial rank-36
        rank = 36
        U = np.zeros((9, rank), dtype=int)
        V = np.zeros((12, rank), dtype=int)
        W = np.zeros((12, rank), dtype=int)
        col = 0
        for i in range(9):
            for j in range(12):
                for k in range(12):
                    if T_expected[i, j, k] != 0:
                        U[i, col] = 1
                        V[j, col] = 1
                        W[k, col] = 1
                        col += 1
    else:
        # Ensure entries are integer or half-integer (they are integers)
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
