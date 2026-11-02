import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
    deadline = time.time() + 110
    n, m, p = 2, 3, 8

    def verify(U, V, W, n, m, p):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        T = np.zeros((n, m, p))
        for c in range(r):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-10)

    def build_2x2x4_rank13():
        """Rank-13 decomposition of M_{2,2,4}.
        A is 2x2 (U indices: i*2+j),
        B is 2x4 (V indices: j*4+k),
        C is 2x4 (W indices: k*2+i).
        """
        rank = 13
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(8)]
        W = [[0]*rank for _ in range(8)]

        # Left block (k=0,1): 7 multiplications
        # M1 = (a00+a11)(b00+b10) -> c00 + c11
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[4][0] = 1
        W[0][0] = 1; W[5][0] = 1

        # M2 = (a10+a11)b00 -> c10 - c11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[4][1] = 1; W[5][1] = -1

        # M3 = a00(b01+b11) -> c01 + c11
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = 1
        W[1][2] = 1; W[5][2] = 1

        # M4 = (a00-a10)(b01+b10) -> c00 + c10
        U[0][3] = 1; U[2][3] = -1
        V[1][3] = 1; V[4][3] = 1
        W[0][3] = 1; W[4][3] = 1

        # M5 = (a01+a10)(b00-b10) -> c00 + c01
        U[1][4] = 1; U[2][4] = 1
        V[0][4] = 1; V[4][4] = -1
        W[0][4] = -1; W[1][4] = 1

        # M6 = (a11-a01)(b01-b11) -> c11
        U[1][5] = -1; U[3][5] = 1
        V[1][5] = 1; V[5][5] = -1
        W[5][5] = 1

        # M7 = (a00+a01)(b11-b01) -> c00
        U[0][6] = 1; U[1][6] = 1
        V[1][6] = -1; V[5][6] = 1
        W[0][6] = 1

        # Right block (k=2,3): 6 multiplications (shared U factors)
        # M8 = (a00+a11)(b02+b12) -> c02 + c13
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[6][7] = 1
        W[4][7] = 1; W[7][7] = 1

        # M9 = (a10+a11)b02 -> c12 - c13
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[6][8] = 1; W[7][8] = -1

        # M10 = a00(b03+b13) -> c03 + c13
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = 1
        W[5][9] = 1; W[7][9] = 1

        # M11 = (a00-a10)(b03+b12) -> c02 + c12
        U[0][10] = 1; U[2][10] = -1
        V[3][10] = 1; V[6][10] = 1
        W[4][10] = 1; W[6][10] = 1

        # M12 = (a01+a10)(b02-b12) -> c02 + c03
        U[1][11] = 1; U[2][11] = 1
        V[2][11] = 1; V[6][11] = -1
        W[4][11] = -1; W[5][11] = 1

        # M13 = (a11-a01)(b03-b13) -> c13
        U[1][12] = -1; U[3][12] = 1
        V[3][12] = 1; V[7][12] = -1
        W[7][12] = 1

        return U, V, W

    def build_1plus2_split():
        """Split m=3 as 1+2: M_{2,1,8} rank 16 + 2*M_{2,2,4} rank 13 each = 42."""
        total_rank = 16 + 2 * 13
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]

        col = 0
        # Part 1: A[:,0] (2x1) * B[0,:] (1x8) -> 16 schoolbook
        for i in range(n):
            for k in range(p):
                U[i*m + 0][col] = 1
                V[0*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1

        # Part 2: A[:,1:3] (2x2) * B[1:3,:] (2x8)
        # Split into two M_{2,2,4}: cols 0-3 and cols 4-7
        U224, V224, W224 = build_2x2x4_rank13()

        for half in range(2):
            k_offset = half * 4
            for r in range(13):
                # Map U224 indices (i*2+j) to global U indices (i*3 + (1+j))
                for i in range(2):
                    U[i*3 + 1][col+r] = U224[i*2 + 0][r]
                    U[i*3 + 2][col+r] = U224[i*2 + 1][r]
                # Map V224 indices (j*4+k) to global V indices ((1+j)*8 + k_offset+k)
                for j_local in range(2):
                    for k_local in range(4):
                        V[(1+j_local)*8 + k_offset + k_local][col+r] = V224[j_local*4 + k_local][r]
                # Map W224 indices (k*2+i) to global W indices ((k_offset+k)*2 + i)
                for k_local in range(4):
                    for i in range(2):
                        W[(k_offset + k_local)*2 + i][col+r] = W224[k_local*2 + i][r]
            col += 13

        return U, V, W

    def build_strassen_44():
        """Fallback: 1+2 split with Strassen: rank 44."""
        rank = 44
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0

        for i in range(n):
            for k in range(p):
                U[i*m + 0][col] = 1
                V[0*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1

        for block in range(4):
            b00 = 1*p + block*2
            b01 = 1*p + block*2 + 1
            b10 = 2*p + block*2
            b11 = 2*p + block*2 + 1
            c00 = (block*2)*n + 0
            c01 = (block*2+1)*n + 0
            c10 = (block*2)*n + 1
            c11 = (block*2+1)*n + 1

            U[1][col] = 1; U[5][col] = 1
            V[b00][col] = 1; V[b11][col] = 1
            W[c00][col] = 1; W[c11][col] = 1
            col += 1

            U[4][col] = 1; U[5][col] = 1
            V[b00][col] = 1
            W[c10][col] = 1; W[c11][col] = -1
            col += 1

            U[1][col] = 1
            V[b01][col] = 1; V[b11][col] = -1
            W[c01][col] = 1; W[c11][col] = 1
            col += 1

            U[5][col] = 1
            V[b10][col] = 1; V[b00][col] = -1
            W[c00][col] = 1; W[c10][col] = 1
            col += 1

            U[1][col] = 1; U[2][col] = 1
            V[b11][col] = 1
            W[c00][col] = -1; W[c01][col] = 1
            col += 1

            U[4][col] = 1; U[1][col] = -1
            V[b00][col] = 1; V[b01][col] = 1
            W[c11][col] = 1
            col += 1

            U[2][col] = 1; U[5][col] = -1
            V[b10][col] = 1; V[b11][col] = 1
            W[c00][col] = 1
            col += 1

        return U, V, W

    def build_schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return U, V, W

    # Try rank-42 construction first
    U, V, W = build_1plus2_split()
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback to rank-44 Strassen
    if time.time() < deadline:
        U, V, W = build_strassen_44()
        if verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}

    # Final fallback: schoolbook
    U, V, W = build_schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
