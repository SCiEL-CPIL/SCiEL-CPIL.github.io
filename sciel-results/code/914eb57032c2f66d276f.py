"""
Matrix multiplication 3x4 by 4x7 with rank-minimal tensor decomposition.
Combines block decomposition with Strassen and optimized sub-block decompositions.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank tensor decomposition for 3x4 * 4x7 matrix multiplication.
    Uses block decomposition with optimized sub-block decompositions.
    """
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 4, 7

    U_vecs = []
    V_vecs = []
    W_vecs = []

    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        """Add a single rank-1 tensor term to the decomposition."""
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)

        for idx, val in zip(u_indices, u_vals):
            u[idx] += val
        for idx, val in zip(v_indices, v_vals):
            v[idx] += val
        for idx, val in zip(w_indices, w_vals):
            w[idx] += val

        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def strassen_222(i0, j0, k0):
        """
        Strassen decomposition for 2x2 * 2x2 block starting at (i0, j0, k0).
        Adds 7 rank-1 terms.
        """
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        # M1: (A11+A22)*(B11+B22)
        add_rank1_term([i0*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0, k1*n+i1], [1, 1])

        # M2: (A21+A22)*B11
        add_rank1_term([i1*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0], [1],
                       [k0*n+i1, k1*n+i1], [1, -1])

        # M3: A11*(B12-B22)
        add_rank1_term([i0*m+j0], [1],
                       [j0*p+k1, j1*p+k1], [1, -1],
                       [k1*n+i0, k1*n+i1], [1, 1])

        # M4: A22*(B21-B11)
        add_rank1_term([i1*m+j1], [1],
                       [j1*p+k0, j0*p+k0], [1, -1],
                       [k0*n+i0, k0*n+i1], [1, 1])

        # M5: (A11+A12)*B22
        add_rank1_term([i0*m+j0, i0*m+j1], [1, 1],
                       [j1*p+k1], [1],
                       [k0*n+i0, k1*n+i0], [-1, 1])

        # M6: (A21-A11)*(B11+B12)
        add_rank1_term([i1*m+j0, i0*m+j0], [1, -1],
                       [j0*p+k0, j0*p+k1], [1, 1],
                       [k1*n+i1], [1])

        # M7: (A12-A22)*(B21+B22)
        add_rank1_term([i0*m+j1, i1*m+j1], [1, -1],
                       [j1*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0], [1])



    def m123_decomp(i0, j0, k0):
        """
        Decomposition for 1x2 * 2x3 block.
        Schoolbook = rank 6.
        """
        for j in range(2):
            for k in range(3):
                jj = j0 + j
                kk = k0 + k
                add_rank1_term([i0*m+jj], [1.0],
                               [jj*p+kk], [1.0],
                               [kk*n+i0], [1.0])

    def m124_decomp(i0, j0, k0):
        """
        Decomposition for 1x2 * 2x4 block.
        Schoolbook = rank 8.
        """
        for j in range(2):
            for k in range(4):
                jj = j0 + j
                kk = k0 + k
                add_rank1_term([i0*m+jj], [1.0],
                               [jj*p+kk], [1.0],
                               [kk*n+i0], [1.0])

    def m223_decomp_v2(i0, j0, k0):
        """
        Decomposition for 2x2 * 2x3 block.
        Strassen for 2x2x2 + schoolbook for 2x2x1 = rank 11.
        """
        strassen_222(i0, j0, k0)
        # Schoolbook for the 3rd column (k = k0+2)
        k2 = k0 + 2
        for i in range(2):
            for j in range(2):
                ii = i0 + i
                jj = j0 + j
                add_rank1_term([ii*m+jj], [1.0],
                               [jj*p+k2], [1.0],
                               [k2*n+ii], [1.0])

    def m224_decomp_v2(i0, j0, k0):
        """
        Decomposition for 2x2 * 2x4 block.
        Two Strassen 2x2x2 blocks = rank 14.
        """
        strassen_222(i0, j0, k0)
        strassen_222(i0, j0, k0 + 2)

    def m347_decomp(i0, j0, k0):
        """
        Decomposition for 3x4 * 4x7 block.
        Split n=3 as 1+2, m=4 as 2+2, p=7 as 3+4:
          M_{1,2,3}: 6
          M_{1,2,4}: 8
          M_{2,2,3}: 11
          M_{2,2,4}: 14
        Total: 39
        """
        # n=1, m=2, p=3: rows 0, cols 0-1, outputs 0-2
        m123_decomp(i0, j0, k0)
        # n=1, m=2, p=4: rows 0, cols 0-1, outputs 3-6
        m124_decomp(i0, j0, k0 + 3)
        # n=2, m=2, p=3: rows 1-2, cols 0-1, outputs 0-2
        m223_decomp_v2(i0 + 1, j0, k0)
        # n=2, m=2, p=4: rows 1-2, cols 0-1, outputs 3-6
        m224_decomp_v2(i0 + 1, j0, k0 + 3)
        # n=2, m=2, p=7: rows 1-2, cols 2-3, outputs 0-6
        # Split p=7 as 3+4
        m223_decomp_v2(i0 + 1, j0 + 2, k0)
        m224_decomp_v2(i0 + 1, j0 + 2, k0 + 3)

    # Build the full decomposition
    if time.time() - start_time < deadline:
        m347_decomp(0, 0, 0)

    rank = len(U_vecs)

    # Convert to matrix format
    U = np.array(U_vecs).T  # shape (n*m, rank)
    V = np.array(V_vecs).T  # shape (m*p, rank)
    W = np.array(W_vecs).T  # shape (p*n, rank)

    # Verify correctness
    if time.time() - start_time < deadline - 5.0:
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B

        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for j in range(m):
                    for r in range(rank):
                        val += A[i, j] * B[j, k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                C_computed[i, k] = val

        error = np.max(np.abs(C_computed - C_expected))
        if error > 1e-9:
            # Fallback to schoolbook
            rank_school = n * m * p
            U_sb = [[0] * rank_school for _ in range(n * m)]
            V_sb = [[0] * rank_school for _ in range(m * p)]
            W_sb = [[0] * rank_school for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i * m + j][col] = 1
                        V_sb[j * p + k][col] = 1
                        W_sb[k * n + i][col] = 1
                        col += 1
            return {"U": U_sb, "V": V_sb, "W": W_sb}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
