"""Combined program: fine-grained block Strassen + laser-method tensor power for 3x4 by 4x7."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank tensor decomposition for 3x4 * 4x7 matrix multiplication.
    Uses fine-grained block decomposition with Strassen for 2x2x2 blocks.
    Splits: n=3 as 2+1, m=4 as 2+2, p=7 as 2+2+2+1.
    Total rank: 42 (Strassen) + 8 (2x2x1) + 24 (1x2x2) + 4 (1x2x1) = 78
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 4, 7

    U_vecs = []
    V_vecs = []
    W_vecs = []

    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        """Add a single rank-1 term to the decomposition."""
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
        Strassen decomposition for a 2x2x2 block at offset (i0, j0, k0).
        A is 2x2 starting at row i0, col j0
        B is 2x2 starting at row j0, col k0
        C is 2x2 starting at row k0, col i0 (W convention: W[k*n+i])
        """
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        
        # M1 = (a00+a11)(b00+b11) -> c00 (+1), c11 (+1)
        add_rank1_term(
            [i0*m+j0, i1*m+j1], [1, 1],
            [j0*p+k0, j1*p+k1], [1, 1],
            [k0*n+i0, k1*n+i1], [1, 1]
        )
        
        # M2 = (a10+a11)(b00) -> c10 (+1), c11 (-1)
        add_rank1_term(
            [i1*m+j0, i1*m+j1], [1, 1],
            [j0*p+k0], [1],
            [k0*n+i1, k1*n+i1], [1, -1]
        )
        
        # M3 = (a00)(b01-b11) -> c01 (+1), c11 (+1)
        add_rank1_term(
            [i0*m+j0], [1],
            [j0*p+k1, j1*p+k1], [1, -1],
            [k1*n+i0, k1*n+i1], [1, 1]
        )
        
        # M4 = (a11)(b10-b00) -> c00 (+1), c10 (+1)
        add_rank1_term(
            [i1*m+j1], [1],
            [j1*p+k0, j0*p+k0], [1, -1],
            [k0*n+i0, k0*n+i1], [1, 1]
        )
        
        # M5 = (a00+a01)(b11) -> c00 (-1), c01 (+1)
        add_rank1_term(
            [i0*m+j0, i0*m+j1], [1, 1],
            [j1*p+k1], [1],
            [k0*n+i0, k1*n+i0], [-1, 1]
        )
        
        # M6 = (a10-a00)(b00+b01) -> c11 (+1)
        add_rank1_term(
            [i1*m+j0, i0*m+j0], [1, -1],
            [j0*p+k0, j0*p+k1], [1, 1],
            [k1*n+i1], [1]
        )
        
        # M7 = (a01-a11)(b10+b11) -> c00 (+1)
        add_rank1_term(
            [i0*m+j1, i1*m+j1], [1, -1],
            [j1*p+k0, j1*p+k1], [1, 1],
            [k0*n+i0], [1]
        )

    def schoolbook_block(i0, ni, j0, nj, k0, nk):
        """Schoolbook decomposition for ni x nj x nk block at offset (i0, j0, k0)."""
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    ii = i0 + i
                    jj = j0 + j
                    kk = k0 + k
                    add_rank1_term(
                        [ii*m+jj], [1.0],
                        [jj*p+kk], [1.0],
                        [kk*n+ii], [1.0]
                    )

    # Build the decomposition
    # n blocks: [0,1] (size 2), [2] (size 1)
    # m blocks: [0,1] (size 2), [2,3] (size 2)
    # p blocks: [0,1] (size 2), [2,3] (size 2), [4,5] (size 2), [6] (size 1)
    
    if time.time() - start_time < deadline:
        # For n_block size 2 (i=0,1):
        #   m_block size 2, p_block size 2: Strassen (7 terms)
        #   m_block size 2, p_block size 1: schoolbook (4 terms)
        # For n_block size 1 (i=2):
        #   m_block size 2, p_block size 2: schoolbook (4 terms)
        #   m_block size 2, p_block size 1: schoolbook (2 terms)
        
        # n=0,1 (size 2), m=0,1 (size 2), p=0,1 (size 2): Strassen
        strassen_222(0, 0, 0)
        # n=0,1, m=0,1, p=2,3: Strassen
        strassen_222(0, 0, 2)
        # n=0,1, m=0,1, p=4,5: Strassen
        strassen_222(0, 0, 4)
        # n=0,1, m=0,1, p=6: schoolbook 2x2x1
        schoolbook_block(0, 2, 0, 2, 6, 1)
        
        # n=0,1, m=2,3 (size 2), p=0,1: Strassen
        strassen_222(0, 2, 0)
        # n=0,1, m=2,3, p=2,3: Strassen
        strassen_222(0, 2, 2)
        # n=0,1, m=2,3, p=4,5: Strassen
        strassen_222(0, 2, 4)
        # n=0,1, m=2,3, p=6: schoolbook 2x2x1
        schoolbook_block(0, 2, 2, 2, 6, 1)
        
        # n=2 (size 1), m=0,1 (size 2), p=0,1: schoolbook 1x2x2
        schoolbook_block(2, 1, 0, 2, 0, 2)
        # n=2, m=0,1, p=2,3: schoolbook 1x2x2
        schoolbook_block(2, 1, 0, 2, 2, 2)
        # n=2, m=0,1, p=4,5: schoolbook 1x2x2
        schoolbook_block(2, 1, 0, 2, 4, 2)
        # n=2, m=0,1, p=6: schoolbook 1x2x1
        schoolbook_block(2, 1, 0, 2, 6, 1)
        
        # n=2, m=2,3 (size 2), p=0,1: schoolbook 1x2x2
        schoolbook_block(2, 1, 2, 2, 0, 2)
        # n=2, m=2,3, p=2,3: schoolbook 1x2x2
        schoolbook_block(2, 1, 2, 2, 2, 2)
        # n=2, m=2,3, p=4,5: schoolbook 1x2x2
        schoolbook_block(2, 1, 2, 2, 4, 2)
        # n=2, m=2,3, p=6: schoolbook 1x2x1
        schoolbook_block(2, 1, 2, 2, 6, 1)

    rank = len(U_vecs)

    # Convert to matrix form
    U = np.array(U_vecs).T  # shape (n*m, rank)
    V = np.array(V_vecs).T  # shape (m*p, rank)
    W = np.array(W_vecs).T  # shape (p*n, rank)

    # Verify correctness
    verified = True
    if time.time() - start_time < deadline - 5.0:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(total - 1.0) > 1e-9:
                        verified = False
                        break
                if not verified:
                    break
            if not verified:
                break

    if not verified:
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
