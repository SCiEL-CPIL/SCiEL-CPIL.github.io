"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    deadline = time.time() + 110

    # Build Strassen-based decomposition
    # A is 3x3, B is 3x8, C is 3x8
    # Partition: A[0:2,0:2] x B[0:2,0:8] uses Strassen (4 blocks of 2x2 x 2x2)
    # Remaining uses schoolbook
    # Total rank: 4*7 + 16 + 16 + 8 = 28 + 40 = 68

    terms = []  # list of (U_indices, V_indices, W_indices) where each is list of (idx, coeff)

    def add_term(u_entries, v_entries, w_entries):
        """Add one rank-1 term. Each entry is list of (flat_index, coefficient)."""
        terms.append((u_entries, v_entries, w_entries))

    # Strassen for 2x2 x 2x2 block:
    # A block: [[a00,a01],[a10,a11]], B block: [[b00,b01],[b10,b11]]
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b10)
    # M7 = (a01-a11)*(b01+b11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    # For each 2x2 x 2x2 Strassen instance, we add 7 terms.
    # U entries: for A[i,j] with coefficient, flat index = i*m + j
    # V entries: for B[j,k] with coefficient, flat index = j*p + k
    # W entries: for C[i,k] with coefficient, flat index = k*n + i

    def strassen_2x2_2x2(a_row0, a_row1, a_col0, a_col1, b_row0, b_row1, b_col0, b_col1):
        """
        a_row0, a_row1: row indices in A for the 2x2 block
        a_col0, a_col1: col indices in A for the 2x2 block
        b_row0, b_row1: row indices in B for the 2x2 block (same as a_col0, a_col1)
        b_col0, b_col1: col indices in B for the 2x2 block
        """
        # A entries: A[a_row0,a_col0], A[a_row0,a_col1], A[a_row1,a_col0], A[a_row1,a_col1]
        # B entries: B[b_row0,b_col0], B[b_row0,b_col1], B[b_row1,b_col0], B[b_row1,b_col1]
        # C entries: C[a_row0,b_col0], C[a_row0,b_col1], C[a_row1,b_col0], C[a_row1,b_col1]

        a00 = (a_row0, a_col0)
        a01 = (a_row0, a_col1)
        a10 = (a_row1, a_col0)
        a11 = (a_row1, a_col1)

        b00 = (b_row0, b_col0)
        b01 = (b_row0, b_col1)
        b10 = (b_row1, b_col0)
        b11 = (b_row1, b_col1)

        c00 = (a_row0, b_col0)
        c01 = (a_row0, b_col1)
        c10 = (a_row1, b_col0)
        c11 = (a_row1, b_col1)

        # M1 = (a00+a11)*(b00+b11)
        add_term([(a00[0]*m+a00[1], 1), (a11[0]*m+a11[1], 1)],
                 [(b00[0]*p+b00[1], 1), (b11[0]*p+b11[1], 1)],
                 [(c00[1]*n+c00[0], 1), (c11[1]*n+c11[0], 1)])

        # M2 = (a10+a11)*b00
        add_term([(a10[0]*m+a10[1], 1), (a11[0]*m+a11[1], 1)],
                 [(b00[0]*p+b00[1], 1)],
                 [(c10[1]*n+c10[0], 1)])

        # M3 = a00*(b01-b11)
        add_term([(a00[0]*m+a00[1], 1)],
                 [(b01[0]*p+b01[1], 1), (b11[0]*p+b11[1], -1)],
                 [(c01[1]*n+c01[0], 1)])

        # M4 = a11*(b10-b00)
        add_term([(a11[0]*m+a11[1], 1)],
                 [(b10[0]*p+b10[1], 1), (b00[0]*p+b00[1], -1)],
                 [(c00[1]*n+c00[0], 1), (c10[1]*n+c10[0], 1)])

        # M5 = (a00+a01)*b11
        add_term([(a00[0]*m+a00[1], 1), (a01[0]*m+a01[1], 1)],
                 [(b11[0]*p+b11[1], 1)],
                 [(c00[1]*n+c00[0], -1), (c01[1]*n+c01[0], 1)])

        # M6 = (a10-a00)*(b00+b10)
        add_term([(a10[0]*m+a10[1], 1), (a00[0]*m+a00[1], -1)],
                 [(b00[0]*p+b00[1], 1), (b10[0]*p+b10[1], 1)],
                 [(c11[1]*n+c11[0], 1)])

        # M7 = (a01-a11)*(b01+b11)
        add_term([(a01[0]*m+a01[1], 1), (a11[0]*m+a11[1], -1)],
                 [(b01[0]*p+b01[1], 1), (b11[0]*p+b11[1], 1)],
                 [(c00[1]*n+c00[0], 1), (c01[1]*n+c01[0], 1)])

    # Apply Strassen to A[0:2,0:2] x B[0:2,0:8]
    # B[0:2,0:8] has 8 columns, split into 4 groups of 2: [0,1], [2,3], [4,5], [6,7]
    for bc in range(0, 8, 2):
        strassen_2x2_2x2(0, 1, 0, 1, 0, 1, bc, bc + 1)

    # Schoolbook for remaining terms:
    # A[0:2,2] x B[2,0:8]: i in {0,1}, j=2, k in 0..7
    for i in range(2):
        for k in range(8):
            add_term([(i * m + 2, 1)],
                     [(2 * p + k, 1)],
                     [(k * n + i, 1)])

    # A[2,0:2] x B[0:2,0:8]: i=2, j in {0,1}, k in 0..7
    for j in range(2):
        for k in range(8):
            add_term([(2 * m + j, 1)],
                     [(j * p + k, 1)],
                     [(k * n + 2, 1)])

    # A[2,2] x B[2,0:8]: i=2, j=2, k in 0..7
    for k in range(8):
        add_term([(2 * m + 2, 1)],
                 [(2 * p + k, 1)],
                 [(k * n + 2, 1)])

    rank = len(terms)

    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r, (u_entries, v_entries, w_entries) in enumerate(terms):
        for (idx, coeff) in u_entries:
            U[idx][r] = coeff
        for (idx, coeff) in v_entries:
            V[idx][r] = coeff
        for (idx, coeff) in w_entries:
            W[idx][r] = coeff

    # Verify correctness with random test
    if time.time() < deadline:
        if not _verify(U, V, W, n, m, p):
            # Fallback to schoolbook
            return _schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, trials=5):
    """Verify the decomposition computes correct matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]

    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B

        # Compute C from decomposition:
        # C[i,k] = sum_r sum_j U[i*m+j, r] * A[i,j] * V[j*p+k, r] * W[k*n+i, r]
        # Actually: C[i,k] = sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_k2 V[j*p+k2,r]*B[j,k2]) * W[k*n+i,r]
        # No, the correct formula is:
        # C[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] * A[i,j]*B[j,k]
        # So we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k

        C_computed = np.zeros((n, p))
        for r in range(rank):
            # u_r: vector over (i,j) pairs
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            # Term r contributes: (A * u_r) outer (v_r * B) ... no
            # Actually: sum_j u_r[i,j]*A[i,j] gives a scalar for each i
            # sum_k v_r[j,k]*B[j,k] gives a scalar for each j
            # This doesn't quite work as a simple outer product.
            #
            # The correct interpretation:
            # C[i,k] = sum_r [sum_j U[i*m+j,r] * A[i,j]] * [sum_{k'} V[j*p+k',r] * B[j,k']] * W[k*n+i,r]
            # But j is free in the V sum, which is wrong.
            #
            # Let me re-derive. The tensor decomposition says:
            # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # And C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
            #           = sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] * A[i,j]*B[j,k]
            #
            # So for each r, the contribution to C[i,k] is:
            # sum_j U[i*m+j,r] * A[i,j] * V[j*p+k,r] * B[j,k] * W[k*n+i,r]
            # = W[k*n+i,r] * sum_j (U[i*m+j,r] * A[i,j]) * (V[j*p+k,r] * B[j,k])
            #
            # This is NOT a simple outer product of two vectors. It's a sum over j of products.
            # So we can't just do (A*U_r) outer (V_r*B).
            #
            # Let me just verify the tensor identity directly.

        # Direct verification of tensor identity:
        # Check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
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
# EVOLVE-BLOCK-END
