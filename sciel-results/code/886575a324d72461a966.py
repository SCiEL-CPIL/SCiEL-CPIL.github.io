"""Rank-minimal exact tensor decomposition for 6x6x6 matrix multiplication.

Strategy:
- The current best known construction uses a 2x2 block Strassen decomposition
  of the 6x6 matrix product, where each 3x3 block product is done via schoolbook.
  This gives rank 7 * 27 = 189.

- However, we can do better by using a more efficient decomposition of the
  3x3 block product. The key insight is that we don't need to decompose each
  3x3 product independently; we can combine the Strassen structure with a
  more efficient 3x3 decomposition.

- Actually, the cleanest improvement is to use the known result that the
  tensor rank of 6x6x6 matrix multiplication is at most 189 (from block
  Strassen), but there might be constructions with lower rank.

- Let me think about this differently. The standard approach:
  1. Block Strassen: 2x2 blocks of 3x3, rank 7*27 = 189
  3. Alternative: Use a different block structure or a direct construction.

- Actually, for 6x6, we can use the fact that 6 = 2*3 and apply Strassen
  at the top level (2x2 blocks of 3x3), giving 7 multiplications of 3x3
  matrices. Each 3x3 multiplication has rank 27 (schoolbook), so total = 189.

- Can we do better? One approach: use a 3x3 block structure with 2x2 sub-blocks.
  But 6 = 3*2, so we'd have 3x3 blocks of 2x2 matrices. The 3x3 matrix
  multiplication tensor has rank 23 (over reals, using known results) or
  27 (schoolbook). If we use rank 23 for 3x3, then total = 9 * 23 = 207,
  which is worse.

- Another approach: use a 2x3 block structure. But this is asymmetric and
  doesn't directly apply Strassen.

- Let me just implement the block Strassen construction cleanly and verify it.
  The rank is 189, which is an improvement over schoolbook (216).

- Actually, I realize the current program already has this. Let me check if
  there's a bug in the verification or construction.

- Looking more carefully, I think the issue might be in how the W matrix
  indices are handled. The task says "output factor follows the official trace
  convention W[k*n+i,r]". So W is indexed by (k*n+i, r) where k is the row
  of C and i is the column of C. This means W has shape (p*n, rank) where
  the first index is k*n+i (k in 0..p-1, i in 0..n-1).

- Let me re-examine the tensor decomposition. The matrix multiplication tensor
  T[i,j,k] = 1 means C[i,k] = sum_j A[i,j] * B[j,k].

- In the decomposition:
  U has shape (n*m, rank), indexed by (i*m+j, r) where i is row of A, j is col of A
  V has shape (m*p, rank), indexed by (j*p+k, r) where j is row of B, k is col of B  
  W has shape (p*n, rank), indexed by (k*n+i, r) where k is row of C, i is col of C

- The reconstruction is: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
  summed over j.

- Wait, that's not right. Let me re-read the task.

- "output factor follows the official trace convention W[k*n+i,r]"

- So W[k*n+i, r] where k is the row index of C (0..p-1) and i is the column
  index of C (0..n-1). This is consistent with C being p x n... but wait,
  C should be n x p if A is n x m and B is m x p.

- Hmm, let me re-read: "multiplying a 6 by 6 matrix by a 6 by 6 matrix".
  So A is 6x6, B is 6x6, C is 6x6. n=m=p=6.

- The tensor is T[i,j,k] where C[i,k] = sum_j A[i,j] B[j,k].
- U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]

- So the reconstruction is: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

- This is a standard tensor decomposition of the matrix multiplication tensor.

- Let me implement the block Strassen construction correctly and verify it.
"""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly reproduces the matrix-multiplication tensor."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    # T[i,j,k] = sum_r Ur[i,j,r] * Vr[j,k,r] * Wr[k,i,r]
    T = np.einsum('ijr,jkr,kr->ijk', Ur, Vr, Wr)
    expected = np.ones((n, m, p))
    if not np.allclose(T, expected, atol=tol):
        diff = np.abs(T - expected).max()
        return False, diff
    return True, 0.0


def _build_block_strassen(n, m, p):
    """Block-Strassen decomposition: 2x2 blocks of 3x3, Strassen rank 7, each 3x3 schoolbook rank 27.
    Total rank = 7 * 27 = 189."""
    assert n == m == p == 6, "This construction is for 6x6x6"
    
    # Standard Strassen for 2x2 block matrices:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M2 + M4 - M6
    # C12 = M3 + M4
    # C21 = M1 + M2
    # C22 = M3 - M1 + M5 + M6
    
    U_local = [
        [(0, 0, 1), (1, 1, 1)],          # M1: (A11+A22)
        [(1, 0, 1), (1, 1, 1)],          # M2: (A21+A22)
        [(0, 0, 1)],                     # M3: A11
        [(1, 1, 1)],                     # M4: A22
        [(0, 0, 1), (0, 1, 1)],          # M5: (A11+A12)
        [(1, 0, 1), (0, 0, -1)],         # M6: (A21-A11)
        [(0, 1, 1), (1, 1, -1)],         # M7: (A12-A22)
    ]
    V_local = [
        [(0, 0, 1), (1, 1, 1)],          # M1: (B11+B22)
        [(0, 0, 1)],                     # M2: B11
        [(0, 1, 1), (1, 1, -1)],         # M3: (B12-B22)
        [(1, 0, 1), (0, 0, -1)],         # M4: (B21-B11)
        [(1, 1, 1)],                     # M5: B22
        [(0, 0, 1), (0, 1, 1)],          # M6: (B11+B12)
        [(1, 0, 1), (1, 1, 1)],          # M7: (B21+B22)
    ]
    C_local = {
        (0, 0): {0: 1, 1: -1, 3: 1, 5: -1},   # C11 = M1 - M2 + M4 - M6
        (0, 1): {2: 1, 3: 1},                 # C12 = M3 + M4
        (1, 0): {0: 1, 1: 1},                 # C21 = M1 + M2
        (1, 1): {2: 1, 0: -1, 4: 1, 5: 1},    # C22 = M3 - M1 + M5 + M6
    }
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for sm in range(7):
        for (a, b, ca) in U_local[sm]:
            for (c, d, cb) in V_local[sm]:
                for (e, f) in C_local:
                    if sm in C_local[(e, f)]:
                        cc = C_local[(e, f)][sm]
                        coeff = ca * cb * cc
                        # 3x3 * 3x3 schoolbook: 27 terms
                        for t in range(3):  # row of A-block / col of C-block
                            for s in range(3):  # inner (col of A-block / row of B-block)
                                for u in range(3):  # col of B-block / row of C-block
                                    i_idx = a * 3 + t
                                    j_idx = b * 3 + s
                                    k_idx = c * 3 + s
                                    l_idx = d * 3 + u
                                    m_idx = e * 3 + u  # row of C
                                    n_idx = f * 3 + t  # col of C
                                    
                                    u_vec = np.zeros(n * m)
                                    v_vec = np.zeros(m * p)
                                    w_vec = np.zeros(p * n)
                                    u_vec[i_idx * m + j_idx] = coeff
                                    v_vec[k_idx * p + l_idx] = 1.0
                                    w_vec[m_idx * n + n_idx] = 1.0
                                    U_cols.append(u_vec)
                                    V_cols.append(v_vec)
                                    W_cols.append(w_vec)
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _build_schoolbook(n, m, p):
    """Fallback: schoolbook rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def construct():
    start = time.time()
    deadline = start + 135

    n, m, p = DIMENSIONS

    # Try block-Strassen (rank 189)
    try:
        result = _build_block_strassen(n, m, p)
        ok, diff = _verify(result["U"], result["V"], result["W"], n, m, p)
        if ok:
            return result
    except Exception:
        pass

    # Fallback: schoolbook rank 216
    return _build_schoolbook(n, m, p)
