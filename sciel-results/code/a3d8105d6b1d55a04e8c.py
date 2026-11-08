"""Tensor decomposition for 2x4x6 matrix multiplication: Strassen core + optimized embeddings."""
import numpy as np
import time
import random

DIMENSIONS = [2, 4, 6]
DEADLINE = 140.0  # safety margin before 150s kill


def _time_left():
    return DEADLINE - time.time()


def _strassen_2x2x2():
    """Return (U,V,W) each shape (4,7) for the rank-7 Strassen 2x2x2 decomposition.

    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c] with the 7 Strassen products.
    """
    # 7 Strassen products M1..M7, each a bilinear form on (i,j) x (j,k).
    # We store, per product c, the U-row (i,j) coefficient and V-row (j,k) coefficient.
    # Strassen:
    # M1 = (A00)*(B01 - B11)
    # M2 = (A10 - A00)*B00
    # M3 = (A00 + A11)*B11
    # M4 = (A11 - A01)*(B00 + B10)
    # M5 = (A01 + A11)*(B10 - B00)
    # M6 = (A00 - A10)*(B01 + B11)
    # M7 = (A10 + A01)*(B11 - B10)
    #
    # U rows index (i,j) in {0,1}x{0,1} -> row = i*2 + j
    # V rows index (j,k) in {0,1}x{0,1} -> row = j*2 + k
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    # M1: A00 * (B01 - B11)
    U[0*2+0, 0] = 1.0
    V[0*2+1, 0] = 1.0
    V[1*2+1, 0] = -1.0
    # M2: (A10 - A00) * B00
    U[1*2+0, 1] = 1.0
    U[0*2+0, 1] = -1.0
    V[0*2+0, 1] = 1.0
    # M3: (A00 + A11) * B11
    U[0*2+0, 2] = 1.0
    U[1*2+1, 2] = 1.0
    V[1*2+1, 2] = 1.0
    # M4: (A11 - A01) * (B00 + B10)
    U[1*2+1, 3] = 1.0
    U[0*2+1, 3] = -1.0
    V[0*2+0, 3] = 1.0
    V[1*2+0, 3] = 1.0
    # M5: (A01 + A11) * (B10 - B00)
    U[0*2+1, 4] = 1.0
    U[1*2+1, 4] = 1.0
    V[1*2+0, 4] = 1.0
    V[0*2+0, 4] = -1.0
    # M6: (A00 - A10) * (B01 + B11)
    U[0*2+0, 5] = 1.0
    U[1*2+0, 5] = -1.0
    V[0*2+1, 5] = 1.0
    V[1*2+1, 5] = 1.0
    # M7: (A10 + A01) * (B11 - B10)
    U[1*2+0, 6] = 1.0
    U[0*2+1, 6] = 1.0
    V[1*2+1, 6] = 1.0
    V[1*2+0, 6] = -1.0
    return U, V


def _build_tensor_factors(U, V, W):
    """Given factor matrices, build the full multiplication tensor T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
    Returns the tensor as a 3D array of shape (n,m,p)."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c]  # length n*m
        vc = V[:, c]  # length m*p
        wc = W[:, c]  # length p*n
        u = uc.reshape(n, m)
        v = vc.reshape(m, p)
        w = wc.reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    return T


def _target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T


def _to_list(M):
    return [[float(x) for x in row] for row in M]


def _verify(U, V, W, tol=1e-8):
    T = _build_tensor_factors(np.asarray(U), np.asarray(V), np.asarray(W))
    Tt = _target_tensor()
    return float(np.max(np.abs(T - Tt))) < tol


def _rank_minimal_candidate():
    """Try to construct a low-rank decomposition using Strassen core + optimized embeddings.
    Returns (U,V,W) as lists or None if not valid."""
    n, m, p = DIMENSIONS
    S_U, S_V = _strassen_2x2x2()  # (4,7) each
    core_rank = 7

    # The 2x4x6 multiplication: A is 2x4 (n x m), B is 4x6 (m x p), C is 2x6 (n x p).
    # W convention: W[k*n + i, c], so W has p*n rows.
    #
    # Block structure: split m=4 into two blocks of 2, p=6 into three blocks of 2.
    # A: rows i in {0,1}, cols j in {0..3}.  Split j into j0 in {0,1} and j1 in {2,3}.
    # B: rows j in {0..3}, cols k in {0..5}. Split j into {0,1},{2,3}; k into {0,1},{2,3},{4,5}.
    #
    # The tensor T[i,j,k] = 1 for all i,j,k. We want sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
    #
    # Use Strassen on the 2x2 (i vs k) block with the "j" dimension as a free parameter.
    # This is the WFTT-style embedding: for each pair of j-blocks and k-blocks we run a
    # 2x2x2 Strassen-like contraction.
    #
    # A cleaner structural model: treat the multiplication as a tensor product of a
    # 2x2x2 Strassen core (rank 7) and a "selection" tensor over the extra dimensions.
    #
    # Concretely, the full tensor T[i,j,k]=1 can be written as a sum over (a,b) where
    # a indexes the 2x2 Strassen block in (i, k) and b indexes the (j, k) embedding.
    #
    # Let's use a direct rank-42 construction first (Strassen per 2x2 block of j,k):
    # There are 2 j-blocks and 3 k-blocks = 6 blocks, each contributing 7 rank-1 terms
    # via Strassen on the (i, k_within_block) 2x2 with j_within_block as scalar.
    #
    # Actually, the simplest verified improvement: split into 2x2x2 sub-tensors.
    # n=2, m=4, p=6. Split m into 2 groups of 2, p into 3 groups of 2.
    # For each (jg, kg) block, we have a 2x2x2 subproblem: A[:, jg*2:(jg+1)*2] (2x2),
    # B[jg*2:(jg+1)*2, kg*2:(kg+1)*2] (2x2), contributing to C[:, kg*2:(kg+1)*2] (2x2).
    # Each 2x2x2 block can use rank-7 Strassen. 6 blocks * 7 = 42.
    #
    # But blocks are not independent because they share the same A and B rows/cols.
    # However, since T[i,j,k]=1 (all ones), the decomposition is separable:
    # T[i,j,k] = 1 = (sum over blocks) of contributions. We can assign each block its
    # own 7 Strassen terms and the linearity ensures correctness IF the embeddings
    # are chosen so that each (i,j,k) gets exactly 1.
    #
    # Let's just build the 42-rank construction and verify.
    U = np.zeros((n * m, 42))
    V = np.zeros((m * p, 42))
    W = np.zeros((p * n, 42))

    for jg in range(2):
        for kg in range(3):
            base = (jg * 3 + kg) * 7
            for c in range(7):
                col = base + c
                # U: for i in {0,1}, j in {jg*2, jg*2+1}
                for i in range(2):
                    for dj in range(2):
                        j = jg * 2 + dj
                        U[i * m + j, col] = S_U[i * 2 + dj, c]
                # V: for j in {jg*2, jg*2+1}, k in {kg*2, kg*2+1}
                for dj in range(2):
                    j = jg * 2 + dj
                    for dk in range(2):
                        k = kg * 2 + dk
                        V[j * p + k, col] = S_V[dj * 2 + dk, c]
                # W: for k in {kg*2, kg*2+1}, i in {0,1}
                for dk in range(2):
                    k = kg * 2 + dk
                    for i in range(2):
                        W[k * n + i, col] = 1.0 if (dk == 0 and i == 0) else (1.0 if (dk == 1 and i == 1) else 0.0)
                        # Actually W for Strassen: W[k*2+i, c] should be the Strassen W.
                        # But Strassen W is identity-like: W[k*2+i] = delta_{k,i} for the 2x2 case?
                        # No: in Strassen, C = [[C00, C01],[C10, C11]] where C00 = M1+M3-M5+M7, etc.
                        # The W matrix in the tensor sense: W[k*2+i, c] is the coefficient
                        # of product c in C[i,k]. So we need the actual Strassen W.
    # The above W assignment is wrong. Let me redo with proper Strassen W.
    # Strassen C[i,k] = sum_c W_c[i,k] * M_c where:
    # C00 = M1 + M3 - M5 + M7
    # C01 = M4 + M5
    # C10 = M2 + M3
    # C11 = M1 - M2 + M4 + M6
    # So W[k*2+i, c] (k is col of C, i is row of C):
    # C[i,k]: i=row, k=col. W index = k*2 + i.
    W_strassen = np.zeros((4, 7))
    # c: 0=M1, 1=M2, 2=M3, 3=M4, 4=M5, 5=M6, 6=M7
    # C00 (i=0,k=0): M1 - M5 + M3 + M7 -> W[0*2+0, c]
    W_strassen[0, 0] = 1; W_strassen[0, 4] = -1; W_strassen[0, 2] = 1; W_strassen[0, 6] = 1
    # C01 (i=0,k=1): M4 + M5 -> W[1*2+0, c]
    W_strassen[2, 3] = 1; W_strassen[2, 4] = 1
    # C10 (i=1,k=0): M2 + M3 -> W[0*2+1, c]
    W_strassen[1, 1] = 1; W_strassen[1, 2] = 1
    # C11 (i=1,k=1): M1 - M2 + M4 + M6 -> W[1*2+1, c]
    W_strassen[3, 0] = 1; W_strassen[3, 1] = -1; W_strassen[3, 3] = 1; W_strassen[3, 5] = 1

    U = np.zeros((n * m, 42))
    V = np.zeros((m * p, 42))
    W = np.zeros((p * n, 42))
    for jg in range(2):
        for kg in range(3):
            base = (jg * 3 + kg) * 7
            for c in range(7):
                col = base + c
                for i in range(2):
                    for dj in range(2):
                        j = jg * 2 + dj
                        U[i * m + j, col] = S_U[i * 2 + dj, c]
                for dj in range(2):
                    j = jg * 2 + dj
                    for dk in range(2):
                        k = kg * 2 + dk
                        V[j * p + k, col] = S_V[dj * 2 + dk, c]
                for dk in range(2):
                    k = kg * 2 + dk
                    for i in range(2):
                        W[k * n + i, col] = W_strassen[dk * 2 + i, c]

    if _verify(U, V, W):
        return _to_list(U), _to_list(V), _to_list(W)
    return None


def construct():
    # First, try the structural 42-rank Strassen-block construction.
    cand = _rank_minimal_candidate()
    if cand is not None:
        U, V, W = cand
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank 48.
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
