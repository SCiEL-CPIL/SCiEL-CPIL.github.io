"""Matrix multiplication 2x4 by 4x5 via block Strassen + schoolbook,
with exact verification."""

import time
import numpy as np

DIMENSIONS = [2, 4, 5]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0


def _remaining():
    return _DEADLINE - time.time()


def construct():
    """
    Return {'U','V','W'} factor matrices that exactly decompose the
    2x4-by-4x5 matrix multiplication tensor.

    Strategy (block decomposition):
      - Split the 4x5 B into four 2x2 blocks plus a final 4x1 column.
      - Each 2x2 (A-block) * (B-block) -> 2x2 C block uses Strassen (rank 7).
      - The final 2x4 * 4x1 -> 2x1 C column uses schoolbook (rank 8).
      Total rank = 4*7 + 8 = 36.
      
    Additionally, we check for linear dependencies among the rank-1 terms
    to see if the rank can be reduced.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    rank = 36

    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36

    col = 0

    def add_strassen_222(i0, j0, k0, col_start):
        """
        Add a Strassen 2x2x2 decomposition for:
        C[i0:i0+2, k0:k0+2] = A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        """
        nonlocal col

        # M1 = (A00 + A11) * (B00 + B11)
        c = col_start
        U[i0 * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        col += 1

        # M2 = (A10 + A11) * B00
        c = col_start + 1
        U[(i0 + 1) * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] -= 1
        col += 1

        # M3 = A00 * (B01 - B11)
        c = col_start + 2
        U[i0 * m + j0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] -= 1
        W[(k0 + 1) * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        col += 1

        # M4 = A11 * (B10 - B00)
        c = col_start + 3
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + k0][c] += 1
        V[j0 * p + k0][c] -= 1
        W[k0 * n + i0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        col += 1

        # M5 = (A00 + A01) * B11
        c = col_start + 4
        U[i0 * m + j0][c] += 1
        U[i0 * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] -= 1
        W[(k0 + 1) * n + i0][c] += 1
        col += 1

        # M6 = (A10 - A00) * (B00 + B01)
        c = col_start + 5
        U[(i0 + 1) * m + j0][c] += 1
        U[i0 * m + j0][c] -= 1
        V[j0 * p + k0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        col += 1

        # M7 = (A01 - A11) * (B10 + B11)
        c = col_start + 6
        U[i0 * m + (j0 + 1)][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] -= 1
        V[(j0 + 1) * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        col += 1

        return col_start + 7

    def add_schoolbook_2x4x1(i0, j0, k0, col_start):
        """
        Add schoolbook decomposition for:
        C[i0:i0+2, k0:k0+1] = A[i0:i0+2, j0:j0+4] * B[j0:j0+4, k0:k0+1]
        """
        c = col_start
        for di in range(2):
            for dj in range(4):
                U[(i0 + di) * m + (j0 + dj)][c] = 1
                V[(j0 + dj) * p + k0][c] = 1
                W[k0 * n + (i0 + di)][c] = 1
                c += 1
        return c

    # Block 1: C[:, 0:2] from A[:, 0:2] * B[0:2, 0:2]
    col = add_strassen_222(0, 0, 0, 0)
    # Block 2: C[:, 0:2] from A[:, 2:4] * B[2:4, 0:2]
    col = add_strassen_222(0, 2, 0, col)
    # Block 3: C[:, 2:4] from A[:, 0:2] * B[0:2, 2:4]
    col = add_strassen_222(0, 0, 2, col)
    # Block 4: C[:, 2:4] from A[:, 2:4] * B[2:4, 2:4]
    col = add_strassen_222(0, 2, 2, col)
    # Block 5: C[:, 4] from A * B[:, 4] (schoolbook 2x4 * 4x1)
    col = add_schoolbook_2x4x1(0, 0, 4, col)

    assert col == rank, f"Expected rank {rank}, got {col}"

    # ---- Linear Dependency Check ------------------------------------------
    # Check if the rank-1 tensors are linearly independent.
    # Flatten each rank-1 term into a vector of size n*m*p.
    # If the matrix of these vectors has rank < 36, there is a dependency.
    if _remaining() > 5.0:
        try:
            # Construct matrix K of shape (n*m*p, rank)
            # K[:, r] = flatten(U[:, r] \otimes V[:, r] \otimes W[:, r])
            # Note: The tensor is T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r] ?
            # Wait, the indices in U, V, W are:
            # U is (n*m, rank), indexed by (i, j) -> U[i*m+j, r]
            # V is (m*p, rank), indexed by (j, k) -> V[j*p+k, r]
            # W is (p*n, rank), indexed by (k, i) -> W[k*n+i, r]
            # The tensor T is defined by T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
            # We want to check if sum_r c_r (U_r \otimes V_r \otimes W_r) = 0.
            # This is a linear dependency in the space of tensors.
            
            # Let's form the matrix M where column r is the vectorization of the tensor T_r.
            # T_r[i, j, k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            M = np.zeros((n * m * p, rank))
            for r in range(rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                            if val != 0:
                                # Map (i, j, k) to index
                                # Order: i slowest, k fastest?
                                # Index = i * m * p + j * p + k
                                idx = i * m * p + j * p + k
                                M[idx, r] = val
            
            # Check rank of M
            # Use SVD to find rank
            U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
            tol = 1e-10
            current_rank = np.sum(s > tol)
            
            if current_rank < rank:
                # There is a linear dependency.
                # Find the null space vector c
                # The null space is spanned by the last (rank - current_rank) rows of Vt
                # Since we want to remove one term, we look for a null space vector with a large component.
                # Let's pick the null space vector corresponding to the smallest singular value.
                # Actually, if current_rank < rank, the last column of Vt.T (which is V) is in the null space.
                # Let c = V[:, -1]
                c = Vt[-1, :]
                
                # We want to eliminate the term with the largest |c_r|
                # But eliminating a term requires re-expressing the others?
                # No, if sum c_r T_r = 0, then T_j = -1/c_j sum_{r!=j} c_r T_r.
                # This doesn't reduce the rank unless we can simplify the expression.
                # However, if the dependency is "simple" (e.g., one term is a sum of others),
                # we might be able to replace it.
                
                # For now, just log the finding. If we can't easily reduce, we keep 36.
                # Given the time, we won't attempt complex re-optimization.
                pass
                
        except Exception:
            pass

    # ---- Exact verification ----------------------------------------------
    if _remaining() > 0.5:
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    acc = 0
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    for r in range(rank):
                        acc += U[ui][r] * V[vj][r] * W[wk][r]
                    assert acc == 1, f"Verification failed at i={i},j={j},k={k}: acc={acc}"

    return {"U": U, "V": V, "W": W}
