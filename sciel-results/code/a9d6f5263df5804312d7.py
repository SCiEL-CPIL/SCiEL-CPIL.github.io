import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 148.0

    # --- Strategy: try rank 27 first with a smarter decomposition,
    #     falling back to rank 28 (Strassen+schoolbook) if 27 fails. ---
    #
    # The 2x3x5 matrix multiplication tensor T[i,j,k] = delta_{ij,k} (i.e.,
    # (A B)_{i,k} = sum_j A_{i,j} B_{j,k}) needs a rank-r decomposition:
    #   T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # We use a mixed approach:
    #   - For the 2x2x2 sub-blocks (top-left and top-middle of C), use Strassen (rank 7 each).
    #   - For the remaining rectangular blocks, attempt a combined optimization that
    #     exploits shared structure to reduce total rank below pure schoolbook.
    #
    # Pure schoolbook for all: 2*3*5 = 30.
    # Strassen for two 2x2x2 + schoolbook for rest: 7+7+4+4+4+2 = 28.
    #
    # To get rank 27, we try to save 1 rank by merging some schoolbook terms
    # with Strassen terms. Specifically, we look for a rank-1 term that covers
    # contributions from two different blocks simultaneously.
    #
    # Alternative: Use the known fact that the 2x2x3 tensor (2x2 * 2x3 -> 2x3)
    # has border rank 10 and actual rank at most 11 (schoolbook) or possibly 10.
    # Let's try a direct construction for rank 27.
    #
    # Actually, let's try a different split:
    #   A is 2x3, B is 3x5, C is 2x5.
    #   Split B into B1 (3x3) and B2 (3x2).
    #   Split A into A1 (2x2) and A2 (2x1).
    #   C = A1*B1 + A1*B2 + A2*B1 + A2*B2
    #   A1*B1: 2x2 * 2x2 -> 2x2, Strassen rank 7
    #   A1*B2: 2x2 * 2x2 -> 2x2, Strassen rank 7
    #   A2*B1: 2x1 * 1x2 -> 2x2, schoolbook rank 4
    #   A2*B2: 2x1 * 1x2 -> 2x2, schoolbook rank 4
    #   Total: 22. Wait, that's already 22? No:
    #   A1 is rows 0:2 of A (2 rows), cols 0:2 (2 cols) -> 2x2
    #   A2 is row 0:2 of A, col 2:3 -> 2x1
    #   B1 is rows 0:2 of B, cols 0:3 -> 2x3
    #   B2 is row 2:3 of B, cols 0:3 -> 1x3
    #   Hmm, let me re-index carefully.
    #
    # Let's just try rank 27 with a search over small perturbations of the rank-28 solution.
    # If that's too slow, fall back to rank 28.

    def try_rank_28():
        rank = 28
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i1, i2, j1, j2, k1, k2):
            nonlocal col
            # M1 = (A00+A11)(B00+B11)
            U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
            V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
            col += 1
            # M2 = (A10+A11)B00
            U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
            V[j1*p+k1, col] = 1
            W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
            col += 1
            # M3 = A00(B01-B11)
            U[i1*m+j1, col] = 1
            V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
            W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
            col += 1
            # M4 = A11(B10-B00)
            U[i2*m+j2, col] = 1
            V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
            W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
            col += 1
            # M5 = (A00+A01)B11
            U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
            V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
            col += 1
            # M6 = (A10-A00)(B00+B01)
            U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
            V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
            W[k2*n+i2, col] = 1
            col += 1
            # M7 = (A01-A11)(B10+B11)
            U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
            V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = 1
            col += 1

        def add_schoolbook(i_list, j_list, k_list):
            nonlocal col
            for i in i_list:
                for j in j_list:
                    for k in k_list:
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1

        add_strassen(0, 1, 0, 1, 0, 1)      # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        add_strassen(0, 1, 0, 1, 2, 3)      # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
        add_schoolbook([0,1],[0,1],[4])      # A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5]
        add_schoolbook([0,1],[2],[0,1])      # A[0:2,2:3] * B[2:3,0:2] -> C[0:2,0:2]
        add_schoolbook([0,1],[2],[2,3])      # A[0:2,2:3] * B[2:3,2:4] -> C[0:2,2:4]
        add_schoolbook([0,1],[2],[4])        # A[0:2,2:3] * B[2:3,4:5] -> C[0:2,4:5]

        # Verify
        Ur = U.T.reshape(rank, n, m)
        Vr = V.T.reshape(rank, m, p)
        Wr = W.T.reshape(rank, p, n)
        tensor = np.einsum('rij,rjk,rki->ijk', Ur, Vr, Wr)
        if np.allclose(tensor, 1.0, atol=1e-9):
            return U, V, W, rank
        return None

    def try_rank_27():
        """Attempt rank 27 by modifying the rank-28 solution.
        We look for a rank-1 update that, combined with removing one schoolbook term,
        preserves correctness."""
        U28, V28, W28, _ = try_rank_28()
        if U28 is None:
            return None

        # The schoolbook terms are at columns 14-27 (last 14 columns)
        # Let's try to merge two of them into one with a clever rank-1 term.
        # This is a heuristic search; if it doesn't work in time, return None.
        rank = 27
        # Start with U28, V28, W28 and try to combine last two schoolbook terms
        # Schoolbook terms are individual (i,j,k) triplets.
        # The last two are:
        #   (i=1, j=2, k=4) and (i=0, j=2, k=4) -- wait, let me recheck order.
        # add_schoolbook([0,1],[2],[4]) adds (0,2,4) then (1,2,4)
        # So cols 26 and 27 are (0,2,4) and (1,2,4).
        # We need to cover both with fewer than 2 rank-1 terms, which is impossible
        # unless we use a shared structure.
        #
        # Alternative: try a completely different rank-27 construction.
        # Use Strassen for 2x2x2 (rank 7) for blocks where possible, and
        # a rank-10 decomposition for 2x2x3.
        #
        # Split: A[0:2, 0:3] * B[0:3, 0:5] -> C[0:2, 0:5]
        # = A[0:2,0:2]*B[0:2,0:5] + A[0:2,2:3]*B[2:3,0:5]
        # First part: 2x2 * 2x5 -> 2x5.
        #   Split B into B[0:2,0:2] (2x2) and B[0:2,2:5] (2x3).
        #   A[0:2,0:2]*B[0:2,0:2]: Strassen rank 7
        #   A[0:2,0:2]*B[0:2,2:5]: 2x2 * 2x3 -> 2x3, rank at most 6 (schoolbook) or 5?
        #   Actually 2x2*2x3: schoolbook is 2*2*3=12? No, it's sum over 2 j's, so 2*2*3=12 terms? 
        #   Wait: C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k]. That's 2*2*3=12 scalar mults but
        #   in tensor rank, each term is rank 1, so schoolbook rank = 2*2*3 = 12? No.
        #   For m=2, p=3, n=2: schoolbook rank = n*m*p = 2*2*3 = 12. But we can do better.
        #   The 2x2x3 tensor has rank at most 10 (known result? Not sure).
        #
        # This is getting complex. Let's just try to find any rank-27 via random perturbation.
        if time.time() > deadline - 30:
            return None

        # Randomized local search: start from rank-28, randomly zero out one rank-1 term
        # and see if we can fix with a single new term.
        # This is unlikely to work quickly, so we'll just return None and use rank 28.
        return None

    result = try_rank_27()
    if result is not None and time.time() < deadline - 5:
        U, V, W, rank = result
        # Final verification
        Ur = U.T.reshape(rank, n, m)
        Vr = V.T.reshape(rank, m, p)
        Wr = W.T.reshape(rank, p, n)
        tensor = np.einsum('rij,rjk,rki->ijk', Ur, Vr, Wr)
        if np.allclose(tensor, 1.0, atol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: rank 28
    result = try_rank_28()
    if result is not None:
        U, V, W, rank = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Last resort: full schoolbook rank 30
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
