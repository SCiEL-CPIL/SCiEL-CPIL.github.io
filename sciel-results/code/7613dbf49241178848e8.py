"""Rank-minimal exact tensor decomposition for 4x4 matrix multiplication.
Starts from Strassen (rank 56) and searches for rank reductions via
rank-1 term removal with exact correction."""

import time
import random
import numpy as np

DIMENSIONS = [4, 4, 4]

# W index convention: W[k*n + i, c] for the scalar multiply term (i,j,k)
# T[i*m+j, j*p+k, k*n+i] = 1  for all i,j,k in [0,4)
def _build_T(n, m, p):
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    return T

def _build_strassen(n, m, p):
    """Strassen 4x4 via 7 block-mults, each block mult via schoolbook 2x2.
    Returns U, V, W as lists of lists, rank 56."""
    assert n == 4 and m == 4 and p == 4
    # Strassen for 2x2 blocks (each block is 2x2):
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    #
    # For a 2x2 scalar block mult X*Y (X,Y 2x2), schoolbook uses 8 terms:
    #   (x00*y00, x00*y01, x01*y00, x01*y01,
    #    x10*y00, x10*y01, x11*y00, x11*y01)
    # producing (c00, c01, c10, c11).
    #
    # We treat each M_l as a "virtual" 2x2 matrix product of two 2x2 "sum" matrices.
    # Each sum matrix is a linear combination of A-blocks (or B-blocks).
    # We need to express each M_l = X_l * Y_l where X_l, Y_l are 2x2 matrices whose
    # entries are linear combos of A-entries / B-entries.
    #
    # A-blocks: A11 (rows 0-1, cols 0-1), A12 (rows 0-1, cols 2-3),
    #           A21 (rows 2-3, cols 0-1), A22 (rows 2-3, cols 2-3)
    # B-blocks: B11, B12, B21, B22 similarly.
    #
    # For M1 = (A11+A22)*(B11+B22):
    #   X1 = A11+A22 (a 2x2 matrix with entries A11[r][c]+A22[r][c])
    #   Y1 = B11+B22
    # For M2 = (A21+A22)*B11:
    #   X2 = A21+A22, Y2 = B11
    # For M3 = A11*(B12-B22):
    #   X3 = A11, Y3 = B12-B22
    # For M4 = A22*(B21-B11):
    #   X4 = A22, Y4 = B21-B11
    # For M5 = (A11+A12)*B22:
    #   X5 = A11+A12, Y5 = B22
    # For M6 = (A21-A11)*(B11+B12):
    #   X6 = A21-A11, Y6 = B11+B12
    # For M7 = (A12-A22)*(B21+B22):
    #   X7 = A12-A22, Y7 = B21+B22
    #
    # Each X_l is a 2x2 matrix whose (r,c) entry is a linear combo of A-entries.
    # Each Y_l is a 2x2 matrix whose (r,c) entry is a linear combo of B-entries.
    # The product X_l * Y_l is a 2x2 matrix whose (r,c) entry is a bilinear form.
    # Schoolbook on this 2x2 product uses 8 scalar multiplications.
    #
    # We build U, V, W for each of the 7*8 = 56 terms.
    # For a term that multiplies X_l[r1][c1] (an A-linear combo) by Y_l[r2][c2]
    # (a B-linear combo), contributing to C_l[r1][r2][c1][c2]... 
    # Actually, the schoolbook product of X (2x2) and Y (2x2) gives C = X*Y where
    # C[r][c] = sum_d X[r][d]*Y[d][c]. So the 8 terms are:
    #   C[0][0] = X[0][0]*Y[0][0] + X[0][1]*Y[1][0]
    #   C[0][1] = X[0][0]*Y[0][1] + X[0][1]*Y[1][1]
    #   C[1][0] = X[1][0]*Y[0][0] + X[1][1]*Y[1][0]
    #   C[1][1] = X[1][0]*Y[0][1] + X[1][1]*Y[1][1]
    #
    # Each scalar mult X[r][d]*Y[d][c] is one rank-1 term.
    # X[r][d] is a linear combo of A-entries; Y[d][c] is a linear combo of B-entries.
    # The term contributes to C_l[r][c], which is a 2x2 block of the final C.
    #
    # Now I need to map block coordinates to global 4x4 coordinates.
    # A11 is rows 0-1, cols 0-1. A12 is rows 0-1, cols 2-3.
    # A21 is rows 2-3, cols 0-1. A22 is rows 2-3, cols 2-3.
    # Similarly for B and C.
    #
    # C11 is rows 0-1, cols 0-1. C12 is rows 0-1, cols 2-3.
    # C21 is rows 2-3, cols 0-1. C22 is rows 2-3, cols 2-3.

    # Define the 7 (X, Y) pairs as 2x2 matrices of (block, sign) entries.
    # Each entry is (block_id, sign) where block_id in {11,12,21,22} and sign in {+1,-1}.
    # X entries are A-blocks, Y entries are B-blocks.
    #
    # M1: X = A11+A22, Y = B11+B22
    #   X = [[(11,+1),(11,+1)],[(22,+1),(22,+1)]]  -- wait, X is 2x2, each entry is one A-block
    #   Actually X = [[A11, A11],[A22, A22]]? No.
    #   X = A11 + A22 means X[r][c] = A11[r][c] + A22[r][c].
    #   So X is a 2x2 matrix where each entry (r,c) is the sum of two A-block entries.
    #   Similarly Y[r][c] = B11[r][c] + B22[r][c].
    #
    # Let me define each X_l as a 2x2 array of lists of (block, sign) pairs.
    # X_l[r][c] = list of (a_block, sign) meaning X_l[r][c] = sum(sign * A[a_block][r][c])
    # Similarly Y_l[r][c] = list of (b_block, sign)

    # M1: (A11+A22)*(B11+B22)
    X1 = [[[(11,1), (22,1)], [(11,1), (22,1)]],
          [[(11,1), (22,1)], [(11,1), (22,1)]]]
    # Wait, that's not right. A11+A22 is a 2x2 matrix where (r,c) entry = A11[r][c]+A22[r][c].
    # So X1[r][c] should be [(11,1), (22,1)] for all r,c. Yes.
    Y1 = [[[(11,1), (22,1)], [(11,1), (22,1)]],
          [[(11,1), (22,1)], [(11,1), (22,1)]]]

    # M2: (A21+A22)*B11
    X2 = [[[(21,1), (22,1)], [(21,1), (22,1)]],
          [[(21,1), (22,1)], [(21,1), (22,1)]]]
    Y2 = [[[(11,1)], [(11,1)]],
          [[(11,1)], [(11,1)]]]

    # M3: A11*(B12-B22)
    X3 = [[[(11,1)], [(11,1)]],
          [[(11,1)], [(11,1)]]]
    Y3 = [[[(12,1), (22,-1)], [(12,1), (22,-1)]],
          [[(12,1), (22,-1)], [(12,1), (22,-1)]]]

    # M4: A22*(B21-B11)
    X4 = [[[(22,1)], [(22,1)]],
          [[(22,1)], [(22,1)]]]
    Y4 = [[[(21,1), (11,-1)], [(21,1), (11,-1)]],
          [[(21,1), (11,-1)], [(21,1), (11,-1)]]]

    # M5: (A11+A12)*B22
    X5 = [[[(11,1), (12,1)], [(11,1), (12,1)]],
          [[(11,1), (12,1)], [(11,1), (12,1)]]]
    Y5 = [[[(22,1)], [(22,1)]],
          [[(22,1)], [(22,1)]]]

    # M6: (A21-A11)*(B11+B12)
    X6 = [[[(21,1), (11,-1)], [(21,1), (11,-1)]],
          [[(21,1), (11,-1)], [(21,1), (11,-1)]]]
    Y6 = [[[(11,1), (12,1)], [(11,1), (12,1)]],
          [[(11,1), (12,1)], [(11,1), (12,1)]]]

    # M7: (A12-A22)*(B21+B22)
    X7 = [[[(12,1), (22,-1)], [(12,1), (22,-1)]],
          [[(12,1), (22,-1)], [(12,1), (22,-1)]]]
    Y7 = [[[(21,1), (22,1)], [(21,1), (22,1)]],
          [[(21,1), (22,1)], [(21,1), (22,1)]]]

    # C-block assignments:
    # C11 = M1+M4-M5+M7  -> block (0,0)
    # C12 = M3+M5        -> block (0,1)
    # C21 = M2+M4        -> block (1,0)
    # C22 = M1-M2+M3+M6  -> block (1,1)
    #
    # For each M_l, it contributes to certain C-blocks with certain signs.
    # C11 gets: M1(+1), M4(+1), M5(-1), M7(+1)
    # C12 gets: M3(+1), M5(+1)
    # C21 gets: M2(+1), M4(+1)
    # C22 gets: M1(+1), M2(-1), M3(+1), M6(+1)

    # Map block (bi, bj) to global row/col offset:
    # block (0,0) -> rows 0-1, cols 0-1
    # block (0,1) -> rows 0-1, cols 2-3
    # block (1,0) -> rows 2-3, cols 0-1
    # block (1,1) -> rows 2-3, cols 2-3

    # For each M_l, for each of the 8 schoolbook terms, and for each C-block it contributes to:
    # The term X_l[r][d]*Y_l[d][c] contributes to C_l[r][c].
    # C_l is a 2x2 block matrix. C_l[r][c] is a 2x2 block of the final C.
    # So the contribution to final C is: for each (r,c) in the 2x2 of C_l,
    #   C_l[r][c] contributes sign * X_l[r][d]*Y_l[d][c] to C-block (bi,bj) at position (r,c).
    #
    # Global row = bi*2 + r, global col = bj*2 + c.
    #
    # U entry: for the A-part X_l[r][d] = sum(sign_a * A[a_block][r][d])
    #   Global A row = a_block_row_offset + r, col = a_block_col_offset + d
    #   A-block (11): rows 0-1, cols 0-1 -> offset (0,0)
    #   A-block (12): rows 0-1, cols 2-3 -> offset (0,2)
    #   A-block (21): rows 2-3, cols 0-1 -> offset (2,0)
    #   A-block (22): rows 2-3, cols 2-3 -> offset (2,2)
    #
    # V entry: for the B-part Y_l[d][c] = sum(sign_b * B[b_block][d][c])
    #   Global B row = b_block_row_offset + d, col = b_block_col_offset + c
    #   B-block (11): rows 0-1, cols 0-1 -> offset (0,0)
    #   B-block (12): rows 0-1, cols 2-3 -> offset (0,2)
    #   B-block (21): rows 2-3, cols 0-1 -> offset (2,0)
    #   B-block (22): rows 2-3, cols 2-3 -> offset (2,2)
    #
    # W entry: for the C-part, contribution to C-block (bi,bj) at (r,c) with sign_w
    #   Global C row = bi*2 + r, col = bj*2 + c
    #   W index = k*n + i where k = col, i = row (per convention W[k*n+i, r])
    #   So W index = (bi*2+c)*4 + (bi*2+r)

    block_offsets = {
        11: (0, 0),
        12: (0, 2),
        21: (2, 0),
        22: (2, 2),
    }

    # For each M_l, list of (c_block, sign) it contributes to
    M_contrib = {
        1: [((0,0), 1), ((1,1), 1)],
        2: [((1,0), 1), ((1,1), -1)],
        3: [((0,1), 1), ((1,1), 1)],
        4: [((0,0), 1), ((1,0), 1)],
        5: [((0,0), -1), ((0,1), 1)],
        6: [((1,1), 1)],
        7: [((0,0), 1)],
    }

    Ms = {1: (X1, Y1), 2: (X2, Y2), 3: (X3, Y3), 4: (X4, Y4),
          5: (X5, Y5), 6: (X6, Y6), 7: (X7, Y7)}

    rank = 7 * 8  # 56
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]

    col = 0
    for l in range(1, 8):
        Xl, Yl = Ms[l]
        for r in range(2):
            for d in range(2):
                for c in range(2):
                    # Term: X_l[r][d] * Y_l[d][c]
                    # U: A-part X_l[r][d]
                    for (a_block, sign_a) in Xl[r][d]:
                        a_row_off, a_col_off = block_offsets[a_block]
                        a_row = a_row_off + r
                        a_col = a_col_off + d
                        U[a_row * 4 + a_col][col] += sign_a
                    # V: B-part Y_l[d][c]
                    for (b_block, sign_b) in Yl[d][c]:
                        b_row_off, b_col_off = block_offsets[b_block]
                        b_row = b_row_off + d
                        b_col = b_col_off + c
                        V[b_row * 4 + b_col][col] += sign_b
                    # W: C-part
                    for (cbi, cbj), sign_w in M_contrib[l]:
                        w_row = cbi * 2 + r
                        w_col = cbj * 2 + c
                        # W index = k*n + i = w_col * 4 + w_row
                        W[w_col * 4 + w_row][col] += sign_w
                    col += 1

    return U, V, W


def _verify(U, V, W, T):
    """Verify that sum_c U[:,c] outer V[:,c] outer W[:,c] == T exactly."""
    n = T.shape
    U = np.array(U, dtype=np.int64)
    V = np.array(V, dtype=np.int64)
    W = np.array(W, dtype=np.int64)
    rank = U.shape[1]
    # Reconstruct tensor
    T_recon = np.zeros(n, dtype=np.int64)
    for c in range(rank):
        T_recon += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    return np.array_equal(T_recon, T)


def _try_remove_term(U, V, W, T, col_idx, deadline):
    """Try to remove term col_idx by re-solving with a free parameter.
    Returns (new_U, new_V, new_W, success)."""
    rank = U.shape[1]
    if rank <= 1:
        return None, None, None, False
    if time.time() > deadline:
        return None, None, None, False

    # Remove column col_idx
    mask = [i for i in range(rank) if i != col_idx]
    U_rem = U[:, mask]
    V_rem = V[:, mask]
    W_rem = W[:, mask]
    new_rank = len(mask)

    # Current reconstruction without the removed term
    T_cur = np.zeros(T.shape, dtype=np.int64)
    for c in range(new_rank):
        T_cur += np.einsum('i,j,k->ijk', U_rem[:, c], V_rem[:, c], W_rem[:, c])

    # Target: T_cur + T_removed = T
    # T_removed = T - T_cur
    T_removed = T - T_cur

    # We need to find a rank-1 tensor u outer v outer w such that
    # u outer v outer w = T_removed, where u, v, w are in the span of
    # the existing columns (to keep the structure) OR we allow a new
    # rank-1 term with free coefficients.
    #
    # Actually, the standard approach: we want to modify the remaining
    # terms to absorb the removed term. This is a linear system:
    # sum_{c in mask} (delta_U_c outer delta_V_c outer delta_W_c) = T_removed
    # where delta_U_c, delta_V_c, delta_W_c are small corrections.
    #
    # This is nonlinear. A simpler approach: check if T_removed is itself
    # a rank-1 tensor. If so, we can add it as a new term, but that doesn't
    # reduce rank. 
    #
    # Better: check if T_removed can be expressed as a linear combination
    # of rank-1 tensors formed by the existing U, V, W columns.
    # I.e., is T_removed in the span of {U_c outer V_c outer W_c : c in mask}?
    # If yes, we can adjust the coefficients. But the coefficients are fixed
    # to 1 in our decomposition... 
    #
    # The correct approach for rank reduction: we have rank-1 terms
    # {u_c outer v_c outer w_c}. We want to see if one of them is in the
    # span of the others. If u_{col} outer v_{col} outer w_{col} =
    # sum_{c in mask} alpha_c * u_c outer v_c outer w_c, then we can
    # absorb it. But this requires the tensors to be linearly dependent,
    # which is a strong condition.
    #
    # A more practical approach: treat the problem as a linear system.
    # Fix U and V, solve for W. Or fix V and W, solve for U.
    #
    # Let's try: fix U_rem and V_rem, solve for W_new such that
    # sum_c U_rem[:,c] outer V_rem[:,c] outer W_new[:,c] = T
    # This is a linear system in W_new.
    #
    # Flatten: for each (i,j,k): sum_c U[i,c]*V[j,c]*W[k,c] = T[i,j,k]
    # This is a linear system: A * w = b where A is (16*16*16) x (16*new_rank)
    # A[(i,j,k), (c, k')] = U[i,c]*V[j,c] if k==k' else 0
    # Actually, W is (16, new_rank), so the unknowns are W[k,c] for k in 0..15, c in 0..new_rank-1.
    # The equation for (i,j,k): sum_c U[i,c]*V[j,c]*W[k,c] = T[i,j,k]
    # This couples all k through the same c. So for each k, we have:
    # sum_c (U[i,c]*V[j,c]) * W[k,c] = T[i,j,k]
    # This is a linear system in W[k,:] for each (i,j), but the same W[k,c] appears
    # for all (i,j). So it's a big linear system.
    #
    # Let me set it up properly. Unknowns: W[k,c] for k in range(16), c in range(new_rank).
    # Total unknowns: 16 * new_rank.
    # Equations: for each (i,j,k): sum_c U[i,c]*V[j,c]*W[k,c] = T[i,j,k]
    # Total equations: 16*16*16 = 4096.
    #
    # The matrix A has shape (4096, 16*new_rank).
    # A[(i,j,k), (k',c)] = U[i,c]*V[j,c] if k==k' else 0.
    # So for a fixed k, the rows (i,j,k) form a 16*16 submatrix where
    # A[(i,j,k), (k,c)] = U[i,c]*V[j,c].
    #
    # This is a Kronecker structure. Let me just build it numerically.

    n_rows = 16 * 16 * 16
    n_cols = 16 * new_rank
    A = np.zeros((n_rows, n_cols), dtype=np.float64)
    b = np.zeros(n_rows, dtype=np.float64)

    for i in range(16):
        for j in range(16):
            for k in range(16):
                row_idx = i * 256 + j * 16 + k
                b[row_idx] = T[i, j, k]
                for c in range(new_rank):
                    col_idx_A = k * new_rank + c
                    A[row_idx, col_idx_A] = U_rem[i, c] * V_rem[j, c]

    # Solve A * w = b
    try:
        w, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None, None, None, False

    # Check if the solution is exact (integer or half-integer)
    w = w.reshape(16, new_rank)
    # Check exactness
    T_check = np.zeros(T.shape, dtype=np.float64)
    for c in range(new_rank):
        T_check += np.einsum('i,j,k->ijk', U_rem[:, c].astype(np.float64),
                              V_rem[:, c].astype(np.float64), w[:, c])
    if not np.allclose(T_check, T, atol=1e-9):
        return None, None, None, False

    # Check if entries are integer or half-integer
    w2 = w * 2
    if not np.allclose(w2, np.round(w2), atol=1e-9):
        return None, None, None, False

    # Convert to integer/half-integer
    W_new = np.round(w2).astype(np.int64) / 2.0
    # Verify exactly
    T_exact = np.zeros(T.shape, dtype=np.int64)
    for c in range(new_rank):
        T_exact += np.einsum('i,j,k->ijk', U_rem[:, c], V_rem[:, c],
                              np.round(w[:, c] * 2).astype(np.int64))
    if not np.array_equal(T_exact, T * 2):
        # Try with the original scale
        return None, None, None, False

    # Scale: we solved for W such that sum U*V*W = T.
    # If W has half-integer entries, that's fine.
    W_list = W_new.tolist()
    # Convert to proper integer/half-integer representation
    # The task says "integer/half-integer entries"
    # Let's keep them as floats that are multiples of 0.5
    W_out = [[round(x, 1) for x in row] for row in W_new]

    U_out = U_rem.tolist()
    V_out = V_rem.tolist()

    # Final exact verification
    U_a = np.array(U_out, dtype=np.int64)
    V_a = np.array(V_out, dtype=np.int64)
    W_a = np.array(W_out, dtype=np.int64) * 2  # scale by 2 to make integers
    T_recon = np.zeros(T.shape, dtype=np.int64)
    for c in range(new_rank):
        T_recon += np.einsum('i,j,k->ijk', U_a[:, c], V_a[:, c], W_a[:, c])
    if not np.array_equal(T_recon, T * 2):
        return None, None, None, False

    return U_out, V_out, W_out, True


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    n, m, p = DIMENSIONS
    T = _build_T(n, m, p)

    # Start from Strassen rank-56
    U, V, W = _build_strassen(n, m, p)
    best = (U, V, W)
    best_rank = 56

    if not _verify(U, V, W, T):
        # Fallback to schoolbook
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
        best = (U, V, W)
        best_rank = rank
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}

    # Try to reduce rank by removing terms
    U_a = np.array(U, dtype=np.int64)
    V_a = np.array(V, dtype=np.int64)
    W_a = np.array(W, dtype=np.int64)

    for attempt in range(100):
        if time.time() > deadline:
            break
        current_rank = U_a.shape[1]
        if current_rank <= 48:  # Don't go below reasonable
            break

        # Try removing a random term
        col_idx = random.randint(0, current_rank - 1)
        new_U, new_V, new_W, success = _try_remove_term(U_a, V_a, W_a, T, col_idx, deadline)
        if success:
            U_a = np.array(new_U, dtype=np.int64)
            V_a = np.array(new_V, dtype=np.int64)
            W_a = np.array(new_W, dtype=np.int64) * 2  # scale
            best = (new_U, new_V, new_W)
            best_rank = U_a.shape[1]
            # Continue trying to reduce further
            if time.time() > deadline:
                break

    # Return best result
    if best_rank < 56:
        U_out, V_out, W_out = best
        # Ensure W is in proper form (may be half-integer)
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        return {"U": U, "V": V, "W": W}
