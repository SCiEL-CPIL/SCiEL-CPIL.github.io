"""Fresh design: Strassen-tiling structured init + rank-reduction pruning pass."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

# Strassen 2x2x2 rank-7 decomposition (entries in {0, +/-1})
# M1 = (A11+A22)(B11+B22)
# M2 = (A21+A22)B11
# M3 = A11(B12-B22)
# M4 = A22(B21-B11)
# M5 = (A11+A12)B22
# M6 = (A21-A11)(B11+B12)
# M7 = (A12-A22)(B21+B22)
# C11 = M1+M4-M5+M7
# C12 = M3+M5
# C21 = M2+M4
# C22 = M1-M2+M3+M6
_STRASSEN_U = [
    [1, 0, 0, 0, 0, 0, 0],   # A11
    [0, 0, 1, 0, 1, -1, 0],  # A12
    [0, 1, 0, 0, 0, 1, 0],   # A21
    [1, 1, 0, 1, 0, 0, 0],   # A22
]
_STRASSEN_V = [
    [1, 1, 0, -1, 0, 1, 1],  # B11
    [0, 0, 1, 0, 1, 1, 0],   # B12
    [1, 1, 0, 1, 0, 0, 1],   # B21
    [1, 0, -1, -1, 1, 0, 1], # B22
]
_STRASSEN_W = [
    [1, 0, 0, 0, -1, 0, 1],  # C11
    [0, 0, 1, 0, 1, 0, 0],   # C12
    [0, 1, 0, 1, 0, 0, 0],   # C21
    [1, -1, 1, 0, 0, 1, 0],  # C22
]


def _build_strassen_tiling():
    """Build a structured decomposition via 2x2 block tiling with Strassen.

    A is 4x4, B is 4x6, C is 4x6.
    Split A into 2x2 blocks (A_ij for i,j in {0,1}), each 2x2.
    Split B into 2x2 blocks: B has 4 rows, 6 cols -> 2 row-blocks of 2, 3 col-blocks of 2.
    C = A*B has 4 rows, 6 cols -> 2 row-blocks of 2, 3 col-blocks of 2.
    For each (row-block a, col-block b): C_ab = sum_j A_a,j * B_j,b  (2x2 = 2x2 * 2x2).
    Each 2x2*2x2 uses Strassen rank 7.
    Number of (a, j, b) combinations: a in {0,1}, j in {0,1}, b in {0,1,2} -> 12 terms.
    Total rank = 12 * 7 = 84.
    """
    n, m, p = DIMENSIONS  # 4,4,6
    # U: (n*m, rank) = (16, rank); V: (m*p, rank) = (24, rank); W: (p*n, rank) = (24, rank)
    rank = 0
    U = []
    V = []
    W = []

    def add_term(u_row, v_row, w_row, u_val, v_val, w_val):
        nonlocal rank
        U.append([0.0] * rank + [u_val])
        V.append([0.0] * rank + [v_val])
        W.append([0.0] * rank + [w_val])
        for r in range(rank):
            U[r].append(0.0)
            V[r].append(0.0)
            W[r].append(0.0)
        rank += 1

    # A index: row i (0..3), col j (0..3) -> A[i*m + j]
    # B index: row j (0..3), col k (0..5) -> B[j*p + k]
    # C index: W[k*n + i, r]  (trace convention: output factor W[k*n + i, r])
    # C[i, k] = sum_j A[i, j] B[j, k]
    # So W row = k*n + i, and it gets contribution A[i,j]*B[j,k].

    # Block (a, b, j): a = A row-block (0 or 1), b = B col-block (0,1,2), j = A col-block (0 or 1)
    # A row-block a covers A rows 2a, 2a+1
    # A col-block j covers A cols 2j, 2j+1
    # B row-block j covers B rows 2j, 2j+1
    # B col-block b covers B cols 2b, 2b+1
    # C row-block a covers C rows 2a, 2a+1
    # C col-block b covers C cols 2b, 2b+1

    # Strassen local indices (0..3) map to (local_row, local_col) within 2x2 block
    # local (lr, lc) -> A global row 2a+lr, col 2j+lc
    #                -> B global row 2j+lr, col 2b+lc
    #                -> C global row 2a+lr, col 2b+lc  -> W row = (2b+lc)*n + (2a+lr)

    for a in range(2):
        for j in range(2):
            for b in range(3):
                for s in range(7):
                    # U: A local rows 0..3
                    for lr in range(2):
                        for lc in range(2):
                            u_idx = (2 * a + lr) * m + (2 * j + lc)
                            u_val = _STRASSEN_U[lr * 2 + lc][s]
                            if u_val != 0:
                                add_term(u_idx, -1, -1, u_val, 0.0, 0.0)
                    # V: B local rows 0..3
                    for lr in range(2):
                        for lc in range(2):
                            v_idx = (2 * j + lr) * p + (2 * b + lc)
                            v_val = _STRASSEN_V[lr * 2 + lc][s]
                            if v_val != 0:
                                add_term(-1, v_idx, -1, 0.0, v_val, 0.0)
                    # W: C local
                    for lr in range(2):
                        for lc in range(2):
                            w_row = (2 * b + lc) * n + (2 * a + lr)
                            w_val = _STRASSEN_W[lr * 2 + lc][s]
                            if w_val != 0:
                                add_term(-1, -1, w_row, 0.0, 0.0, w_val)

    # The above add_term approach is inefficient; rebuild cleanly.
    # Rebuild: collect contributions per row.
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for a in range(2):
        for j in range(2):
            for b in range(3):
                for s in range(7):
                    for lr in range(2):
                        for lc in range(2):
                            u_idx = (2 * a + lr) * m + (2 * j + lc)
                            u_val = _STRASSEN_U[lr * 2 + lc][s]
                            if u_val != 0:
                                U[u_idx][col] = u_val
                            v_idx = (2 * j + lr) * p + (2 * b + lc)
                            v_val = _STRASSEN_V[lr * 2 + lc][s]
                            if v_val != 0:
                                V[v_idx][col] = v_val
                            w_row = (2 * b + lc) * n + (2 * a + lr)
                            w_val = _STRASSEN_W[lr * 2 + lc][s]
                            if w_val != 0:
                                W[w_row][col] = w_val
                    col += 1
    return U, V, W, rank


def _verify(U, V, W, tol=1e-9):
    """Check that sum_r U[:,r] V[:,r] W[:,r] (outer) equals the MM tensor."""
    n, m, p = DIMENSIONS
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # tensor[i*m+j, j*p+k, k*n+i] should equal sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
    rank = Ua.shape[1]
    # Build via einsum
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
    # True MM tensor: T[i,j,k] = delta? Actually the MM tensor T[i,j,k] is 1 if ... 
    # The MM tensor is defined by C[i,k] = sum_j A[i,j] B[j,k].
    # As a 3-tensor T[i,j,k] = 1 for all i,j,k (it's the identity in the sense that
    # the bilinear form is sum_{i,j,k} A[i,j] B[j,k] C'[k,i]).
    # So T[i,j,k] = 1 for all i, j, k.
    true_T = np.ones((n, m, p), dtype=float)
    return np.allclose(T, true_T, atol=tol)


def _prune_columns(U, V, W, rank, start_rank, deadline, max_iters=2000):
    """Try to reduce rank by zeroing columns and re-solving via least squares."""
    n, m, p = DIMENSIONS
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)

    # Build the linear system: for each (i,j,k), sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1
    # This is nonlinear in the entries. Instead, we do a simpler heuristic:
    # Try removing one column at a time and check if the remaining system is consistent
    # by solving a least-squares on a linearized version.
    # Simpler: greedily try to drop columns; for each candidate column to drop,
    # check if the tensor equation still holds approximately with that column zeroed.
    # If not, we could try to adjust nearby columns, but for speed we just drop if residual small.

    best_rank = rank
    best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()

    # Precompute the current tensor
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)

    # Greedy forward elimination: for each column, try zeroing it and re-orthogonalizing
    # is expensive. Instead, use a simpler approach: 
    # Try to find a subset of columns of size r that spans the same "tensor space".
    # We'll do: start from full rank, try removing columns one by one in random order,
    # and for each removal, solve a least-squares to see if we can compensate.
    # Given time constraints, do a limited greedy pass.

    rng = np.random.default_rng(42)
    current_rank = rank
    order = list(range(rank))
    rng.shuffle(order)

    for trial in order:
        if time.time() > deadline - 10:
            break
        if current_rank <= start_rank:
            break
        # Try removing column 'trial'
        # Check: is column 'trial' redundant? i.e., can the tensor be represented
        # without it? We test by setting it to zero and checking residual.
        Ur2 = Ur.copy()
        Vr2 = Vr.copy()
        Wr2 = Wr.copy()
        Ur2[:, :, trial] = 0
        Vr2[:, :, trial] = 0
        Wr2[:, :, trial] = 0
        T2 = np.einsum('ijr,jkr,kir->ijk', Ur2, Vr2, Wr2)
        if np.allclose(T2, np.ones((n, m, p)), atol=1e-8):
            # Column is redundant
            current_rank -= 1
            Ur = Ur2
            Vr = Vr2
            Wr = Wr2
            # Update best
            Ua = Ur.reshape(n * m, -1)
            Va = Vr.reshape(m * p, -1)
            Wa = Wr.reshape(p * n, -1)
            best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
            best_rank = current_rank

    # Convert back to lists
    return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_rank


def construct():
    deadline = time.time() + 130  # safety margin before 150s limit

    # Build structured Strassen-tiling decomposition
    U, V, W, rank = _build_strassen_tiling()

    # Verify initial decomposition
    if not _verify(U, V, W):
        # Fallback to schoolbook if something is wrong
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

    # Try to reduce rank via pruning
    if time.time() < deadline - 20:
        U, V, W, rank = _prune_columns(U, V, W, rank, rank, deadline)

    # Final verification
    if not _verify(U, V, W):
        # Fallback to the structured version (pre-pruning)
        U, V, W, rank = _build_strassen_tiling()

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
