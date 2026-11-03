"""Fresh design: Strassen block decomposition + time-budgeted rank reduction."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 5, 5]

DEADLINE = 150.0
MARGIN = 8.0          # safety margin before the 150s kill
_SEED = 20240614


def _mul(U, V, W, n, m, p):
    """Reconstruct the product tensor T[i,k,j] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    T = np.zeros((n, p, m))
    for c in range(U.shape[1]):
        T += np.einsum('ij,jk,ki->ijk', U[:, c], V[:, c], W[:, c])
    return T


def _strassen(U, V, W, n, m, p, si, sj, ii, jj):
    """Build a Strassen-style decomposition for a subblock.
    si, sj = row splits of the first two factor dimensions (n and m).
    ii, jj = column splits (m and p).
    Returns (U, V, W) as dense float arrays (rows x rank).
    """
    U = []
    V = []
    W = []
    for (a, b, c, d) in [(1, 1, 1, 1), (1, -1, 1, 1), (1, 1, -1, 1),
                          (1, 1, 1, -1), (1, -1, 1, -1), (1, 1, -1, -1),
                          (-1, 1, 1, 1)]:
        U.append(np.outer(np.outer(np.ones(si), a * np.ones(ii)), np.ones(si)))
        V.append(np.outer(np.outer(np.ones(sj), b * np.ones(jj)), np.ones(sj)))
        W.append(np.outer(np.outer(np.ones(ii), c * np.ones(ii)), d * np.ones(jj)))
    # Each column: U rows = si*ii, V rows = sj*jj, W rows = ii*jj
    # Reshape each into (rows, 1) then stack along columns
    U = np.concatenate([x.reshape(-1, 1) for x in U], axis=1)
    V = np.concatenate([x.reshape(-1, 1) for x in V], axis=1)
    W = np.concatenate([x.reshape(-1, 1) for x in W], axis=1)
    return U, V, W


def _schoolbook_block(n, m, p):
    """Rank n*m*p schoolbook for a block."""
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _build_initial():
    """Construct the full 3x5x5 decomposition by splitting into blocks.
    First factor A (3x5): rows split 2|1, cols split 2|3.
    Second factor B (5x5): rows split 2|3, cols split 2|3.
    Block (2,2)x(2,2) -> Strassen (rank 8).
    Block (2,2)x(3,3) -> Strassen with si=2,sj=2,ii=2,jj=3 (rank 8).
    Block (1,2)x(2,2) -> Strassen with si=1,sj=2,ii=1,jj=2 (rank 8).
    Block (1,2)x(3,3) -> Strassen with si=1,sj=2,ii=1,jj=3 (rank 8).
    Total rank = 32.
    """
    n, m, p = 3, 5, 5
    # Define the four blocks: (i_start, i_size, j_start, j_size) for A, and for B
    # A rows: i in [0,2) size 2, i in [2,3) size 1
    # A cols: j in [0,2) size 2, j in [2,5) size 3
    # B rows: j in [0,2) size 2, j in [2,5) size 3
    # B cols: k in [0,2) size 2, k in [2,5) size 3
    blocks = [
        # (i0, si, j0, sj, k0, ii, jj)
        (0, 2, 0, 2, 0, 2, 2),   # (2x2) x (2x2)
        (0, 2, 0, 2, 2, 2, 3),   # (2x2) x (3x3)  -> A rows 0:2, cols 0:2 ; B rows 0:2, cols 2:5
        (2, 1, 0, 2, 0, 1, 2),   # (1x2) x (2x2)
        (2, 1, 0, 2, 2, 1, 3),   # (1x2) x (3x3)
    ]
    # Wait: I need to reconsider. The block (i0,si) x (j0,sj) of A multiplies
    # block (j0,sj) x (k0,jj) of B. So for each A-block (i0,si,j0,sj),
    # the B-block is (j0,sj,k0,jj). Let me restructure.
    # A blocks: (i0, si, j0, sj)
    # For each, B blocks: (j0, sj, k0, jj)
    # So I need to iterate over A-blocks and B-blocks independently.
    
    # A row splits: [0,2), [2,3)
    # A col splits: [0,2), [2,5)
    # B row splits: [0,2), [2,5)
    # B col splits: [0,2), [2,5)
    
    a_rows = [(0, 2), (2, 1)]
    a_cols = [(0, 2), (2, 3)]
    b_cols = [(0, 2), (2, 3)]
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    for (i0, si) in a_rows:
        for (j0, sj) in a_cols:
            for (k0, jj) in b_cols:
                # A-block: rows i0:i0+si, cols j0:j0+sj
                # B-block: rows j0:j0+sj, cols k0:k0+jj
                Ub, Vb, Wb = _strassen(None, None, None, si, sj, sj, jj)
                # Ub has shape (si*sj, 8), Vb has shape (sj*jj, 8), Wb has shape (jj*si, 8)
                # Place into global positions
                U_parts.append((i0 * m + j0, si * sj, Ub))
                V_parts.append((j0 * p + k0, sj * jj, Vb))
                W_parts.append((k0 * n + i0, jj * si, Wb))
    
    total_rank = sum(8 for _ in range(len(a_rows) * len(a_cols) * len(b_cols)))
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for (i0, si) in a_rows:
        for (j0, sj) in a_cols:
            for (k0, jj) in b_cols:
                Ub, Vb, Wb = _strassen(None, None, None, si, sj, sj, jj)
                U[i0 * m + j0: i0 * m + j0 + si * sj, col:col+8] = Ub
                V[j0 * p + k0: j0 * p + k0 + sj * jj, col:col+8] = Vb
                W[k0 * n + i0: k0 * n + i0 + jj * si, col:col+8] = Wb
                col += 8
    
    return U, V, W, total_rank


def _to_list(U, V, W):
    """Convert numpy arrays to list-of-lists."""
    def conv(A):
        return [[float(x) for x in row] for row in A]
    return {"U": conv(U), "V": conv(V), "W": conv(W)}


def _try_reduce(U, V, W, n, m, p, deadline):
    """Try to reduce rank by dropping columns and doing local swaps.
    Returns best (U, V, W) found."""
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    rng = np.random.RandomState(_SEED)
    
    # First: try dropping each column (greedy, keep best)
    for c in range(U.shape[1]):
        if time.time() > deadline - MARGIN:
            break
        # Check if column c is redundant
        # A column is redundant if its contribution is zero in the tensor
        # i.e., U[:,c] * V[:,c] * W[:,c] == 0 tensor
        # More precisely, if the outer product U[:,c] ⊗ V[:,c] ⊗ W[:,c] is in the span
        # of the other columns. For now, just check if the column is all zeros
        # or if removing it doesn't change the tensor (expensive check)
        pass
    
    # Better approach: iterative rank reduction via random linear combinations
    # Try to find a column that can be expressed as a linear combination of others
    # This is expensive. Instead, use a simpler heuristic:
    # Try removing each column and check if the tensor is preserved (within tolerance)
    
    for c in range(U.shape[1]):
        if time.time() > deadline - MARGIN:
            break
        U_test = np.delete(U, c, axis=1)
        V_test = np.delete(V, c, axis=1)
        W_test = np.delete(W, c, axis=1)
        # Check if the removed column's contribution is zero
        contrib = np.einsum('ij,jk,ki->ijk', U[:, c], V[:, c], W[:, c])
        if np.max(np.abs(contrib)) < 1e-10:
            U, V, W = U_test, V_test, W_test
            best_rank = U.shape[1]
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Now try local swaps: for each pair of columns, try to replace one with a
    # linear combination that reduces rank. This is complex, so we'll do a simpler
    # random search: try random linear combinations of two columns and see if
    # the tensor is preserved.
    
    # Actually, let's try a more systematic approach: for each column, try to
    # express it as a linear combination of two other columns.
    for c in range(U.shape[1]):
        if time.time() > deadline - MARGIN:
            break
        for c1, c2 in itertools.combinations(range(U.shape[1]), 2):
            if c1 == c or c2 == c:
                continue
            # Try to find alpha, beta such that U[:,c] = alpha*U[:,c1] + beta*U[:,c2]
            # and V[:,c] = alpha*V[:,c1] + beta*V[:,c2] and W[:,c] = alpha*W[:,c1] + beta*W[:,c2]
            # This is a system of linear equations. Solve for alpha, beta.
            # Use least squares on the combined system
            A_mat = np.column_stack([
                np.concatenate([U[:, c1], V[:, c1], W[:, c1]]),
                np.concatenate([U[:, c2], V[:, c2], W[:, c2]])
            ])
            b_vec = np.concatenate([U[:, c], V[:, c], W[:, c]])
            try:
                sol, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                if residuals.size > 0 and residuals[0] < 1e-8:
                    # Column c is in the span of c1 and c2
                    # Remove column c
                    U = np.delete(U, c, axis=1)
                    V = np.delete(V, c, axis=1)
                    W = np.delete(W, c, axis=1)
                    best_rank = U.shape[1]
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break  # Restart the loop since indices changed
            except np.linalg.LinAlgError:
                pass
    
    return best_U, best_V, best_W, best_rank


def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x5x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    deadline = time.time() + DEADLINE - MARGIN
    
    # Build initial Strassen-based decomposition
    U, V, W, rank = _build_initial()
    
    # Verify initial decomposition
    T_target = np.zeros((n, p, m))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i, k, j] = 1
    
    T_initial = _mul(U, V, W, n, m, p)
    if not np.allclose(T_initial, T_target, atol=1e-10):
        # Fallback to schoolbook
        U, V, W = _schoolbook_block(n, m, p)
        rank = n * m * p
    
    # Try to reduce rank
    if time.time() < deadline:
        U, V, W, rank = _try_reduce(U, V, W, n, m, p, deadline)
    
    # Convert to list-of-lists
    return _to_list(U, V, W)
