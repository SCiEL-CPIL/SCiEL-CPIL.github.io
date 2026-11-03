"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill


def _elapsed():
    return time.time() - _START_TIME


def _time_left():
    return _TIME_LIMIT - _elapsed()


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition is exact (vectorized).
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for valid (i,j,k)
    And 0 for invalid triples.
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)

    R = U.shape[1]
    if V.shape[1] != R or W.shape[1] != R:
        return False

    # Vectorized: compute full tensor via einsum
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # Reshape U -> (n, m, R), V -> (m, p, R), W -> (p, n, R)
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)

    # T[i, j, k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)

    # Check valid entries (all i,j,k are valid since n,m,p are the dims)
    # Every (i,j,k) in range is a valid entry and should equal 1
    if np.any(np.abs(T - 1.0) > tol):
        return False
    return True


def _build_54_rank(n, m, p):
    """
    Build 2x5x6 decomposition using block Strassen. Rank 54.
    Split j into {0,1}, {2,3}, {4}. Split k into {0,1}, {2,3}, {4,5}.
    """
    rank = 54
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]

    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
            U[0 * m + j0, col] += 1
            U[1 * m + j1, col] += 1
            V[j0 * p + k0, col] += 1
            V[j1 * p + k1, col] += 1
            W[k0 * n + 0, col] += 1
            W[k1 * n + 1, col] += 1
            col += 1

            # P2 = (A_1,j0 + A_1,j1) B_j0,k0
            U[1 * m + j0, col] += 1
            U[1 * m + j1, col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + 1, col] += 1
            W[k1 * n + 1, col] -= 1
            col += 1

            # P3 = A_0,j0 (B_j0,k1 - B_j1,k1)
            U[0 * m + j0, col] += 1
            V[j0 * p + k1, col] += 1
            V[j1 * p + k1, col] -= 1
            W[k1 * n + 0, col] += 1
            W[k1 * n + 1, col] += 1
            col += 1

            # P4 = A_1,j1 (B_j1,k0 - B_j0,k0)
            U[1 * m + j1, col] += 1
            V[j1 * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + 0, col] += 1
            W[k0 * n + 1, col] += 1
            col += 1

            # P5 = (A_0,j0 + A_0,j1) B_j1,k1
            U[0 * m + j0, col] += 1
            U[0 * m + j1, col] += 1
            V[j1 * p + k1, col] += 1
            W[k0 * n + 0, col] -= 1
            W[k1 * n + 0, col] += 1
            col += 1

            # P6 = (A_1,j0 - A_0,j0)(B_j0,k0 + B_j0,k1)
            U[1 * m + j0, col] += 1
            U[0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + k1, col] += 1
            W[k1 * n + 1, col] += 1
            col += 1

            # P7 = (A_0,j1 - A_1,j1)(B_j1,k0 + B_j1,k1)
            U[0 * m + j1, col] += 1
            U[1 * m + j1, col] -= 1
            V[j1 * p + k0, col] += 1
            V[j1 * p + k1, col] += 1
            W[k0 * n + 0, col] += 1
            col += 1

    # Remaining j=4
    for k in range(p):
        U[0 * m + 4, col] += 1
        V[4 * p + k, col] += 1
        W[k * n + 0, col] += 1
        col += 1

        U[1 * m + 4, col] += 1
        V[4 * p + k, col] += 1
        W[k * n + 1, col] += 1
        col += 1

    assert col == rank, f"Expected col={rank}, got {col}"
    return U, V, W


def _try_eliminate_term(U, V, W, n, m, p, term_idx, tol=1e-9):
    """
    Try to prove term `term_idx` is redundant: can it be expressed as a
    linear combination of the other terms?
    Build a small linear system: find coefficients c_r such that
      sum_{r != term_idx} c_r * (U[:,r] * V[:,r] * W[:,r]) = U[:,term_idx] * V[:,term_idx] * W[:,term_idx]
    where the product is taken over the full index space.
    """
    R = U.shape[1]
    if R <= 1:
        return None

    nm, mp, pn = n * m, m * p, p * n

    # The "target" tensor for the term we want to eliminate
    # target[i,j,k] = U[i*m+j, term_idx] * V[j*p+k, term_idx] * W[k*n+i, term_idx]
    # We want: target = sum_{r != term_idx} c_r * T_r  where T_r[i,j,k] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]

    # Build the system efficiently using vectorized tensor operations.
    # Flatten the full tensor space: index = i*(m*p) + j*p + k
    total = n * m * p

    # Build T_r for all r != term_idx as columns of a matrix (total, R-1)
    # And target as a vector (total,)
    # T_r[i,j,k] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]

    # Precompute reshaped
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)

    # Full tensor for all r: (n, m, p, R)
    T_all = np.einsum('ijr,jkr,kir->ijk r', U3, V3, W3)  # (n,m,p,R)
    T_all = T_all.reshape(total, R)

    target = T_all[:, term_idx]

    # Build coefficient matrix: columns r != term_idx
    others = [r for r in range(R) if r != term_idx]
    A = T_all[:, others]  # (total, R-1)
    b = target  # (total,)

    # Solve A c = b
    try:
        c, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-10)
    except Exception:
        return None

    # Check residual
    if len(residuals) > 0 and residuals[0] > tol:
        return None

    # Verify: reconstruct target from c
    recon = A @ c
    if np.max(np.abs(recon - b)) > tol:
        return None

    # Build the new decomposition with term_idx removed
    # New U = U[:, others], etc.
    new_U = U[:, others]
    new_V = V[:, others]
    new_W = W[:, others]

    # Verify the full decomposition still holds
    if not _verify_decomposition(new_U, new_V, new_W, n, m, p, tol):
        return None

    return new_U, new_V, new_W


def _greedy_elimination(U, V, W, n, m, p, tol=1e-9):
    """
    Iteratively try to eliminate terms until no more can be eliminated.
    """
    R = U.shape[1]
    eliminated = set()

    while True:
        if _time_left() < 5:
            break

        found = False
        for term_idx in range(R):
            if term_idx in eliminated:
                continue
            if _time_left() < 3:
                break

            result = _try_eliminate_term(U, V, W, n, m, p, term_idx, tol)
            if result is not None:
                new_U, new_V, new_W = result
                # Remove term_idx from the matrices
                mask = [r for r in range(R) if r != term_idx]
                U = U[:, mask]
                V = V[:, mask]
                W = W[:, mask]
                R = U.shape[1]
                eliminated.discard(term_idx)
                found = True
                break  # Restart the loop with the new R

        if not found:
            break

    return U, V, W


def _snap_to_half_integer(x, tol=1e-6):
    """Snap a value to the nearest half-integer if close."""
    half = round(x * 2) / 2.0
    if abs(x - half) < tol:
        return half
    return x


def _snap_matrix(M, tol=1e-6):
    """Snap all entries of matrix to nearest half-integer if close."""
    return np.array([[_snap_to_half_integer(x) for x in row] for row in M])


def construct():
    n, m, p = DIMENSIONS

    # Start with the known rank 54 solution
    base_U, base_V, base_W = _build_54_rank(n, m, p)

    # Verify base solution
    if not _verify_decomposition(base_U, base_V, base_W, n, m, p):
        # Fallback to schoolbook if verification fails
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

    best_rank = base_U.shape[1]
    best_U, best_V, best_W = base_U.copy(), base_V.copy(), base_W.copy()

    # Try greedy term elimination
    if _time_left() > 10:
        elim_U, elim_V, elim_W = _greedy_elimination(base_U.copy(), base_V.copy(), base_W.copy(), n, m, p)
        if elim_U.shape[1] < best_rank:
            best_rank = elim_U.shape[1]
            best_U, best_V, best_W = elim_U, elim_V, elim_W

    # Snap to half-integers if possible
    U_snap = _snap_matrix(best_U)
    V_snap = _snap_matrix(best_V)
    W_snap = _snap_matrix(best_W)

    if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
