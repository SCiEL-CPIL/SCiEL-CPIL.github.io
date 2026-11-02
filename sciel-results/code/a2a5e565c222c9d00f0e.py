import numpy as np
import time

DIMENSIONS = [2, 2, 9]
_START_TIME = time.time()
_TIMEOUT = 135  # safety margin before 150s kill


def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _rank_of_terms(U, V, W, indices, T, n, m, p):
    """
    Check if the terms at 'indices' span the tensor T.
    Returns True if T is in the span of the given rank-1 terms.
    """
    if not indices:
        return np.allclose(T, 0)
    
    # Build the matrix whose columns are the vectorized rank-1 tensors
    # T_recon = sum_r U[:,r] * V[:,r] * W[:,r]
    # Each term r contributes outer(U[:,r], V[:,r]) * W[:,r] reshaped
    # More precisely: term r contributes to T[i,j,k] the product U[i,r]*V[j,r]*W[k,r]
    # So the vectorized form is: vec(term_r) = U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    T_flat = T.flatten()
    
    # Number of terms
    r_count = len(indices)
    dim = len(T_flat)
    
    # Build matrix M where M[:, r] = vec(U[:,idx_r] ⊗ V[:,idx_r] ⊗ W[:,idx_r])
    M = np.zeros((dim, r_count), dtype=float)
    for idx, term_r in enumerate(indices):
        M[:, idx] = np.kron(np.kron(Ua[:, term_r], Va[:, term_r]), Wa[:, term_r])
    
    # Check if T_flat is in the column space of M
    # Use least squares: minimize ||M @ x - T_flat||
    # T is in span if residual is ~0
    try:
        x, residuals, rank, sv = np.linalg.lstsq(M, T_flat, rcond=1e-10)
        if len(residuals) > 0:
            return residuals[0] < 1e-8
        # If M is full column rank and lstsq didn't report residual, check directly
        recon = M @ x
        return np.allclose(recon, T_flat, atol=1e-8)
    except np.linalg.LinAlgError:
        return False


def _build_rank_32(n, m, p):
    """
    Build a rank-32 decomposition using Strassen for 4 blocks of 2x2x2
    and standard multiplication for 1 block of 2x2x1.
    """
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        # M1: (A00+A11)*(B00+B11)
        U[0][col] = 1
        U[3][col] = 1
        V[0*p + k0][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1

        # M2: (A10+A11)*B00
        U[2][col] = 1
        U[3][col] = 1
        V[0*p + k0][col] = 1
        W[k0*n + 1][col] = 1
        W[k1*n + 1][col] = -1
        col += 1

        # M3: A00*(B10-B00)
        U[0][col] = 1
        V[0*p + k1][col] = 1
        V[1*p + k1][col] = -1
        W[k1*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1

        # M4: A11*(B01-B11)
        U[3][col] = 1
        V[1*p + k0][col] = 1
        V[0*p + k0][col] = -1
        W[k0*n + 0][col] = 1
        W[k0*n + 1][col] = 1
        col += 1

        # M5: (A00+A01)*B11
        U[0][col] = 1
        U[1][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = -1
        W[k1*n + 0][col] = 1
        col += 1

        # M6: (A10-A00)*(B00+B01)
        U[2][col] = 1
        U[0][col] = -1
        V[0*p + k0][col] = 1
        V[0*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1

        # M7: (A01-A11)*(B10+B11)
        U[1][col] = 1
        U[3][col] = -1
        V[1*p + k0][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        col += 1

    def add_standard(k):
        nonlocal col
        # Standard 2x2x1 multiplication
        # C0 = A00*B0k + A01*B1k
        # C1 = A10*B0k + A11*B1k
        # Term 1: A00 * B0k -> C0
        U[0][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1
        
        # Term 2: A01 * B1k -> C0
        U[1][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1

        # Term 3: A10 * B0k -> C1
        U[2][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1

        # Term 4: A11 * B1k -> C1
        U[3][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1

    # 4 blocks of 2 columns using Strassen (rank 7 each)
    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    
    # 1 block of 1 column using standard (rank 4)
    add_standard(8)

    return U, V, W


def _validate(U, V, W, n, m, p):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)

    T = _build_tensor(n, m, p)
    T_recon = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
    return np.allclose(T, T_recon, atol=1e-10)


def _try_remove_terms(U, V, W, n, m, p):
    """
    Try to remove individual terms from the decomposition.
    Returns the best (lowest-rank) valid decomposition found.
    """
    T = _build_tensor(n, m, p)
    rank = len(U[0])
    
    # Try removing each single term
    best_indices = list(range(rank))
    
    for i in range(rank):
        if time.time() - _START_TIME > _TIMEOUT - 30:
            break
        indices = [j for j in range(rank) if j != i]
        if _rank_of_terms(U, V, W, indices, T, n, m, p):
            best_indices = indices
            break  # Found one removable term, stop at rank-1
    
    # If we found a removable term, try to remove another from the reduced set
    if len(best_indices) < rank:
        for i in range(len(best_indices)):
            if time.time() - _START_TIME > _TIMEOUT - 20:
                break
            remaining = [best_indices[j] for j in range(len(best_indices)) if j != i]
            if _rank_of_terms(U, V, W, remaining, T, n, m, p):
                best_indices = remaining
                break
    
    if len(best_indices) < rank:
        # Rebuild with only the remaining terms
        U_new = [[U[i][r] for r in best_indices] for i in range(n * m)]
        V_new = [[V[i][r] for r in best_indices] for i in range(m * p)]
        W_new = [[W[i][r] for r in best_indices] for i in range(p * n)]
        return U_new, V_new, W_new
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS

    U, V, W = _build_rank_32(n, m, p)

    # Try to reduce rank by removing redundant terms
    U, V, W = _try_remove_terms(U, V, W, n, m, p)

    return {"U": U, "V": V, "W": W}
