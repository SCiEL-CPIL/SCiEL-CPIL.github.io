import numpy as np
import time

DIMENSIONS = [2, 2, 9]
_START_TIME = time.time()
_TIMEOUT = 135  # safety margin before 150s kill


def _build_rank_32(n, m, p):
    """
    Build a rank-32 decomposition using Strassen for 4 blocks of 2x2x2
    and schoolbook for 1 block of 2x2x1.
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

    def add_trivial(k):
        nonlocal col
        U[0][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1

        U[1][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1

        U[2][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1

        U[3][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)

    return U, V, W


def _validate(U, V, W, n, m, p):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    T_recon = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
    return np.allclose(T, T_recon, atol=1e-10)

def _reduce_rank(U, V, W, n, m, p, max_iter=200):
    """
    Attempts to reduce the rank by finding linear dependencies among the terms.
    Returns a new (U, V, W) if a lower rank is found, else None.
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    rank = Ua.shape[1]
    if rank <= 1:
        return None

    # Target Tensor
    T_target = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
    T_flat = T_target.flatten()

    # Flatten the terms
    # Term r corresponds to vector v_r = (U[:,r] * V[:,r] * W[:,r]) flattened
    # Actually, the contribution is sum_r U[:,r] V[:,r] W[:,r]
    # We can view this as a linear combination of basis tensors.
    
    # Let's form a matrix M where columns are the flattened tensor contributions of each term.
    # Contribution of term r is outer product of U[:,r], V[:,r], W[:,r]
    # Flatten this into a vector of size (n*m * m*p * p*n)
    
    # This might be large: 4 * 18 * 18 = 1296 dimensions. Rank 32.
    # Matrix size 1296 x 32.
    
    terms = []
    for r in range(rank):
        u = Ua[:, r]
        v = Va[:, r]
        w = Wa[:, r]
        # Construct tensor T_r
        # T_r[i,j,k] = u[i] * v[j] * w[k]
        # Flatten it
        t_r = np.einsum('i,j,k->ijk', u, v, w).flatten()
        terms.append(t_r)
    
    Terms = np.column_stack(terms) # Shape (1296, 32)
    
    # We want to find a subset of columns that spans T_flat.
    # First, check if T_flat is in the span of Terms.
    # If not, the decomposition is invalid (should not happen).
    
    # Greedy selection:
    # 1. Start with all terms.
    # 2. Try to remove terms one by one if T_flat is still in the span of the remaining terms.
    #    To check if T_flat is in span of remaining terms, we can project T_flat onto the orthogonal complement of the remaining terms' span.
    #    Or simply, solve least squares for T_flat using the remaining terms and check residual.
    
    current_indices = list(range(rank))
    
    # To speed up, we can precompute the projection matrix or use SVD.
    # Let's try a simple loop: try removing each term, check validity.
    # Since rank is small (32), this is feasible.
    
    improved = False
    for _ in range(max_iter):
        if time.time() - _START_TIME > _TIMEOUT - 10:
            break
            
        removed = False
        for idx in range(len(current_indices)):
            # Try removing current_indices[idx]
            test_indices = current_indices[:idx] + current_indices[idx+1:]
            if not test_indices:
                continue
                
            # Check if T_flat is in span of Terms[:, test_indices]
            # Use np.linalg.lstsq
            M_sub = Terms[:, test_indices]
            try:
                coeffs, residuals, rank_sub, s = np.linalg.lstsq(M_sub, T_flat, rcond=None)
                recon = M_sub @ coeffs
                if np.allclose(recon, T_flat, atol=1e-9):
                    # We can remove this term
                    current_indices = test_indices
                    removed = True
                    improved = True
                    break # Restart loop with new set
            except Exception:
                pass
        
        if not removed:
            break
            
    if not improved:
        return None
        
    # Reconstruct U, V, W with reduced rank
    new_rank = len(current_indices)
    new_U = Ua[:, current_indices]
    new_V = Va[:, current_indices]
    new_W = Wa[:, current_indices]
    
    # Convert back to list of lists of floats
    return new_U.tolist(), new_V.tolist(), new_W.tolist()

def construct():
    n, m, p = DIMENSIONS

    U, V, W = _build_rank_32(n, m, p)

    # Try to reduce rank
    reduced = _reduce_rank(U, V, W, n, m, p)
    
    if reduced:
        U, V, W = reduced
        if _validate(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        else:
            # Fallback to original if validation fails
            U, V, W = _build_rank_32(n, m, p)
    else:
        # Validate original
        if not _validate(U, V, W, n, m, p):
             # Fallback to full schoolbook
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

    return {"U": U, "V": V, "W": W}
