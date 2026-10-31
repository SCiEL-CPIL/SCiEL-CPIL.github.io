import time
import numpy as np

DIMENSIONS = [4, 4, 4]


def _build_strassen_49():
    """Build the rank-49 Strassen decomposition for 4x4x4."""
    n, m, p = DIMENSIONS

    # 2x2x2 Strassen decomposition (rank 7)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))

    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1

    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1

    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1

    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1

    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    rank_full = 49
    U = np.zeros((16, rank_full))
    V = np.zeros((16, rank_full))
    W = np.zeros((16, rank_full))

    beta = np.zeros((7, 2, 2))
    gamma = np.zeros((7, 2, 2))
    delta = np.zeros((7, 2, 2))

    beta[0, 0, 0] = 1; beta[0, 1, 1] = 1
    gamma[0, 0, 0] = 1; gamma[0, 1, 1] = 1
    delta[0, 0, 0] = 1; delta[0, 1, 1] = 1

    beta[1, 1, 0] = 1; beta[1, 1, 1] = 1
    gamma[1, 0, 0] = 1
    delta[1, 1, 0] = 1; delta[1, 1, 1] = -1

    beta[2, 0, 0] = 1
    gamma[2, 0, 1] = 1; gamma[2, 1, 1] = -1
    delta[2, 0, 1] = 1; delta[2, 1, 1] = 1

    beta[3, 1, 1] = 1
    gamma[3, 1, 0] = 1; gamma[3, 0, 0] = -1
    delta[3, 0, 0] = 1; delta[3, 1, 0] = 1

    beta[4, 0, 0] = 1; beta[4, 0, 1] = 1
    gamma[4, 1, 1] = 1
    delta[4, 0, 0] = -1; delta[4, 0, 1] = 1

    beta[5, 1, 0] = 1; beta[5, 0, 0] = -1
    gamma[5, 0, 0] = 1; gamma[5, 0, 1] = 1
    delta[5, 1, 1] = 1

    beta[6, 0, 1] = 1; beta[6, 1, 1] = -1
    gamma[6, 1, 0] = 1; gamma[6, 1, 1] = 1
    delta[6, 0, 0] = 1

    for s in range(7):
        for t in range(7):
            c = 7 * s + t
            for i in range(4):
                for j in range(4):
                    I = i // 2; i_prime = i % 2
                    J = j // 2; j_prime = j % 2
                    row_u = i * 4 + j
                    coeff = beta[s, I, J] * U2[i_prime * 2 + j_prime, t]
                    if coeff != 0:
                        U[row_u, c] = coeff
            for j in range(4):
                for k in range(4):
                    J = j // 2; j_prime = j % 2
                    K = k // 2; k_prime = k % 2
                    row_v = j * 4 + k
                    coeff = gamma[s, J, K] * V2[j_prime * 2 + k_prime, t]
                    if coeff != 0:
                        V[row_v, c] = coeff
            for k in range(4):
                for i in range(4):
                    K = k // 2; k_prime = k % 2
                    I = i // 2; i_prime = i % 2
                    row_w = k * 4 + i
                    coeff = delta[s, I, K] * W2[k_prime * 2 + i_prime, t]
                    if coeff != 0:
                        W[row_w, c] = coeff
    return U, V, W


def _build_target_tensor():
    """Build the 4x4x4 matrix multiplication tensor."""
    T = np.zeros((16, 16, 16))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                T[i * 4 + j, j * 4 + k, k * 4 + i] = 1.0
    return T


def _reconstruct(U, V, W):
    """Reconstruct tensor from factor matrices: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]."""
    R = U.shape[1]
    T = np.zeros((16, 16, 16))
    for r in range(R):
        T += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    return T


def _verify(U, V, W, T, tol=1e-9):
    """Check exact decomposition."""
    R = U.shape[1]
    for r in range(R):
        pass
    # Efficient check: compute full reconstruction and compare
    recon = _reconstruct(U, V, W)
    return np.max(np.abs(recon - T)) < tol


def _greedy_rank_reduce(U, V, W, T, tol=1e-9, deadline=None):
    """
    Try to reduce rank by finding linearly dependent rank-1 terms.
    A rank-1 term r is redundant if its contribution can be expressed as a
    linear combination of the others. We check this by seeing if removing
    column r still allows the remaining columns to span the target tensor.
    
    Strategy: For each column, check if the tensor it contributes is in the
    span of the other contributions. If so, remove it.
    """
    R = U.shape[1]
    cols = list(range(R))
    changed = True
    while changed and (deadline is None or time.time() < deadline - 10.0):
        changed = False
        for idx in range(len(cols)):
            r = cols[idx]
            # Try removing column r
            remaining = [c for c in cols if c != r]
            if len(remaining) == 0:
                continue
            # Check if the contribution of column r is in the span of others
            # The contribution of column r is the tensor outer(U[:,r], V[:,r], W[:,r])
            # We need to check if this tensor is in the span of the other tensors.
            # This is equivalent to checking if the vector [U[:,r] ⊗ V[:,r] ⊗ W[:,r]]
            # is in the span of [U[:,c] ⊗ V[:,c] ⊗ W[:,c] for c in remaining]
            # But that's a 4096-dim check which is expensive.
            # 
            # Alternative: check if we can find coefficients alpha_c such that
            # U[:,r] = sum_c alpha_c * U[:,c], V[:,r] = sum_c alpha_c * V[:,c], 
            # W[:,r] = sum_c alpha_c * W[:,c] simultaneously.
            # This is too restrictive.
            #
            # Better: directly check if the tensor contribution can be matched.
            # contrib_r = outer(U[:,r], V[:,r], W[:,r])
            # We need sum_c beta_c * outer(U[:,c], V[:,c], W[:,c]) = contrib_r
            # This is a linear system in beta.
            # Flatten: 4096 equations, len(remaining) unknowns.
            # For len(remaining) < 4096, this is overdetermined.
            # We can check if the system is consistent.
            
            # Actually, a simpler sufficient condition: if column r is a linear
            # combination of the others in ALL THREE matrices simultaneously,
            # then its tensor contribution is in the span.
            # Check: is U[:,r] in span of {U[:,c]} AND V[:,r] in span of {V[:,c]}
            # AND W[:,r] in span of {W[:,c]}?
            # If yes with the SAME coefficients, then we can remove it.
            
            # Let's check if the three vectors are in the spans
            # Build the matrix of other columns
            U_rem = U[:, remaining]  # 16 x len(remaining)
            V_rem = V[:, remaining]
            W_rem = W[:, remaining]
            
            # Check if U[:,r] is in column space of U_rem
            # Use least squares
            try:
                coeff_u, _, rank_u, _ = np.linalg.lstsq(U_rem, U[:, r], rcond=None)
                if np.linalg.norm(U_rem @ coeff_u - U[:, r]) > tol:
                    continue
                coeff_v, _, rank_v, _ = np.linalg.lstsq(V_rem, V[:, r], rcond=None)
                if np.linalg.norm(V_rem @ coeff_v - V[:, r]) > tol:
                    continue
                coeff_w, _, rank_w, _ = np.linalg.lstsq(W_rem, W[:, r], rcond=None)
                if np.linalg.norm(W_rem @ coeff_w - W[:, r]) > tol:
                    continue
                
                # Now check if the SAME coefficients work for the tensor
                # i.e., if sum_c coeff_u[c] * outer(U[:,c],V[:,c],W[:,c]) == outer(U[:,r],V[:,r],W[:,r])
                # This requires coeff_u == coeff_v == coeff_w (approximately)
                # Actually no - we need the tensor contributions to match, not the individual vectors.
                # Let's just verify by reconstruction.
                
                # Remove column r and verify
                U_new = U[:, remaining]
                V_new = V[:, remaining]
                W_new = W[:, remaining]
                if _verify(U_new, V_new, W_new, T, tol):
                    U = U_new
                    V = V_new
                    W = W_new
                    cols = remaining
                    changed = True
                    break
            except Exception:
                continue
    return U, V, W


def _randomized_rank_search(U, V, W, T, deadline):
    """
    Try to find a lower-rank decomposition by randomly perturbing the
    coefficient space and checking if we can reduce rank.
    """
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # First, do greedy reduction
    U, V, W = _greedy_rank_reduce(U, V, W, T, deadline=deadline)
    if U.shape[1] < best_rank and _verify(U, V, W, T):
        best_U, best_V, best_W = U, V, W
        best_rank = U.shape[1]
    
    return best_U, best_V, best_W, best_rank


def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    n, m, p = DIMENSIONS

    # Build the rank-49 Strassen decomposition
    U, V, W = _build_strassen_49()
    T = _build_target_tensor()

    # Verify the initial decomposition
    if not _verify(U, V, W, T):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank
    U, V, W = _greedy_rank_reduce(U, V, W, T, deadline=deadline)

    if time.time() > deadline - 20.0:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
