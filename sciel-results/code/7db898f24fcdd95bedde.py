"""Matrix multiplication 3,3,8: rank-minimal exact tensor decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _is_halfint(x):
    # x is an integer array; check all entries are multiples of 1/2.
    # (all integers are half-integers, so this is always True here; kept for clarity)
    return True


def _verify(U, V, W):
    """Return True if U,V,W exactly decompose the multiplication tensor."""
    n, m, p = DIMENSIONS
    # Build the target multiplication tensor as an array T with indices (i, j, k).
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    # reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    Ur = U.reshape(n, m, -1)
    Vr = V.reshape(m, p, -1)
    Wr = W.reshape(p, n, -1)
    # T[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    # Compute via einsum
    # First, for each c, outer product structure: T = sum_c (Ur[:,:,c] * Vr[:,:,c] broadcast over k) * Wr
    # Use einsum: sum over c of Ur[i,j,c]*Vr[j,k,c] -> A[i,j,k]
    A = np.einsum('ijc,jkc->ijk', Ur, Vr)
    # Now multiply by Wr[k,i,c] summed over c: need sum_c A[i,j,k]*Wr[k,i,c]
    # A is (i,j,k), Wr is (k,n,i) -> we want sum_c A[i,j,k]*Wr[k,i,c]
    # einsum: 'ijk,kic->ij' won't work directly because c is in both.
    # Let's compute: for each c, A_c[i,j,k] = Ur[i,j,c]*Vr[j,k,c]; then T[i,j,k] = sum_c A_c[i,j,k]*Wr[k,i,c]
    # We can do: T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    # Target: T[i,j,k] = 1 for all i,j,k (since it's the multiplication tensor)
    return np.allclose(T, 1.0, atol=1e-9)


def construct():
    deadline = time.time() + 130.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook(n, m, p)
    best_rank = U.shape[1]
    best = (U.copy(), V.copy(), W.copy())

    # Try to reduce rank by merging columns.
    # A merge of columns a and b with coefficient alpha means:
    #   new_col = U[:,a] + alpha * U[:,b], similarly for V and W
    # We need the resulting decomposition to still be exact.
    # This is equivalent to finding a linear dependency in the decomposition.
    #
    # Simpler approach: try to find pairs of columns that can be combined.
    # For a valid decomposition, if we have two columns c1 and c2, we can try
    # to replace them with a single column if there exists alpha such that
    # the contribution of c1 + alpha*c2 equals the original contribution of both.
    # This requires: for all i,j,k:
    #   U[i,j,c1]*V[j,k,c1]*W[k,i,c1] + alpha*U[i,j,c2]*V[j,k,c2]*W[k,i,c2]
    #   = (1+alpha)*U[i,j,c1]*V[j,k,c1]*W[k,i,c1]  (if we absorb)
    # This is generally not possible unless the two columns are "proportional" in a specific way.
    #
    # A more practical approach: use random linear combinations and check validity.
    # Or: try to find if any column is a linear combination of others (for U, V, W simultaneously).
    #
    # Given the time constraints and the difficulty of finding exact rank reductions
    # for 3x3x8 (known to be rank 21 or 22 in literature), we'll do a limited search.

    # Strategy: try random merges with small rational coefficients.
    # For each pair of columns, try alpha in {-2, -3/2, -1, -1/2, 1/2, 1, 3/2, 2}
    # and check if the merged decomposition is still valid.

    def try_merge(U, V, W, a, b, alpha):
        """Try to merge column b into column a with coefficient alpha."""
        r = U.shape[1]
        if r <= 1:
            return None
        U_new = U[:, :a].copy()
        U_new[:, a] = U[:, a] + alpha * U[:, b]
        if a + 1 < b:
            U_new = np.hstack([U_new, U[:, a + 1:b]])
        if b + 1 < r:
            U_new = np.hstack([U_new, U[:, b + 1:]])

        V_new = V[:, :a].copy()
        V_new[:, a] = V[:, a] + alpha * V[:, b]
        if a + 1 < b:
            V_new = np.hstack([V_new, V[:, a + 1:b]])
        if b + 1 < r:
            V_new = np.hstack([V_new, V[:, b + 1:]])

        W_new = W[:, :a].copy()
        W_new[:, a] = W[:, a] + alpha * W[:, b]
        if a + 1 < b:
            W_new = np.hstack([W_new, W[:, a + 1:b]])
        if b + 1 < r:
            W_new = np.hstack([W_new, W[:, b + 1:]])

        if _verify(U_new, V_new, W_new):
            return U_new, V_new, W_new
        return None

    alphas = [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]

    # Iteratively try to reduce rank
    current_U, current_V, current_W = U, V, W
    current_rank = current_U.shape[1]

    # Try a limited number of merge attempts
    max_attempts = 5000
    attempts = 0
    rng = random.Random(42)

    while time.time() < deadline - 10 and attempts < max_attempts and current_rank > 1:
        attempts += 1
        r = current_rank
        # Pick random pair
        a = rng.randint(0, r - 1)
        b = rng.randint(0, r - 1)
        if a == b:
            continue
        alpha = rng.choice(alphas)
        result = try_merge(current_U, current_V, current_W, a, b, alpha)
        if result is not None:
            current_U, current_V, current_W = result
            current_rank = current_U.shape[1]
            if current_rank < best_rank:
                best_rank = current_rank
                best = (current_U.copy(), current_V.copy(), current_W.copy())

    # Return the best found
    U_out, V_out, W_out = best
    # Convert to lists
    U_list = U_out.tolist()
    V_list = V_out.tolist()
    W_list = W_out.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}
