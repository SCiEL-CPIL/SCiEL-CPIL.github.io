"""Matrix multiplication 3x3 by 3x7 - rank minimization via structured decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140  # safety margin

    def build_base():
        """Build schoolbook decomposition: rank = n*m*p = 63."""
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank

    def verify(U, V, W):
        """Verify decomposition: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
        r = U.shape[1]
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def to_list(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Start with schoolbook
    U, V, W, rank = build_base()

    # Try to reduce rank by merging columns
    # Two columns c1, c2 can be merged if their U, V, W rows are "compatible"
    # i.e., there exist scalars a, b such that a*U[:,c1] + b*U[:,c2] has at most
    # the same support, and similarly for V and W.
    # A simpler heuristic: if two columns have identical U-row support (same i*m+j nonzero)
    # AND identical W-row support (same k*n+i nonzero), they can potentially be merged.

    # For our base: column for (i,j,k) has U nonzero only at i*m+j, V only at j*p+k, W only at k*n+i.
    # Two columns (i,j,k1) and (i,j,k2) share the same U row (i*m+j) but different V and W rows.
    # They can't simply be merged unless we allow linear combinations.

    # Better approach: try to find a lower-rank decomposition directly.
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # We want sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.

    # Key observation: for fixed i, the equation is:
    # sum_c (sum_j U[i*m+j,c]) ... no, that's not separable.
    # Actually: for fixed (i,k): sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is a system of n*p = 21 equations (one per (i,k)) but with n*m*p = 63 constraints
    # (one per (i,j,k)).

    # Let me try a structured construction:
    # Use the fact that T is all-ones. If I can find vectors u_i (length r), v_j (length r), w_k (length r)
    # such that sum_c u_i[c]*v_j[c]*w_k[c] = 1 for all i,j,k.
    # But our U, V, W are not separable in that simple way due to the vectorized indexing.

    # Actually, the vectorized form means U[i*m+j,c] couples i and j. For the all-ones tensor,
    # we need: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1.

    # Let me try a rank-21 construction based on the lower bound max(n*m, m*p, p*n) = 21.
    # Idea: use r = 21 columns. For each column c (0..20), assign it to a unique (j,k) pair
    # (there are m*p = 21 such pairs). Then:
    # V[j*p+k, c] = 1 if c == j*p+k, else 0.
    # For U[i*m+j, c]: we need it to depend on i and j. Since c indexes (j,k), for a given c = j'*p+k',
    # U[i*m+j, c] should be nonzero only when j == j'. Let U[i*m+j, c] = a[i] when j == j', else 0.
    # For W[k*n+i, c]: should be nonzero only when k == k'. Let W[k*n+i, c] = b[i] when k == k', else 0.
    # Then sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = sum_c [a[i] if j==j'_c][1 if c==j*p+k][b[i] if k==k'_c]
    # = a[i]*b[i] when c = j*p+k (the unique c matching both j and k)
    # = a[i]*b[i]
    # We need a[i]*b[i] = 1 for all i. So set a[i]=1, b[i]=1.
    # This gives rank = m*p = 21!

    # Let's verify this construction:
    r = m * p  # 21
    U2 = np.zeros((n * m, r), dtype=float)
    V2 = np.zeros((m * p, r), dtype=float)
    W2 = np.zeros((p * n, r), dtype=float)
    for c in range(r):
        j = c // p
        k = c % p
        # V: V[j*p+k, c] = 1
        V2[j * p + k, c] = 1.0
        # U: for all i, U[i*m+j, c] = 1
        for i in range(n):
            U2[i * m + j, c] = 1.0
        # W: for all i, W[k*n+i, c] = 1
        for i in range(n):
            W2[k * n + i, c] = 1.0

    if verify(U2, V2, W2):
        return to_list(U2, V2, W2)

    # If that didn't work (it should), fall back to schoolbook
    return to_list(U, V, W)
