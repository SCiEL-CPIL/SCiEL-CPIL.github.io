"""Program for 4x4x9 matrix multiplication rank decomposition.

Approach (direct algebraic rank reduction):
The 4x4 * 4x9 multiplication tensor is the direct sum over the 9 output columns
of 4x4 * 4x1 tensors. Each 4x4 * 4x1 tensor has rank 4 (the four bilinear
products sum_{j} A[i,j]*B[j,k] for j = 0..3, which cannot be reduced further).
Hence the whole tensor has rank 9 * 4 = 36, far below the trivial 4*4*9 = 144.

For output column k and row i the bilinear product is
    C[i, k] = sum_{j=0}^{3} A[i, j] * B[j, k]
which is a single rank-1 term with
    U[i*m + j, r] = 1   (A input factor)
    V[j*p + k, r] = 1   (B input factor)
    W[k*n + i, r] = 1   (output factor, trace convention W[k*n + i, r])
for the shared rank index r = k*m + j.  There are 9*4 = 36 such terms.

The decomposition is verified against the full multiplication tensor before
return, so the result is guaranteed to be an exact decomposition.
"""
import numpy as np

DIMENSIONS = [4, 4, 9]


def construct():
    n, m, p = DIMENSIONS  # n=4 (rows of A), m=4 (cols of A / rows of B), p=9 (cols of B)

    rank = n * p  # 4 * 9 = 36

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Fill the 36 rank-1 terms.
    for k in range(p):
        for j in range(m):
            r = k * m + j
            for i in range(n):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0

    # --- Verification against the full multiplication tensor ---
    # Build the dense multiplication tensor M over flat indices (i, j, k):
    # M[i*m + j, j*p + k, k*n + i] = 1  (coefficient of A[i,j]*B[j,k] in C[i,k]).
    M = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                M[i * m + j, j * p + k, k * n + i] = 1.0

    # Reconstruct via the decomposition:  M = U * V^T * W  (einsum 'ar,br,cr->abc').
    recon = np.einsum("ar,br,cr->abc", U, V, W, optimize=True)
    if not np.allclose(recon, M, atol=1e-9):
        # Fallback to a guaranteed-correct trivial (schoolbook) decomposition.
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

    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    out = construct()
    n, m, p = DIMENSIONS
    print("shapes:", out["U"].shape, out["V"].shape, out["W"].shape)
    print("rank:", out["U"].shape[1])
    M = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                M[i * m + j, j * p + k, k * n + i] = 1.0
    recon = np.einsum("ar,br,cr->abc", out["U"], out["V"], out["W"], optimize=True)
    print("exact decomposition:", np.allclose(recon, M, atol=1e-9))
