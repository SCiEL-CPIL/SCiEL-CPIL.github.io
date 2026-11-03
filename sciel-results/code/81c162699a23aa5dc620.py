import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# --- Precompute 2x2x2 Strassen factors (rank 7) ---
def _strassen_2x2():
    U2 = np.zeros((2, 2, 7), dtype=float)
    V2 = np.zeros((2, 2, 7), dtype=float)
    W2 = np.zeros((2, 2, 7), dtype=float)
    # r=0
    U2[0, 0, 0] = 1; U2[1, 1, 0] = 1
    V2[0, 0, 0] = 1; V2[1, 1, 0] = 1
    W2[0, 0, 0] = 1; W2[1, 1, 0] = 1
    # r=1
    U2[1, 0, 1] = 1; U2[1, 1, 1] = 1
    V2[0, 0, 1] = 1
    W2[0, 1, 1] = 1; W2[1, 1, 1] = -1
    # r=2
    U2[0, 0, 2] = 1
    V2[0, 1, 2] = 1; V2[1, 1, 2] = -1
    W2[1, 0, 2] = 1; W2[1, 1, 2] = 1
    # r=3
    U2[1, 1, 3] = 1
    V2[1, 0, 3] = 1; V2[0, 0, 3] = -1
    W2[0, 0, 3] = 1; W2[0, 1, 3] = 1
    # r=4
    U2[0, 0, 4] = 1; U2[0, 1, 4] = 1
    V2[1, 1, 4] = 1
    W2[0, 0, 4] = -1; W2[1, 0, 4] = 1
    # r=5
    U2[1, 0, 5] = 1; U2[0, 0, 5] = -1
    V2[0, 0, 5] = 1; V2[0, 1, 5] = 1
    W2[1, 1, 5] = 1
    # r=6
    U2[0, 1, 6] = 1; U2[1, 1, 6] = -1
    V2[1, 0, 6] = 1; V2[1, 1, 6] = 1
    W2[0, 0, 6] = 1
    return U2, V2, W2

_U2, _V2, _W2 = _strassen_2x2()

# --- Build 4x4x4 Strassen factors (rank 49) ---
def _strassen_4x4():
    U4 = np.zeros((4, 4, 49), dtype=float)
    V4 = np.zeros((4, 4, 49), dtype=float)
    W4 = np.zeros((4, 4, 49), dtype=float)
    for i in range(4):
        for j in range(4):
            i1, i0 = i // 2, i % 2
            j1, j0 = j // 2, j % 2
            for s in range(7):
                for r in range(7):
                    U4[i, j, 7*s + r] = _U2[i1, j1, s] * _U2[i0, j0, r]
    for j in range(4):
        for k in range(4):
            j1, j0 = j // 2, j % 2
            k1, k0 = k // 2, k % 2
            for s in range(7):
                for r in range(7):
                    V4[j, k, 7*s + r] = _V2[j1, k1, s] * _V2[j0, k0, r]
    for k in range(4):
        for i in range(4):
            k1, k0 = k // 2, k % 2
            i1, i0 = i // 2, i % 2
            for s in range(7):
                for r in range(7):
                    W4[k, i, 7*s + r] = _W2[k1, i1, s] * _W2[k0, i0, r]
    return U4, V4, W4

_U4, _V4, _W4 = _strassen_4x4()


def _build_4x4x8():
    """Build 4x4x8 factors by duplicating the 4x4x4 factors for the two halves of B."""
    rank = 98
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((32, rank), dtype=float)

    for i in range(4):
        for j in range(4):
            for R in range(49):
                U[i*4 + j, R] = _U4[i, j, R]
                U[i*4 + j, 49 + R] = _U4[i, j, R]
    for j in range(4):
        for k in range(4):
            for R in range(49):
                V[j*8 + k, R] = _V4[j, k, R]
                V[j*8 + (4+k), 49 + R] = _V4[j, k, R]
    for k in range(4):
        for i in range(4):
            for R in range(49):
                W[k*4 + i, R] = _W4[k, i, R]
                W[(4+k)*4 + i, 49 + R] = _W4[k, i, R]
    return U, V, W


def _tensor_check(U, V, W, atol=1e-9):
    """Verify that the rank-1 decomposition equals the matrix multiplication tensor."""
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    n, m, p = 4, 4, 8
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    r = U.shape[1]
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    # Expected: for each (i,k), result[i,k,j] = sum over... 
    # Actually the tensor T[i,j,k] = delta (the multiplication tensor has T[i,j,k]=1 iff it's the (i,k) output from (i,j) input... 
    # For matrix multiplication C = A @ B, C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is T[i,j,k] = 1 for all (i,j,k) as a "multiplication" pattern? No.
    # The standard tensor: T[i,j,k] = 1 means the (i,j) input of A and (j,k) input of B produce (i,k) output.
    # Actually the tensor entries are all 1 in the sense that T[i,j,k] = 1 for every triple (i,j,k).
    # Wait, no. The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k.
    # Because C[i,k] = sum_j A[i,j] B[j,k], so the tensor coefficient for (i,j,k) is 1.
    expected = np.ones((n, p, m))  # T[i,k,j] = 1
    # But our einsum gives result[i,k,j] where j is the "inner" index
    # Let me re-examine: U3[i,j,r], V3[j,k,r], W3[k,i,r] -> result[i,k,r summed]
    # This computes sum_r U3[i,j,r]*V3[j,k,r]*W3[k,i,r] but summed over r only, not j
    # I need to sum over j as well. Let me fix.
    return None  # placeholder, will fix below


def _verify(U, V, W, atol=1e-9):
    """Proper verification of the 4x4x8 matrix multiplication tensor."""
    n, m, p = 4, 4, 8
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    # T[i,k,j] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]  (sum over j and r)
    # We want T[i,k,j] = 1 for all i,k,j
    result = np.einsum('ijr,jkr,kir->ikr', U3, V3, W3)  # this sums over r only
    # Need to also sum over j:
    result = np.einsum('ijr,jkr,kir->ik', U3, V3, W3)  # sums over j and r
    # Hmm, that gives a 2D array. The tensor should be 3D.
    # Let me think again. The matrix multiplication tensor for C=A@B where A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1 for all i,j,k.
    # The decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # So we check: for all i,j,k, sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=atol)


def _try_reduce(U, V, W, deadline, atol=1e-9):
    """Try to remove redundant rank-1 columns. Returns reduced (U, V, W) or original."""
    n, m, p = 4, 4, 8
    keep = list(range(U.shape[1]))
    # Greedy: try removing each column, keep removal if still valid
    # To be efficient, iterate and test
    for idx in range(len(keep) - 1, -1, -1):
        if time.time() > deadline - 5.0:
            break
        # Try removing keep[idx]
        new_keep = keep[:idx] + keep[idx+1:]
        if len(new_keep) == 0:
            continue
        U_t = U[:, new_keep]
        V_t = V[:, new_keep]
        W_t = W[:, new_keep]
        if _verify(U_t, V_t, W_t, atol=atol):
            keep = new_keep
    if len(keep) == U.shape[1]:
        return U, V, W
    return U[:, keep], V[:, keep], W[:, keep]


def _schoolbook():
    n, m, p = 4, 4, 8
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Build 4x4x8 factors (rank 98)
    U, V, W = _build_4x4x8()

    # Verify the base construction
    if not _verify(U, V, W, atol=1e-9):
        # Fallback to schoolbook
        U, V, W = _schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank by removing redundant columns
    if time.time() < deadline - 10.0:
        U, V, W = _try_reduce(U, V, W, deadline, atol=1e-9)

    # Final safety check
    if not _verify(U, V, W, atol=1e-9):
        U, V, W = _schoolbook()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
