"""Block-composition tensor decomposition for 3x4x8 matrix multiplication."""
import numpy as np

DIMENSIONS = [3, 4, 8]


def _combine(parts):
    """parts: list of (U, V, W) each as 2-D arrays. Concatenate U and V,
    stack W along axis 1 (sum of contributions)."""
    U = np.concatenate([p[0] for p in parts], axis=1)
    V = np.concatenate([p[1] for p in parts], axis=1)
    W = np.concatenate([p[2] for p in parts], axis=1)
    return U, V, W


def _place(U, V, W, i0, j0, k0, n, m, p):
    """Embed a block decomposition for C[i0:i0+nb, k0:k0+nb] = A[i0:i0+nb, j0:j0+mb] B[j0:j0+mb, k0:k0+nb]."""
    nb, mb = U.shape[0] // 0 + 0, 0  # placeholders to keep linters quiet
    nb = U.shape[0]
    mb = V.shape[0]
    nb2 = W.shape[0]
    rank = U.shape[1]
    Uf = np.zeros((n * m, rank))
    Vf = np.zeros((m * p, rank))
    Wf = np.zeros((p * n, rank))
    for u in range(nb):
        for uu in range(rank):
            Uf[(i0 + u) * m + (j0 + u) % mb, uu] = U[u, uu]
    for v in range(mb):
        for vv in range(rank):
            Vf[(j0 + v) * p + (k0 + v) % nb2, vv] = V[v, vv]
    for w in range(nb2):
        for ww in range(rank):
            Wf[(k0 + w) * n + (i0 + w) % nb, ww] = W[w, ww]
    return Uf, Vf, Wf


def _matmul_decomp(n, m, p):
    """Return (U, V, W) for multiplying n x m by m x p."""
    # Base: 2x2x2 Strassen rank 7
    if n == 2 and m == 2 and p == 2:
        S = np.array([
            [1, 1, 1, -1, 1, -1, 1],
            [-1, -1, 1, 1, -1, 1, -1],
        ])
        T = np.array([
            [1, -1, 1, 1, 1, -1, -1],
            [1, 1, -1, -1, -1, -1, 1],
        ])
        W = np.array([
            [0, 1, 1, 0, 0, 1, 0],
            [1, 0, 0, 1, 1, 0, 1],
        ])
        return S, T, W
    # 1 x m x p: each output element is one inner product -> m*p rank-1 terms
    if n == 1:
        U = np.zeros((m, m * p))
        V = np.zeros((m * p, m * p))
        W = np.zeros((p, m * p))
        for j in range(m):
            for k in range(p):
                c = j * p + k
                U[j, c] = 1
                V[c, c] = 1
                W[k, c] = 1
        return U, V, W
    # 1 x 1 x p: rank 1
    if n == 1 and m == 1:
        U = np.zeros((1, p))
        V = np.zeros((p, p))
        W = np.zeros((p, p))
        for k in range(p):
            U[0, k] = 1
            V[k, k] = 1
            W[k, k] = 1
        return U, V, W

    # 2 x 2 x p (p even): two independent 2x2x2 blocks
    if n == 2 and m == 2 and p % 2 == 0:
        half = p // 2
        parts = []
        for b in range(half):
            U, V, W = _matmul_decomp(2, 2, 2)
            Uf, Vf, Wf = _place(U, V, W, 0, 0, 2 * b, 2, 2, p)
            parts.append((Uf, Vf, Wf))
        return _combine(parts)

    # n x 2 x 2 (n even): two independent n x 2 x 2? no, split A rows in 2s
    # General: split into (2 or 1) x 2 x 2 row blocks if m==2,p==2 and n even
    if m == 2 and p == 2 and n % 2 == 0:
        parts = []
        for b in range(n // 2):
            U, V, W = _matmul_decomp(2, 2, 2)
            Uf, Vf, Wf = _place(U, V, W, 2 * b, 0, 0, n, 2, 2)
            parts.append((Uf, Vf, Wf))
        return _combine(parts)

    # General recursion:
    # If n even: split A rows into two halves of n/2.
    if n % 2 == 0:
        h = n // 2
        U1, V1, W1 = _matmul_decomp(h, m, p)
        U2, V2, W2 = _matmul_decomp(h, m, p)
        U1f, V1f, W1f = _place(U1, V1, W1, 0, 0, 0, n, m, p)
        U2f, V2f, W2f = _place(U2, V2, W2, h, 0, 0, n, m, p)
        return _combine([(U1f, V1f, W1f), (U2f, V2f, W2f)])
    # n odd (>=3): split into (n-1) rows + 1 row.
    if n >= 3:
        h = n - 1
        U1, V1, W1 = _matmul_decomp(h, m, p)
        U2, V2, W2 = _matmul_decomp(1, m, p)
        U1f, V1f, W1f = _place(U1, V1, W1, 0, 0, 0, n, m, p)
        U2f, V2f, W2f = _place(U2, V2, W2, h, 0, 0, n, m, p)
        return _combine([(U1f, V1f, W1f), (U2f, V2f, W2f)])
    # Fallback: schoolbook
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


def construct():
    n, m, p = DIMENSIONS
    # Split B into two 4x4 blocks -> two 3x4x4 products.
    U1, V1, W1 = _matmul_decomp(3, 4, 4)
    U2, V2, W2 = _matmul_decomp(3, 4, 4)
    U1f, V1f, W1f = _place(U1, V1, W1, 0, 0, 0, 3, 4, 8)
    U2f, V2f, W2f = _place(U2, V2, W2, 0, 0, 4, 3, 4, 8)
    U, V, W = _combine([(U1f, V1f, W1f), (U2f, V2f, W2f)])
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    n, m, p = DIMENSIONS
    res = construct()
    U, V, W = res["U"], res["V"], res["W"]
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C = A @ B
    Ct = W @ (V @ B) @ (U @ A).T if False else (W.T @ np.kron(V.T, U.T)).reshape(p * n, n * p)
    # Proper contraction: sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    val = 0
    for c in range(U.shape[1]):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        val += np.einsum("ij,jk,ki->ik", u, v, w)
    print("rank:", U.shape[1])
    print("err:", np.max(np.abs(val - C)))
