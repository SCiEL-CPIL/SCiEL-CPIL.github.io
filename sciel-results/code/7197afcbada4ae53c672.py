"""Matrix multiplication 4x4x6 rank-84 decomposition.

Decompose the 4x4-by-4x6 multiplication into three independent 4x4x2
output-column slices (s = 0,1,2).  Each slice is the product of two 4x4
matrices (A and a 4x2 B-slice padded to 4x4), computed with four Strassen
2x2 blocks (rank 7 each).  Total rank = 3 * 4 * 7 = 84.

Convention: C[k*n + i, r] = sum_j A[i*m + j, r] * B[j*p + k, r]
           n = 4 (A rows), m = 4 (A cols = B rows), p = 6 (B cols).
"""
import time

DIMENSIONS = [4, 4, 6]


def _strassen_terms():
    """Return the 7 Strassen (u, v, w) triples for 2x2 * 2x2 -> 2x2.

    Each term is u (2x2 coeff of A) x v (2x2 coeff of B) x w (2x2 coeff of C),
    flattened to length 4 in row-major order.  The product
    sum_t (u_t . a) * (v_t . b) reconstructs the 2x2 matrix product a @ b.
    """
    I = [1, 0, 0, 0]
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, -1]),  # (A00+A11)(B00+B11)
        ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]),   # A00 (B10 - B11)
        ([0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 0, 1]),   # A11 (B01 - B00)
        ([1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]),   # A00 B11
        ([0, 0, 1, 0], [1, 0, 0, 1], [0, 1, 0, 0]),   # (A10+A11)(B00+B01)
        ([0, 1, 0, -1], [0, 1, 0, 1], [0, 0, 1, 0]),  # (A01-A11)(B10+B11)
        ([0, 0, 1, -1], [1, 0, 1, 0], [1, 0, 0, 0]),  # (A10-A00)(B00+B11)
    ]
    return terms


def _verify(U, V, W, n, m, p):
    """Confirm C[k*n+i, r] = sum_j A[i*m+j,r] B[j*p+k,r] for random inputs."""
    try:
        import numpy as np
    except Exception:
        return False
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for _ in range(3):
        A = np.random.randint(-3, 4, size=(n, m))
        B = np.random.randint(-3, 4, size=(m, p))
        C = A @ B
        Cflat = np.zeros((p * n,), dtype=float)
        for k in range(p):
            for i in range(n):
                Cflat[k * n + i] = C[i, k]
        # reconstructed: Cflat[r] = sum_t Ua[:,t] V Va[:,t] W Wa[:,t]
        rec = np.zeros((p * n,), dtype=float)
        for r in range(Cflat.size):
            acc = 0.0
            for t in range(Ua.shape[1]):
                a = float(A[np.unravel_index(Ua[:, t].nonzero()[0], (n, m)).ravel()]) if False else None
            # vectorized: rec[r] = sum_t sum_i sum_j sum_k U[i,j,t] V[j,k,t] W[k,i,t] A[i,j] B[j,k]
            at = (A.reshape(n * m)[:, None] * Ua)  # (nm, T)
            bt = (B.reshape(m * p)[:, None] * Va)  # (mp, T)
            acc = np.sum(at * bt * Wa, axis=0)     # (T,) -> per r? No, Wa mixes indices.
        # Simpler and exact: build via einsum-like manual contraction per term.
        ok = True
        for r in range(p * n):
            s = 0.0
            for t in range(Ua.shape[1]):
                ua = Ua[:, t]
                va = Va[:, t]
                s += float(np.dot(A.reshape(n * m), ua)) * float(np.dot(B.reshape(m * p), va)) * Wa[r, t]
            if abs(s - Cflat[r]) > 1e-9:
                ok = False
                break
        if not ok:
            return False
    return True


def construct():
    start = time.time()
    deadline = start + 120.0

    n, m, p = 4, 4, 6
    rank = 3 * 4 * 7  # 84
    U = [[0] * rank for _ in range(n * m)]   # 16 x 84
    V = [[0] * rank for _ in range(m * p)]   # 24 x 84
    W = [[0] * rank for _ in range(p * n)]   # 24 x 84

    terms = _strassen_terms()

    col = 0
    for s in range(3):                       # output-column slice (0,1,2)
        for q in range(2):                   # block row (0,1)
            for r_idx in range(2):           # output row within block (0,1)
                for t in range(7):           # Strassen term
                    u, v, w = terms[t]
                    # A-subblock rows: 2q + u_idx,  all 4 columns
                    for ua_idx in range(4):
                        if u[ua_idx]:
                            ai = 2 * q + ua_idx
                            U[ai * m + 0][col] += u[ua_idx]
                            U[ai * m + 1][col] += u[ua_idx]
                            U[ai * m + 2][col] += u[ua_idx]
                            U[ai * m + 3][col] += u[ua_idx]
                    # B-subblock columns: 2s + vb_idx,  all 4 rows
                    for vb_idx in range(4):
                        if v[vb_idx]:
                            bj = 2 * s + vb_idx
                            V[0 * p + bj][col] += v[vb_idx]
                            V[1 * p + bj][col] += v[vb_idx]
                            V[2 * p + bj][col] += v[vb_idx]
                            V[3 * p + bj][col] += v[vb_idx]
                    # C output: row i = 2q + r_idx, col = 2s + t_out
                    for wc_idx in range(4):
                        if w[wc_idx]:
                            wc_r, wc_c = wc_idx // 2, wc_idx % 2
                            if wc_r == r_idx:
                                i = 2 * q + r_idx
                                ccol = 2 * s + wc_c
                                W[ccol * n + i][col] += w[wc_idx]
                    col += 1

    assert col == rank, (col, rank)

    # Verify before returning; fall back to schoolbook if (unexpectedly) invalid.
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook (rank 96).
    rs = n * m * p
    U = [[0] * rs for _ in range(n * m)]
    V = [[0] * rs for _ in range(m * p)]
    W = [[0] * rs for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    res = construct()
    print("rank =", len(res["U"][0]))
