import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = 2, 3, 3
    U_size = n * m   # 6
    V_size = m * p   # 9
    W_size = p * n   # 6

    # Target tensor T[u, v, w]
    T = np.zeros((U_size, V_size, W_size))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    start_time = time.time()
    deadline = start_time + 112.0

    # Precompute flattened views of T
    T_flat = T.reshape(U_size, V_size * W_size)
    T_flat_v = T.transpose(1, 0, 2).reshape(V_size, U_size * W_size)
    T_flat_w = T.transpose(2, 0, 1).reshape(W_size, U_size * V_size)

    # ---- Exact verification with integer arithmetic ----
    def exact_decompose(FU, FV, FW):
        """FU/FV/FW are lists of Fraction (length = rank). Return bool."""
        for u in range(U_size):
            for v in range(V_size):
                s = Fraction(0)
                for c in range(len(FU)):
                    s += FU[c][u] * FV[c][v] * FW[c]
                target = T[u, v, 0]  # not used; we check all w
                # check all w
                for w in range(W_size):
                    val = Fraction(0)
                    for c in range(len(FU)):
                        val += FU[c][u] * FV[c][v] * FW[c][w]
                    if val != int(T[u, v, w]):
                        return False
        return True

    def reconstruct(X, denom_max=1024):
        """X: 2D float array (rows x rank). Return list of lists of Fraction,
        or None if any entry cannot be represented with small denominator."""
        out = []
        for r in range(X.shape[0]):
            row = []
            for c in range(X.shape[1]):
                x = X[r, c]
                if abs(x) < 1e-7:
                    row.append(Fraction(0))
                    continue
                # continued fraction
                try:
                    fr = Fraction(x).limit_denominator(denom_max)
                except Exception:
                    return None
                if abs(fr - x) > 1e-6:
                    return None
                row.append(fr)
            out.append(row)
        return out

    def try_rank_exact(rank, denom_max=1024):
        """Run NNLS-ALS for this rank; if it converges to ~zero residual,
        reconstruct to exact rationals and verify. Return (U,V,W) as lists
        of lists of float (the exact rational values) or None."""
        def nnls_als(seed, max_iter=400):
            rng = np.random.RandomState(seed)
            U = np.abs(rng.randn(U_size, rank)) + 1e-4
            V = np.abs(rng.randn(V_size, rank)) + 1e-4
            W = np.abs(rng.randn(W_size, rank)) + 1e-4
            prev_cost = np.inf
            for it in range(max_iter):
                # Build A (rank x V*W): A[c] = outer(V[:,c], W[:,c])
                A = np.zeros((rank, V_size * W_size))
                for c in range(rank):
                    A[c] = np.outer(V[:, c], W[:, c]).ravel()
                B = np.zeros((rank, U_size * W_size))
                for c in range(rank):
                    B[c] = np.outer(U[:, c], W[:, c]).ravel()
                C = np.zeros((rank, U_size * V_size))
                for c in range(rank):
                    C[c] = np.outer(U[:, c], V[:, c]).ravel()

                # NNLS solves: min ||X - A^T y|| s.t. y>=0  for each row of X
                U_new = np.linalg.lstsq(A.T, T_flat.T, rcond=None)[0].T
                V_new = np.linalg.lstsq(B.T, T_flat_v.T, rcond=None)[0].T
                W_new = np.linalg.lstsq(C.T, T_flat_w.T, rcond=None)[0].T
                # project to non-negative
                U_new = np.maximum(U_new, 0.0)
                V_new = np.maximum(V_new, 0.0)
                W_new = np.maximum(W_new, 0.0)

                cost = np.sum((T - np.einsum('uc,vc,wc->uvw', U_new, V_new, W_new))**2)
                if abs(prev_cost - cost) < 1e-14:
                    break
                prev_cost = cost
                U, V, W = U_new, V_new, W_new
            return U, V, W, prev_cost

        for seed in range(200):
            if time.time() > deadline:
                return None
            try:
                U, V, W, cost = nnls_als(seed)
            except Exception:
                continue
            # check near-exact float residual
            T_calc = np.einsum('uc,vc,wc->uvw', U, V, W)
            resid = np.max(np.abs(T_calc - T))
            if resid < 1e-5:
                # reconstruct to exact rationals
                FU = reconstruct(U, denom_max)
                FV = reconstruct(V, denom_max)
                FW = reconstruct(W, denom_max)
                if FU is None or FV is None or FW is None:
                    continue
                if exact_decompose(FU, FV, FW):
                    # convert to float lists
                    Uf = [[float(x) for x in row] for row in FU]
                    Vf = [[float(x) for x in row] for row in FV]
                    Wf = [[float(x) for x in row] for row in FW]
                    return Uf, Vf, Wf
        return None

    # ---- Known rank-18 (schoolbook) exact decomposition as fallback ----
    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(U_size)]
        V = [[0.0] * rank for _ in range(V_size)]
        W = [[0.0] * rank for _ in range(W_size)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return U, V, W

    # Search ranks in increasing order
    best = None
    for rank in range(7, 19):
        if time.time() > deadline - 10.0:
            break
        result = try_rank_exact(rank)
        if result is not None:
            best = result
            break  # smallest rank found

    if best is None:
        U, V, W = schoolbook()
    else:
        U, V, W = best

    return {"U": U, "V": V, "W": W}
