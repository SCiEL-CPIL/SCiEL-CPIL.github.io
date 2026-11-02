import time
import sys
import numpy as np
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 140  # safety margin before 150s kill

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def remaining():
    return max(0.0, TIME_LIMIT - (time.time() - START_TIME))

def strassen_decomposition():
    """Rank-28 Strassen-based decomposition (known valid)."""
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    alpha = [
        {},
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1},
    ]
    beta = [
        {},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1},
    ]
    w_coeff = [
        {},
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1},
        {(0,0): 1, (1,0): 1},
        {(0,0): -1, (0,1): 1},
        {(1,1): 1},
        {(0,0): 1},
    ]

    for s in range(1, 8):
        for t in range(4):
            c = 4 * (s - 1) + t
            k_a = 0 if t in [0, 2] else 1
            r_b = 1 if t in [1, 3] else 0
            c_b = 1 if t in [2, 3] else 0
            k_c = 0 if t in [0, 1] else 1

            for (i, j), val in alpha[s].items():
                if val != 0:
                    u_idx = i * m + 2 * j + k_a
                    U[u_idx, c] += val

            for (i, j), val in beta[s].items():
                if val != 0:
                    v_row = 2 * i + r_b
                    v_col = 2 * j + c_b
                    v_idx = v_row * p + v_col
                    V[v_idx, c] += val

            for (i, j), val in w_coeff[s].items():
                if val != 0:
                    w_col = 2 * j + k_c
                    w_row = i
                    w_idx = w_col * n + w_row
                    W[w_idx, c] += val

    return U, V, W


def residual(U, V, W, n=2, m=4, p=4):
    """Sum of squared errors of the reconstructed tensor vs. all-ones."""
    rank = U.shape[1]
    T_recon = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.dot(
                    U[i * m + j, :],
                    V[j * p + k, :] * W[k * n + i, :]
                )
    return np.sum((T_recon - 1.0) ** 2)


def max_err(U, V, W, n=2, m=4, p=4):
    rank = U.shape[1]
    T_recon = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.dot(
                    U[i * m + j, :],
                    V[j * p + k, :] * W[k * n + i, :]
                )
    return np.max(np.abs(T_recon - 1.0))


def verify_decomposition(U, V, W, n=2, m=4, p=4):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    me = max_err(U_arr, V_arr, W_arr, n, m, p)
    return me < 1e-9, me


def polish(U, V, W, n=2, m=4, p=4, iters=200, tol=1e-13):
    """Gauss-Newton / least-squares polish: fix two factors, solve third exactly.
    Cycles (U|V,W), (V|U,W), (W|U,V) until residual < tol."""
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    rank = U.shape[1]

    for it in range(iters):
        if time_up():
            break
        prev = residual(U, V, W, n, m, p)
        if prev < tol:
            break

        # Solve for W given U,V: for each (i,k): dot(W[k*n+i], U[i*m+j]*V[j*p+k]) over j? No.
        # The equation: for each (i,j,k): sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1
        # This is bilinear in (U,V) per r times W. Fixing U,V makes it linear in W:
        # For each (i,k): sum_r W[k*n+i,r] * (sum_j U[i*m+j,r]*V[j*p+k,r]) = 1  -- NO, j is summed inside? 
        # Actually the constraint is per (i,j,k), not summed over j. So fixing U,V gives, for each (i,k),
        # a system over r: sum_r W[k*n+i,r] * A_{(i,k),r} where A depends on j too -> not separable.
        # So we cannot solve W in closed form fixing U,V because j couples. 
        # Instead do a full least-squares on W by linearizing: treat each (i,j,k) as one equation.
        # Unknowns: W (p*n*rank). Equation for (i,j,k): sum_r W[k*n+i,r] * U[i*m+j,r]*V[j*p+k,r] = 1.
        # Build matrix M of size (n*m*p) x (p*n*rank): M[(i,j,k), (k*n+i)*rank + r] = U[i*m+j,r]*V[j*p+k,r]
        # Solve M @ w_vec = 1 by least squares.
        N = n * m * p
        M = np.zeros((N, p * n * rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    base = (i * m + j) * p + (j * p + k)  # not needed; use flat index
                    idx = (i * m + j) * p + k
                    for r in range(rank):
                        M[idx, (k * n + i) * rank + r] = U[i * m + j, r] * V[j * p + k, r]
        w_vec, *_ = np.linalg.lstsq(M, np.ones(N), rcond=None)
        W = w_vec.reshape(p * n, rank)

        # Solve for U given V,W: M2[idx, (i*m+j)*rank + r] = V[j*p+k,r]*W[k*n+i,r]
        M2 = np.zeros((N, n * m * rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = (i * m + j) * p + k
                    for r in range(rank):
                        M2[idx, (i * m + j) * rank + r] = V[j * p + k, r] * W[k * n + i, r]
        u_vec, *_ = np.linalg.lstsq(M2, np.ones(N), rcond=None)
        U = u_vec.reshape(n * m, rank)

        # Solve for V given U,W: M3[idx, (j*p+k)*rank + r] = U[i*m+j,r]*W[k*n+i,r]
        M3 = np.zeros((N, m * p * rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = (i * m + j) * p + k
                    for r in range(rank):
                        M3[idx, (j * p + k) * rank + r] = U[i * m + j, r] * W[k * n + i, r]
        v_vec, *_ = np.linalg.lstsq(M3, np.ones(N), rcond=None)
        V = v_vec.reshape(m * p, rank)

        cur = residual(U, V, W, n, m, p)
        if cur < tol or abs(prev - cur) < 1e-16:
            break

    return U, V, W, residual(U, V, W, n, m, p)


def snap_to_rational(U, V, W, scale=2):
    """Round entries to nearest multiple of 1/scale (half-integers) if close."""
    def snap(A):
        B = (A * scale).round() / scale
        if np.max(np.abs(A - B)) < 1e-6:
            return B
        return A
    return snap(U), snap(V), snap(W)


def search_rank(r, n=2, m=4, p=4, budget=30.0, seed=0):
    """Try to find an exact rank-r decomposition. Returns (U,V,W) or None."""
    if time_up():
        return None
    t0 = time.time()
    rng = np.random.default_rng(seed)
    best = None
    best_res = np.inf

    # Initialize candidates: from Strassen truncated/padded + random
    U0, V0, W0 = strassen_decomposition()
    inits = []
    # Use Strassen columns (first r) as a strong init
    inits.append((U0[:, :r].copy(), V0[:, :r].copy(), W0[:, :r].copy()))
    # A few random inits
    for _ in range(3):
        inits.append((
            rng.normal(size=(n * m, r)),
            rng.normal(size=(m * p, r)),
            rng.normal(size=(p * n, r)),
        ))

    for (Ui, Vi, Wi) in inits:
        if time_up() or (time.time() - t0) > budget:
            break
        # Polish from this init
        U, V, W, res = polish(Ui, Vi, Wi, n, m, p, iters=300, tol=1e-13)
        if res < best_res:
            best_res = res
            best = (U.copy(), V.copy(), W.copy())
        if best_res < 1e-10:
            break

    if best is not None and best_res < 1e-10:
        U, V, W = best
        U, V, W = snap_to_rational(U, V, W)
        me = max_err(U, V, W, n, m, p)
        if me < 1e-9:
            return U, V, W
    return None


def construct():
    n, m, p = 2, 4, 4

    # Baseline: Strassen rank 28
    U, V, W = strassen_decomposition()
    valid, err = verify_decomposition(U, V, W, n, m, p)
    best_rank = 28
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    if not valid:
        # Fallback naive
        rank = 32
        U = np.zeros((n * m, rank)); V = np.zeros((m * p, rank)); W = np.zeros((p * n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        best_rank = 32
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Search for lower ranks: 24,25,26,27
    for r in [24, 25, 26, 27]:
        if time_up():
            break
        if r >= best_rank:
            break
        found = search_rank(r, n, m, p, budget=min(30.0, remaining() * 0.5), seed=r)
        if found is not None:
            Uf, Vf, Wf = found
            valid, err = verify_decomposition(Uf, Vf, Wf, n, m, p)
            if valid:
                best_rank = r
                best_result = {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}

    return best_result


if __name__ == "__main__":
    result = construct()
    print(f"Best rank: {len(result['U'][0])}")
    valid, err = verify_decomposition(result['U'], result['V'], result['W'])
    print(f"Valid: {valid}, Max error: {err}")
