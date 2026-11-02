import time
import numpy as np

DIMENSIONS = [3, 4, 4]
_START = time.time()
_TIME_LIMIT = 140.0

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    r0 = n * m * p

    # Schoolbook decomposition
    U0 = np.zeros((nm, r0))
    V0 = np.zeros((mp, r0))
    W0 = np.zeros((pn, r0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1

    best_rank, best_U, best_V, best_W = r0, U0, V0, W0

    # Try to find a lower-rank decomposition using algebraic approach
    for r in range(r0 - 1, max(r0 - 8, 28), -1):
        if time.time() - _START > _TIME_LIMIT - 5:
            break
        res = _try_rank_algebraic(n, m, p, r)
        if res is not None:
            best_rank, best_U, best_V, best_W = r, res[0], res[1], res[2]
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _try_rank_algebraic(n, m, p, r, n_trials=5):
    nm, mp, pn = n * m, m * p, p * n
    idx = [(i * m + j, j * p + k, k * n + i)
           for i in range(n) for j in range(m) for k in range(p)]
    nm, mp, pn = n * m, m * p, p * n

    for t in range(n_trials):
        if time.time() - _START > _TIME_LIMIT - 5:
            return None
        
        rng = np.random.RandomState(t * 7919 + r + 12345)
        
        # Initialize with random matrices
        U = rng.randn(nm, r)
        V = rng.randn(mp, r)
        W = rng.randn(pn, r)
        
        # Normalize columns to avoid scale issues
        for c in range(r):
            for mat in [U, V, W]:
                norm = np.linalg.norm(mat[:, c])
                if norm > 1e-10:
                    mat[:, c] /= norm
        
        # Use alternating least squares with better convergence
        for it in range(200):
            if time.time() - _START > _TIME_LIMIT - 5:
                return None
            
            # Update U
            Ug = np.zeros_like(U)
            for ui, vi, wi in idx:
                vw = V[vi] * W[wi]
                s = np.dot(U[ui], vw)
                e = 1.0 - s
                Ug[ui] += e * vw
            # Solve for U update
            for ui in range(nm):
                if np.linalg.norm(Ug[ui]) > 1e-12:
                    # Gradient descent step
                    lr = 0.01 / (1.0 + it * 0.005)
                    U[ui] += lr * Ug[ui]
            
            # Update V
            Vg = np.zeros_like(V)
            for ui, vi, wi in idx:
                uw = U[ui] * W[wi]
                s = np.dot(U[ui] * V[vi], W[wi])
                e = 1.0 - s
                Vg[vi] += e * uw
            for vi in range(mp):
                if np.linalg.norm(Vg[vi]) > 1e-12:
                    lr = 0.01 / (1.0 + it * 0.005)
                    V[vi] += lr * Vg[vi]
            
            # Update W
            Wg = np.zeros_like(W)
            for ui, vi, wi in idx:
                uv = U[ui] * V[vi]
                s = np.dot(U[ui], V[vi] * W[wi])
                e = 1.0 - s
                Wg[wi] += e * uv
            for wi in range(pn):
                if np.linalg.norm(Wg[wi]) > 1e-12:
                    lr = 0.01 / (1.0 + it * 0.005)
                    W[wi] += lr * Wg[wi]
            
            # Check convergence
            loss = 0.0
            for ui, vi, wi in idx:
                s = np.dot(U[ui], V[vi] * W[wi])
                loss += (s - 1.0) ** 2
            
            if loss < 1e-12:
                # Try to round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if _verify(n, m, p, U_r, V_r, W_r):
                    return (U_r, V_r, W_r)
                
                # If rounding doesn't work, try finer rounding
                for scale in [4, 8, 16]:
                    U_r = np.round(U * scale) / scale
                    V_r = np.round(V * scale) / scale
                    W_r = np.round(W * scale) / scale
                    if _verify(n, m, p, U_r, V_r, W_r):
                        return (U_r, V_r, W_r)
        
        # Final attempt with high-precision polish
        if time.time() - _START < _TIME_LIMIT - 5:
            for it in range(100):
                if time.time() - _START > _TIME_LIMIT - 5:
                    return None
                Ug = np.zeros_like(U)
                Vg = np.zeros_like(V)
                Wg = np.zeros_like(W)
                loss = 0.0
                for ui, vi, wi in idx:
                    vw = V[vi] * W[wi]
                    uv = U[ui] * V[vi]
                    uw = U[ui] * W[wi]
                    s = np.dot(U[ui], vw)
                    e = s - 1.0
                    loss += e * e
                    Ug[ui] += e * vw
                    Vg[vi] += e * uw
                    Wg[wi] += e * uv
                
                if loss < 1e-14:
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if _verify(n, m, p, U_r, V_r, W_r):
                        return (U_r, V_r, W_r)
                
                lr = 0.005 / (1.0 + it * 0.01)
                U -= lr * Ug
                V -= lr * Vg
                W -= lr * Wg

    return None

def _verify(n, m, p, U, V, W):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vi = j * p + k
                wi = k * n + i
                val = np.dot(U[ui], V[vi] * W[wi])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True
