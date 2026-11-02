"""
Rank-minimal decomposition for 2x2 by 2x10 matrix multiplication.

Uses numerical search to find the minimum rank tensor decomposition,
falling back to the known rank-35 Strassen-based solution.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    T = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, i*p+k] = 1.0
    return T

def _strassen_35(n, m, p):
    rank = 35
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for t in range(5):
        cb = t*7
        c = cb; U[0][c]=1; U[3][c]=1; V[2*t][c]=1; V[10+2*t+1][c]=1; W[4*t][c]=1; W[4*t+3][c]=1
        c = cb+1; U[2][c]=1; U[3][c]=1; V[2*t][c]=1; W[4*t+1][c]=1; W[4*t+3][c]=-1
        c = cb+2; U[0][c]=1; V[2*t+1][c]=1; V[10+2*t+1][c]=-1; W[4*t+2][c]=1; W[4*t+3][c]=1
        c = cb+3; U[3][c]=1; V[10+2*t][c]=1; V[2*t][c]=-1; W[4*t][c]=1; W[4*t+1][c]=1
        c = cb+4; U[0][c]=1; U[1][c]=1; V[10+2*t+1][c]=1; W[4*t][c]=-1; W[4*t+2][c]=1
        c = cb+5; U[2][c]=1; U[0][c]=-1; V[2*t][c]=1; V[2*t+1][c]=1; W[4*t+3][c]=1
        c = cb+6; U[1][c]=1; U[3][c]=-1; V[10+2*t][c]=1; V[10+2*t+1][c]=1; W[4*t][c]=1
    return U, V, W

def _snap_to_half(x):
    """Snap value to nearest half-integer."""
    return round(x * 2) / 2

def _try_rank_numerical(T, r, max_iters=2000, seed=42):
    """Try to find rank-r decomposition via gradient descent."""
    rng = np.random.RandomState(seed)
    dim_u, dim_v, dim_w = T.shape
    
    for attempt in range(2):
        U = rng.randn(dim_u, r) * 0.3
        V = rng.randn(dim_v, r) * 0.3
        W = rng.randn(dim_w, r) * 0.3
        lr = 0.005
        best_err = np.inf
        best = None
        
        for it in range(max_iters):
            recon = np.einsum('ia,ja,ka->ijk', U, V, W)
            diff = recon - T
            err = np.sum(diff**2)
            if err < best_err:
                best_err = err
                best = (U.copy(), V.copy(), W.copy())
            if err < 1e-14:
                break
            dU = 2.0 * np.einsum('ijk,ja,ka->ia', diff, V, W)
            dV = 2.0 * np.einsum('ijk,ia,ka->ja', diff, U, W)
            dW = 2.0 * np.einsum('ijk,ia,ja->ka', diff, U, V)
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        if best_err < 1e-12:
            return best, best_err
    return best, best_err

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Fallback: rank-35 Strassen
    U35, V35, W35 = _strassen_35(n, m, p)
    fallback = {"U": U35, "V": V35, "W": W35}
    
    deadline = time.time() + 120
    
    # Try lower ranks
    for r in [30, 31, 32, 33, 34]:
        if time.time() > deadline:
            break
        params, err = _try_rank_numerical(T, r, max_iters=1500, seed=r)
        if params is not None and err < 1e-10:
            U, V, W = params
            # Snap to half-integers and verify
            U_s = np.array([[_snap_to_half(x) for x in row] for row in U])
            V_s = np.array([[_snap_to_half(x) for x in row] for row in V])
            W_s = np.array([[_snap_to_half(x) for x in row] for row in W])
            recon = np.einsum('ia,ja,ka->ijk', U_s, V_s, W_s)
            if np.max(np.abs(recon - T)) < 1e-10:
                U_l = [[float(x) for x in row] for row in U_s]
                V_l = [[float(x) for x in row] for row in V_s]
                W_l = [[float(x) for x in row] for row in W_s]
                return {"U": U_l, "V": V_l, "W": W_l}
    
    return fallback
# EVOLVE-BLOCK-END
