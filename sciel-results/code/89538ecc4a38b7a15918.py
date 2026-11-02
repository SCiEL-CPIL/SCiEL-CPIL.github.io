import numpy as np
import time
import sys

def _build_rank14():
    """Construct the rank-14 Strassen-based decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 7 terms for 2x2*2x2
    terms = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]
    
    for block in range(2):
        base = block * 7
        for t in range(7):
            r = base + t
            uc, vc, wc = terms[t]
            for (idx, val) in uc:
                U[idx, r] += val
            for (idx, val) in vc:
                j, k_loc = idx // 2, idx % 2
                k = k_loc + block * 2
                V[j * p + k, r] += val
            for (idx, val) in wc:
                k_loc, i = idx // 2, idx % 2
                k = k_loc + block * 2
                W[k * n + i, r] += val
    return U, V, W

def _verify(U, V, W, tol=1e-9):
    n, m, p = 2, 2, 4
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank13(time_budget):
    """Attempt to find rank-13 decomposition via alternating minimization with structured init."""
    n, m, p = 2, 2, 4
    rank = 13
    start = time.time()
    
    # Multiple restarts with different seeds
    for seed in range(20):
        if time.time() - start > time_budget:
            break
        np.random.seed(seed)
        
        # Structured initialization: start from rank-14 and zero out one column,
        # then redistribute
        U14, V14, W14 = _build_rank14()
        U = U14[:, :rank].copy()
        V = V14[:, :rank].copy()
        W = W14[:, :rank].copy()
        
        # Add small random perturbation to break symmetry
        U += np.random.randn(*U.shape) * 0.1
        V += np.random.randn(*V.shape) * 0.1
        W += np.random.randn(*W.shape) * 0.1
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(500):
            if time.time() - start > time_budget:
                break
            
            # Solve U
            for i in range(n):
                for j in range(m):
                    Aij = np.zeros((p, rank))
                    for k in range(p):
                        Aij[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U[i*m+j, :], _, _, _ = np.linalg.lstsq(Aij, np.ones(p), rcond=None)
                    except:
                        pass
            
            # Solve V
            for j in range(m):
                for k in range(p):
                    Ajk = np.zeros((n, rank))
                    for i in range(n):
                        Ajk[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :], _, _, _ = np.linalg.lstsq(Ajk, np.ones(n), rcond=None)
                    except:
                        pass
            
            # Solve W
            for i in range(n):
                for k in range(p):
                    Aik = np.zeros((m, rank))
                    for j in range(m):
                        Aik[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :], _, _, _ = np.linalg.lstsq(Aik, np.ones(m), rcond=None)
                    except:
                        pass
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err += (s - 1.0)**2
            err = np.sqrt(err)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-10:
                return best_U, best_V, best_W
    
    if best_err < 1e-10:
        return best_U, best_V, best_W
    return None

def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    # Try to find rank 13
    U, V, W = _try_rank13(time_budget=120)
    if U is not None and _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to rank 14
    if time.time() > deadline:
        U, V, W = _build_rank14()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U, V, W = _build_rank14()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
