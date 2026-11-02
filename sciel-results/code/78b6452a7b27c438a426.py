"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build non-zero tensor indices
    nz = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nz.append((i * m + j, j * p + k, k * n + i))
    
    # Schoolbook
    rank_s = n * m * p
    Us = np.zeros((nm, rank_s))
    Vs = np.zeros((mp, rank_s))
    Ws = np.zeros((pn, rank_s))
    for idx, (a, b, c) in enumerate(nz):
        Us[a, idx] = 1
        Vs[b, idx] = 1
        Ws[c, idx] = 1
    
    best = (rank_s, Us, Vs, Ws)
    deadline = time.time() + 110
    
    # Try lower ranks with alternating minimization
    for rank in range(rank_s - 1, 30, -1):
        if time.time() > deadline:
            break
        for trial in range(5):
            if time.time() > deadline:
                break
            np.random.seed(trial * 1000 + rank)
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            converged = False
            for it in range(100):
                if time.time() > deadline:
                    break
                
                # Solve W
                M = (U[:, None, :] * V[None, :, :]).reshape(nm * mp, rank)
                Tf = np.zeros((nm * mp, pn))
                for (a, b, c) in nz:
                    Tf[a * mp + b, c] = 1
                try:
                    W, _, _, _ = np.linalg.lstsq(M, Tf, rcond=None)
                except:
                    break
                
                # Solve U
                N = (V[:, None, :] * W[None, :, :]).reshape(mp * pn, rank)
                Tr = np.zeros((nm, mp * pn))
                for (a, b, c) in nz:
                    Tr[a, b * pn + c] = 1
                U_new = np.zeros((nm, rank))
                for a in range(nm):
                    try:
                        U_new[a], _, _, _ = np.linalg.lstsq(N.T, Tr[a], rcond=None)
                    except:
                        break
                U = U_new
                
                # Check
                if it % 30 == 0:
                    res = sum((np.dot(U[a], V[b] * W[c]) - 1)**2 for (a,b,c) in nz)
                    if res < 1e-8:
                        converged = True
                        break
            
            if converged:
                res = sum((np.dot(U[a], V[b] * W[c]) - 1)**2 for (a,b,c) in nz)
                if res < 1e-6:
                    best = (rank, U, V, W)
                    break
    
    r, U, V, W = best
    
    # Round to half-integers if close
    for arr in [U, V, W]:
        rounded = np.round(arr * 2) / 2
        if np.max(np.abs(arr - rounded)) < 1e-3:
            arr[:] = rounded
    
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
# EVOLVE-BLOCK-END

