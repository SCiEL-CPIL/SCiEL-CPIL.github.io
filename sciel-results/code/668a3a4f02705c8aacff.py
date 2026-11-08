"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]
import numpy as np
from itertools import product
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build tensor T where T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    def try_rank(R, seed=42):
        """Try to find rank-R decomposition using ALS."""
        np.random.seed(seed)
        U = np.random.randn(nm, R) * 0.1
        V = np.random.randn(mp, R) * 0.1
        W = np.random.randn(pn, R) * 0.1
        
        for iteration in range(200):
            # Solve for W
            W_new = np.zeros_like(W)
            for c in range(pn):
                Z = np.zeros((nm * mp, R))
                idx = 0
                for a in range(nm):
                    for b in range(mp):
                        Z[idx, :] = U[a, :] * V[b, :]
                        idx += 1
                rhs = T[:, :, c].flatten()
                try:
                    w, _, _, _ = np.linalg.lstsq(Z, rhs, rcond=None)
                    W_new[c, :] = w
                except:
                    pass
            W = W_new
            
            # Solve for U
            U_new = np.zeros_like(U)
            for a in range(nm):
                Z = np.zeros((mp * pn, R))
                idx = 0
                for b in range(mp):
                    for c in range(pn):
                        Z[idx, :] = V[b, :] * W[c, :]
                        idx += 1
                rhs = T[a, :, :].flatten()
                try:
                    u, _, _, _ = np.linalg.lstsq(Z, rhs, rcond=None)
                    U_new[a, :] = u
                except:
                    pass
            U = U_new
            
            # Solve for V
            V_new = np.zeros_like(V)
            for b in range(mp):
                Z = np.zeros((nm * pn, R))
                idx = 0
                for a in range(nm):
                    for c in range(pn):
                        Z[idx, :] = U[a, :] * W[c, :]
                        idx += 1
                rhs = T[:, b, :].flatten()
                try:
                    v, _, _, _ = np.linalg.lstsq(Z, rhs, rcond=None)
                    V_new[b, :] = v
                except:
                    pass
            V = V_new
        
        # Verify
        T_recon = np.zeros((nm, mp, pn))
        for r in range(R):
            T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        if np.allclose(T_recon, T, atol=1e-5):
            return True, U, V, W
        return False, U, V, W
    
    # Try ranks from 1 to 18
    for R in range(1, 19):
        found, U, V, W = try_rank(R)
        if found:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
