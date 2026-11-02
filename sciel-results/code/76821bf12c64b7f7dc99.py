"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Build the multiplication tensor
    T = np.zeros((n*m, m*p, n*p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((n*p, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    def verify(U, V, W):
        r = U.shape[1]
        T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.allclose(T_recon, T, atol=1e-9)
    
    # Try to find lower rank using structured search
    # Strategy: use the fact that we can add linear combinations
    
    # For each pair of schoolbook terms, check if they can be merged
    # Term t1: (i1,j1,k1), Term t2: (i2,j2,k2)
    # They can be merged if there exist scalars such that
    # e_{i1m+j1}⊗e_{j1p+k1}⊗e_{k1n+i1} + e_{i2m+j2}⊗e_{j2p+k2}⊗e_{k2n+i2}
    # can be replaced by a single term (which requires them to be proportional, which they're not)
    
    # So direct merging doesn't work. We need more sophisticated approaches.
    
    # Try: use Strassen's idea of 2x2x2 = 7
    # Construct a 2x2x2 subproblem and see if it helps
    
    # For now, return schoolbook as we haven't found a better solution
    U_list = [[float(best_U[i, j]) for j in range(best_U.shape[1])] for i in range(best_U.shape[0])]
    V_list = [[float(best_V[i, j]) for j in range(best_V.shape[1])] for i in range(best_V.shape[0])]
    W_list = [[float(best_W[i, j]) for j in range(best_W.shape[1])] for i in range(best_W.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

