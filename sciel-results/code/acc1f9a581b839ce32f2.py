import numpy as np
from itertools import product
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build tensor T
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
                
    # Precompute flattened RHS matrices for faster ALS
    # R_V[b, :] = T[:, b, :].flatten()
    R_V = np.zeros((mp, nm * pn))
    for b in range(mp):
        R_V[b, :] = T[:, b, :].flatten()
        
    # R_U[a, :] = T[a, :, :].flatten()
    R_U = np.zeros((nm, mp * pn))
    for a in range(nm):
        R_U[a, :] = T[a, :, :].flatten()
        
    # R_W[c, :] = T[:, :, c].flatten()
    R_W = np.zeros((pn, nm * mp))
    for c in range(pn):
        R_W[c, :] = T[:, :, c].flatten()

    # Schoolbook solution
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
                
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    start_time = time.time()
    deadline = start_time + 115 # Safety margin

    # Try ranks from a lower bound up
    # Lower bound for 3x5x5 is likely around 40-60, but let's start low.
    # Max rank is 75.
    for rank in range(30, rank_school + 1):
        if time.time() > deadline:
            break
            
        found = False
        # Reduce trials if time is tight
        num_trials = 10 if rank > 50 else 20
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
                
            np.random.seed(trial * 1000 + rank)
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # ALS
            for it in range(100):
                if time.time() > deadline:
                    break
                    
                # Update V
                # R_V = V @ M_V.T  => V = R_V @ pinv(M_V.T)
                M_V = np.zeros((nm * pn, rank))
                for r in range(rank):
                    M_V[:, r] = np.outer(U[:, r], W[:, r]).flatten()
                # Solve R_V = V @ M_V.T
                # M_V.T is (rank, nm*pn)
                # V is (mp, rank)
                # R_V is (mp, nm*pn)
                # V = R_V @ (M_V.T)^+
                V = np.linalg.lstsq(M_V.T, R_V.T, rcond=None)[0].T
                
                # Update U
                # R_U = U @ M_U.T
                M_U = np.zeros((mp * pn, rank))
                for r in range(rank):
                    M_U[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                U = np.linalg.lstsq(M_U.T, R_U.T, rcond=None)[0].T
                
                # Update W
                # R_W = W @ M_W.T
                M_W = np.zeros((nm * mp, rank))
                for r in range(rank):
                    M_W[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                W = np.linalg.lstsq(M_W.T, R_W.T, rcond=None)[0].T
                
            # Check validity
            # Reconstruct T
            T_rec = np.zeros((nm, mp, pn))
            for r in range(rank):
                T_rec += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                
            err = np.max(np.abs(T_rec - T))
            
            if err < 1e-6:
                # Try rounding to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                T_rec2 = np.zeros((nm, mp, pn))
                for r in range(rank):
                    T_rec2 += np.einsum('i,j,k->ijk', U_rounded[:, r], V_rounded[:, r], W_rounded[:, r])
                err2 = np.max(np.abs(T_rec2 - T))
                
                if err2 < 1e-9:
                    best_rank = rank
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    found = True
                    break
                else:
                    # Try integer rounding
                    U_rounded2 = np.round(U)
                    V_rounded2 = np.round(V)
                    W_rounded2 = np.round(W)
                    T_rec3 = np.zeros((nm, mp, pn))
                    for r in range(rank):
                        T_rec3 += np.einsum('i,j,k->ijk', U_rounded2[:, r], V_rounded2[:, r], W_rounded2[:, r])
                    err3 = np.max(np.abs(T_rec3 - T))
                    if err3 < 1e-9:
                        best_rank = rank
                        best_U = U_rounded2
                        best_V = V_rounded2
                        best_W = W_rounded2
                        found = True
                        break
                        
        if found:
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
