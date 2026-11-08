"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS
    best_rank = n * m * p
    best_result = None
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find lower rank decompositions
    for rank in range(25, best_rank):
        if time.time() > deadline - 15.0:
            break
        
        found = False
        for trial in range(10):
            if time.time() > deadline - 5.0:
                break
            
            np.random.seed(42 + rank * 100 + trial)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # ALS optimization
            for iteration in range(300):
                if time.time() > deadline - 3.0:
                    break
                
                # Update U
                for c in range(rank):
                    M_c = np.outer(V[:,c], W[:,c])
                    denom = np.sum(M_c**2)
                    if denom < 1e-12:
                        continue
                    TC = np.einsum('ijk,jk->i', T, M_c)
                    # Compute MC vector
                    MC = np.array([np.sum(M_c * np.outer(V[:,c2], W[:,c2])) for c2 in range(rank)])
                    Uc = TC / denom
                    for c2 in range(rank):
                        if c2 != c:
                            Uc -= U[:,c2] * MC[c2] / denom
                    U[:,c] = Uc
                
                # Update V
                for c in range(rank):
                    M_c = np.outer(U[:,c], W[:,c])
                    denom = np.sum(M_c**2)
                    if denom < 1e-12:
                        continue
                    TC = np.einsum('ijk,ik->j', T, M_c)
                    MC = np.array([np.sum(M_c * np.outer(U[:,c2], W[:,c2])) for c2 in range(rank)])
                    Vc = TC / denom
                    for c2 in range(rank):
                        if c2 != c:
                            Vc -= V[:,c2] * MC[c2] / denom
                    V[:,c] = Vc
                
                # Update W
                for c in range(rank):
                    M_c = np.outer(U[:,c], V[:,c])
                    denom = np.sum(M_c**2)
                    if denom < 1e-12:
                        continue
                    TC = np.einsum('ijk,ij->k', T, M_c)
                    MC = np.array([np.sum(M_c * np.outer(U[:,c2], V[:,c2])) for c2 in range(rank)])
                    Wc = TC / denom
                    for c2 in range(rank):
                        if c2 != c:
                            Wc -= W[:,c2] * MC[c2] / denom
                    W[:,c] = Wc
                
                # Check error every 10 iterations
                if iteration % 10 == 0 or iteration == 299:
                    recon = np.einsum('ic,jc,kc->ijk', U, V, W)
                    err = np.linalg.norm(T - recon)
                    if err < 1e-4:
                        # Snap to half-integers
                        U_snap = np.round(U * 2) / 2
                        V_snap = np.round(V * 2) / 2
                        W_snap = np.round(W * 2) / 2
                        recon_snap = np.einsum('ic,jc,kc->ijk', U_snap, V_snap, W_snap)
                        if np.linalg.norm(T - recon_snap) < 1e-4:
                            U_list = [[float(U_snap[i,c]) for c in range(rank)] for i in range(n*m)]
                            V_list = [[float(V_snap[j,c]) for c in range(rank)] for j in range(m*p)]
                            W_list = [[float(W_snap[k,c]) for c in range(rank)] for k in range(p*n)]
                            if best_result is None or rank < best_rank:
                                best_rank = rank
                                best_result = {"U": U_list, "V": V_list, "W": W_list}
                            found = True
                            break
        
        if found:
            break
    
    if best_result is None:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
