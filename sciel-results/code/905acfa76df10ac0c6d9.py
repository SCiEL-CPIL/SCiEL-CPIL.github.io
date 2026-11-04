"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    def verify(U, V, W, tol=1e-9):
        """Verify decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
        U_np = np.array(U) if not isinstance(U, np.ndarray) else U
        V_np = np.array(V) if not isinstance(V, np.ndarray) else V
        W_np = np.array(W) if not isinstance(W, np.ndarray) else W
        rank = U_np.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_sb = np.zeros((n*m, rank_school), dtype=float)
    V_sb = np.zeros((m*p, rank_school), dtype=float)
    W_sb = np.zeros((p*n, rank_school), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank using structured random search
    np.random.seed(42)
    
    # Try ranks from 60 to 119
    for target_rank in range(60, best_rank):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Structured random: each term covers a block
            U = np.zeros((n*m, target_rank), dtype=float)
            V = np.zeros((m*p, target_rank), dtype=float)
            W = np.zeros((p*n, target_rank), dtype=float)
            
            for r in range(target_rank):
                # Assign each term to a random subset of indices
                # U: choose which (i,j) pairs this term affects
                ui = np.random.randint(0, n, 2)
                uj = np.random.randint(0, m, 2)
                vk = np.random.randint(0, p, 2)
                
                for ii in ui:
                    for jj in uj:
                        U[ii*m+jj, r] = 1.0
                for jj in uj:
                    for kk in vk:
                        V[jj*p+kk, r] = 1.0
                for kk in vk:
                    for ii in ui:
                        W[kk*n+ii, r] = 1.0
            
            if verify(U, V, W):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                found = True
                break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
