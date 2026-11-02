"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 3
    deadline = time.time() + 140  # 10s safety margin
    
    # Target tensor: T[i,j,k] = 1 for all i,j,k
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Start with schoolbook rank-27
    rank_27 = n * m * p
    U27 = np.zeros((n*m, rank_27))
    V27 = np.zeros((m*p, rank_27))
    W27 = np.zeros((p*n, rank_27))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U27[i*m+j, col] = 1
                V27[j*p+k, col] = 1
                W27[k*n+i, col] = 1
                col += 1
    
    # Verify schoolbook
    def check(U, V, W):
        rank = U.shape[1]
        # Build tensor
        T = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            # u[i,j] * v[j,k] * w[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        return np.allclose(T, 1.0, atol=1e-9)
    
    if check(U27, V27, W27):
        # Try to find lower rank via search
        best_rank = rank_27
        best_U, best_V, best_W = U27, V27, W27
        
        # Try ranks from 23 to 26
        for target_rank in range(23, rank_27):
            if time.time() > deadline - 30:
                break
            # Use random initialization and local search
            found = False
            for attempt in range(20):
                if time.time() > deadline - 30:
                    break
                U = np.zeros((n*m, target_rank))
                V = np.zeros((m*p, target_rank))
                W = np.zeros((p*n, target_rank))
                # Initialize with random ±1/2, 0, 1
                for r in range(target_rank):
                    U[:, r] = np.random.choice([0, 0.5, 1, -0.5, -1], size=n*m)
                    V[:, r] = np.random.choice([0, 0.5, 1, -0.5, -1], size=m*p)
                    W[:, r] = np.random.choice([0, 0.5, 1, -0.5, -1], size=p*n)
                if check(U, V, W):
                    found = True
                    break
            
            if found:
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
        
        # Convert to lists
        U_list = [[float(U27[i, r]) if best_rank == rank_27 else float(best_U[i, r]) for r in range(best_rank)] for i in range(n*m)]
        V_list = [[float(V27[i, r]) if best_rank == rank_27 else float(best_V[i, r]) for r in range(best_rank)] for i in range(m*p)]
        W_list = [[float(W27[i, r]) if best_rank == rank_27 else float(best_W[i, r]) for r in range(best_rank)] for i in range(p*n)]
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback: return schoolbook
        U_list = [[float(U27[i, r]) for r in range(rank_27)] for i in range(n*m)]
        V_list = [[float(V27[i, r]) for r in range(rank_27)] for i in range(m*p)]
        W_list = [[float(W27[i, r]) for r in range(rank_27)] for i in range(p*n)]
        return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
