import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Schoolbook rank = n*m*p = 45
    rank_school = n * m * p
    
    def schoolbook():
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    def to_list(mat):
        return [[int(x) if x == int(x) else float(x) for x in row] for row in mat]
    
    def verify(U, V, W, tol=1e-9):
        rank = U.shape[1]
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        T_computed = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_computed += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(T, T_computed, atol=tol)

    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Try to find a lower rank decomposition
    best_rank = rank_school
    best_U, best_V, best_W = schoolbook()

    # Try ranks from 44 down to 30
    for target_rank in range(44, 29, -1):
        if time.time() > deadline - 10:
            break
        
        # Try a few seeds
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values
            rng = np.random.RandomState(seed)
            U = rng.randn(n * m, target_rank)
            V = rng.randn(m * p, target_rank)
            W = rng.randn(p * n, target_rank)
            
            # ALS iterations
            for it in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                for a in range(n * m):
                    M_full = np.einsum('br,cr->bcr', V, W).reshape(target_rank, -1)
                    rhs = T[a, :, :].ravel()
                    U[a, :] = np.linalg.lstsq(M_full.T, rhs, rcond=None)[0]
                
                # Update V
                for b in range(m * p):
                    M_full = np.einsum('ar,cr->acr', U, W).reshape(target_rank, -1)
                    rhs = T[:, b, :].ravel()
                    V[b, :] = np.linalg.lstsq(M_full.T, rhs, rcond=None)[0]
                
                # Update W
                for c in range(p * n):
                    M_full = np.einsum('ar,br->abr', U, V).reshape(target_rank, -1)
                    rhs = T[:, :, c].ravel()
                    W[c, :] = np.linalg.lstsq(M_full.T, rhs, rcond=None)[0]
                
                # Compute error
                T_computed = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T_computed - T)
                if err < 1e-6:
                    break
            
            if err < 1e-6:
                if verify(U, V, W, tol=1e-9):
                    best_rank = target_rank
                    best_U, best_V, best_W = U, V, W
                    break
        
        if best_rank < rank_school:
            break

    if best_rank < rank_school:
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
    else:
        U_school, V_school, W_school = schoolbook()
        return {"U": to_list(U_school), "V": to_list(V_school), "W": to_list(W_school)}
