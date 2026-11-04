"""Find rank-minimal exact tensor decomposition for 2x5 x 5x6 matrix multiplication."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 5, 6]
_START_TIME = time.time()
DEADLINE = _START_TIME + 140

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _verify_numpy(U, V, W, n, m, p):
    """Vectorized verification."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _rationalize(U, V, W, n, m, p):
    """Try to round to nearest half-integers and verify."""
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_r[i*m+j], V_r[j*p+k] * W_r[k*n+i])
                if abs(val - 1) > 1e-9:
                    return None
    
    rank = U.shape[1]
    U_list = [[float(U_r[i,j]) for j in range(rank)] for i in range(U_r.shape[0])]
    V_list = [[float(V_r[i,j]) for j in range(rank)] for i in range(V_r.shape[0])]
    W_list = [[float(W_r[i,j]) for j in range(rank)] for i in range(W_r.shape[0])]
    return {"U": U_list, "V": V_list, "W": W_list}

def _try_rank_als(n, m, p, rank, deadline, seed_offset=0):
    """Alternating least squares to find a rank decomposition."""
    max_restarts = 30
    max_iter = 150
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        rng = np.random.RandomState(12345 + restart * 7919 + seed_offset)
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U rows
            for i in range(n):
                for j in range(m):
                    M = np.empty((p, rank))
                    for k in range(p):
                        M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass
            
            # Update V rows
            for j in range(m):
                for k in range(p):
                    N_mat = np.empty((n, rank))
                    for i in range(n):
                        N_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(N_mat.T, np.ones(n), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass
            
            # Update W rows
            for k in range(p):
                for i in range(n):
                    P_mat = np.empty((m, rank))
                    for j in range(m):
                        P_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(P_mat.T, np.ones(m), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass
            
            # Check convergence
            if it % 10 == 9:
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            max_err = max(max_err, abs(val - 1))
                
                if max_err < 1e-7:
                    result = _rationalize(U, V, W, n, m, p)
                    if result:
                        return result
                elif max_err < 1e-4:
                    # Try a final polish
                    for _ in range(50):
                        for i in range(n):
                            for j in range(m):
                                M = np.empty((p, rank))
                                for k in range(p):
                                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                                try:
                                    U[i * m + j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                                except:
                                    pass
                        for j in range(m):
                            for k in range(p):
                                N_mat = np.empty((n, rank))
                                for i in range(n):
                                    N_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                                try:
                                    V[j * p + k, :] = np.linalg.lstsq(N_mat.T, np.ones(n), rcond=None)[0]
                                except:
                                    pass
                        for k in range(p):
                            for i in range(n):
                                P_mat = np.empty((m, rank))
                                for j in range(m):
                                    P_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                                try:
                                    W[k * n + i, :] = np.linalg.lstsq(P_mat.T, np.ones(m), rcond=None)[0]
                                except:
                                    pass
                        if time.time() > deadline:
                            break
                    result = _rationalize(U, V, W, n, m, p)
                    if result:
                        return result
    
    return None

def _try_rank_sparse(n, m, p, rank, deadline, seed_offset=0):
    """Try sparse random initializations with values in {-1, -0.5, 0, 0.5, 1}."""
    vals = [-1, -0.5, 0, 0.5, 1]
    max_restarts = 40
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        rng = np.random.RandomState(777 + restart * 31 + seed_offset)
        U = np.array([[rng.choice(vals) for _ in range(rank)] for _ in range(n*m)])
        V = np.array([[rng.choice(vals) for _ in range(rank)] for _ in range(m*p)])
        W = np.array([[rng.choice(vals) for _ in range(rank)] for _ in range(p*n)])
        
        if _verify_numpy(U, V, W, n, m, p):
            U_list = [[float(U[i,j]) for j in range(rank)] for i in range(U.shape[0])]
            V_list = [[float(V[i,j]) for j in range(rank)] for i in range(V.shape[0])]
            W_list = [[float(W[i,j]) for j in range(rank)] for i in range(W.shape[0])]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try ranks from 55 down to 35
    for R in range(55, 34, -1):
        if time.time() > DEADLINE - 15:
            break
        
        # First try sparse random
        result = _try_rank_sparse(n, m, p, R, DEADLINE - 10, seed_offset=R)
        if result is not None:
            best = result
            best_rank = R
            break
        
        # Then try ALS
        if time.time() > DEADLINE - 10:
            break
        result = _try_rank_als(n, m, p, R, DEADLINE - 10, seed_offset=R * 100)
        if result is not None:
            best = result
            best_rank = R
            break
    
    return best
