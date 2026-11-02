"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Compute the tensor: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _snap_to_half_integer(x, tol=1e-9):
    """Snap a value to the nearest half-integer if close."""
    half = round(x * 2) / 2.0
    if abs(x - half) < tol:
        return half
    return x

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    
    # Try to find a lower rank decomposition
    # Known: r(2,5,5) = 24, so r(2,5,6) <= 24 + something
    # Let's try ranks from 25 upward
    
    np.random.seed(42)
    
    for target_rank in range(25, 61):
        if time.time() > deadline - 30:
            break
        
        attempts = 0
        max_attempts = 500 if target_rank < 35 else 200
        
        while attempts < max_attempts and time.time() < deadline - 30:
            attempts += 1
            
            # Random initialization
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Normalize columns to avoid scale issues
            for c in range(target_rank):
                norm_U = np.linalg.norm(U[:, c])
                norm_V = np.linalg.norm(V[:, c])
                norm_W = np.linalg.norm(W[:, c])
                if norm_U < 1e-10 or norm_V < 1e-10 or norm_W < 1e-10:
                    continue
                # Scale so that product of norms is 1
                scale = (norm_U * norm_V * norm_W) ** (-1/3)
                U[:, c] *= scale
                V[:, c] *= scale
                W[:, c] *= scale
            
            # Check if close to a valid solution
            # Compute the error
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(target_rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        max_err = max(max_err, abs(s - 1.0))
            
            if max_err < 0.5:
                # Try to refine with local search
                for _ in range(50):
                    if time.time() > deadline - 30:
                        break
                    # Perturb
                    c = np.random.randint(target_rank)
                    U[:, c] += 0.1 * np.random.randn(n*m)
                    V[:, c] += 0.1 * np.random.randn(m*p)
                    W[:, c] += 0.1 * np.random.randn(p*n)
                    
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for cc in range(target_rank):
                                    s += U[i*m+j, cc] * V[j*p+k, cc] * W[k*n+i, cc]
                                max_err = max(max_err, abs(s - 1.0))
                    
                    if max_err < 1e-8:
                        # Snap to half-integers
                        U_snap = np.array([[_snap_to_half_integer(x) for x in row] for row in U])
                        V_snap = np.array([[_snap_to_half_integer(x) for x in row] for row in V])
                        W_snap = np.array([[_snap_to_half_integer(x) for x in row] for row in W])
                        
                        if _verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), n, m, p):
                            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                        break
    
    return best_result
# EVOLVE-BLOCK-END
