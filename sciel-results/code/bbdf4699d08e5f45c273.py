"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s to be safe
    
    # The multiplication tensor: T[i, j, k] = 1 for all i,j,k
    # We need U (n*m x rank), V (m*p x rank), W (p*n x rank) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Schoolbook: rank = n*m*p = 60
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Check if a decomposition is exact (integer or half-integer)
    def is_exact(U, V, W, tol=1e-10):
        # Check all entries are integer or half-integer
        for mat in [U, V, W]:
            for val in mat.flatten():
                if abs(val) < tol:
                    continue
                # Check if val is close to integer or half-integer
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2.0
                if abs(val - nearest_int) > tol and abs(val - nearest_half) > tol:
                    return False
        return True
    
    # Verify the decomposition is correct
    def verify(U, V, W, tol=1e-8):
        rank = U.shape[1]
        # For each i,j,k check: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try to find a low-rank decomposition using ALS
    def try_rank(rank, max_iters=200, n_restarts=5, tol=1e-12):
        if time.time() > deadline - 5:
            return None
        
        best_result = None
        best_residual = float('inf')
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Scale to help convergence
            norm_U = np.linalg.norm(U)
            norm_V = np.linalg.norm(V)
            norm_W = np.linalg.norm(W)
            if norm_U > 0: U /= norm_U
            if norm_V > 0: V /= norm_V
            if norm_W > 0: W /= norm_W
            
            prev_residual = float('inf')
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each row i*m+j, solve least squares
                for i in range(n):
                    for j in range(m):
                        # Residual for this (i,j): 1 - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for all k
                        # We want: U[i*m+j,:] * (V[j,:] outer W[:,i]) = ones(p)
                        # Let A = V[j*p:(j+1)*p, :] outer W[:, i] -> shape (p, rank)
                        A = np.zeros((p, rank))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[i, :]  # W[k*n+i] = W[i] if we index W as W[k*n+i]
                        # Actually W is indexed as W[k*n + i], so for fixed i, W[:, i] doesn't work directly
                        # Let me re-index: W has shape (p*n, rank), W[k*n + i, c]
                        # For fixed i, the vector W_i = W[::n, i] has shape (p, rank)? No.
                        # W[k*n + i] for k=0..p-1 gives rows: i, n+i, 2n+i, ...
                        W_i = W[i::n, :]  # shape (p, rank)
                        A = np.zeros((p, rank))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W_i[k, :]
                        # Solve: U[i*m+j, :] @ A.T = ones(p)
                        # A.T @ A is rank x rank, A.T @ ones(p) is rank
                        try:
                            ATA = A.T @ A
                            ATb = A.T @ np.ones(p)
                            U[i*m+j, :] = np.linalg.solve(ATA + 1e-12*np.eye(rank), ATb)
                        except:
                            pass
                
                # Update V: for each row j*p+k
                for j in range(m):
                    for k in range(p):
                        # For fixed j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i
                        # Let B = U[:, j] outer W[k*n+k, :] ... 
                        # U[i*m+j, c] for i=0..n-1: U[::m, j] shape (n, rank)
                        U_j = U[::m, j]  # shape (n, rank)
                        W_k = W[k*n, :]  # shape (rank,)
                        B = np.zeros((n, rank))
                        for i in range(n):
                            B[i, :] = U_j[i, :] * W_k
                        try:
                            BTB = B.T @ B
                            BTb = B.T @ np.ones(n)
                            V[j*p+k, :] = np.linalg.solve(BTB + 1e-12*np.eye(rank), BTb)
                        except:
                            pass
                
                # Update W: for each row k*n+i
                for k in range(p):
                    for i in range(n):
                        # For fixed k,i: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all j
                        # Let C = U[i, :] outer V[::p, k] ...
                        U_i = U[i*m:(i+1)*m, :]  # shape (m, rank)
                        V_k = V[k::p, :]  # shape (m, rank)
                        C = np.zeros((m, rank))
                        for j in range(m):
                            C[j, :] = U_i[j, :] * V_k[j, :]
                        try:
                            CTC = C.T @ C
                            CTb = C.T @ np.ones(m)
                            W[k*n+i, :] = np.linalg.solve(CTC + 1e-12*np.eye(rank), CTb)
                        except:
                            pass
                
                # Compute residual
                residual = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(rank):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            residual += (s - 1.0)**2
                residual = np.sqrt(residual / (n*m*p))
                
                if residual < best_residual:
                    best_residual = residual
                    if is_exact(U, V, W, tol=1e-8) and verify(U, V, W, tol=1e-6):
                        # Found exact solution!
                        return U, V, W, rank
                    best_result = (U.copy(), V.copy(), W.copy(), rank)
                
                if residual < tol:
                    break
                if abs(prev_residual - residual) < 1e-15:
                    break
                prev_residual = residual
        
        # Check if best_result is exact
        if best_result is not None:
            U, V, W, r = best_result
            if is_exact(U, V, W, tol=1e-8) and verify(U, V, W, tol=1e-6):
                return U, V, W, r
        
        return None
    
    # Search from low rank upward
    # Known: rank >= n*p = 12 (lower bound from one mode)
    # Try ranks from 12 up to 60
    for rank in range(12, 61):
        if time.time() > deadline - 10:
            break
        result = try_rank(rank, max_iters=100, n_restarts=3)
        if result is not None:
            U, V, W, r = result
            # Convert to list of lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    U, V, W, rank = schoolbook()
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
