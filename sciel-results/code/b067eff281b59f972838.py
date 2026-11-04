"""
Matrix multiplication tensor decomposition for 3x4x6.
Uses an alternating projection / sparse factorization approach to seek
a rank-minimal exact decomposition. Falls back to schoolbook rank 72.

The tensor T for C = AB (A: 3x4, B: 4x6, C: 3x6) is defined by:
  C[i,k] = sum_j A[i,j] * B[j,k]

In the trace convention with output indexed as W[k*n + i, r]:
  sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all valid (i,j,k)

This is the standard matrix multiplication tensor. The schoolbook rank is n*m*p = 72.

We attempt to find a lower-rank decomposition using:
1. Structured search based on known tensor identities
2. Alternating least squares with integer/half-integer rounding
3. Greedy column removal / merging
"""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 72
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = schoolbook_rank
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    # Verify a decomposition
    def verify(U, V, W, tol=1e-6):
        rank = U.shape[1]
        # Check: sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c] for all a,b,c
        # Efficient: for each r, outer product U[:,r] * V[:,r] * W[:,r]
        recon = np.zeros((n * m, m * p, p * n), dtype=float)
        for r in range(rank):
            recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.max(np.abs(recon - T)) < tol
    
    # Try to find a decomposition of rank r
    def try_rank(r, max_iters=200, seed=42):
        rng = np.random.RandomState(seed)
        
        # Initialize with random factors
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        W = rng.randn(p * n, r)
        
        # Normalize columns to prevent blowup
        for col in range(r):
            norms = [np.linalg.norm(U[:, col]), np.linalg.norm(V[:, col]), np.linalg.norm(W[:, col])]
            avg = np.mean(norms)
            if avg > 0:
                U[:, col] /= avg
                V[:, col] /= avg
                W[:, col] /= avg
        
        for it in range(max_iters):
            # Alternating least squares
            # Fix V, W: solve for U
            # T[:, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
            # For each a: T[a, :, :] = sum_r U[a, r] * (V * W)[:, :, r]
            # This is a linear system in U[a, :]
            
            # Build the system: for each a, we have (m*p*p*n) equations
            # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
            # Let's form matrix M of shape (m*p*p*n, r) where M[b*p*n + c, r] = V[b, r] * W[c, r]
            M = np.zeros((m * p * p * n, r), dtype=float)
            for b in range(m * p):
                for c in range(p * n):
                    M[b * p * n + c, :] = V[b, :] * W[c, :]
            
            # For each a, solve M^T M * U[a, :] = M^T * T[a, :, :].flatten()
            # Actually, least squares: min ||M @ U[a,:] - T[a,:,:].flatten()||
            A_mat = M.T  # (r, m*p*p*n)
            AAT = A_mat @ M  # (r, r)
            
            for a in range(n * m):
                rhs = A_mat @ T[a, :, :].flatten()
                try:
                    sol = np.linalg.solve(AAT + 1e-8 * np.eye(r), rhs)
                except np.linalg.LinAlgError:
                    sol = np.linalg.lstsq(AAT, rhs, rcond=None)[0]
                U[a, :] = sol
            
            # Fix U, W: solve for V
            # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
            # For each b: T[:, b, c] = sum_r V[b, r] * (U * W)[a, c, r]
            M2 = np.zeros((n * m * p * n, r), dtype=float)
            for a in range(n * m):
                for c in range(p * n):
                    M2[a * p * n + c, :] = U[a, :] * W[c, :]
            
            AAT2 = M2.T @ M2
            for b in range(m * p):
                rhs = M2.T @ T[:, b, :].flatten()
                try:
                    sol = np.linalg.solve(AAT2 + 1e-8 * np.eye(r), rhs)
                except np.linalg.LinAlgError:
                    sol = np.linalg.lstsq(AAT2, rhs, rcond=None)[0]
                V[b, :] = sol
            
            # Fix U, V: solve for W
            # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
            # For each c: T[a, b, c] = sum_r W[c, r] * (U * V)[a, b, r]
            M3 = np.zeros((n * m * m * p, r), dtype=float)
            for a in range(n * m):
                for b in range(m * p):
                    M3[a * m * p + b, :] = U[a, :] * V[b, :]
            
            AAT3 = M3.T @ M3
            for c in range(p * n):
                rhs = M3.T @ T[:, :, c].flatten()
                try:
                    sol = np.linalg.solve(AAT3 + 1e-8 * np.eye(r), rhs)
                except np.linalg.LinAlgError:
                    sol = np.linalg.lstsq(AAT3, rhs, rcond=None)[0]
                W[c, :] = sol
            
            # Check convergence
            if it % 10 == 0:
                recon = np.zeros((n * m, m * p, p * n), dtype=float)
                for r_idx in range(r):
                    recon += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
                err = np.max(np.abs(recon - T))
                if err < 1e-4:
                    # Try rounding to integers/half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if verify(U_r, V_r, W_r):
                        return U_r, V_r, W_r
                    # Also try pure integer rounding
                    U_i = np.round(U)
                    V_i = np.round(V)
                    W_i = np.round(W)
                    if verify(U_i, V_i, W_i):
                        return U_i, V_i, W_i
        
        return None
    
    # Try ranks below schoolbook
    # For 3x4x6, known lower bounds suggest rank >= 54 (from asymptotic bounds)
    # But exact optimal is unknown. Try a few ranks.
    
    best_result = None
    start_time = time.time()
    
    # Try a few candidate ranks
    candidate_ranks = [60, 58, 56, 54, 52, 50]
    
    for r in candidate_ranks:
        if time.time() - start_time > 100:
            break
        result = try_rank(r, max_iters=100, seed=random.randint(0, 10000))
        if result is not None:
            best_result = result
            break
    
    # If no better decomposition found, return schoolbook
    if best_result is None:
        U, V, W = schoolbook()
    else:
        U, V, W = best_result
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
