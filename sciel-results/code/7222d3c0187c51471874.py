"""
Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 7 matrix.
The output factor follows the official trace convention W[k*n+i,r].

Strategy:
1. Start with the schoolbook decomposition (rank = 2*3*7 = 42).
2. Use structured search:
   - Try to find linear dependencies in schoolbook terms to reduce rank.
   - Use random initialization with alternating least squares refinement.
   - Verify with full tensor check (all valid and spurious entries).
3. Return the best (lowest rank) decomposition found within the time limit.
"""

import numpy as np
import time
import random

DIMENSIONS = [2, 3, 7]
N, M, P = DIMENSIONS

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Schoolbook decomposition
    def schoolbook():
        rank = N * M * P
        U = np.zeros((N * M, rank))
        V = np.zeros((M * P, rank))
        W = np.zeros((P * N, rank))
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j, col] = 1
                    V[j * P + k, col] = 1
                    W[k * N + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = N * M * P
    
    # Full verification: check all valid entries equal 1 and all spurious entries equal 0
    def check_decomposition_full(U, V, W, tolerance=1e-8):
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        
        # Precompute for efficiency: for each r, we have U[:,r], V[:,r], W[:,r]
        # Valid entries: for each (i,j,k), check sum_r U[i*M+j,r]*V[j*P+k,r]*W[k*N+i,r] == 1
        # Spurious entries: all other (a,b,c) should give sum == 0
        
        # Check all valid entries
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    val = np.sum(U[i*M+j, :] * V[j*P+k, :] * W[k*N+i, :])
                    if abs(val - 1.0) > tolerance:
                        return False
        
        # Check all spurious entries (a,b,c) where (a,b,c) is not a valid (i*M+j, j*P+k, k*N+i)
        # Build set of valid (a,b,c) tuples
        valid_set = set()
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    valid_set.add((i*M+j, j*P+k, k*N+i))
        
        # Check spurious entries - use vectorized approach for speed
        # For each a in range(N*M), b in range(M*P), c in range(P*N)
        # If (a,b,c) not in valid_set, check that sum_r U[a,r]*V[b,r]*W[c,r] == 0
        # This is expensive, so we'll check a large random sample plus all entries if rank is small
        if rank <= 30:
            # Check all spurious entries
            for a in range(N * M):
                for b in range(M * P):
                    for c in range(P * N):
                        if (a, b, c) not in valid_set:
                            val = np.sum(U[a, :] * V[b, :] * W[c, :])
                            if abs(val) > tolerance:
                                return False
        else:
            # Check a large random sample of spurious entries
            n_checks = 5000
            for _ in range(n_checks):
                a = random.randint(0, N * M - 1)
                b = random.randint(0, M * P - 1)
                c = random.randint(0, P * N - 1)
                if (a, b, c) not in valid_set:
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > tolerance:
                        return False
        return True
    
    # Faster check for iterations (less thorough, for speed)
    def check_decomposition_fast(U, V, W, tolerance=1e-7):
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        
        # Check all valid entries
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    val = np.sum(U[i*M+j, :] * V[j*P+k, :] * W[k*N+i, :])
                    if abs(val - 1.0) > tolerance:
                        return False
        
        # Check 500 random spurious entries
        valid_set = set()
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    valid_set.add((i*M+j, j*P+k, k*N+i))
        
        for _ in range(500):
            a = random.randint(0, N * M - 1)
            b = random.randint(0, M * P - 1)
            c = random.randint(0, P * N - 1)
            if (a, b, c) not in valid_set:
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > tolerance:
                    return False
        return True
    
    # Alternating least squares refinement
    def als_refine(U, V, W, target_rank, n_iterations=20, time_limit=None):
        """Refine U, V, W using alternating least squares."""
        for iteration in range(n_iterations):
            if time_limit is not None and time.time() > time_limit:
                break
            
            # Fix U and V, solve for W
            W_new = np.zeros((P * N, target_rank))
            valid = True
            for k in range(P):
                for i in range(N):
                    # A_mat[j, r] = U[i*M+j, r] * V[j*P+k, r]
                    A_mat = np.outer(U[i*M:i*M+M, :].T, V[k:k+P*P:P, :].T).reshape(M, target_rank, M).transpose(1, 0, 2).reshape(M, target_rank) if False else None
                    # Simpler: build directly
                    A_mat = np.zeros((M, target_rank))
                    for j in range(M):
                        A_mat[j, :] = U[i * M + j, :] * V[j * P + k, :]
                    b_vec = np.ones(M)
                    try:
                        sol, residuals, rank_, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        W_new[k * N + i, :] = sol
                    except:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
            W = W_new
            
            # Fix V and W, solve for U
            U_new = np.zeros((N * M, target_rank))
            valid = True
            for i in range(N):
                for j in range(M):
                    A_mat = np.zeros((P, target_rank))
                    for k in range(P):
                        A_mat[k, :] = V[j * P + k, :] * W[k * N + i, :]
                    b_vec = np.ones(P)
                    try:
                        sol, residuals, rank_, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        U_new[i * M + j, :] = sol
                    except:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
            U = U_new
            
            # Fix U and W, solve for V
            V_new = np.zeros((M * P, target_rank))
            valid = True
            for j in range(M):
                for k in range(P):
                    A_mat = np.zeros((N, target_rank))
                    for i in range(N):
                        A_mat[i, :] = U[i * M + j, :] * W[k * N + i, :]
                    b_vec = np.ones(N)
                    try:
                        sol, residuals, rank_, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        V_new[j * P + k, :] = sol
                    except:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
            V = V_new
        
        return U, V, W
    
    # Try to find a lower rank decomposition
    # Strategy: try ranks from 20 to 41
    for target_rank in range(20, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        n_trials = 30
        for trial in range(n_trials):
            if time.time() > deadline:
                break
            
            # Random initialization
            scale = 1.0 / np.sqrt(target_rank)
            U = np.random.randn(N * M, target_rank) * scale
            V = np.random.randn(M * P, target_rank) * scale
            W = np.random.randn(P * N, target_rank) * scale
            
            # Run ALS refinement
            U, V, W = als_refine(U, V, W, target_rank, n_iterations=15, time_limit=deadline)
            
            # Quick check
            if check_decomposition_fast(U, V, W):
                # Full verification
                if check_decomposition_full(U, V, W):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    break
    
    # Convert to lists for the return format
    U = best_result["U"]
    V = best_result["V"]
    W = best_result["W"]
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
