"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

import numpy as np
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def _verify_decomposition(n, m, p, U, V, W, tol=1e-6):
    """Verify that the given factors decompose the matmul tensor correctly.
    U: shape (n*m, rank), V: shape (m*p, rank), W: shape (p*n, rank)
    Convention: W index is k*n + i, U is i*m + j, V is j*p + k.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _als_decomposition(n, m, p, rank, max_iters=200, tol=1e-10, deadline=None):
    """Alternating Least Squares for tensor decomposition.
    
    We want to find U, V, W such that:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k
    
    The ALS approach alternates between optimizing each factor given the others.
    """
    if deadline is None:
        deadline = time.time() + 100
    
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    
    # Normalize initial factors
    for c in range(rank):
        norm = np.sqrt(np.sum(U[:, c]**2) * np.sum(V[:, c]**2) * np.sum(W[:, c]**2))
        if norm > 1e-10:
            scale = 1.0 / (norm ** (1.0/3.0))
            U[:, c] *= scale
            V[:, c] *= scale
            W[:, c] *= scale
    
    prev_err = float('inf')
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline - 2:
            break
        
        # Compute current error
        err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                    err += (val - 1.0) ** 2
        
        if err < tol:
            break
        
        if abs(prev_err - err) < 1e-14:
            break
        
        prev_err = err
        
        # Optimize W given U and V
        # For each (k, i), we need: sum_c W[k*n+i, c] * (U[i*m+j, c] * V[j*p+k, c]) = 1 for all j
        # This is a system: A @ W_row = 1 where A[j, c] = U[i*m+j, c] * V[j*p+k, c]
        for i in range(n):
            for k in range(p):
                A_mat = np.zeros((m, rank))
                for j in range(m):
                    for c in range(rank):
                        A_mat[j, c] = U[i*m + j, c] * V[j*p + k, c]
                
                # Solve A_mat @ W_row = ones(m)
                if rank <= m:
                    sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                else:
                    # Underdetermined: find min-norm solution
                    sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                
                W[k*n + i, :] = sol
        
        # Optimize U given V and W
        # For each (i, j), we need: sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1 for all k
        # This is a system: B @ U_row = 1 where B[k, c] = V[j*p+k, c] * W[k*n+i, c]
        for i in range(n):
            for j in range(m):
                B_mat = np.zeros((p, rank))
                for k in range(p):
                    for c in range(rank):
                        B_mat[k, c] = V[j*p + k, c] * W[k*n + i, c]
                
                if rank <= p:
                    sol, _, _, _ = np.linalg.lstsq(B_mat, np.ones(p), rcond=None)
                else:
                    sol, _, _, _ = np.linalg.lstsq(B_mat, np.ones(p), rcond=None)
                
                U[i*m + j, :] = sol
        
        # Optimize V given U and W
        # For each (j, k), we need: sum_c V[j*p+k, c] * (U[i*m+j, c] * W[k*n+i, c]) = 1 for all i
        # This is a system: C @ V_row = 1 where C[i, c] = U[i*m+j, c] * W[k*n+i, c]
        for j in range(m):
            for k in range(p):
                C_mat = np.zeros((n, rank))
                for i in range(n):
                    for c in range(rank):
                        C_mat[i, c] = U[i*m + j, c] * W[k*n + i, c]
                
                if rank <= n:
                    sol, _, _, _ = np.linalg.lstsq(C_mat, np.ones(n), rcond=None)
                else:
                    sol, _, _, _ = np.linalg.lstsq(C_mat, np.ones(n), rcond=None)
                
                V[j*p + k, :] = sol
    
    # Final error check
    final_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                final_err += (val - 1.0) ** 2
    
    converged = final_err < 1e-6
    return U, V, W, final_err, converged

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 4*4*6 = 96
    
    # Create the schoolbook solution as fallback
    U_sch = np.zeros((n * m, schoolbook_rank))
    V_sch = np.zeros((m * p, schoolbook_rank))
    W_sch = np.zeros((p * n, schoolbook_rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U_sch[i * m + j, c] = 1.0
                V_sch[j * p + k, c] = 1.0
                W_sch[k * n + i, c] = 1.0
    
    best_rank = schoolbook_rank
    best_U = U_sch.copy()
    best_V = V_sch.copy()
    best_W = W_sch.copy()
    
    # Try to find lower rank decompositions
    # Start from a reasonable upper bound and work down
    # Known results: for 4x4*4x6, best known is around 72-80
    # Try ranks from 80 down to 64
    
    random.seed(42)
    np.random.seed(42)
    
    for target_rank in range(80, 63, -1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        found = False
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            U, V, W, err, converged = _als_decomposition(
                n, m, p, target_rank, 
                max_iters=150, tol=1e-10,
                deadline=deadline - 3
            )
            
            if converged and err < 1e-6:
                # Verify
                if _verify_decomposition(n, m, p, U, V, W, tol=1e-6):
                    best_rank = target_rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    found = True
                    break
        
        if found:
            continue
        else:
            # If we can't find a solution at this rank, try a bit lower
            # but don't go too far down
            if target_rank <= 70:
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
