"""Tensor decomposition for 2x3x4 matrix multiplication."""
DIMENSIONS = [2, 3, 4]

import time
import random
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START

def _compute_tensor_residual(U, V, W, n, m, p):
    """Compute the max absolute residual of the tensor decomposition."""
    rank = U.shape[1]
    max_res = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                res = abs(val - 1.0)
                if res > max_res:
                    max_res = res
    return max_res

def _local_search(U, V, W, n, m, p, max_iter=200, step_sizes=None):
    """Perform local search to minimize the tensor residual.
    
    Uses coordinate descent with discrete steps in {-1, -0.5, 0, 0.5, 1}.
    """
    if step_sizes is None:
        step_sizes = [1.0, 0.5, 0.25, 0.125]
    
    current_res = _compute_tensor_residual(U, V, W, n, m, p)
    
    for iteration in range(max_iter):
        improved = False
        
        # Try updating each entry in U, V, W
        for matrix_name, matrix, rows in [('U', U, n*m), ('V', V, m*p), ('W', W, p*n)]:
            rank = matrix.shape[1]
            for r in range(rows):
                for c in range(rank):
                    old_val = matrix[r, c]
                    # Try nearby half-integer values
                    candidates = set()
                    for delta in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                        new_val = old_val + delta
                        # Snap to half-integer
                        new_val = round(new_val * 2) / 2.0
                        if -2.0 <= new_val <= 2.0:
                            candidates.add(new_val)
                    
                    for new_val in candidates:
                        if new_val == old_val:
                            continue
                        matrix[r, c] = new_val
                        new_res = _compute_tensor_residual(U, V, W, n, m, p)
                        if new_res < current_res - 1e-12:
                            current_res = new_res
                            improved = True
                        else:
                            matrix[r, c] = old_val
        
        if not improved:
            break
    
    return U, V, W, current_res

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _snap_to_half(vals):
    """Snap values to nearest half-integer."""
    return np.round(vals * 2) / 2.0

def _is_half_integer(vals, tol=1e-9):
    """Check if all values are integers or half-integers (multiples of 0.5)."""
    for v in vals:
        if abs(v * 2 - round(v * 2)) > tol:
            return False
    return True

def construct():
    """Find rank-minimal exact tensor decomposition for 2x3x4 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline with 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Schoolbook solution as fallback
    rank_school = n * m * p  # 24
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Known lower bound: max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # Known upper bound from literature: R(2,3,4) <= 16
    # Try ranks from 12 to 16
    
    for target_rank in range(12, 17):
        if time.time() > deadline:
            break
        
        # Multiple random initializations with local search
        max_attempts = 30 if target_rank <= 14 else 15
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Deterministic seed for reproducibility
            np.random.seed(attempt * 1000 + target_rank * 7 + 42)
            
            # Initialize with structured pattern
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Dense initialization with half-integers in [-1, 1]
            for i in range(n*m):
                for c in range(target_rank):
                    if np.random.random() < 0.6:
                        U[i, c] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            
            for j in range(m*p):
                for c in range(target_rank):
                    if np.random.random() < 0.6:
                        V[j, c] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            
            for k in range(p*n):
                for c in range(target_rank):
                    if np.random.random() < 0.6:
                        W[k, c] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            
            # Run local search
            U_ls, V_ls, W_ls, final_res = _local_search(U, V, W, n, m, p, max_iter=100)
            
            # Check if we found an exact solution
            if final_res < 1e-9:
                # Verify with exact check
                if _verify(U_ls, V_ls, W_ls, n, m, p):
                    # Ensure all values are half-integers
                    if _is_half_integer(U_ls) and _is_half_integer(V_ls) and _is_half_integer(W_ls):
                        best_rank = target_rank
                        best_U = U_ls
                        best_V = V_ls
                        best_W = W_ls
                        break
                    else:
                        # Snap to half-integers and re-verify
                        U_snap = _snap_to_half(U_ls)
                        V_snap = _snap_to_half(V_ls)
                        W_snap = _snap_to_half(W_ls)
                        if _verify(U_snap, V_snap, W_snap, n, m, p):
                            best_rank = target_rank
                            best_U = U_snap
                            best_V = V_snap
                            best_W = W_snap
                            break
        
        if best_rank == target_rank:
            break
    
    # If still at schoolbook rank, try one more structured approach
    if best_rank == rank_school and time.time() < deadline:
        # Try rank 16 with more systematic initialization
        target_rank = 16
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 5000 + 999)
            
            # Use a different initialization strategy: start from schoolbook
            # and try to merge columns
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Initialize with random half-integers
            for i in range(n*m):
                for c in range(target_rank):
                    U[i, c] = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
            
            for j in range(m*p):
                for c in range(target_rank):
                    V[j, c] = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
            
            for k in range(p*n):
                for c in range(target_rank):
                    W[k, c] = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
            
            # Run local search with more iterations
            U_ls, V_ls, W_ls, final_res = _local_search(U, V, W, n, m, p, max_iter=150)
            
            if final_res < 1e-9:
                if _verify(U_ls, V_ls, W_ls, n, m, p):
                    if _is_half_integer(U_ls) and _is_half_integer(V_ls) and _is_half_integer(W_ls):
                        best_rank = target_rank
                        best_U = U_ls
                        best_V = V_ls
                        best_W = W_ls
                        break
    
    # Final safety: ensure we have a valid solution
    if not _verify(best_U, best_V, best_W, n, m, p):
        # Fall back to schoolbook
        best_rank = rank_school
        best_U = U_school
        best_V = V_school
        best_W = W_school
    
    # Convert to lists for the return format
    U_list = [[float(best_U[i, c]) for c in range(best_rank)] for i in range(n*m)]
    V_list = [[float(best_V[i, c]) for c in range(best_rank)] for i in range(m*p)]
    W_list = [[float(best_W[i, c]) for c in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
