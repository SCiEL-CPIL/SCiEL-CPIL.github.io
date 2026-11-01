"""Tensor decomposition for 2x3x4 matrix multiplication."""
DIMENSIONS = [2, 3, 4]

import time
import random
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START

def _verify(U, V, W):
    """Verify that the decomposition exactly reproduces the tensor."""
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    rank = U.shape[1]
    # Check tensor: T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # The tensor for matrix multiplication (2x3 * 3x4) is T[i,j,k] = 1 if all indices valid, 0 otherwise
    # Actually the tensor is the identity mapping: C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _is_half_integer(vals):
    """Check if all values are integers or half-integers (multiples of 0.5)."""
    for v in vals:
        if abs(v * 2 - round(v * 2)) > 1e-9:
            return False
    return True

def _snap_to_half(vals):
    """Snap values to nearest half-integer."""
    return np.round(vals * 2) / 2.0

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
    
    # Try to find lower rank solutions
    # The minimum rank for 2x3x4 matrix multiplication:
    # Lower bound: max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # Known: R(2,3,4) is known to be 16 (from literature)
    # Let's try to find solutions from rank 12 upward
    
    # Strategy: Use random initialization and gradient-free optimization
    # to find rank-r decompositions
    
    # For each candidate rank, try multiple random initializations
    for target_rank in range(12, min(rank_school, 24)):
        if time.time() > deadline:
            break
        
        max_attempts = 50 if target_rank <= 16 else 20
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization with small integer/half-integer values
            np.random.seed(attempt * 1000 + target_rank)
            
            U = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(n*m, target_rank))
            V = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(m*p, target_rank))
            W = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(p*n, target_rank))
            
            # Compute the residual
            # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
            residual = 0.0
            max_res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j], V[j*p+k]) * W[k*n+i]
                        # Wait, that's not right. Let me reconsider.
                        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(target_rank))
                        res = abs(val - 1.0)
                        residual += res * res
                        max_res = max(max_res, res)
            
            if max_res < 1e-9:
                # Found a valid solution
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                break
        
        if best_rank == target_rank:
            break
    
    # If no better solution found, also try a structured approach
    # Pad 2x3x4 to 4x4x4 and use known 4x4x4 decomposition? No, that's more complex.
    
    # Try a specific known structure: exploit the fact that n=2 is small
    # For 2x3x4, we can think of it as two 3x4 multiplications (rows)
    # Or as two 2x3 matrices being multiplied by a 3x4 matrix
    
    # Let's try a more systematic search for rank 12-16
    # using the fact that entries should be half-integers
    
    if best_rank == rank_school and time.time() < deadline:
        # Try structured random search for lower ranks
        for target_rank in range(12, 17):
            if time.time() > deadline:
                break
            
            for attempt in range(100):
                if time.time() > deadline:
                    break
                
                np.random.seed(attempt * 777 + target_rank * 13)
                
                # Initialize with sparse structure
                U = np.zeros((n*m, target_rank))
                V = np.zeros((m*p, target_rank))
                W = np.zeros((p*n, target_rank))
                
                # Fill randomly with half-integers
                for i in range(n*m):
                    for c in range(target_rank):
                        if np.random.random() < 0.3:
                            U[i, c] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                
                for j in range(m*p):
                    for c in range(target_rank):
                        if np.random.random() < 0.3:
                            V[j, c] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                
                for k in range(p*n):
                    for c in range(target_rank):
                        if np.random.random() < 0.3:
                            W[k, c] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                
                # Check if this works
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(target_rank))
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_rank = target_rank
                    best_U = U
                    best_V = V
                    best_W = W
                    break
            
            if best_rank == target_rank:
                break
    
    # Convert to lists for the return format
    U_list = [[float(best_U[i, c]) for c in range(best_rank)] for i in range(n*m)]
    V_list = [[float(best_V[i, c]) for c in range(best_rank)] for i in range(m*p)]
    W_list = [[float(best_W[i, c]) for c in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
