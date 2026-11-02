"""Rewritten program for matrix multiplication 2x3 by 3x4 using group action / character method."""
import numpy as np
from itertools import product
import time
import random

DIMENSIONS = [2, 3, 4]
GLOBAL_START_TIME = time.time()
DEADLINE = GLOBAL_START_TIME + 140  # 140s safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    
    Strategy: Use numerical search to find a valid decomposition, then verify exactness.
    We search for the minimum rank r such that there exist matrices U (6xr), V (12xr), W (8xr)
    with entries in {-1, -0.5, 0, 0.5, 1} satisfying:
    sum_c U[i*3+j, c] * V[j*4+k, c] * W[k*2+i, c] = 1 for all i,j,k
    """
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Try ranks from 12 upward with deadline guard
    for rank in range(12, 25):
        if time.time() > DEADLINE - 10:  # 10s safety margin
            break
        result = _find_decomposition(rank, n, m, p)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _find_decomposition(rank, n, m, p):
    """
    Find a rank-r decomposition using random search with linear solve for W.
    Multiple initialization strategies and a polish pass.
    """
    start_time = time.time()
    
    # Candidate values: integers and half-integers
    values = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    # Different initialization strategies
    n_restarts_strategy1 = 150  # Dense random
    n_restarts_strategy2 = 100  # Sparse with more non-zeros
    n_restarts_strategy3 = 100  # Structured patterns
    
    # Strategy 1: Dense random initialization
    for restart in range(n_restarts_strategy1):
        if time.time() > DEADLINE - 5:
            return None
        
        # Random initialization of U and V with higher density
        U = np.array([[random.choice(values) for _ in range(rank)] for _ in range(n * m)])
        V = np.array([[random.choice(values) for _ in range(rank)] for _ in range(m * p)])
        
        result = _solve_and_verify(U, V, n, m, p)
        if result is not None:
            return result
    
    # Strategy 2: Sparse but with guaranteed non-zero entries per row
    for restart in range(n_restarts_strategy2):
        if time.time() > DEADLINE - 5:
            return None
        
        # Ensure each row has at least a few non-zero entries
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        for row in range(n * m):
            n_nonzero = random.randint(2, min(rank, 5))
            cols = random.sample(range(rank), n_nonzero)
            for c in cols:
                U[row, c] = random.choice([v for v in values if v != 0.0])
        for row in range(m * p):
            n_nonzero = random.randint(2, min(rank, 5))
            cols = random.sample(range(rank), n_nonzero)
            for c in cols:
                V[row, c] = random.choice([v for v in values if v != 0.0])
        
        result = _solve_and_verify(U, V, n, m, p)
        if result is not None:
            return result
    
    # Strategy 3: Structured initialization based on Kronecker patterns
    for restart in range(n_restarts_strategy3):
        if time.time() > DEADLINE - 5:
            return None
        
        # Initialize with block structure
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        
        # Distribute rank across blocks
        base_rank = rank // (n * m)
        remainder = rank % (n * m)
        
        col_offset = 0
        for i in range(n):
            for j in range(m):
                block_size = base_rank + (1 if (i * m + j) < remainder else 0)
                if block_size > 0:
                    for c in range(col_offset, col_offset + block_size):
                        U[i * m + j, c] = random.choice([v for v in values if v != 0.0])
                col_offset += block_size
        
        col_offset = 0
        for j in range(m):
            for k in range(p):
                block_size = rank // (m * p)
                if block_size > 0:
                    for c in range(col_offset, col_offset + block_size):
                        V[j * p + k, c] = random.choice([v for v in values if v != 0.0])
                col_offset += block_size
        
        result = _solve_and_verify(U, V, n, m, p)
        if result is not None:
            return result
    
    return None


def _solve_and_verify(U, V, n, m, p):
    """
    Solve for W given U and V, then verify and polish if needed.
    """
    rank = U.shape[1]
    
    # Solve for W: for each (k, i), we need
    # sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1 for all j
    # This is a linear system in W[k*n+i, :]
    
    W = np.zeros((p * n, rank))
    valid = True
    
    for k in range(p):
        for i in range(n):
            # Build the linear system: for each j in 0..m-1
            # sum_c W[k*n+i, c] * (U[i*m+j, c] * V[j*p+k, c]) = 1
            A_sys = np.zeros((m, rank))
            for j in range(m):
                for c in range(rank):
                    A_sys[j, c] = U[i * m + j, c] * V[j * p + k, c]
            
            b_sys = np.ones(m)
            
            # Check if system is solvable
            if np.linalg.matrix_rank(A_sys) < m:
                valid = False
                break
            
            # Solve A_sys * W[k*n+i, :]^T = b_sys
            try:
                W[k * n + i, :] = np.linalg.lstsq(A_sys, b_sys, rcond=None)[0]
            except:
                valid = False
                break
        
        if not valid:
            break
    
    if not valid:
        return None
    
    # Round W to nearest value in the candidate set
    W_rounded = np.array([[round(W[row, col] / 0.5) * 0.5 for col in range(rank)] for row in range(p * n)])
    
    # Check if rounded W is valid
    if _is_valid(U, V, W_rounded, n, m, p):
        return _convert_to_list(U, V, W_rounded)
    
    # Try a polish pass: adjust U and V slightly and re-solve
    result = _polish(U, V, W_rounded, n, m, p)
    if result is not None:
        return result
    
    return None


def _polish(U, V, W, n, m, p):
    """
    Polish pass: try small perturbations to U and V to find a valid decomposition.
    """
    rank = U.shape[1]
    values = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    # Try perturbing individual entries
    for attempt in range(50):
        if time.time() > DEADLINE - 3:
            return None
        
        # Pick a random entry in U or V to perturb
        if random.random() < 0.5:
            row = random.randint(0, U.shape[0] - 1)
            col = random.randint(0, rank - 1)
            U_perturbed = U.copy()
            U_perturbed[row, col] = random.choice(values)
            V_perturbed = V
        else:
            row = random.randint(0, V.shape[0] - 1)
            col = random.randint(0, rank - 1)
            U_perturbed = U
            V_perturbed = V.copy()
            V_perturbed[row, col] = random.choice(values)
        
        # Re-solve for W
        W_new = np.zeros((p * n, rank))
        valid = True
        
        for k in range(p):
            for i in range(n):
                A_sys = np.zeros((m, rank))
                for j in range(m):
                    for c in range(rank):
                        A_sys[j, c] = U_perturbed[i * m + j, c] * V_perturbed[j * p + k, c]
                
                b_sys = np.ones(m)
                
                try:
                    W_new[k * n + i, :] = np.linalg.lstsq(A_sys, b_sys, rcond=None)[0]
                except:
                    valid = False
                    break
            
            if not valid:
                break
        
        if not valid:
            continue
        
        # Round and verify
        W_rounded = np.array([[round(W_new[row, col] / 0.5) * 0.5 for col in range(rank)] for row in range(p * n)])
        
        if _is_valid(U_perturbed, V_perturbed, W_rounded, n, m, p):
            return _convert_to_list(U_perturbed, V_perturbed, W_rounded)
    
    return None


def _is_valid(U, V, W, n, m, p):
    """
    Check if the decomposition is valid:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _convert_to_list(U, V, W):
    """Convert numpy arrays to lists of lists."""
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }


# EVOLVE-BLOCK-END
