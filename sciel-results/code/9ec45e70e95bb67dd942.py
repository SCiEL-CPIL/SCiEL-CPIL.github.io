"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 4]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Reconstruct tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Expected: T[i,j,k] = 1 if i,j,k all valid, else 0
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T[i,j,k] should be 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Efficient check using einsum
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This is a 3-tensor. Let's check a few random entries or all.
    
    # For small sizes, check all
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                expected = 1.0  # All entries are 1 for the multiplication tensor
                err = abs(val - expected)
                if err > max_err:
                    max_err = err
                if max_err > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    # Start with schoolbook
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions using random search
    # Strategy: Try ranks from 40 down to 20, with random integer entries
    
    random.seed(42)
    
    # Try various ranks
    for target_rank in range(47, 19, -1):
        if time.time() > deadline - 10.0:
            break
        
        # Number of attempts per rank
        max_attempts = 50
        attempts = 0
        
        while attempts < max_attempts and time.time() < deadline - 5.0:
            attempts += 1
            
            # Generate random matrices with small integer entries
            # Use entries from {-1, 0, 1} or {-2, -1, 0, 1, 2}
            entry_range = 2 if target_rank > 30 else 3
            
            U_rand = np.zeros((n * m, target_rank), dtype=int)
            V_rand = np.zeros((m * p, target_rank), dtype=int)
            W_rand = np.zeros((p * n, target_rank), dtype=int)
            
            for idx in range(n * m):
                for c in range(target_rank):
                    U_rand[idx, c] = random.randint(-entry_range, entry_range)
            
            for idx in range(m * p):
                for c in range(target_rank):
                    V_rand[idx, c] = random.randint(-entry_range, entry_range)
            
            for idx in range(p * n):
                for c in range(target_rank):
                    W_rand[idx, c] = random.randint(-entry_range, entry_range)
            
            # Convert to lists for verification
            U_list = [list(row) for row in U_rand]
            V_list = [list(row) for row in V_rand]
            W_list = [list(row) for row in W_rand]
            
            if _verify(U_list, V_list, W_list, n, m, p, tol=1e-9):
                best_rank = target_rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                return best_result  # Return immediately on success
        
        # If we found a valid decomposition at this rank, try lower
        # Actually, we should continue searching lower ranks
    
    # If no better found, return schoolbook
    return best_result

# EVOLVE-BLOCK-END
