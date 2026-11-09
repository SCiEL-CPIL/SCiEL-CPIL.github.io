"""Tensor decomposition for 3x4 x 4x6 matrix multiplication."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    def verify_fast(U, V, W, rank):
        """Verify decomposition using numpy einsum for speed."""
        U_arr = np.array(U, dtype=object)
        V_arr = np.array(V, dtype=object)
        W_arr = np.array(W, dtype=object)
        
        # Reshape to 3D tensors
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        # Compute sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This is a tensor contraction
        result = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        # Check all entries equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if result[i, j, k] != 1:
                        return False
        return True
    
    def try_rank_structured(rank, max_attempts=50):
        """Try to find a decomposition using structured random search."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Use small integers for better chance of exact solutions
            U = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(p*n)]
            
            if verify_fast(U, V, W, rank):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Schoolbook solution as fallback
    U_school = [[0] * schoolbook_rank for _ in range(n*m)]
    V_school = [[0] * schoolbook_rank for _ in range(m*p)]
    W_school = [[0] * schoolbook_rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank
    
    # Try ranks from lower bound upward
    lower_bound = max(n*m, m*p, p*n)
    
    for rank in range(lower_bound, schoolbook_rank):
        if time.time() > deadline - 10:
            break
        
        result = try_rank_structured(rank, max_attempts=100)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    return best_result
