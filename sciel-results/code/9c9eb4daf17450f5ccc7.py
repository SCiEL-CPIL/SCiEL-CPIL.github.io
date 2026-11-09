"""Tensor decomposition for 3x5 * 5x5 matrix multiplication.
Searches for a rank-minimal exact decomposition over half-integers.
"""
import time
import random
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 15s margin for safety
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Build the target tensor
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1 for all i,j,k
    # We'll work with the matricized form for numerical methods
    # T as a matrix of shape (n*m) x (m*p * p*n)
    # But we need to find U, V, W such that sum_r U[:,r] V[:,r]^T W[:,r]^T matches T
    
    # Alternative: use the fact that for fixed i, the tensor slice is a matrix multiplication
    # Let's try a structured decomposition
    
    # For the 3x5x5 case, we can use the following approach:
    # Split i into i1 (first 1 row) and i2 (last 2 rows)
    # The tensor is block diagonal in a certain sense
    
    # Try to find a decomposition using random search with exact verification
    # We'll try ranks from 30 to 74 (less than schoolbook)
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly reproduces the tensor."""
        # Check that for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # And for all other triples, the sum should be 0
        # But the tensor is only defined for valid (i,j,k) triples
        # Actually, the tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
        # and T[a, b, c] = 0 otherwise (where a,b,c are not consistent)
        
        # We need to check that the decomposition matches the tensor
        # The tensor is a 3D array of shape (n*m, m*p, p*n)
        # T[a, b, c] = 1 if there exist i,j,k such that a = i*m+j, b = j*p+k, c = k*n+i
        # and 0 otherwise
        
        # Build the target tensor
        T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Compute the decomposition
        # D[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        D = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check if D matches T
        if not np.allclose(D, T, atol=1e-10):
            return False
        return True
    
    def try_rank(R):
        """Try to find a decomposition of rank R using random search."""
        # Initialize with random half-integers in [-2, 2]
        U = np.random.uniform(-2, 2, (n*m, R))
        V = np.random.uniform(-2, 2, (m*p, R))
        W = np.random.uniform(-2, 2, (p*n, R))
        
        # Round to half-integers
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Verify
        if verify_decomposition(U, V, W, R):
            return U, V, W
        
        return None
    
    # Try a few random searches for lower ranks
    for R in range(40, 75):
        if time.time() > deadline:
            break
        
        # Try a few random attempts for this rank
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            result = try_rank(R)
            if result is not None:
                U, V, W = result
                # Convert to list format
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                
                # Verify once more
                if verify_decomposition(np.array(U_list), np.array(V_list), np.array(W_list), R):
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no better decomposition found, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
