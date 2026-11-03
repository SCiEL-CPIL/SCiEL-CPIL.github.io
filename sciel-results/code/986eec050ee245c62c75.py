"""Rewritten program for matrix multiplication tensor decomposition (3x5 by 5x7)."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# Known best ranks for matrix multiplication (lower bounds from literature)
# For 3x5 * 5x7, the schoolbook rank is 105.
# We know that for square matrices, the rank can be lower.
# For rectangular matrices, we can try to find better decompositions.
# 
# Key insight: The rank of matrix multiplication M_{n,m,p} is at least 
# max(n*m, m*p, p*n) = max(15, 35, 21) = 35.
# The schoolbook algorithm gives rank n*m*p = 105.
# 
# For small cases, we can use known results:
# - 2x2x2 has rank 7 (Strassen)
# - 2x2x2 has lower bound 7
# - For 3x3x3, best known is 23 (Landsberg)
# 
# For 3x5x7, we need to search for a better decomposition.
# 
# Strategy: Use a randomized approach with verification.
# We'll try to find a decomposition of rank R where R < 105.
# 
# The decomposition requires:
# U: (n*m, R) = (15, R)
# V: (m*p, R) = (35, R)  
# W: (p*n, R) = (21, R)
# such that for all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,j,k}
# 
# Actually, the condition is that the tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
# should equal 1 if (i,j,k) corresponds to a valid multiplication term, and 0 otherwise.
# 
# Wait, let me re-read the task. The tensor for matrix multiplication C = A * B where
# A is n x m and B is m x p, C is n x p.
# C[i,k] = sum_j A[i,j] * B[j,k]
# 
# The tensor T has indices (i,j,k) and T[i,j,k] = 1 for all valid (i,j,k), i.e., 
# T[i,j,k] = 1 if 0<=i<n, 0<=j<m, 0<=k<p.
# 
# The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
# 
# So we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
# 
# This is a bilinear rank problem. The schoolbook algorithm uses rank n*m*p with
# U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1 for the corresponding r.
# 
# To improve, we need to find a lower rank decomposition.
# 
# For now, let's implement a basic search that tries to find a better rank.
# We'll use a greedy approach with random initialization and iterative improvement.

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 15
    mp = m * p  # 35
    pn = p * n  # 21
    
    # Schoolbook rank
    schoolbook_rank = nm * mp  # Wait, that's not right.
    # Actually schoolbook rank is n*m*p = 105
    # Let me re-derive:
    # For each (i,j,k), we have a term. There are n*m*p = 105 terms.
    # So schoolbook rank = 105.
    
    # Let's try to find a better rank.
    # Lower bound: max(nm, mp, pn) = max(15, 35, 21) = 35
    
    # Try ranks from lower bound upward
    best_result = None
    best_rank = n * m * p  # Start with schoolbook
    
    # First, create the schoolbook solution as fallback
    U_school = [[0] * best_rank for _ in range(nm)]
    V_school = [[0] * best_rank for _ in range(mp)]
    W_school = [[0] * best_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find better ranks
    for rank in range(36, min(105, 80)):  # Try up to rank 79
        if time.time() > deadline - 30:
            break
        
        # Try a few random initializations for this rank
        for attempt in range(20):
            if time.time() > deadline - 30:
                break
            
            # Random initialization
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Verify if this is a valid decomposition
            # We need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is equivalent to: the tensor T = U ⊗ V ⊗ W (with appropriate indexing) = 1 tensor
            
            # Let's compute the left-hand side for a few random (i,j,k) to check
            valid = True
            for _ in range(50):
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
                ui = U[i * m + j]
                vj = V[j * p + k]
                wk = W[k * n + i]
                val = np.dot(ui, vj * wk)
                if abs(val - 1.0) > 1e-6:
                    valid = False
                    break
            
            if valid:
                # Convert to lists
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = rank
                break
        
        if best_rank < n * m * p:
            break
    
    return best_result
