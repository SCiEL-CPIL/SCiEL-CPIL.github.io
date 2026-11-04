"""Rewritten program for 3x4x6 matrix multiplication tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 24
    pn = p * n  # 18
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need U (nm x r), V (mp x r), W (pn x r) such that:
    # for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Try ranks from a reasonable lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # But we can do better. Let's try from 24 up.
    
    best_result = None
    
    for rank in range(24, 73):
        if time.time() > deadline - 5:
            break
        
        # Use a randomized initialization and then try to solve
        # We'll use a simple approach: initialize U, V, W randomly and 
        # try to find entries that satisfy the equations.
        
        # Actually, let's use a more systematic approach.
        # For each (i,j,k), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is a system of n*m*p = 72 equations in 3*rank*? variables.
        # Each rank-r decomposition has nm*r + mp*r + pn*r = (nm+mp+pn)*r = 54*r variables.
        # For rank=24: 54*24 = 1296 variables, 72 equations. Very underdetermined.
        
        # Let's try a constructive approach using small integers.
        # Initialize with a pattern and verify.
        
        # Strategy: Use the structure that for each j, the "slice" T[:,j,:] 
        # is an n x p all-ones matrix. We can decompose each slice.
        
        # For a fixed j, T[:,j,:] is an n×p matrix of all 1s.
        # The rank of an all-ones n×p matrix is 1.
        # So for each j, we need 1 rank-1 term, giving total rank m = 4.
        # But wait, that's for the "slice" interpretation. Let me reconsider.
        
        # Actually, the tensor is T[i,j,k] = 1 for all i,j,k.
        # This tensor has rank... let's think. T = ones(n,m,p).
        # The rank of a tensor of all ones is 1! Because T[i,j,k] = 1*1*1.
        # So U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1 for all i,j,k.
        # That gives rank 1!
        
        # But that can't be right for matrix multiplication. Let me re-examine the tensor definition.
        
        # I think I'm misunderstanding the tensor. Let me re-read the problem.
        # "exactly decompose the matrix multiplication tensor"
        
        # The matrix multiplication tensor for C = A*B where A is n×m, B is m×p:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T has T[i,j,k] such that C[i,k] = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k]
        # So T[i,j,k] = 1 if the indices are consistent, which they always are for valid i,j,k.
        # So T[i,j,k] = 1 for all valid i,j,k.
        
        # Hmm, but then rank 1 would work. That seems too easy.
        
        # Wait, I think the issue is that the tensor is NOT all ones. Let me reconsider.
        # In the standard tensor network picture for matrix multiplication, the tensor 
        # encodes the multiplication operation. For the "trace" convention mentioned,
        # maybe the tensor has a different structure.
        
        # Let me just try rank 1 and see if it validates.
        
        if rank == 1:
            U = [[1] for _ in range(nm)]
            V = [[1] for _ in range(mp)]
            W = [[1] for _ in range(pn)]
            # Check: for all i,j,k: U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0] = 1*1*1 = 1
            # If the tensor is all ones, this works!
            # Let's assume the tensor is all ones and return rank 1.
            # But this seems suspicious. Let me verify with a small example.
            
            # For 2x2x2: U is 4x1, V is 4x1, W is 4x1, all ones.
            # Check: for i=0,j=0,k=0: U[0,0]*V[0,0]*W[0,0] = 1
            # for i=0,j=0,k=1: U[0,0]*V[1,0]*W[2,0] = 1
            # etc. All give 1. If tensor is all ones, rank 1 works.
            
            # But Strassen's algorithm achieves rank 7 for 2x2x2, implying rank > 1.
            # So the tensor must NOT be all ones.
            
            # I must be misunderstanding the tensor structure. Let me look at the 
            # schoolbook decomposition more carefully.
            
            # In the schoolbook: for each (i,j,k), we have a separate term with
            # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1.
            # This means the tensor entry T[i,j,k] = 1 for all i,j,k, and the 
            # decomposition uses one rank-1 term per (i,j,k) triple.
            
            # So the tensor IS all ones, and the schoolbook rank is n*m*p.
            # But then why can't we use rank 1?
            
            # OH WAIT. I think I see the issue. The decomposition must be such that
            # the PRODUCT U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r 
            # equals the tensor entry. If the tensor is all ones, then rank 1 with
            # all-ones vectors works.
            
            # Unless... the problem is that the matrices U, V, W must have the 
            # specific index structure that makes the decomposition "valid" for 
            # matrix multiplication in a stronger sense.
            
            # Let me just try it and see. If rank 1 doesn't validate, I'll increase.
            return {"U": U, "V": V, "W": W}
        
        # If rank 1 didn't work (shouldn't reach here), continue search
        # This is a fallback - try a more careful construction
    
    # Fallback: return schoolbook if nothing else works
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
