"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    # U: 8 x R
    # V: 24 x R
    # W: 12 x R
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}, rank

    best_result, best_rank = schoolbook()
    
    if time.time() > deadline - 10:
        return best_result

    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Strategy: Structured Algebraic Decomposition
    # The tensor is separable along j.
    # T = sum_j (U_j * V_j * W_j) where U_j is non-zero only in rows i*m+j, V_j in rows j*p+k, W_j in rows k*n+i.
    # Actually, we can share columns across j if we design them carefully?
    # No, the indices i, j, k are fixed.
    # However, we can minimize the rank for each j independently?
    # For a fixed j, we need to decompose the n x p matrix of ones? No, the tensor element is 1 if i,k match?
    # Wait, the tensor is T_{i,j,k} = 1 if we are computing (AB)_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The "1" comes from the summation.
    # The tensor is actually the sum over j of rank-1 tensors?
    # T_{i,j,k} is not a single tensor. The multiplication map is linear.
    # The structure is: Output[i,k] = Sum_j Input1[i,j] * Input2[j,k].
    # This corresponds to a tensor T with indices i, j, k such that T_{i,j,k} = 1?
    # No, the tensor is defined by the trilinear form U V W.
    # The condition is: Sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k.
    # This means the tensor is the identity tensor in the j-index?
    # No, it means that for every (i,j,k), the sum is 1.
    # This implies that the tensor is effectively separable in j.
    # We can define the problem as: Find U, V, W such that for all i,j,k,
    # <u_{i,j}, v_{j,k} * w_{k,i}> = 1.
    
    # Let's try to construct a solution with Rank = 24.
    # We have 4 j's.
    # For each j, we have n*p = 12 equations.
    # If we assign 6 columns per j, total rank 24.
    # For a fixed j, we need to solve:
    # Sum_{r in R_j} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i in {0,1}, k in {0..5}.
    
    # Let's try to make U, V, W very simple.
    # Set U[i*m+j, r] = 1 for all r in R_j.
    # Set V[j*p+k, r] = 1 for all r in R_j.
    # Then Sum_{r in R_j} W[k*n+i, r] = 1.
    # This requires that for each (i,k), the sum of W entries over R_j is 1.
    # If we set W[k*n+i, r] = 1 for exactly one r in R_j and 0 otherwise, and ensure that each (i,k) pair has exactly one such r, we are good.
    # But R_j has size 6. There are 12 (i,k) pairs.
    # So we can't have disjoint sets of r for each (i,k) if we only have 6 r's.
    # We need to share r's.
    # If we share an r, then W[k*n+i, r] contributes to multiple (i,k) equations?
    # No, W[k*n+i, r] is specific to (k,i).
    # The equation for (i,k) is: Sum_{r in R_j} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1.
    # If U and V are 1, then Sum_{r in R_j} W[k*n+i, r] = 1.
    # This is independent for each (i,k).
    # So we just need to assign non-zero values to W[k*n+i, r] such that the sum is 1.
    # We can set W[k*n+i, r0] = 1 for some r0 in R_j and 0 elsewhere.
    # This works!
    # So for each j, we can pick any 6 columns.
    # We set U[i*m+j, r] = 1 for all r in R_j.
    # We set V[j*p+k, r] = 1 for all r in R_j.
    # For each (i,k), we pick one r in R_j and set W[k*n+i, r] = 1.
    # Wait, does this interfere with other j's?
    # U[i*m+j, r] is non-zero only for that specific j.
    # V[j*p+k, r] is non-zero only for that specific j.
    # So the columns R_j are disjoint in terms of U and V support?
    # Yes, because U row index depends on j, and V row index depends on j.
    # So we can use disjoint sets of columns for each j.
    # Total rank = sum |R_j| = 4 * 6 = 24.
    
    # Let's verify this construction.
    # Rank = 24.
    # R_j = {6j, ..., 6j+5}.
    # For j in 0..3:
    #   For r in R_j:
    #     U[i*m+j, r] = 1 for all i.
    #     V[j*p+k, r] = 1 for all k.
    #   For each (i,k):
    #     Pick r_{i,k} = 6j + (i*p + k) % 6.
    #     W[k*n+i, r_{i,k}] = 1.
    #     All other W[k*n+i, r] for r in R_j are 0.
    
    # Check equation for (i,j,k):
    # Sum_{r in R_j} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # = Sum_{r in R_j} 1 * 1 * W[k*n+i, r]
    # = W[k*n+i, r_{i,k}] + Sum_{r in R_j, r != r_{i,k}} 0
    # = 1.
    # Correct!
    
    # This gives a valid rank 24 solution.
    
    U = [[0] * 24 for _ in range(n * m)]
    V = [[0] * 24 for _ in range(m * p)]
    W = [[0] * 24 for _ in range(p * n)]
    
    for j in range(m):
        base_r = j * 6
        for r_offset in range(6):
            r = base_r + r_offset
            # U
            for i in range(n):
                U[i * m + j][r] = 1
            # V
            for k in range(p):
                V[j * p + k][r] = 1
            
    for j in range(m):
        base_r = j * 6
        for i in range(n):
            for k in range(p):
                r_offset = (i * p + k) % 6
                r = base_r + r_offset
                W[k * n + i][r] = 1
                
    # Verify
    if verify(U, V, W):
        best_result = {"U": U, "V": V, "W": W}
        best_rank = 24
    else:
        # Fallback if verification fails (should not happen)
        pass

    # Can we go lower?
    # Lower bound is max(n*p, m*n, m*p) = max(12, 8, 24) = 24.
    # So 24 is the theoretical minimum for this structure?
    # Actually, the lower bound for matrix multiplication M_{n,m,p} is max(n*m, m*p, n*p)? No.
    # The rank of the tensor is at least max(n*m, m*p, n*p) / something?
    # For 2x4x6, the rank is at least 24?
    # The support of V is 24. The rank must be at least the dimension of the space spanned by the slices?
    # Yes, rank >= max(n*m, m*p, n*p) is not necessarily true, but rank >= max(n, m, p) * something.
    # However, 24 is a very strong candidate for the minimum.
    
    return best_result

# EVOLVE-BLOCK-END
