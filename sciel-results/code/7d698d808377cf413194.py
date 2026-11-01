"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_construct_rank12(n, m, p):
    """Try to construct a rank-12 decomposition for 2x3x4."""
    # For n=2, m=3, p=4: U is 6x12, V is 12x12, W is 8x12
    # We need: sum_c U[i*3+j,c] * V[j*4+k,c] * W[k*2+i,c] = 1 for all i,j,k
    
    # Try a structured construction
    # Idea: use the 3 values of j to separate contributions
    # For each (i,k), we need sum over j of A[i,j]*B[j,k]
    
    # Let's try: for each j, create 4 terms (one for each k), but that's 12 total
    # U[i*3+j, j*4+k] = 1, V[j*4+k, j*4+k] = 1, W[k*2+i, j*4+k] = 1
    # But that's just schoolbook rearranged, still rank 12? No, schoolbook is rank 24.
    
    # The issue: in schoolbook, each (i,j,k) triple gets its own term.
    # For rank 12, we need to share terms across i values.
    
    # Since n=2, we have i=0 and i=1. 
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # If we could compute for each k: some combination that gives both C[0,k] and C[1,k]
    # We need 4 such combinations (one per k), each producing 2 outputs.
    # But each "term" in the tensor decomposition is rank-1, so it contributes 
    # a product U_val * V_val * W_val to exactly one (i,j,k) position.
    
    # Actually, let me reconsider. The tensor T has T[i,j,k] = delta_{i,k valid} 
    # No, T[i,j,k] = 1 for all valid (i,j,k) where the multiplication uses A[i,j]*B[j,k]
    
    # The rank-12 lower bound: consider fixing i=0. Then we have a 3x4 subproblem.
    # The rank of the 3x4 multiplication tensor (for one row of A) is at least max(3,4)=4.
    # Similarly for i=1. But they share the B matrix.
    
    # Let me try a direct numerical search with small integer values.
    
    # Given time constraints, let me try a few structured patterns.
    
    # Pattern: for each j in 0..2 and k in 0..3, define term c = j*4+k
    # U[i*3+j, c] should depend on i somehow
    # V[j*4+k, c] = 1
    # W[k*2+i, c] should depend on i
    
    # We need: sum_c U[i*3+j,c]*V[j*4+k,c]*W[k*2+i,c] = 1
    # If V[j*4+k, j*4+k] = 1 and V[j*4+k, c] = 0 for c != j*4+k, then:
    # U[i*3+j, j*4+k] * 1 * W[k*2+i, j*4+k] = 1
    # So U[i*3+j, j*4+k] * W[k*2+i, j*4+k] = 1 for all i,j,k
    # This means for fixed j,k: U[i*3+j, j*4+k]*W[k*2+i, j*4+k] = 1
    # For i=0: U[j, j*4+k]*W[k*2, j*4+k] = 1
    # For i=1: U[3+j, j*4+k]*W[k*2+1, j*4+k] = 1
    # We can set U[j, j*4+k] = 1, W[k*2, j*4+k] = 1, U[3+j, j*4+k] = 1, W[k*2+1, j*4+k] = 1
    # But wait, that gives 12 terms, each contributing to exactly one (i,j,k).
    # That's the same as schoolbook! Rank 12? No, schoolbook has 24 terms.
    
    # I'm confusing myself. Let me recount.
    # Schoolbook: for each (i,j,k), one term. That's 2*3*4=24 terms.
    # In my pattern above, for each (j,k), I'm trying to handle both i=0 and i=1.
    # But a single rank-1 term U[a,c]*V[b,c]*W[d,c] contributes to exactly one 
    # (i,j,k) position where a=i*m+j, b=j*p+k, d=k*n+i.
    # So one term can only contribute to one (i,j,k).
    
    # Therefore, to get the sum to 1 for each (i,j,k), and 0 elsewhere, 
    # we need at least one term per (i,j,k), giving rank >= 24? 
    # No! Multiple terms can contribute to the same (i,j,k).
    
    # The constraint is: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # There's no constraint that the sum must be 0 for "invalid" positions because 
    # those positions don't exist in the tensor. The tensor only has entries for valid (i,j,k).
    
    # So the problem is: find matrices U (6xR), V (12xR), W (8xR) such that 
    # for all i in 0..1, j in 0..2, k in 0..3:
    #   sum_c U[i*3+j,c] * V[j*4+k,c] * W[k*2+i,c] = 1
    
    # This is a system of 24 equations in 6R + 12R + 8R = 26R unknowns.
    # For R=12: 24 equations, 312 unknowns. Very underdetermined.
    
    # Let me try to find a solution with small integers.
    
    # Strategy: set V to be identity-like, then solve for U and W.
    # If V is 12x12 identity, then for each (i,j,k):
    #   U[i*3+j, j*4+k] * W[k*2+i, j*4+k] = 1
    # We can set U[i*3+j, j*4+k] = 1 and W[k*2+i, j*4+k] = 1
    # But we need to check: does this work?
    # For (i,j,k): only term c=j*4+k is nonzero in the sum (since V is identity),
    # so sum = U[i*3+j, j*4+k] * V[j*4+k, j*4+k] * W[k*2+i, j*4+k] = 1*1*1 = 1. Yes!
    
    # Wait, but V is 12x12 and has rows indexed by j*p+k = j*4+k. 
    # V[j*4+k, c] for c in 0..11. If V is identity, V[j*4+k, j*4+k]=1, else 0.
    # Then for (i,j,k), the sum is over c, but only c=j*4+k contributes.
    # So we need U[i*3+j, j*4+k] * W[k*2+i, j*4+k] = 1.
    # Set both to 1. This works!
    
    # But wait, is V allowed to be identity? V has shape (m*p, rank) = (12, 12).
    # Yes, identity is fine.
    
    # Let me verify this construction:
    rank = 12
    U = [[0] * rank for _ in range(n * m)]  # 6 x 12
    V = [[0] * rank for _ in range(m * p)]  # 12 x 12
    W = [[0] * rank for _ in range(p * n)]  # 8 x 12
    
    for j in range(m):
        for k in range(p):
            c = j * p + k
            V[c][c] = 1  # V is identity
            for i in range(n):
                U[i * m + j][c] = 1
                W[k * n + i][c] = 1
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Try rank 12 first
    if time.time() < deadline:
        result = _try_construct_rank12(n, m, p)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
