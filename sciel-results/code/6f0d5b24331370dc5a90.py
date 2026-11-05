"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 63
    
    # Build target tensor: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # In flattened form: T[i*m+j, j*p+k, k*n+i] = 1
    
    # Strategy: Use a structured decomposition based on partitioning.
    # Key idea: For each pair (i, k), we need to handle all j values.
    # If we use one term per (i, j, k), that's 63 terms (schoolbook).
    # Can we do better?
    
    # Let's try a "Strassen-like" approach for the j dimension.
    # For fixed (i, k), we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j.
    # This is a matrix multiplication: (U_i * V_k) * W_k = 1 for all j.
    # Where U_i is a 3x r matrix (rows j), V_k is a 3x r matrix (rows j), W_k is a scalar.
    # Actually, for fixed (i,k), the contribution is:
    #   For each j: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c.
    #   = (U_i[:, c] * V_k[:, c]) * W_k[c] where U_i[j,c] = U[i*m+j, c], etc.
    
    # Let's try a different structure. Use rank = 3 * 7 * 2 = 42.
    # For each (i, k), use 2 terms that together cover all j.
    
    # Term (i, k, 0): 
    #   U[i*m+j, (i,k,0)] = 1 for j=0,1; 0 for j=2
    #   V[j*p+k, (i,k,0)] = 1 for j=0,1; 0 for j=2
    #   W[k*n+i, (i,k,0)] = 1
    #   Contribution to (i',j',k'): nonzero only if i'=i, k'=k, and j' in {0,1}
    #   Value: 1 if j' in {0,1}
    
    # Term (i, k, 1):
    #   U[i*m+j, (i,k,1)] = 0 for j=0,1; 1 for j=2
    #   V[j*p+k, (i,k,1)] = 0 for j=0,1; 1 for j=2
    #   W[k*n+i, (i,k,1)] = 1
    #   Contribution: 1 if j'=2
    
    # Total for (i,j,k): 1 (from one of the two terms). Good for target positions.
    
    # But check non-target positions:
    # Position (a, b, c) where a = i*m + j_a, b = j_b*p + k, c = k*n + i
    # (so a//m = i, b%p = k, c = k*n+i, but j_a may differ from j_b)
    # Contribution from term (i, k, 0): U[a, (i,k,0)] * V[b, (i,k,0)] * W[c, (i,k,0)]
    #   = (1 if j_a in {0,1} else 0) * (1 if j_b in {0,1} else 0) * 1
    #   = 1 if j_a in {0,1} AND j_b in {0,1}
    # Contribution from term (i, k, 1):
    #   = (1 if j_a == 2) * (1 if j_b == 2) * 1
    #   = 1 if j_a == 2 AND j_b == 2
    # Total: 1 if (j_a in {0,1} AND j_b in {0,1}) OR (j_a == 2 AND j_b == 2)
    #        = 1 if j_a == j_b (since if j_a in {0,1}, j_b in {0,1} includes j_a=j_b and j_a!=j_b)
    # Wait, if j_a=0 and j_b=1, both in {0,1}, so contribution = 1. But (j_a, j_b) = (0,1) is NOT a target!
    # Target requires j_a = j_b. So this doesn't work.
    
    # The problem is that we can't distinguish j_a from j_b with independent U and V patterns.
    
    # We need a structure where the product U[a,c]*V[b,c] is 1 iff j_a = j_b (for fixed i,k).
    # This is like a "diagonal" pattern.
    
    # For 3x3, we can use 3 terms per (i,k) to get the diagonal:
    # Term (i,k,j): U[i*m+j', (i,k,j)] = delta(j', j), V[j'*p+k, (i,k,j)] = delta(j', j), W[k*n+i, (i,k,j)] = 1
    # This is the schoolbook: 3 terms per (i,k), total 3*7*3 = 63.
    
    # To do better, we need to share terms across different (i,k) pairs or use more clever patterns.
    
    # Let's try a global structure. Use rank = 9 * 7 = 63... that's still schoolbook.
    
    # What if we use rank = 3 * 7 = 21 with more complex patterns?
    # Term (i, k): 
    #   U[i*m+j, (i,k)] = f_j(i,k)
    #   V[j*p+k, (i,k)] = g_j(i,k)
    #   W[k*n+i, (i,k)] = h(i,k)
    # Need: f_j * g_j * h = 1 for all j (target positions)
    # And: f_{j_a} * g_{j_b} * h = 0 for j_a != j_b (non-target positions with same i,k)
    # The second condition requires f_{j_a} * g_{j_b} = 0 for j_a != j_b.
    # This means the vectors f and g must be "orthogonal" in the sense that 
    # f_a * g_b = 0 for a != b. This is only possible if f and g are "diagonal" 
    # in some basis, which for 3 elements means we need at least 3 nonzero entries.
    # So we can't do it with 1 term per (i,k).
    
    # What about 2 terms per (i,k)? Rank = 2 * 3 * 7 = 42.
    # Term (i, k, 0) and (i, k, 1):
    #   Need: f^0_j * g^0_j * h^0 + f^1_j * g^1_j * h^1 = 1 for all j
    #   And: f^0_{j_a} * g^0_{j_b} * h^0 + f^1_{j_a} * g^1_{j_b} * h^1 = 0 for j_a != j_b
    #   
    #   Let's try: f^0 = (1, 1, 0), g^0 = (1, 1, 0), h^0 = 1
    #                f^1 = (0, 0, 1), g^1 = (0, 0, 1), h^1 = 1
    #   Target: j=0: 1*1*1 + 0 = 1. j=1: 1*1*1 + 0 = 1. j=2: 0 + 1*1*1 = 1. Good.
    #   Non-target j_a=0, j_b=1: 1*1*1 + 0 = 1. BAD! Should be 0.
    
    # The issue is that f^0 and g^0 have overlapping support.
    
    # Try: f^0 = (1, 0, 0), g^0 = (1, 0, 0), h^0 = 1
    #       f^1 = (0, 1, 0), g^1 = (0, 1, 0), h^1 = 1
    #       f^2 = (0, 0, 1), g^2 = (0, 0, 1), h^2 = 1
    # That's 3 terms per (i,k), back to 63.
    
    # It seems hard to beat 63 with simple structures. Let's just return schoolbook.
    
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
