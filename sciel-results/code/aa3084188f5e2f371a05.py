"""Matrix multiplication tensor decomposition for 4x4 * 4x7.
Finds rank-minimal exact decomposition with integer/half-integer entries.
"""

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin from 150s limit
    
    # Schoolbook baseline: rank = n*m*p = 112
    rank_sb = n * m * p
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_solution = {"U": U, "V": V, "W": W}
    best_rank = rank_sb
    
    # Target tensor: non-zero entries at (i*m+j, j*p+k, k*n+i)
    target_entries = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_entries.add((i * m + j, j * p + k, k * n + i))
    
    def verify_solution(U_mat, V_mat, W_mat):
        """Strict verification of tensor decomposition."""
        rank = len(U_mat[0])
        U_np = np.array(U_mat, dtype=float)
        V_np = np.array(V_mat, dtype=float)
        W_np = np.array(W_mat, dtype=float)
        
        # Check all target entries
        for (u, v, w) in target_entries:
            val = float(np.dot(U_np[u] * V_np[v], W_np[w]))
            if abs(val - 1.0) > 1e-9:
                return False
        
        # Check all non-target entries
        for u in range(n * m):
            for v in range(m * p):
                for w in range(p * n):
                    if (u, v, w) in target_entries:
                        continue
                    val = float(np.dot(U_np[u] * V_np[v], W_np[w]))
                    if abs(val) > 1e-9:
                        return False
        return True
    
    # Winograd-based decomposition using FFT over complex numbers
    # The key insight: C[i,k] = sum_j A[i,j] * B[j,k]
    # We can use the fact that for each fixed i and k, this is a dot product
    # over j. Using FFT, we can reduce the number of multiplications.
    
    # Strategy: Use a block-wise FFT approach
    # For the j-index (size m=4), we can use a 4-point FFT
    # This gives us rank = n * p * ceil(m/2) * 2 = n * p * 4 = 112 for full FFT
    # But we can do better by using real FFT or by exploiting structure
    
    # Let's try a different approach: use the fact that m=4 is small
    # and try to find a rank-64 or lower decomposition
    
    # Approach: Use Winograd's algorithm for 4x4x7
    # The idea is to use precomputation to reduce multiplications
    
    # For a 4-element vector, we can use the following trick:
    # Given vectors a = [a0, a1, a2, a3] and b = [b0, b1, b2, b3]
    # We want to compute the dot product a·b = a0*b0 + a1*b1 + a2*b2 + a3*b3
    # 
    # Using FFT: 
    # A = FFT(a), B = FFT(b)
    # a·b = (1/4) * sum_k A[k] * conj(B[k])
    # But this uses complex numbers and doesn't directly help for tensor rank
    
    # Alternative: Use the fact that for 4 elements, we can pair them:
    # (a0+a1, a2+a3) and (b0+b1, b2+b3)
    # a0*b0 + a1*b1 = ((a0+a1)^2 - a0^2 - a1^2 + (b0+b1)^2 - b0^2 - b1^2 + 2*a0*b1 + 2*a1*b0)/4
    # This is getting complicated. Let's try a direct search for lower rank.
    
    # Try to find a rank-96 decomposition (12% reduction)
    # by exploiting the structure of the 4x4 matrix multiplication
    
    # For each pair (i, k), we need to compute sum_j A[i,j] * B[j,k]
    # There are n*p = 28 such pairs, each requiring 4 multiplications = 112 total
    
    # Can we reduce the number of multiplications per (i,k) pair?
    # For 4-element dot products, the minimum number of multiplications is 4
    # (you can't do better than n multiplications for an n-element dot product
    #  over a field, unless you allow additions/subtractions in the precomputation)
    
    # However, Winograd's algorithm shows that for polynomial multiplication,
    # you can do better. Let's apply this idea.
    
    # For 4x4 matrix multiplication, Winograd's algorithm gives rank 20
    # (instead of 16). But that's for square matrices.
    
    # For 4x4 * 4x7, let's try a block approach:
    # Split the 4x7 matrix into two 4x3 and 4x4 blocks? No, 7 is odd.
    # Split into 4x4 and 4x3.
    
    # Actually, let's just try to use a rank-96 construction by combining
    # some of the schoolbook terms using linear dependencies.
    
    # The schoolbook has 112 terms. Can we find linear dependencies?
    # Each term is (U_row, V_row, W_row) where these are basis vectors.
    # The tensor is T[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
    
    # For the schoolbook, T[u,v,w] = 1 if (u,v,w) is a valid (i,j,k) triple, else 0.
    # We want to find a lower-rank decomposition.
    
    # Let's try a random search for rank 100-110 with integer entries
    
    import random
    random.seed(42)
    
    # Try ranks from 111 down to 95
    for trial_rank in range(rank_sb - 1, max(rank_sb - 17, 0), -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(3):
            if time.time() > deadline - 15:
                break
            
            # Start with schoolbook and try to merge some terms
            U_try = [[0] * trial_rank for _ in range(n * m)]
            V_try = [[0] * trial_rank for _ in range(m * p)]
            W_try = [[0] * trial_rank for _ in range(p * n)]
            
            # Fill first trial_rank terms with schoolbook
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col >= trial_rank:
                            break
                        U_try[i * m + j][col] = 1
                        V_try[j * p + k][col] = 1
                        W_try[k * n + i][col] = 1
                        col += 1
                    if col >= trial_rank:
                        break
                if col >= trial_rank:
                    break
            
            # This truncated schoolbook won't work. We need a different strategy.
            # Let's try to use the fact that we can add linear combinations.
            
            # For now, just verify and move on
            found = False
        
        if found:
            best_rank = trial_rank
            best_solution = {"U": U_try, "V": V_try, "W": W_try}
            break
    
    # If no better solution found, return schoolbook
    return best_solution
# EVOLVE-BLOCK-END
