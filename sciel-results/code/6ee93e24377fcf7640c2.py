import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Strategy: Split the 4 middle indices into two groups of 2.
    # Group 1: j in {0,1} -> A[:, 0:2] (2x2) * B[0:2, :] (2x8) -> C1 (2x8)
    # Group 2: j in {2,3} -> A[:, 2:4] (2x2) * B[2:4, :] (2x8) -> C2 (2x8)
    # C = C1 + C2
    #
    # Each M_{2,2,8} subproblem has rank 16 (4 scalar mults per output column, 8 columns)
    # Total rank = 2 * 16 = 32
    #
    # For M_{2,2,8} with A' (2x2), B' (2x8), C' (2x8):
    #   C'[i,k] = A'[i,0]*B'[0,k] + A'[i,1]*B'[1,k]
    #   For each k: 4 independent products -> rank 4 per k, 8 k's -> rank 32? 
    #   No wait, that's schoolbook. Let me recount.
    #
    # Actually for M_{2,2,8}:
    #   C'[0,k] = A'[0,0]*B'[0,k] + A'[0,1]*B'[1,k]
    #   C'[1,k] = A'[1,0]*B'[0,k] + A'[1,1]*B'[1,k]
    #   Each k gives 4 products, but they're all independent across k.
    #   So rank = 4 * 8 = 32? No, that's the schoolbook count.
    #
    # The rank of the tensor is the minimum number of rank-1 terms.
    # For M_{2,2,p}, the rank is 2p (not 4p) because we can pair up the multiplications.
    # 
    # Let me think more carefully. The tensor T_{i,j,k} for M_{2,2,p} has T=1 for all valid (i,j,k).
    # We need: sum_s U[2i+j,s] * V[2j+k,s] * W[2k+i,s] = 1 for all i,j,k.
    #
    # For p=1: M_{2,2,1}. C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0], C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    # This needs 4 products in schoolbook, but rank is 4? Or can we do better?
    # Actually M_{2,2,1} has rank 4 (it's the sum of 4 rank-1 tensors, and you can't do better).
    #
    # Hmm, so for M_{2,2,8}, rank = 8 * 4 = 32? That's the same as schoolbook.
    #
    # I think the issue is that M_{2,2,p} doesn't benefit from Strassen because the "square" dimension is only 2.
    # Strassen works for M_{2,2,2} (rank 7 vs 8 schoolbook).
    #
    # So let's go back to the Strassen approach but verify it's correct.
    # 
    # The current rank-56 solution splits into 8 blocks of M_{2,2,2}, each rank 7.
    # That seems correct and hard to beat without a fundamentally new algorithm.
    
    # Let me just return the known good rank-56 solution, which is verified correct.
    U, V, W = _build_rank_56(n, m, p)
    return {"U": U, "V": V, "W": W}

def _build_rank_56(n, m, p):
    """
    Build the rank-56 decomposition using Strassen on 2x2 blocks.
    8 blocks of 2x2x2, each with rank 7 (Strassen).
    Total rank = 56.
    """
    total_rank = 56
    U = [[0] * total_rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 56
    
    for u in range(2):
        for v in range(4):
            base_col = 7 * (4 * u + v)
            
            # A indices (2x4 matrix, row-major: index = i*m + j = i*4 + j)
            a11_idx = 0 * 4 + 2 * u
            a12_idx = 0 * 4 + 2 * u + 1
            a21_idx = 1 * 4 + 2 * u
            a22_idx = 1 * 4 + 2 * u + 1
            
            # B indices (4x8 matrix, row-major: index = j*p + k = j*8 + k)
            b11_idx = (2 * u) * 8 + 2 * v
            b12_idx = (2 * u) * 8 + 2 * v + 1
            b21_idx = (2 * u + 1) * 8 + 2 * v
            b22_idx = (2 * u + 1) * 8 + 2 * v + 1
            
            # W indices (output convention W[k*n+i] = W[2k+i])
            # C[0, 2v], C[0, 2v+1], C[1, 2v], C[1, 2v+1]
            c11_idx = (2 * v) * 2 + 0
            c12_idx = (2 * v + 1) * 2 + 0
            c21_idx = (2 * v) * 2 + 1
            c22_idx = (2 * v + 1) * 2 + 1
            
            # Strassen's 7 products for 2x2 matrix multiplication:
            # M1 = (A11+A22)(B11+B22)
            U[a11_idx][base_col] += 1
            U[a22_idx][base_col] += 1
            V[b11_idx][base_col] += 1
            V[b22_idx][base_col] += 1
            W[c11_idx][base_col] += 1
            W[c22_idx][base_col] += 1
            
            # M2 = (A21+A22)B11
            U[a21_idx][base_col + 1] += 1
            U[a22_idx][base_col + 1] += 1
            V[b11_idx][base_col + 1] += 1
            W[c21_idx][base_col + 1] += 1
            W[c22_idx][base_col + 1] -= 1
            
            # M3 = A11(B12-B22)
            U[a11_idx][base_col + 2] += 1
            V[b12_idx][base_col + 2] += 1
            V[b22_idx][base_col + 2] -= 1
            W[c12_idx][base_col + 2] += 1
            W[c22_idx][base_col + 2] += 1
            
            # M4 = A22(B21-B11)
            U[a22_idx][base_col + 3] += 1
            V[b21_idx][base_col + 3] += 1
            V[b11_idx][base_col + 3] -= 1
            W[c11_idx][base_col + 3] += 1
            W[c21_idx][base_col + 3] += 1
            
            # M5 = (A11+A12)B22
            U[a11_idx][base_col + 4] += 1
            U[a12_idx][base_col + 4] += 1
            V[b22_idx][base_col + 4] += 1
            W[c11_idx][base_col + 4] -= 1
            W[c12_idx][base_col + 4] += 1
            
            # M6 = (A21-A11)(B11+B12)
            U[a21_idx][base_col + 5] += 1
            U[a11_idx][base_col + 5] -= 1
            V[b11_idx][base_col + 5] += 1
            V[b12_idx][base_col + 5] += 1
            W[c22_idx][base_col + 5] += 1
            
            # M7 = (A12-A22)(B21+B22)
            U[a12_idx][base_col + 6] += 1
            U[a22_idx][base_col + 6] -= 1
            V[b21_idx][base_col + 6] += 1
            V[b22_idx][base_col + 6] += 1
            W[c11_idx][base_col + 6] += 1
            
    return U, V, W
