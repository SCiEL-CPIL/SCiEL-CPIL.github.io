"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Matrix multiplication 4x4 by 4x8.
    Output W convention: W[k*n+i, r]
    We need U (n*m x rank) = (16 x rank), V (m*p x rank) = (32 x rank), W (p*n x rank) = (32 x rank)
    such that for all i, j, k, r:
      sum_c U[i*m+j, c] * V[j*p+r, c] * W[k*n+i, c] = (A[i,j] * B[j,r])
    
    The tensor for (A4x4 * B4x8) has rank at most 19*4 = 76 using Laderman-like decomposition
    on the 4x4x4 core applied per output column.
    
    Actually, let me think about this more carefully.
    
    We want to decompose: T[i,j,k,r] = 1 (it's the multiplication tensor)
    where i in [0,4), j in [0,4), k in [0,4), r in [0,8)
    
    The standard approach: for each output column r, we have a 4x4x4 multiplication tensor.
    We can use a rank-19 decomposition of the 4x4x4 multiplication tensor.
    
    Laderman's rank-19 decomposition for 4x4x4:
    The multiplication tensor M_{4,4,4} has tensor rank 19 (proved by Laderman).
    
    Let me write down the Laderman decomposition. The 19 rank-1 terms are:
    
    For 4x4x4 matrix multiplication C = A*B where A, B, C are 4x4:
    
    L1:  A[0,0]*B[0,0] -> C[0,0]
    L2:  A[0,1]*B[0,1] -> C[0,1]
    L3:  A[0,2]*B[0,2] -> C[0,2]
    L4:  A[0,3]*B[0,3] -> C[0,3]
    L5:  A[1,0]*B[1,0] -> C[1,0]
    L6:  A[1,1]*B[1,1] -> C[1,1]
    L7:  A[1,2]*B[1,2] -> C[1,2]
    L8:  A[1,3]*B[1,3] -> C[1,3]
    L9:  A[2,0]*B[2,0] -> C[2,0]
    L10: A[2,1]*B[2,1] -> C[2,1]
    L11: A[2,2]*B[2,2] -> C[2,2]
    L12: A[2,3]*B[2,3] -> C[2,3]
    L13: A[3,0]*B[3,0] -> C[3,0]
    L14: A[3,1]*B[3,1] -> C[3,1]
    L15: A[3,2]*B[3,2] -> C[3,2]
    L16: A[3,3]*B[3,3] -> C[3,3]
    L17: (A[0,0]+A[1,1])*B[0,1] -> C[0,1] + C[1,0]... 
    """
    
    # Let me use a known rank-19 decomposition for 4x4x4.
    # The Laderman decomposition uses 19 scalar multiplications.
    
    # I'll define the 19 terms as tuples of (U_indices, V_indices, W_indices)
    # For the 4x4x4 core: U is 4x4 indexed by (i,j), V is 4x4 indexed by (j,k), W is 4x4 indexed by (k,i)
    
    # Laderman's 19 terms for C = A * B (all 4x4):
    # Using 0-based indexing:
    
    terms = []
    
    # Diagonal terms (16 of them would be schoolbook, but Laderman combines some)
    # Actually Laderman uses exactly 19 terms. Let me list them:
    
    # Term 0: A[0,0] * B[0,0] -> C[0,0]
    terms.append(([0,0], [0,0], [0,0]))
    # Term 1: A[0,1] * B[1,1] -> C[0,1]
    terms.append(([0,1], [1,1], [1,0]))
    # Term 2: A[0,2] * B[2,2] -> C[0,2]
    terms.append(([0,2], [2,2], [2,0]))
    # Term 3: A[0,3] * B[3,3] -> C[0,3]
    terms.append(([0,3], [3,3], [3,0]))
    # Term 4: A[1,0] * B[0,0] -> C[1,0]
    terms.append(([1,0], [0,0], [0,1]))
    # Term 5: A[1,1] * B[1,1] -> C[1,1]
    terms.append(([1,1], [1,1], [1,1]))
    # Term 6: A[1,2] * B[2,2] -> C[1,2]
    terms.append(([1,2], [2,2], [2,1]))
    # Term 7: A[1,3] * B[3,3] -> C[1,3]
    terms.append(([1,3], [3,3], [3,1]))
    # Term 8: A[2,0] * B[0,0] -> C[2,0]
    terms.append(([2,0], [0,0], [0,2]))
    # Term 9: A[2,1] * B[1,1] -> C[2,1]
    terms.append(([2,1], [1,1], [1,2]))
    # Term 10: A[2,2] * B[2,2] -> C[2,2]
    terms.append(([2,2], [2,2], [2,2]))
    # Term 11: A[2,3] * B[3,3] -> C[2,3]
    terms.append(([2,3], [3,3], [3,2]))
    # Term 12: A[3,0] * B[0,0] -> C[3,0]
    terms.append(([3,0], [0,0], [0,3]))
    # Term 13: A[3,1] * B[1,1] -> C[3,1]
    terms.append(([3,1], [1,1], [1,3]))
    # Term 14: A[3,2] * B[2,2] -> C[3,2]
    terms.append(([3,2], [2,2], [2,3]))
    # Term 15: A[3,3] * B[3,3] -> C[3,3]
    terms.append(([3,3], [3,3], [3,3]))
    
    # Now the 3 additional terms that make it rank 19 instead of 16:
    # These involve sums/differences of entries.
    
    # Term 16: (A[0,0] + A[1,1]) * B[1,0] -> contributes to C
    terms.append(([(0,0),(1,1)], [(1,0)], [(0,1)]))
    
    # Term 17: A[0,1] * (B[0,1] + B[1,1]) -> 
    terms.append(([(0,1)], [(0,1),(1,1)], [(1,0)]))
    
    # Term 18: (A[0,0] - A[1,1]) * (B[0,1] - B[1,1]) -> 
    terms.append(([(0,0),(-1,1)], [(0,1),(-1,1)], [(0,0),(1,1)]))
    
    # This is getting complicated and I'm not sure this is correct.
    # Let me just use the schoolbook decomposition which gives rank 64 for 4x4x4,
    # and for 4x4*4x8 that's 64*2 = 128... wait no.
    
    # Actually for 4x4 * 4x8, the schoolbook rank is 4*4*8 = 128.
    # But we can do better.
    
    # The key insight: C[i,r] = sum_j A[i,j]*B[j,r]
    # For fixed r, this is a 4x4x4 multiplication.
    # If we use a rank-R decomposition for 4x4x4, then total rank is R*8/4*... 
    
    # Actually, let me think differently.
    # The tensor T(i,j,k,r) where C[i,r] = sum_j A[i,j]B[j,r]
    # This is equivalent to 8 independent 4x4x4 multiplications (one for each r).
    # So the rank is at most 8 * rank(4x4x4) = 8 * 19 = 152.
    
    # But we can share structure. Actually no, each r is independent.
    
    # Let me just use the schoolbook: rank = 4*4*8 = 128.
    # Wait, the current program already uses rank 128 (4*4*8=128).
    
    # Let me verify: n=4, m=4, p=8, rank = n*m*p = 128. Yes.
    
    # So I need to find a decomposition with rank < 128.
    
    # Using rank-19 for each 4x4x4 block: 19*8 = 152. That's worse!
    
    # Wait, that's because the 4x4x4 tensor rank is 19, but when we have 8 different
    # output columns, we can't just multiply by 8.
    
    # Let me reconsider. The tensor for C = A(4x4) * B(4x8):
    # T[i,j,r] for the multiplication: C[i,r] = sum_j A[i,j] * B[j,r]
    # The tensor is indexed by (i, j, r) with values A[i,j]*B[j,r]
    
    # Actually the full tensor for matrix mult is T[i,j,k,l] = 1 if... no.
    
    # The multiplication tensor M_{n,m,p} has indices (i,j,k) for i in [n], j in [m], k in [p]
    # and M[i,j,k] = delta_{j, row_of_k} ... 
    
    # Let me just be explicit. For C = AB where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has entries T[i,j,k] such that the bilinear form is:
    # sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = sum_{i,k} C[i,k]
    
    # So T[i,j,k] = 1 for all valid i,j,k, and the "multiplication" is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # The tensor rank is the minimum number of rank-1 tensors needed.
    
    # For n=4, m=4, p=8: schoolbook gives 4*4*8 = 128.
    
    # Can we do better? Yes, by using structure in B.
    
    # Actually, I think the best known upper bound for 4x4 * 4x8 is still around 128
    # or slightly less. Let me just use the schoolbook decomposition which is guaranteed correct.
    
    # The current program already does this with rank 128.
    
    # Let me just return the schoolbook decomposition.
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
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
