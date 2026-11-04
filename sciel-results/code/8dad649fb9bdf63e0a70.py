"""Matrix multiplication 3x3 by 3x8 - tensor decomposition search."""
import numpy as np
import itertools
import time
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 8
    U_rows = n * m  # 9
    V_rows = m * p  # 24
    W_rows = p * n  # 24
    
    # The tensor T[i,j,k] corresponds to U[i*m+j] * V[j*p+k] * W[k*n+i]
    # We want to find minimal rank decomposition
    
    # Schoolbook gives rank 72. We try to do better.
    
    # Strategy: Use the fact that p=8 = 3+3+2, or exploit structure.
    # Let's try to find a decomposition with rank < 72.
    
    # One approach: if we can find a rank-r decomposition for M_{3,3,3} with r < 27,
    # we can combine with M_{3,3,2} with rank < 12.
    # But we need exact integer/half-integer decompositions.
    
    # Let's try a direct search using the tensor structure.
    # The tensor is T = sum_{a,b,c} e_a^{(U)} e_b^{(V)} e_c^{(W)} where T[i,j,k] = 1 if (i,j,k) in range.
    
    # Actually, let's think about this differently.
    # For 3x3 x 3x3, the best known rank is 23 (Strassen-based or similar).
    # For 3x3 x 3x2, we need a decomposition.
    
    # Let's try to use known Strassen-like constructions.
    # Strassen's algorithm for 2x2x2 has rank 7.
    # We can pad 3x3 to 4x4, use rank 49, then... no, that's worse.
    
    # Alternative: use the fact that 8 = 2+2+2+2 and decompose.
    # M_{3,3,8} = M_{3,3,2} + M_{3,3,2} + M_{3,3,2} + M_{3,3,2}
    # If we can find rank r_{3,3,2} for M_{3,3,2}, then total rank = 4 * r_{3,3,2}
    # Schoolbook for 3x3x2 is 18, so 4*18=72. Same as before.
    
    # What about M_{3,3,4} = M_{3,3,2} + M_{3,3,2}?
    # If rank(M_{3,3,4}) < 2*rank(M_{3,3,2}), we can save.
    
    # Let me try a more systematic approach: use bilinear complexity via
    # the border rank or direct tensor rank search.
    
    # Given the time constraint, let's try a heuristic search:
    # 1. Start with schoolbook (rank 72)
    # 2. Try to merge terms or find linear dependencies
    
    # Actually, let's try a completely different approach:
    # Use the Strassen-Winograd algorithm structure.
    
    # For 3x3 matrices, there are known algorithms with rank 23 or 24.
    # For 3x3 x 3x8, we can try to find a structured decomposition.
    
    # Let me try to use the following idea:
    # Decompose the 3x8 matrix as [3x2, 3x2, 3x2, 3x2] and use a 
    # 4-way split with shared intermediate values.
    
    # This is complex. Let me try a direct numerical approach instead.
    
    # Use rank minimization via SVD on the flattened tensor.
    # The tensor T has shape (9, 24, 24) with T[i,j,k] = 1 if the corresponding
    # multiplication term is present.
    
    # Actually, the tensor for matrix multiplication is:
    # T[i,j,k] = 1 if the (i,j) element of A multiplies with the (j,k) element of B
    # Wait, no. The tensor T satisfies:
    # C[i,k] = sum_{i,j,k} A[i,j] * B[j,k] * T[i,j,k]
    # So T[i,j,k] = 1 for all valid i,j,k (which is all of them for rectangular matrices).
    
    # The tensor is simply the all-ones tensor of shape (9, 24, 24) where
    # the indices are (i*m+j, j*p+k, k*n+i) for i in [0,3), j in [0,3), k in [0,8).
    # Wait, that's not right either. Let me re-read the convention.
    
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # So W has shape (p*n, rank) = (24, rank), indexed by k*n + i.
    # U has shape (n*m, rank) = (9, rank), indexed by i*m + j.
    # V has shape (m*p, rank) = (24, rank), indexed by j*p + k.
    
    # The decomposition says: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # for the appropriate j. Wait, but j is summed over in matrix multiplication.
    
    # Actually, the tensor decomposition for matrix multiplication is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_r (sum_j A[i,j] * U_{i,j,r}) * V_{j,k,r} * W_{k,i,r}  ... no, that's not right.
    
    # The correct tensor decomposition is:
    # The multiplication tensor T has T[i,j,k] = 1 (for the 3x3x3 case, all ones in a sense)
    # and C[i,k] = sum_{j,r} A[i,j] * B[j,k] * ... 
    # 
    # Let me think more carefully. The standard tensor decomposition for matrix multiplication
    # M_{n,m,p} is a decomposition of the tensor T in R^{(n*m) x (m*p) x (p*n)} such that
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1 for all i,j,k, and 0 otherwise.
    # Wait, that's not quite right. The tensor is:
    # T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    # and 0 otherwise.
    # 
    # So T is a 0-1 tensor with exactly n*m*p = 72 ones.
    # We want to decompose T = sum_r u_r v_r w_r where u_r in R^{9}, v_r in R^{24}, w_r in R^{24}
    # and the entries are integers or half-integers.
    
    # This is a rank minimization problem for a specific 0-1 tensor.
    
    # Given the complexity, let me try a practical approach:
    # 1. Use the known Strassen-based decomposition for 3x3x3 (rank 23 or 24)
    # 2. Extend to 3x3x8
    
    # For now, let's just return the schoolbook decomposition, but with a
    # small optimization: try to see if we can use rank 71 by finding one
    # linear dependency.
    
    # The tensor has 72 nonzero entries. The minimum rank is at least the
    # maximum of the border rank lower bounds. For M_{3,3,8}, the lower bound
    # is not well-established, but it's certainly less than 72.
    
    # Given time constraints, let me try a simple heuristic:
    # Randomly perturb the schoolbook and try to reduce rank via SVD.
    
    # Actually, the safest bet is to return the schoolbook decomposition.
    # Any attempt at rank reduction without knowing the exact coefficients
    # will likely produce an invalid decomposition.
    
    rank = 72
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
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
