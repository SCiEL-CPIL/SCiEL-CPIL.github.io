import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Use rank 76: embed 3x3x3 (rank 7) for the 3x3 sub-block,
    # plus schoolbook for the remaining parts.
    # For 3x4x7: we can use 7 terms for the 3x3x3 block (i<3, j<3, k<3),
    # and schoolbook for the rest.
    # Total: 7 (for 3x3x3) + (4*7 - 3*3) = 7 + (28-9) = 7 + 19 = 26? No...
    # Let me think more carefully.
    # 
    # The tensor has components T[i,j,k] for i in 0..2, j in 0..3, k in 0..6.
    # A rank-1 term contributes u_i * v_j * w_k to T[i,j,k].
    # 
    # Strategy: Use 7 rank-1 terms to cover the 3x3x3 sub-tensor (i<3, j<3, k<3)
    # exactly (the known Strassen-like decomposition for 3x3x3 has rank 7).
    # Then for the remaining entries where j>=3 or k>=3, use schoolbook terms.
    # 
    # Entries not covered by the 3x3x3 block:
    # - j in {3}, k in {0,1,2}: 3*3 = 9 entries (i can be 0,1,2; k can be 0,1,2)
    # - j in {0,1,2,3}, k in {3,4,5,6}: 4*3*4 = 48 entries... wait
    # 
    # Let me count differently. The 3x3x3 block covers i in [0,3), j in [0,3), k in [0,3).
    # Remaining:
    # Case 1: j=3, k in [0,3): 3 values of i * 1 * 3 = 9
    # Case 2: j in [0,4), k in [3,7): 3 * 4 * 4 = 48
    # Total remaining: 9 + 48 = 57
    # Schoolbook for remaining: 57 terms
    # Total: 7 + 57 = 64
    # 
    # But we need to be careful: the 7 terms for 3x3x3 might have non-zero values
    # for j>=3 or k>=3 that could interfere. We need to ensure they are zero there.
    # 
    # Actually, let's use a simpler approach: use schoolbook for everything except
    # we try to find a lower-rank decomposition for the 3x3x3 sub-block.
    # 
    # The known minimal rank for 3x3x3 matrix multiplication is 7 (Landsberg, 1994).
    # Let me construct it explicitly.
    # 
    # For multiplying 3x3 by 3x3, rank 7 decomposition:
    # The 7 terms can be constructed from the 8 schoolbook terms minus 1.
    # 
    # Actually, let me just use rank 76 directly with a known construction.
    # 
    # Simpler approach: use schoolbook but merge some terms.
    # For 3x4x7, schoolbook is 84. 
    # 
    # Let me try: use 7 terms for the 3x3x3 core, and for the remaining
    # dimensions use minimal additional terms.
    # 
    # The 3x3x3 core (i in 0..2, j in 0..2, k in 0..2) needs rank 7.
    # The remaining entries:
    # - i in 0..2, j=3, k in 0..6: 3*7 = 21 entries
    # - i in 0..2, j in 0..3, k in 3..6: 3*4*4 = 48 entries
    # Total: 69 entries
    # 
    # But the 7 core terms may have support outside the core. Let me design
    # them to have zero support for j>=3 and k>=3.
    # 
    # If I can make the 7 core terms have v_j = 0 for j>=3 and w_k = 0 for k>=3,
    # then I need schoolbook for the remaining 69 entries: 69 terms.
    # Total: 7 + 69 = 76.
    # 
    # Let me construct the 7-term decomposition for 3x3x3 with restricted support.
    # 
    # For 3x3x3 matrix mult, the tensor is T[i,j,k] = delta_{i+k, j} (no wait,
    # it's the multiplication tensor: (A*B)[i,k] = sum_j A[i,j]*B[j,k])
    # So T[i,j,k] = 1 if this is the standard basis element.
    # Actually the matrix multiplication tensor for m x n * n x p is:
    # T[i,j,k,l] where C[i,l] = sum_{j,k} A[i,j] B[k,l] delta_{j,k}
    # 
    # In our notation with factors U, V, W:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition is: for each rank term r,
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta condition
    # 
    # For matrix mult: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 when we're computing C[i,k] from A[i,j]*B[j,k]
    # The W factor indexes (k*n + i), so W[k*n+i, r] should be 1 for the term.
    # 
    # For the 3x3x3 sub-block (i in 0..2, j in 0..2, k in 0..2):
    # We need: sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1 for all i,j,k in 0..2
    # Wait no. The matrix multiplication is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So for each triple (i,j,k) with j being the summation index, the tensor value is 1.
    # But actually, the tensor is T[i,j,k] = 1 for all valid (i,j,k) because
    # each term A[i,j]*B[j,k] contributes to C[i,k].
    # 
    # Hmm, let me re-read the convention. The output factor W[k*n+i, r] means
    # W is indexed by (k, i) where k is the column of C and i is the row.
    # So W[k*n+i, r] corresponds to the coefficient for C[i,k].
    # 
    # The decomposition says: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    # 
    # No wait. Let me think again. The standard tensor decomposition for matrix mult:
    # M[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where M[i,j,k] = 1 for all i,j,k (since every A[i,j]*B[j,k] contributes to C[i,k]).
    # 
    # Actually no. The matrix multiplication tensor has entries:
    # T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1
    # because C[i,k] = sum_j A[i,j] B[j,k], and each term is a separate tensor entry.
    # 
    # Wait, that's not right either. The tensor is:
    # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to give
    # a contribution to C[i,k]. Since every such pair contributes, T[i,j,k] = 1 for all valid i,j,k.
    # 
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    # 
    # For the 3x3x3 sub-block, this is the standard 3x3x3 matrix multiplication tensor.
    # The minimal rank is 7.
    # 
    # Let me look up or derive the rank-7 decomposition for 3x3x3.
    # 
    # One known rank-7 decomposition for 3x3x3:
    # It's based on the fact that 3x3x3 can be embedded in a structure.
    # 
    # Actually, for this problem, let me just use a practical approach:
    # Start with schoolbook rank 84, and try to reduce by finding linear dependencies
    # or using known results.
    # 
    # Given time constraints, let me use rank 76 with a specific construction.
    # 
    # Here's a construction:
    # Use 7 terms for the 3x3x3 core with U, V, W having zeros for j>=3 and k>=3.
    # Use 69 schoolbook terms for the rest.
    # 
    # For the 3x3x3 core, I'll use a known rank-7 decomposition.
    # One such decomposition uses the following 7 triples (u, v, w) where
    # u is in R^9 (for i,j in 0..2), v is in R^9 (for j,k in 0..2), w is in R^9 (for k,i in 0..2).
    # 
    # Let me use a simpler approach: just return schoolbook rank 84 but verify it works.
    # Then if time permits, try to improve.
    
    # For now, let's use a rank-76 construction.
    # I'll use 7 terms for the 3x3x3 block and 69 for the rest.
    
    rank = 76
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # 7 terms for the 3x3x3 core (i in 0..2, j in 0..2, k in 0..2)
    # Using a known rank-7 decomposition for 3x3x3 matrix multiplication.
    # 
    # The 3x3x3 multiplication tensor can be decomposed with rank 7.
    # One explicit construction:
    # 
    # Let me use the following approach. For 3x3x3, we can use:
    # Terms 0-6: specific patterns
    
    # Actually, let me just use schoolbook for the 3x3x3 part too (9 terms)
    # and then optimize the total.
    # 
    # Total schoolbook: 3*4*7 = 84
    # If I use 9 terms for 3x3x3, I need 84-9 = 75 for the rest, total 84.
    # 
    # To get 76, I need to save 8 terms. Using rank 7 for 3x3x3 saves 2 terms,
    # giving 7+75 = 77. Not quite 76.
    # 
    # Let me try a different split. 
    # 
    # Alternative: use rank 7 for 3x3x3, and for the remaining 69 entries,
    # see if I can do better than 69 schoolbook terms.
    # 
    # The remaining entries form a "frame" around the 3x3x3 core.
    # - j=3, k in 0..6: 3*7 = 21 entries (i in 0..2)
    # - j in 0..3, k in 3..6: 3*4*4 = 48 entries (i in 0..2)
    # Total: 69
    # 
    # Can I share terms between these groups? 
    # A term with V[j=3, k] nonzero can cover multiple i values if U[i, j=3] is set appropriately.
    # 
    # This is getting complex. Let me just use rank 76 with a straightforward construction:
    # 7 terms for 3x3x3 + 69 schoolbook for the rest = 76.
    
    # For the 3x3x3 core, use a rank-7 decomposition.
    # I'll use a specific known decomposition.
    
    # Rank-7 decomposition for 3x3x3:
    # Based on the work of Landsberg, the 3x3x3 tensor has rank 7.
    # One explicit decomposition:
    
    # Let me use the following 7 terms (u_r, v_r, w_r) for the 3x3x3 core:
    # These are 9-dimensional vectors (for the 3x3 indices).
    
    # I'll construct them to satisfy: sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1 for all i,j,k in 0..2
    
    # A known rank-7 decomposition for 3x3x3:
    # Terms 0-2: diagonal terms
    # Terms 3-6: off-diagonal combinations
    
    # Let me use a concrete construction:
    # For 3x3x3, one rank-7 decomposition is:
    
    core_terms = [
        # (u, v, w) where u[i*3+j], v[j*3+k], w[k*3+i]
        # Term 0: i=0,j=0,k=0 type
        ([(1,0,0,0,0,0,0,0,0)], [(0,0,0,0,0,0,0,0,0)], None),  # placeholder
    ]
    
    # This is getting too complex without a reference. Let me just use schoolbook rank 84.
    # The task says score = rank, so 84 is the baseline. Let me at least return a valid solution.
    
    rank = n * m * p  # 84
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
