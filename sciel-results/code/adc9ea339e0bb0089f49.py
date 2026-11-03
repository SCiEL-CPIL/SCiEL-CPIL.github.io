"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # We'll try to find a rank-minimal decomposition using a structured approach.
    # Strategy: Use Strassen for 2x2 blocks and optimize the remaining parts.
    # 
    # Key insight: For 2x3x8, we can:
    # 1. Split B into two 3x4 blocks
    # 2. For each 3x4 block, use a hybrid decomposition
    #
    # Let's try to find a better decomposition by considering the structure more carefully.
    #
    # Alternative approach: Use a global optimization over the tensor structure.
    # The tensor T[i,j,k] = delta[i,k] for i in {0,1}, j in {0,1,2}, k in {0,...,7}
    #
    # We want to find rank r such that T = sum_t U_t ⊗ V_t ⊗ W_t
    # where U_t is in R^{6}, V_t in R^{24}, W_t in R^{16}
    #
    # For small ranks, we can try to find explicit decompositions.
    #
    # Known bounds:
    # - Schoolbook: 2*3*8 = 48
    # - Current best in code: 44
    # - Theoretical lower bound: max(2*3, 3*8, 2*8) = 24, but this is not tight
    #
    # Let's try a different structural decomposition:
    # Group the 8 columns of B into 4 pairs: (k0,k1), (k2,k3), (k4,k5), (k6,k7)
    # For each pair, we have a 2x3x2 subproblem.
    #
    # For 2x3x2: 
    #   Split j into {0,1} and {2}
    #   j in {0,1}: 2x2x2, rank 7 (Strassen)
    #   j = 2: 2x1x2, rank 4 (schoolbook: 2*2)
    #   Total per pair: 7 + 4 = 11
    #   Total for 4 pairs: 4*11 = 44
    #
    # That gives the same rank 44. Let's see if we can do better.
    #
    # What if we use a different splitting for the 2x1x2 part?
    # 2x1x2: C[i,k] = A[i,2] * B[2,k]
    # This is an outer product of A[:,2] (2x1) and B[2,:] (1x2)
    # Rank is exactly 2*2 = 4. Can't do better.
    #
    # So the bottleneck is the 2x2x2 part at rank 7.
    #
    # Alternative: Can we share computation across the 4 pairs?
    # The A matrix is the same for all pairs, only B differs.
    # So we can precompute some combinations of A rows.
    #
    # Let's try: Precompute 7 Strassen factors for A (2x2 part):
    #   m1 = A00 + A11
    #   m2 = A10 + A11  
    #   m3 = A00
    #   m4 = A11
    #   m5 = A00 + A01
    #   m6 = A10 - A00
    #   m7 = A01 - A11
    #
    # Then for each k-pair, we need to combine these with B factors.
    # But the B factors depend on which k-pair we're in.
    #
    # Actually, the Strassen decomposition for each 2x2x2 block is:
    #   C00 = m1*B00 - m4*(B10-B00) - m5*(B11-B01) + ... 
    # This doesn't simplify easily to sharing.
    #
    # Let me try a completely different approach: use the fact that n=2 is small.
    # C is 2x8, so we have 16 output entries.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We can write this as:
    # C[0,:] = A[0,0]*B[0,:] + A[0,1]*B[1,:] + A[0,2]*B[2,:]
    # C[1,:] = A[1,0]*B[0,:] + A[1,1]*B[1,:] + A[1,2]*B[2,:]
    #
    # Let u = A[0,:], v = A[1,:], so C[0,:] = u0*B[0,:] + u1*B[1,:] + u2*B[2,:]
    # and C[1,:] = v0*B[0,:] + v1*B[1,:] + v2*B[2,:]
    #
    # This is like a bilinear map. The rank is the minimum number of rank-1 bilinear forms.
    #
    # For this specific case, let's try to find a rank < 44 decomposition.
    #
    # Idea: Use the structure that B has 8 columns. Group them as B = [B1 | B2] where
    # B1, B2 are 3x4. Then C = [A*B1 | A*B2].
    #
    # For A*B1 (2x4): 
    #   Use Strassen on the 2x2x2 blocks: 2 blocks, each rank 7 = 14
    #   Plus the j=2 part: 2x1x4, rank 8
    #   Total: 22
    #
    # Same for A*B2: 22
    # Total: 44
    #
    # To beat 44, we need a more clever approach.
    #
    # Let's try: What if we don't split B into two 3x4 blocks, but instead
    # use a global Strassen-like approach?
    #
    # Consider: C = A * B
    # Let A = [A1 | A2] where A1 is 2x2 (cols 0,1) and A2 is 2x1 (col 2)
    # Let B = [B1; B2] where B1 is 2x8 (rows 0,1) and B2 is 1x8 (row 2)
    #
    # C = A1 * B1 + A2 * B2
    #
    # A1 * B1 is 2x2 * 2x8 = 2x8. We can use Strassen on 2x2x2 blocks:
    #   4 blocks of 2x2x2, each rank 7 = 28
    #
    # A2 * B2 is 2x1 * 1x8 = 2x8. Rank 16.
    #
    # Total: 28 + 16 = 44. Same again.
    #
    # It seems 44 is hard to beat with these simple splits. Let me try a more
    # sophisticated approach using the actual tensor structure.
    #
    # The tensor T has entries T[i,j,k] = 1 if i==k (for the diagonal), but wait,
    # no. For matrix multiplication, T[i,j,k] = delta[i,k] is not right.
    # 
    # Actually, the multiplication tensor for (n,m,p) = (2,3,8) is:
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    # So T is a 2x3x8 tensor where T[i,j,k] = 1 for all i,j,k? No.
    #
    # The tensor is: T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * delta_{i,k}... no.
    #
    # For matrix mult, the tensor is T[i,j,k] = 1 if the (i,k) entry of C
    # depends on A[i,j] and B[j,k]. So T[i,j,k] = 1 for all valid i,j,k.
    # Wait, that's not right either.
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,...,7}.
    # No! T[i,j,k] is the coefficient of A[i,j]*B[j,k] in C[i,k], which is 1.
    # But C[i,k] only depends on the triple (i,j,k) where the j index is summed.
    #
    # The tensor is: T = sum_{i,j,k} A[i,j] B[j,k] e_{i,k}
    # So T[i,j,k] = 1 for all i,j,k.
    #
    # Wait, that means the tensor is the all-ones tensor of shape 2x3x8?
    # No, that's not right. Let me think again.
    #
    # The bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T is defined by T(A,B) = C, so T[i,j,k] is such that
    # C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k.
    #
    # But that can't be right because then the rank would be 2*3*8 = 48 (schoolbook).
    #
    # Ah, I see the issue. The tensor T is not all-ones. It's:
    # T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # So T[i,j,k] = 1 for all valid (i,j,k).
    #
    # But the rank of the all-ones tensor of shape 2x3x8 is indeed 48 (since it's
    # a sum of 48 rank-1 tensors e_i ⊗ e_j ⊗ e_k).
    #
    # However, we can do better because of the structure. The key is that C[i,k]
    # is a linear combination over j, so we can use Strassen-like tricks.
    #
    # Let me just implement the rank-44 decomposition cleanly and see if I can
    # find a better one through optimization.
    
    # For now, let's return the known rank-44 decomposition.
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 decomposition
    def add_strassen(i0, j0, k0):
        nonlocal col
        # m1 = (A00 + A11)(B00 + B11)
        U[i0 * m + j0][col] += 1
        U[(i0 + 1) * m + (j0 + 1)][col] += 1
        V[j0 * p + k0][col] += 1
        V[(j0 + 1) * p + (k0 + 1)][col] += 1
        W[k0 * n + i0][col] += 1
        W[(k0 + 1) * n + (i0 + 1)][col] += 1
        col += 1
        
        # m2 = (A10 + A11) * B00
        U[(i0 + 1) * m + j0][col] += 1
        U[(i0 + 1) * m + (j0 + 1)][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + (i0 + 1)][col] += 1
        W[(k0 + 1) * n + (i0 + 1)][col] += -1
        col += 1
        
        # m3 = A00 * (B01 - B11)
        U[i0 * m + j0][col] += 1
        V[j0 * p + (k0 + 1)][col] += 1
        V[(j0 + 1) * p + (k0 + 1)][col] += -1
        W[(k0 + 1) * n + i0][col] += 1
        W[(k0 + 1) * n + (i0 + 1)][col] += 1
        col += 1
        
        # m4 = A11 * (B10 - B00)
        U[(i0 + 1) * m + (j0 + 1)][col] += 1
        V[(j0 + 1) * p + k0][col] += 1
        V[j0 * p + k0][col] -= 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + (i0 + 1)][col] += 1
        col += 1
        
        # m5 = (A00 + A01) * B11
        U[i0 * m + j0][col] += 1
        U[i0 * m + (j0 + 1)][col] += 1
        V[(j0 + 1) * p + (k0 + 1)][col] += 1
        W[k0 * n + i0][col] -= 1
        W[(k0 + 1) * n + i0][col] += 1
        col += 1
        
        # m6 = (A10 - A00) * (B00 + B01)
        U[(i0 + 1) * m + j0][col] += 1
        U[i0 * m + j0][col] -= 1
        V[j0 * p + k0][col] += 1
        V[j0 * p + (k0 + 1)][col] += 1
        W[(k0 + 1) * n + (i0 + 1)][col] += 1
        col += 1
        
        # m7 = (A01 - A11) * (B10 + B11)
        U[i0 * m + (j0 + 1)][col] += 1
        U[(i0 + 1) * m + (j0 + 1)][col] -= 1
        V[(j0 + 1) * p + k0][col] += 1
        V[(j0 + 1) * p + (k0 + 1)][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1
    
    # Schoolbook 2x1x4: A[:,j] (2x1) * B[j,:] (1x4)
    def add_2x1x4(i0, j, k0):
        nonlocal col
        for ii in range(2):
            for kk in range(4):
                U[(i0 + ii) * m + j][col] = 1
                V[j * p + (k0 + kk)][col] = 1
                W[(k0 + kk) * n + (i0 + ii)][col] = 1
                col += 1
    
    # Process k-block 0: k in 0..3
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_2x1x4(0, 2, 0)
    
    # Process k-block 1: k in 4..7
    add_strassen(0, 0, 4)
    add_strassen(0, 0, 6)
    add_2x1x4(0, 2, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
