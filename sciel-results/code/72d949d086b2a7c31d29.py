"""Rewritten program using a distinct construction strategy for 2x3x8 matrix multiplication."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 2, 3, 8
    rank = 44
    
    # Build tensor T of shape (n, m, p) where T[i,j,k] = 1
    # This represents the matrix multiplication tensor for multiplying
    # an (n x m) matrix by an (m x p) matrix.
    # The decomposition U, V, W satisfy:
    #   T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # We split p=8 into four groups of p=2.
    # For each p=2 group, we have a 2x3x2 sub-tensor.
    # For each 2x3x2, we split m=3 into m=2 and m=1.
    #   m=2, p=2: 2x2x2, rank 7 (Strassen)
    #   m=1, p=2: 2x1x2, rank 2 (schoolbook: two scalar products)
    # Total per p=2 group: 7 + 2 = 9
    # Total for 4 groups: 36... wait, that's only 36.
    # 
    # Actually let me reconsider. The current program uses rank 44.
    # Let me think about what's minimal.
    # 
    # 2x3x4: We can think of this as multiplying (2x3) by (3x4).
    # Split m=3 into m=2 + m=1, split p=4 into p=2 + p=2.
    #   (m=2, p=2): 2x2x2, rank 7
    #   (m=2, p=2): 2x2x2, rank 7  
    #   (m=1, p=2): 2x1x2, rank 2
    #   (m=1, p=2): 2x1x2, rank 2
    # Total: 7+7+2+2 = 18
    # 
    # But the current program says 22 for 2x3x4. Let me re-examine.
    # The current program splits p=4 into two p=2 blocks, and for each:
    #   m=2 part: Strassen 2x2x2 rank 7
    #   m=1 part: schoolbook 2x1x2 rank 2
    # So per p=2 block within a p=4: 7+2 = 9
    # For p=4: 9+9 = 18
    # For p=8: 18+18 = 36
    # 
    # But the current program uses 44. Let me re-read...
    # 
    # Current program: "For m=2, p=4 part, split p=4 into two p=2 blocks.
    # Each p=2 block is 2x2x2, rank 7. So m=2, p=4 part rank is 7+7=14.
    # m=1, p=4 part is 2x1x4, rank 8 (Schoolbook)."
    # 
    # Ah, they use schoolbook for the m=1, p=4 part which is rank 8 (2*4=8 scalar products).
    # If instead we split p=4 into two p=2 blocks for the m=1 part too:
    #   m=1, p=2: rank 2 each, so m=1, p=4: rank 4
    # Total per p=4: 14 + 4 = 18
    # Total for p=8: 36
    #
    # Can we do even better? What about using a smarter decomposition for 2x3x2?
    # 2x3x2: multiply (2x3) by (3x2). Rank is at most 2*2=4 (schoolbook on p dimension)
    # or at most 2*3=6 (schoolbook on m dimension).
    # Actually, the minimal rank for 2x3x2... let me think.
    # The tensor has dimensions 2x3x2 = 12 entries. 
    # Schoolbook: 6 multiplications (2*3 scalars in A times... no).
    # Actually for matrix mult (n x m) * (m x p), schoolbook rank is n*m*p = 12.
    # But we can do better.
    # 
    # For 2x3x2: We can view this as computing 2 outputs each being a dot product of length 3.
    # So 2 dot products of length 3 = 6 scalar multiplications. Rank 6.
    # Can we do less? The tensor T[i,j,k] = delta_{j,j} for the appropriate indices...
    # Actually T is the standard matrix multiplication tensor.
    # 
    # Let me just use rank 36 with the improved splitting and verify it works.
    
    # Strategy: For each of 4 p=2 blocks:
    #   Use Strassen (rank 7) for the 2x2x2 part (i in 0..1, j in 0..1, k in 0..1)
    #   Use 2 schoolbook terms for the 2x1x2 part (i in 0..1, j=2, k in 0..1)
    # Total: 4 * (7 + 2) = 36
    
    def add_strassen_2x2x2(i_base, j_base, k_base):
        """Add Strassen decomposition for A[i_base:i_base+2, j_base:j_base+2] * B[j_base:j_base+2, k_base:k_base+2]"""
        nonlocal col
        # U indices: (i_base+di)*m + (j_base+dj)
        # V indices: (j_base+dj)*p + (k_base+dk)
        # W indices: (k_base+dk)*n + (i_base+di)
        
        # m1: (a+d)(e+h)
        U[(i_base+0)*m + (j_base+0), col] += 1
        U[(i_base+1)*m + (j_base+1), col] += 1
        V[(j_base+0)*p + (k_base+0), col] += 1
        V[(j_base+1)*p + (k_base+1), col] += 1
        W[(k_base+0)*n + (i_base+0), col] += 1
        W[(k_base+1)*n + (i_base+1), col] += 1
        col += 1
        
        # m2: (c+d)e
        U[(i_base+1)*m + (j_base+0), col] += 1
        U[(i_base+1)*m + (j_base+1), col] += 1
        V[(j_base+0)*p + (k_base+0), col] += 1
        W[(k_base+0)*n + (i_base+1), col] += 1
        W[(k_base+1)*n + (i_base+1), col] += -1
        col += 1
        
        # m3: a(f-h)
        U[(i_base+0)*m + (j_base+0), col] += 1
        V[(j_base+0)*p + (k_base+1), col] += 1
        V[(j_base+1)*p + (k_base+1), col] += -1
        W[(k_base+1)*n + (i_base+0), col] += 1
        W[(k_base+1)*n + (i_base+1), col] += 1
        col += 1
        
        # m4: d(g-e)
        U[(i_base+1)*m + (j_base+1), col] += 1
        V[(j_base+1)*p + (k_base+0), col] += 1
        V[(j_base+0)*p + (k_base+0), col] += -1
        W[(k_base+0)*n + (i_base+0), col] += 1
        W[(k_base+0)*n + (i_base+1), col] += 1
        col += 1
        
        # m5: (a+b)h
        U[(i_base+0)*m + (j_base+0), col] += 1
        U[(i_base+0)*m + (j_base+1), col] += 1
        V[(j_base+1)*p + (k_base+1), col] += 1
        W[(k_base+0)*n + (i_base+0), col] += -1
        W[(k_base+1)*n + (i_base+0), col] += 1
        col += 1
        
        # m6: (c-a)(e+f)
        U[(i_base+1)*m + (j_base+0), col] += 1
        U[(i_base+0)*m + (j_base+0), col] += -1
        V[(j_base+0)*p + (k_base+0), col] += 1
        V[(j_base+0)*p + (k_base+1), col] += 1
        W[(k_base+1)*n + (i_base+1), col] += 1
        col += 1
        
        # m7: (b-d)(g+h)
        U[(i_base+0)*m + (j_base+1), col] += 1
        U[(i_base+1)*m + (j_base+1), col] += -1
        V[(j_base+1)*p + (k_base+0), col] += 1
        V[(j_base+1)*p + (k_base+1), col] += 1
        W[(k_base+0)*n + (i_base+0), col] += 1
        col += 1
    
    def add_schoolbook_2x1x2(i_base, j_val, k_base):
        """Add schoolbook for A[i_base:i_base+2, j_val] * B[j_val, k_base:k_base+2]"""
        nonlocal col
        for di in range(2):
            for dk in range(2):
                U[(i_base+di)*m + j_val, col] = 1
                V[j_val*p + (k_base+dk), col] = 1
                W[(k_base+dk)*n + (i_base+di), col] = 1
                col += 1
    
    # Process 4 blocks of p=2
    for block in range(4):
        k_base = block * 2
        # Strassen for j=0,1
        add_strassen_2x2x2(0, 0, k_base)
        # Schoolbook for j=2
        add_schoolbook_2x1x2(0, 2, k_base)
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
