import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # We will use a block-based decomposition that reduces the rank below the trivial n*m*p.
    # The trivial rank is 50.
    # We decompose the 5x5 matrix multiplication into blocks.
    # Let's try to use the rank of 2x2x2 (which is 7) for the 2x2x2 sub-blocks if possible, 
    # but the dimensions are 2x5x5.
    # 
    # A known lower bound for matrix multiplication rank R(n,m,p) is max(nm, mp, pn).
    # Here max(10, 25, 10) = 25.
    # The trivial upper bound is 50.
    # 
    # We can view the 2x5x5 multiplication as combining smaller multiplications.
    # Specifically, we can partition the 5x5 matrices into blocks.
    # However, a simpler approach is to use the fact that R(2,5,5) <= R(2,2,2) * something? No.
    # 
    # Let's use the "sum of products" decomposition.
    # We know R(2,2,2) = 7.
    # Can we embed 2x2x2 into 2x5x5?
    # If we take the first 2 columns of A and first 2 rows of B, we get a 2x2x2 block.
    # But we need to cover all 5x5.
    #
    # Let's try a specific construction based on the identity:
    # R(n,m,p) <= R(n1,m1,p1) * ... ?
    #
    # Actually, there is a known result that R(2,5,5) = 25 is NOT possible (lower bound is 25, but usually higher).
    # Wait, the lower bound for 2x5x5 is max(2*5, 5*5, 5*2) = 25.
    # Is it possible to achieve 25?
    # If we could achieve 25, the score would be 50/25 = 2.0.
    #
    # Let's try to construct a rank-25 decomposition.
    # This requires that the tensor is "slice rank" 25?
    # For square matrices, rank is often higher.
    #
    # Let's look at the seed program which uses rank 50.
    # Can we do better?
    #
    # Consider the decomposition of 5x5 multiplication using 2x2 blocks and 1x1 blocks.
    # 5 = 2 + 2 + 1.
    # We have blocks:
    # A = [A11 A12 A13; A21 A22 A23] where A11, A21 are 2x2, A12, A22 are 2x2, A13, A23 are 2x1.
    # This partitioning is messy for B (5x5).
    #
    # Alternative: Use the fact that R(2,5,5) <= 2 * R(1,5,5) ? No.
    #
    # Let's try a random search or a structured search for a smaller rank.
    # Since we need an exact decomposition, we can set up the system of equations.
    # But solving for rank 25 is hard.
    #
    # Let's try a heuristic:
    # We know R(2,2,2)=7.
    # If we split the 5x5 into four 2x2 blocks and four 2x1/1x2/1x1 blocks, it's complex.
    #
    # However, there is a simple improvement:
    # The trivial decomposition has rank 50.
    # We can use the identity:
    # (AB)_{ik} = sum_j A_{ij} B_{jk}
    #
    # Let's try to use the rank-7 decomposition for 2x2x2 on the "corner" 2x2x2 blocks.
    # We have 2x5x5.
    # Let's split the 5x5 B into B = [B1 B2 B3] where B1 is 2x5, B2 is 2x5, B3 is 1x5? No, B is 5x5.
    # Split B into columns: B1 (2 cols), B2 (2 cols), B3 (1 col).
    # Split A into rows? A is 2x5.
    #
    # Actually, let's just use a known upper bound.
    # R(2,5,5) is at most 2 * R(1,5,5)?
    # R(1,5,5) is the rank of vector-matrix multiplication? No, that's just dot products.
    #
    # Let's stick to a safe improvement if possible.
    # If we cannot find a better one, we return 50.
    #
    # Wait, the prompt asks for a "stronger algorithm".
    # A Gröbner basis solver is too slow for 120s for rank 25.
    #
    # Let's try a different structure:
    # We can use the fact that R(2,5,5) <= R(2,2,2) + R(2,3,3)?
    # If we partition the 5x5 into a 2x2 block and a 3x3 block (plus cross terms).
    # This is not a direct sum because of cross terms.
    #
    # However, we can use the "block diagonal" argument if we can diagonalize the cross terms? No.
    #
    # Let's try to use the rank 49? No, must be exact.
    #
    # Actually, there is a known decomposition for 2x5x5 with rank 25?
    # If the matrix multiplication tensor is "decomposable" into 25 rank-1 tensors.
    # This is equivalent to saying the matrix multiplication map is a "tensor of rank 25".
    # For 2x2x2, rank is 7 (not 8).
    # For 2x3x3, rank is?
    #
    # Given the constraints and the risk of failure, I will implement a search for a rank slightly lower than 50,
    # but if it fails, return 50.
    #
    # Actually, a very simple improvement:
    # The seed program uses rank 50.
    # We can try to use the fact that some entries might be zero? No, general matrices.
    #
    # Let's try to use the rank-7 decomposition for the 2x2x2 sub-tensor embedded in the 2x5x5.
    # We can define the decomposition as a sum of:
    # 1. The 2x2x2 block at (0:2, 0:2, 0:2) using rank 7.
    # 2. The remaining terms using rank 1.
    #
    # The 2x5x5 tensor has 50 entries.
    # The 2x2x2 sub-tensor covers 8 entries.
    # If we use rank 7 for these 8, we save 1 rank.
    # The remaining 42 entries need rank 42?
    # Total rank = 7 + 42 = 49.
    #
    # Is this valid?
    # The tensor is T = T_1 + T_2 + ...
    # If we decompose T_{0:2, 0:2, 0:2} with 7 terms, and the rest with 42 terms,
    # the total rank is 49.
    #
    # Let's verify if we can just "add" the decompositions.
    # Yes, rank(A+B) <= rank(A) + rank(B).
    # So if we split the tensor into two parts:
    # T1: non-zero only when i<2, j<2, k<2.
    # T2: non-zero otherwise.
    # Then rank(T) <= rank(T1) + rank(T2).
    #
    # T1 is the 2x2x2 multiplication tensor. Rank(T1) = 7.
    # T2 is the 2x5x5 tensor with the 2x2x2 corner zeroed out.
    # What is the rank of T2?
    # T2 has 50 - 8 = 42 non-zero entries.
    # We can decompose T2 using 42 rank-1 terms (one for each non-zero entry).
    # So rank(T2) <= 42.
    #
    # Total rank <= 7 + 42 = 49.
    #
    # This is a valid improvement!
    #
    # Let's implement this.
    #
    # Step 1: Define the 2x2x2 rank-7 decomposition (Strassen).
    # Step 2: Define the rank-1 terms for the remaining 42 entries.
    # Step 3: Combine them.
    #
    # Strassen 2x2x2 decomposition:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 - M4 + M6 + M7
    # C12 = M3 + M5 + M7
    # C21 = M2 + M4 + M6
    # C22 = M1 - M3 + M2 - M5
    #
    # We need to map these to the U, V, W format.
    # U has shape (2*5, R), V (5*5, R), W (5*2, R).
    # Indices:
    # U[i*m + j], V[j*p + k], W[k*n + i].
    # Here n=2, m=5, p=5.
    #
    # For the 2x2x2 block (i in 0..1, j in 0..1, k in 0..1):
    # The terms are:
    # 1. (A00 + A11) * (B00 + B11) -> U[0, r]=1, U[1, r]=1; V[0, r]=1, V[1, r]=1; W[0, r]=1 (for C00), etc.
    # Wait, the output C is 2x2.
    # C00 gets contributions.
    #
    # Let's construct the 7 terms for the block.
    # Term 1: (A00+A11)(B00+B11)
    #   U: [1,1,0,0,0] (for A00, A11)
    #   V: [1,0,1,0,0] (for B00, B11) -- wait, V index is j*p+k.
    #   j=0,k=0 -> 0. j=1,k=1 -> 6.
    #   So V[0]=1, V[6]=1.
    #   W: This product contributes to all C_ik? No.
    #   In Strassen, M1 is used in C00, C11.
    #   So W[0, r] += 1 (for C00), W[2, r] += 1 (for C11).
    #   Wait, W index is k*n + i.
    #   C00: k=0, i=0 -> 0.
    #   C11: k=1, i=1 -> 1*2+1 = 3.
    #
    # This is getting complicated to code manually for the block.
    #
    # Alternative:
    # We can just use the rank-1 decomposition for ALL 50 terms, but try to merge them?
    # No, the block decomposition is the way.
    #
    # Let's write a helper to add a rank-1 term.
    #
    # We will generate the 42 rank-1 terms for the "rest".
    # The "rest" are all (i,j,k) except (0,0,0)..(1,1,1).
    #
    # And the 7 terms for the block.
    #
    # Total rank = 49.
    #
    # Let's implement this.

    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. Add the 42 rank-1 terms for the non-block entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip the 2x2x2 block
                if i < 2 and j < 2 and k < 2:
                    continue
                
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # 2. Add the 7 Strassen terms for the 2x2x2 block
    # We need to map Strassen's M1..M7 to U, V, W contributions.
    # Strassen terms for 2x2x2 (indices 0,1):
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # M2 = (A10+A11)B00 -> contributes to C10, C11
    # M3 = A00(B01-B11) -> contributes to C01
    # M4 = A11(B10-B00) -> contributes to C10, C11
    # M5 = (A00+A01)B11 -> contributes to C01, C11
    # M6 = (A10-A00)(B00+B01) -> contributes to C00, C10
    # M7 = (A01-A11)(B10+B11) -> contributes to C00, C01
    
    # Note: The indices in Strassen are usually:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Our indices:
    # A[i,j], B[j,k], C[k,i] ?? No, C[i,k] = sum_j A[i,j] B[j,k].
    # So C is 2x2.
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    
    # Let's re-verify Strassen for this convention.
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5 + M7
    # C10 = M2 + M4 + M6
    # C11 = M1 - M3 + M2 - M5
    
    # Let's add these 7 terms.
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, signs):
        nonlocal col
        if col >= rank:
            return
        for idx in u_indices:
            U[idx][col] = 1
        for idx in v_indices:
            V[idx][col] = 1
        for idx, sign in zip(w_indices, signs):
            W[idx][col] = sign
        col += 1
        
    # Term 1: M1 = (A00+A11)(B00+B11)
    # U: A00 (0), A11 (1) -> indices 0, 1
    # V: B00 (0), B11 (1*5+1=6) -> indices 0, 6
    # W: C00 (0), C11 (1*2+1=3) -> signs 1, 1
    add_term([0, 1], [0, 6], [0, 3], [1, 1])
    
    # Term 2: M2 = (A10+A11)B00
    # U: A10 (1*5+0=5), A11 (1) -> indices 5, 1
    # V: B00 (0) -> index 0
    # W: C10 (0*2+1=1), C11 (3) -> signs 1, 1
    add_term([5, 1], [0], [1, 3], [1, 1])
    
    # Term 3: M3 = A00(B01-B11)
    # U: A00 (0)
    # V: B01 (1), B11 (6) -> signs 1, -1
    # W: C01 (1) -> sign 1
    add_term([0], [1, 6], [1], [1])
    # Wait, V signs?
    # The term is A00 * (B01 - B11).
    # So V[1] = 1, V[6] = -1.
    # My helper only sets 1. I need to handle signs in V too.
    
    # Let's rewrite the helper to accept signs for all.
    col = 42 # Reset col to after the 42 terms
    
    def add_term_signed(u_vals, v_vals, w_vals):
        nonlocal col
        if col >= rank:
            return
        for idx, val in u_vals.items():
            U[idx][col] = val
        for idx, val in v_vals.items():
            V[idx][col] = val
        for idx, val in w_vals.items():
            W[idx][col] = val
        col += 1
        
    # Term 1: M1
    add_term_signed({0:1, 1:1}, {0:1, 6:1}, {0:1, 3:1})
    
    # Term 2: M2
    add_term_signed({5:1, 1:1}, {0:1}, {1:1, 3:1})
    
    # Term 3: M3
    add_term_signed({0:1}, {1:1, 6:-1}, {1:1})
    
    # Term 4: M4 = A11(B10-B00)
    add_term_signed({1:1}, {5:1, 0:-1}, {1:1, 3:-1}) # C10, C11
    
    # Term 5: M5 = (A00+A01)B11
    add_term_signed({0:1, 1:1}, {6:1}, {1:1, 3:-1}) # C01, C11
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    add_term_signed({5:1, 0:-1}, {0:1, 1:1}, {0:1, 1:1}) # C00, C10
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    add_term_signed({1:1, 5:-1}, {5:1, 6:1}, {0:1, 1:1}) # C00, C01
    
    # Check col
    # 42 + 7 = 49.
    
    return {"U": U, "V": V, "W": W}

