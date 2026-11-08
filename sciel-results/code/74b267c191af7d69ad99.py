"""Program for matrix multiplication 3x3x4 tensor decomposition."""
DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    # We aim for a rank-minimal exact decomposition.
    # The current seed uses rank 34.
    # A simple block decomposition:
    # A is 3x3, B is 3x4.
    # We can view B as two 3x2 blocks: B1 (cols 0,1) and B2 (cols 2,3).
    # C = A*B = [A*B1, A*B2].
    # A*B1 is 3x2. A*B2 is 3x2.
    
    # Let's try to use a known low-rank algorithm for 3x3x2 or similar?
    # Actually, the seed uses a mix of Strassen (rank 7 for 2x2x2) and direct terms.
    # The seed structure:
    # Left part (cols 0,1 of B):
    #   A11 (2x2) * B1L (2x2) -> uses 7 terms (Strassen)
    #   A12 (2x1) * B2L (1x2) -> 4 terms (direct)
    #   A21 (1x2) * B1L (2x2) -> 4 terms (direct)
    #   A22 (1x1) * B2L (1x2) -> 2 terms (direct)
    #   Total left: 7+4+4+2 = 17 terms.
    # Right part (cols 2,3 of B):
    #   Same structure: 17 terms.
    # Total: 34 terms.
    
    # Can we do better?
    # The tensor rank of 3x3x4 multiplication is at least max(9, 12, 12) = 12.
    # Known bounds: R(3,3,4) <= 34 (this seed).
    # Is there a known better bound?
    # For 3x3x3, rank is 23.
    # 3x3x4 can be seen as 3x3x3 + 3x3x1.
    # R(3,3,3) = 23. R(3,3,1) = 9 (direct). Total 32?
    # But they might share terms if we structure it right?
    # Or use a different block size.
    
    # Let's try a 2x2x2 Strassen for the top-left 2x2 block of C, 
    # and handle the rest efficiently.
    
    # Alternative: Use the known rank-23 algorithm for 3x3x3 on the first 3 columns of B,
    # and then add terms for the 4th column.
    # B = [B_3col, b_4].
    # C = [A*B_3col, A*b_4].
    # A*B_3col is 3x3. Rank 23.
    # A*b_4 is 3x1. This is a vector-matrix product.
    # A*b_4 = sum_j A[:,j] * b_4[j].
    # This requires 3*3=9 terms if done directly?
    # Wait, the tensor for A(3x3) * b(3x1) -> c(3x1) is of size 3x3x1.
    # The rank is 9 (each c_i is a sum of 3 products).
    # So total rank = 23 + 9 = 32.
    
    # Can we do better than 32?
    # Maybe interleave the 4th column into the 3x3x3 algorithm?
    # Or use a different decomposition.
    
    # Let's try to construct the 3x3x3 rank-23 algorithm and add the 4th column.
    # However, implementing the full rank-23 algorithm for 3x3x3 is complex.
    # The seed's approach is simpler and achieves 34.
    # 32 is an improvement.
    
    # Let's see if we can get the 3x3x3 rank 23 terms.
    # This is non-trivial to code from scratch correctly.
    # The seed's 34 is based on 2x2x2 Strassen (7) + direct for the rest.
    
    # Let's try a hybrid:
    # Use Strassen for the 2x2x2 blocks where possible.
    # The seed already does this.
    
    # Is there a rank 31 or 30 algorithm?
    # According to literature, the rank of 3x3x4 matrix multiplication is at most 34.
    # Some sources suggest 32 is achievable.
    
    # Let's attempt the 3x3x3 (rank 23) + 3x3x1 (rank 9) = 32 strategy.
    # I need the rank-23 decomposition for 3x3x3.
    # This is a known result but the specific coefficients are complex.
    
    # Alternative: Stick to the seed's logic but optimize.
    # The seed uses 17 terms for each half (2 columns of B).
    # For 2 columns of B, we compute 3x2 output.
    # Can we compute 3x2 with fewer than 17 terms?
    # 3x2x3 multiplication (A 3x3, B 3x2 -> C 3x2).
    # Rank of 3x2x3?
    # Direct: 3*3*2 = 18.
    # Seed uses 17.
    # Is 16 possible?
    
    # Let's look at the seed's 17 terms for 3x3 * 3x2 -> 3x2.
    # A11 (2x2) * B11 (2x2) -> 7 terms (Strassen).
    # A12 (2x1) * B21 (1x2) -> 4 terms.
    # A21 (1x2) * B11 (2x2) -> 4 terms.
    # A22 (1x1) * B21 (1x2) -> 2 terms.
    # Total 17.
    
    # Can we reduce the "direct" parts?
    # A12 * B21:
    # a02*b20 -> c00
    # a02*b21 -> c01
    # a12*b20 -> c10
    # a12*b21 -> c11
    # These are 4 distinct products. Can't combine with Strassen easily unless we expand A11.
    
    # What if we use a different block partition?
    # 1x3x3? No, Strassen needs 2x2.
    
    # Let's try to implement the 3x3x3 rank-23 algorithm.
    # Reference: "A new algorithm for 3x3 matrix multiplication" or similar.
    # Actually, the rank 23 for 3x3x3 is due to Stothers (1971) or similar improvements.
    # Wait, Strassen is 7 for 2x2. 
    # 3x3x3 rank is 23.
    
    # Since coding the full 23-term algorithm is error-prone without a reference implementation,
    # and the seed is stable at 34, let's see if we can find a simpler improvement.
    
    # What if we use 2x2x2 Strassen for the top-left 2x2x2 block of the 3x3x4 problem?
    # That covers C[0:2, 0:2].
    # Then we need C[0:2, 2:4], C[2:3, 0:4].
    
    # Let's stick to the seed's structure but check for any obvious redundancies.
    # The seed is 34.
    
    # Is there a known rank 31 algorithm for 3x3x4?
    # If I can't find one quickly, I should return the seed's logic to ensure validity,
    # or try the 32-rank approach if I can find the 3x3x3 coefficients.
    
    # Let's try to find a simpler 32-rank construction.
    # Maybe 3x3x3 (rank 23) + 3x3x1 (rank 9) is the standard bound.
    # Without the explicit 23-term coefficients, I cannot implement it.
    
    # However, I can try to optimize the seed's 17-term 3x3x2 block.
    # Is there a 16-term algorithm for 3x3x2?
    # 3x3x2 tensor.
    # Dimensions: A(3x3), B(3x2), C(3x2).
    # Direct: 18.
    # Seed: 17.
    # Lower bound: max(9, 6, 6) = 9.
    # It is known that R(3,3,2) <= 17.
    # Can it be 16?
    
    # Let's assume the seed is near-optimal for this specific simple construction.
    # I will return the seed's logic to ensure correctness, as I cannot verify a new 32-rank
    # implementation without the specific coefficients for the 3x3x3 rank-23 algorithm.
    
    # Wait, I can try to use the fact that B is 3x4.
    # Maybe I can share terms between the left and right halves?
    # The left half uses B columns 0,1. The right half uses B columns 2,3.
    # The A indices are the same.
    # The Strassen terms for A11*B1L and A11*B1R are separate because B indices differ.
    # But the U vectors are the same!
    # U for P1 left: [1,0,0,0,1,0,0,0,0]
    # U for P1 right: [1,0,0,0,1,0,0,0,0]
    # They are identical.
    # But V and W are different.
    # In tensor decomposition, a term is (u, v, w).
    # If u is the same but v,w are different, they are distinct terms.
    # So no sharing there.
    
    # I will output the seed program as it is valid and scores 1.0588.
    # Any change without a verified better algorithm risks breaking it.
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    rank = 34
    U = [[0] * rank for _ in range(n * m)]  # 9 x 34
    V = [[0] * rank for _ in range(m * p)]  # 12 x 34
    W = [[0] * rank for _ in range(p * n)]  # 12 x 34
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Strassen Left (A11 * B1L)
    # A11 indices: 0,1,3,4
    # B1L indices: 0,1,4,5
    # C1L indices: 0,3,1,4
    
    # P1
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # P2
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    # P3
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # P4
    U[4][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # P5
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    # P6
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # P7
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1
    
    # A12 * B2L
    # A12 indices: 2, 5
    # B2L indices: 8, 9
    # C1L contribution:
    # C[0,0]->0, C[0,1]->3, C[1,0]->1, C[1,1]->4
    
    # T1: A[0,2] B[2,0] C[0,0]
    U[2][col] = 1
    V[8][col] = 1
    W[0][col] = 1
    col += 1
    
    # T2: A[0,2] B[2,1] C[0,1]
    U[2][col] = 1
    V[9][col] = 1
    W[3][col] = 1
    col += 1
    
    # T3: A[1,2] B[2,0] C[1,0]
    U[5][col] = 1
    V[8][col] = 1
    W[1][col] = 1
    col += 1
    
    # T4: A[1,2] B[2,1] C[1,1]
    U[5][col] = 1
    V[9][col] = 1
    W[4][col] = 1
    col += 1
    
    # A21 * B1L
    # A21 indices: 6, 7
    # B1L indices: 0,1,4,5
    # C2L contribution:
    # C[2,0]->2, C[2,1]->5
    
    # T1: A[2,0] B[0,0] C[2,0]
    U[6][col] = 1
    V[0][col] = 1
    W[2][col] = 1
    col += 1
    
    # T2: A[2,0] B[0,1] C[2,1]
    U[6][col] = 1
    V[1][col] = 1
    W[5][col] = 1
    col += 1
    
    # T3: A[2,1] B[1,0] C[2,0]
    U[7][col] = 1
    V[4][col] = 1
    W[2][col] = 1
    col += 1
    
    # T4: A[2,1] B[1,1] C[2,1]
    U[7][col] = 1
    V[5][col] = 1
    W[5][col] = 1
    col += 1
    
    # A22 * B2L
    # A22 index: 8
    # B2L indices: 8, 9
    # C2L contribution:
    # C[2,0]->2, C[2,1]->5
    
    # T1: A[2,2] B[2,0] C[2,0]
    U[8][col] = 1
    V[8][col] = 1
    W[2][col] = 1
    col += 1
    
    # T2: A[2,2] B[2,1] C[2,1]
    U[8][col] = 1
    V[9][col] = 1
    W[5][col] = 1
    col += 1
    
    # Strassen Right (A11 * B1R)
    # A11 indices: 0,1,3,4
    # B1R indices: 2,3,6,7
    # C1R indices: 6,9,7,10
    
    # P1
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[6][col] = 1; W[10][col] = 1
    col += 1
    
    # P2
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[7][col] = 1; W[10][col] = -1
    col += 1
    
    # P3
    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[9][col] = 1; W[10][col] = 1
    col += 1
    
    # P4
    U[4][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    # P5
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[6][col] = -1; W[9][col] = 1
    col += 1
    
    # P6
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[10][col] = 1
    col += 1
    
    # P7
    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[6][col] = 1
    col += 1
    
    # A12 * B2R
    # A12 indices: 2, 5
    # B2R indices: 10, 11
    # C1R contribution:
    # C[0,2]->6, C[0,3]->9, C[1,2]->7, C[1,3]->10
    
    # T1: A[0,2] B[2,2] C[0,2]
    U[2][col] = 1
    V[10][col] = 1
    W[6][col] = 1
    col += 1
    
    # T2: A[0,2] B[2,3] C[0,3]
    U[2][col] = 1
    V[11][col] = 1
    W[9][col] = 1
    col += 1
    
    # T3: A[1,2] B[2,2] C[1,2]
    U[5][col] = 1
    V[10][col] = 1
    W[7][col] = 1
    col += 1
    
    # T4: A[1,2] B[2,3] C[1,3]
    U[5][col] = 1
    V[11][col] = 1
    W[10][col] = 1
    col += 1
    
    # A21 * B1R
    # A21 indices: 6, 7
    # B1R indices: 2,3,6,7
    # C2R contribution:
    # C[2,2]->8, C[2,3]->11
    
    # T1: A[2,0] B[0,2] C[2,2]
    U[6][col] = 1
    V[2][col] = 1
    W[8][col] = 1
    col += 1
    
    # T2: A[2,0] B[0,3] C[2,3]
    U[6][col] = 1
    V[3][col] = 1
    W[11][col] = 1
    col += 1
    
    # T3: A[2,1] B[1,2] C[2,2]
    U[7][col] = 1
    V[6][col] = 1
    W[8][col] = 1
    col += 1
    
    # T4: A[2,1] B[1,3] C[2,3]
    U[7][col] = 1
    V[7][col] = 1
    W[11][col] = 1
    col += 1
    
    # A22 * B2R
    # A22 index: 8
    # B2R indices: 10, 11
    # C2R contribution:
    # C[2,2]->8, C[2,3]->11
    
    # T1: A[2,2] B[2,2] C[2,2]
    U[8][col] = 1
    V[10][col] = 1
    W[8][col] = 1
    col += 1
    
    # T2: A[2,2] B[2,3] C[2,3]
    U[8][col] = 1
    V[11][col] = 1
    W[11][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
