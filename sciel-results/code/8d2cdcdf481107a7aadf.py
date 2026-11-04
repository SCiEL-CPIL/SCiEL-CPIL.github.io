"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # We aim for rank 77.
    # Decomposition:
    # 1. C[:, 0:4] = A @ B[:, 0:4] using Strassen 4x4x4 (Rank 49)
    # 2. C[:, 4:6] = A @ B[:, 4:6] using Strassen 4x4x2 (Rank 28)
    #    4x4x2 can be seen as two 2x2x2 Strassen blocks? No.
    #    4x4 * 4x2.
    #    Split A into 2x2 blocks (4 blocks). Split B (4x2) into 2x2 blocks (2 blocks).
    #    Actually, let's split the 4x4 A into four 2x2 blocks: A11, A12, A21, A22.
    #    B is 4x2. Split into B1 (2x2, top 2 rows), B2 (2x2, bottom 2 rows).
    #    C is 4x2. Split into C1 (2x2, top 2 rows), C2 (2x2, bottom 2 rows).
    #    C1 = A11*B1 + A12*B2
    #    C2 = A21*B1 + A22*B2
    #    This is 4 products of 2x2 matrices.
    #    Each 2x2x2 product has rank 7 (Strassen).
    #    Total rank = 4 * 7 = 28.
    #    This is valid because the blocks are independent (no shared intermediate sums across the 4 products 
    #    that would allow further compression in this simple block structure, but 28 is the standard 
    #    "block Strassen" rank for 4x4x2).
    
    total_rank = 77
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Helper to add a 2x2x2 Strassen decomposition
    # A block at rows (ri, ri+1), cols (cj, cj+1)
    # B block at rows (rj, rj+1), cols (ck, ck+1)
    # C block at rows (rk, rk+1), cols (ci, ci+1)
    # Note: W index is k*n + i.
    # W[k*n+i, r] corresponds to C[i, k] in standard matrix notation?
    # Task says: W[k*n+i, r].
    # Standard MM: C[i, k] = sum_j A[i, j] B[j, k].
    # So W index (k, i) holds the coefficient for C[i, k].
    
    def add_strassen_2x2(ri, cj, rj, ck, rk, ci, offset):
        """
        Adds 7 rank-1 terms for C_block = A_block * B_block.
        A_block is 2x2: rows ri, ri+1; cols cj, cj+1.
        B_block is 2x2: rows rj, rj+1; cols ck, ck+1.
        C_block is 2x2: rows rk, rk+1; cols ci, ci+1.
        """
        # Strassen formulas for 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Indices in U (n*m), V (m*p), W (p*n)
        # U[i*m + j]
        # V[j*p + k]
        # W[k*n + i]
        
        # Term 1: M1
        # U: A00 (ri, cj) + A11 (ri+1, cj+1)
        # V: B00 (rj, ck) + B11 (rj+1, ck+1)
        # W: C00 (rk, ci) + C11 (rk+1, ci+1)
        r = offset
        U[ri*m + cj, r] += 1
        U[(ri+1)*m + (cj+1), r] += 1
        V[rj*p + ck, r] += 1
        V[(rj+1)*p + (ck+1), r] += 1
        W[rk*n + ci, r] += 1
        W[(rk+1)*n + (ci+1), r] += 1
        
        # Term 2: M2
        # U: A10 (ri+1, cj) + A11 (ri+1, cj+1)
        # V: B00 (rj, ck)
        # W: C10 (rk+1, ci) + C11 (rk+1, ci+1) -> Wait, C10 is row 1, col 0. C11 is row 1, col 1.
        # Formula: C10 = M2 + M4. C11 = M1 - M2 + M3 + M6.
        # So M2 contributes +1 to C10 and -1 to C11.
        r = offset + 1
        U[(ri+1)*m + cj, r] += 1
        U[(ri+1)*m + (cj+1), r] += 1
        V[rj*p + ck, r] += 1
        W[(rk+1)*n + ci, r] += 1
        W[(rk+1)*n + (ci+1), r] += -1
        
        # Term 3: M3
        # U: A00 (ri, cj)
        # V: B01 (rj, ck+1) - B11 (rj+1, ck+1)
        # W: C01 (rk, ci+1) + C11 (rk+1, ci+1)
        # Formula: C01 = M3 + M5. C11 = ... + M3 ...
        # So M3 contributes +1 to C01 and +1 to C11.
        r = offset + 2
        U[ri*m + cj, r] += 1
        V[rj*p + (ck+1), r] += 1
        V[(rj+1)*p + (ck+1), r] += -1
        W[rk*n + (ci+1), r] += 1
        W[(rk+1)*n + (ci+1), r] += 1
        
        # Term 4: M4
        # U: A11 (ri+1, cj+1)
        # V: B10 (rj+1, ck) - B00 (rj, ck)
        # W: C00 (rk, ci) + C10 (rk+1, ci)
        # Formula: C00 = ... + M4 ... C10 = M2 + M4.
        # So M4 contributes +1 to C00 and +1 to C10.
        r = offset + 3
        U[(ri+1)*m + (cj+1), r] += 1
        V[(rj+1)*p + ck, r] += 1
        V[rj*p + ck, r] += -1
        W[rk*n + ci, r] += 1
        W[(rk+1)*n + ci, r] += 1
        
        # Term 5: M5
        # U: A00 (ri, cj) + A01 (ri, cj+1)
        # V: B11 (rj+1, ck+1)
        # W: C00 (rk, ci) - M5 ... C01 (rk, ci+1) + M5
        # Formula: C00 = ... - M5 ... C01 = M3 + M5.
        # So M5 contributes -1 to C00 and +1 to C01.
        r = offset + 4
        U[ri*m + cj, r] += 1
        U[ri*m + (cj+1), r] += 1
        V[(rj+1)*p + (ck+1), r] += 1
        W[rk*n + ci, r] += -1
        W[rk*n + (ci+1), r] += 1
        
        # Term 6: M6
        # U: A10 (ri+1, cj) - A00 (ri, cj)
        # V: B00 (rj, ck) + B01 (rj, ck+1)
        # W: C11 (rk+1, ci+1)
        # Formula: C11 = ... + M6.
        # So M6 contributes +1 to C11.
        r = offset + 5
        U[(ri+1)*m + cj, r] += 1
        U[ri*m + cj, r] += -1
        V[rj*p + ck, r] += 1
        V[rj*p + (ck+1), r] += 1
        W[(rk+1)*n + (ci+1), r] += 1
        
        # Term 7: M7
        # U: A01 (ri, cj+1) - A11 (ri+1, cj+1)
        # V: B10 (rj+1, ck) + B11 (rj+1, ck+1)
        # W: C00 (rk, ci)
        # Formula: C00 = ... + M7.
        # So M7 contributes +1 to C00.
        r = offset + 6
        U[ri*m + (cj+1), r] += 1
        U[(ri+1)*m + (cj+1), r] += -1
        V[(rj+1)*p + ck, r] += 1
        V[(rj+1)*p + (ck+1), r] += 1
        W[rk*n + ci, r] += 1

    # Part 1: 4x4 * 4x4 (cols 0-3)
    # We can decompose this into 4x4x4 Strassen.
    # However, implementing full 4x4x4 Strassen recursively is complex.
    # Alternative: Use the fact that 4x4x4 rank is 49.
    # We can construct it by splitting 4x4x4 into blocks?
    # 4x4x4 = 4 * (2x2x2) + 4 * (2x2x2) ... no.
    # Standard 4x4x4 Strassen uses 7 products of 2x2x2 matrices? No, that's rank 7*7=49.
    # Yes! 4x4x4 Strassen is based on 7 multiplications of 2x2 matrices.
    # So we can implement 4x4x4 Strassen by calling add_strassen_2x2 7 times?
    # No, the inputs to the 2x2x2 Strassen are linear combinations of the 4x4 blocks.
    # This requires pre-processing U and V to form the sums/differences.
    # This is what the "recursive" approach does.
    # Let's try to implement the 4x4x4 Strassen explicitly.
    
    # 4x4 A:
    # A11 (0,0) A12 (0,2)
    # A21 (2,0) A22 (2,2)
    # 4x4 B:
    # B11 (0,0) B12 (0,2)
    # B21 (2,0) B22 (2,2)
    # 4x4 C:
    # C11 (0,0) C12 (0,2)
    # C21 (2,0) C22 (2,2)
    
    # Strassen 4x4:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2 * 2x2 product.
    # We need to express Mi as a sum of rank-1 tensors.
    # But wait, Mi is the product of two 2x2 matrices.
    # So each Mi contributes 7 rank-1 terms (using Strassen for 2x2x2).
    # Total 7 * 7 = 49 terms.
    
    # We need to map the "virtual" 2x2 matrices in M1...M7 to the actual U, V, W indices.
    # For M1:
    # A_part = A11 + A22.
    # B_part = B11 + B22.
    # C_part = C11, C22 (with coefficients).
    
    # This is tricky because the "A_part" is a linear combination.
    # In the U matrix, we can simply add the coefficients to the appropriate rows.
    # For M1, the U vector for the "A" part of the 2x2x2 Strassen will have non-zeros at A11 and A22 positions.
    # Specifically, if we call add_strassen_2x2 for M1, we need to pass the indices of the 2x2 blocks.
    # But add_strassen_2x2 assumes the inputs are single 2x2 blocks.
    # Here the input is a SUM of two blocks.
    # So we cannot just call add_strassen_2x2 once.
    # We must expand the Strassen formula for M1, which itself is a product of sums.
    
    # Let's do it manually for M1 = (A11+A22)(B11+B22).
    # Let X = A11+A22, Y = B11+B22.
    # Z = X Y.
    # Z is computed by Strassen 2x2x2 on X and Y.
    # The U coefficients for X are:
    #   X00 = A11_00 + A22_00
    #   X01 = A11_01 + A22_01
    #   X10 = A11_10 + A22_10
    #   X11 = A11_11 + A22_11
    # Similarly for Y.
    # So when we add the rank-1 terms for Z, the U vector will have entries corresponding to A11 and A22.
    
    # This suggests we can write a helper `add_strassen_2x2_combined` that takes lists of (block_idx, coeff) for A, B, and C.
    # But that's complex.
    
    # Alternative: Just use Schoolbook for the 4x4x4 part?
    # Schoolbook 4x4x4 is rank 64.
    # Total rank 64 + 28 = 92.
    # 92 is worse than 77.
    
    # I will implement the 4x4x4 Strassen by expanding the 7 terms.
    # Each term is a 2x2x2 Strassen.
    # I will create a function `add_strassen_2x2_with_coeffs` that allows scaling the input blocks.
    
    def add_strassen_2x2_scaled(A_blocks, B_blocks, C_blocks, offset):
        """
        A_blocks: list of (row_idx, col_idx, coeff)
        B_blocks: list of (row_idx, col_idx, coeff)
        C_blocks: list of (row_idx, col_idx, coeff)
        """
        # This is getting too complicated to generate dynamically in 150s.
        # Let's fall back to a safer, known-good rank 77 construction:
        # Use Strassen for 4x4x4? No, let's use the "3-way" split?
        
        # Actually, there is a simpler rank 77 construction for 4x4x6:
        # Split B into B1 (4x4) and B2 (4x2).
        # C1 = A B1 (4x4x4). Rank 49 (Strassen).
        # C2 = A B2 (4x4x2). Rank 28 (4x Strassen 2x2x2).
        # Total 77.
        
        # I will implement the 4x4x4 Strassen by explicitly coding the 49 terms.
        # This is tedious but safe if I copy the structure from a known source or derive it carefully.
        
        # Let's try to derive M1 = (A11+A22)(B11+B22) explicitly.
        # A11 is at (0,0), A22 is at (2,2).
        # B11 is at (0,0), B22 is at (2,2).
        # C11 is at (0,0), C22 is at (2,2).
        
        # M1 contributes to C11 (coeff 1) and C22 (coeff 1).
        
        # Strassen for X=A11+A22, Y=B11+B22, Z=M1.
        # Z00 = M1_00 + M1_04 - M1_05 + M1_07 (using local indices 00, 04, 05, 07? No, standard 2x2 indices)
        # Let's stick to the standard 2x2 Strassen terms M1..M7 for the 2x2 product.
        
        # I will implement a generic `add_2x2_strassen` that takes explicit U, V, W coefficient dictionaries.
        
        pass

    # Given the complexity and risk, I will use the Schoolbook for the 4x4x4 part (Rank 64) 
    # and Strassen for the 4x4x2 part (Rank 28). Total 92.
    # Wait, 92 is worse than 96 (Schoolbook total).
    # But 77 is better.
    
    # Is there a rank 76 or lower?
    # Known lower bound for 4x4x4 is 49.
    # Known lower bound for 4x4x2 is 28? 
    # 4x4x2 tensor rank is at least 4*4*2 / max(4,4,2) = 8? No.
    # Submultiplicativity: r(4,4,2) <= r(4,4,4) = 49? No.
    # r(4,4,2) <= 4 * r(1,1,2) = 4? No.
    # r(4,4,2) is known to be 28?
    # Actually, 4x4x2 can be done with 4 2x2x2 Strassen blocks = 28.
    # Can it be done with less?
    # The tensor rank of 4x4x2 is at least 16 (schoolbook).
    # It is known that r(4,4,2) = 28? Or lower?
    # If r(4,4,2) < 28, we could get below 77.
    # But 28 is a very standard block decomposition.
    
    # Let's try to implement the 4x4x4 Strassen correctly.
    # I will hardcode the 49 terms for 4x4x4.
    # This is a known algorithm.
    
    # 4x4x4 Strassen Decomposition
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2 matrix product.
    # We will generate the 7 rank-1 terms for each Mi.
    # For M1:
    #   A_part = A11 + A22
    #   B_part = B11 + B22
    #   C_part = C11 (coeff 1), C22 (coeff 1)
    
    # I will write a helper that adds a 2x2x2 Strassen decomposition given the "virtual" blocks.
    
    def add_strassen_2x2_virtual(A_blocks, B_blocks, C_blocks, offset):
        """
        A_blocks: list of (i, j, coeff) for A
        B_blocks: list of (j, k, coeff) for B
        C_blocks: list of (k, i, coeff) for C
        """
        # Strassen 2x2 terms:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We need to sum the contributions from each A_block, B_block to the Strassen terms.
        # This is equivalent to:
        # U_vec = sum_{(i,j,c) in A_blocks} c * e_{i,j}
        # V_vec = sum_{(j,k,c) in B_blocks} c * e_{j,k}
        # W_vec = sum_{(k,i,c) in C_blocks} c * e_{k,i}
        
        # But Strassen uses linear combinations of A and B elements.
        # E.g., M1 uses A00, A11, B00, B11.
        # So we need to extract the coefficients for A00, A11, etc. from A_blocks.
        
        # Let's build a map for A: (i, j) -> coeff
        # A is 2x2, so indices are (0,0), (0,1), (1,0), (1,1).
        # But our A_blocks are given in global indices.
        # We need to map global (i, j) to local (0,0)..(1,1).
        # This requires knowing the base indices.
        
        # This function is too generic. Let's hardcode the 49 terms.
        pass

    # Okay, I will use the Schoolbook for 4x4x4 (Rank 64) and Strassen for 4x4x2 (Rank 28).
    # Total Rank 92.
    # This is better than 96.
    
    rank_total = 92
    U = np.zeros((n * m, rank_total), dtype=float)
    V = np.zeros((m * p, rank_total), dtype=float)
    W = np.zeros((p * n, rank_total), dtype=float)
    
    # Part 1: 4x4 * 4x4 Schoolbook (Rank 64)
    col_offset = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*m + j, col_offset] = 1
                V[j*p + k, col_offset] = 1
                W[k*n + i, col_offset] = 1
                col_offset += 1
                
    # Part 2: 4x4 * 4x2 Strassen (Rank 28)
    # B is 4x2 (cols 4, 5).
    # A is 4x4.
    # C is 4x2 (cols 4, 5).
    # Split A into 4 2x2 blocks: A11(0,0), A12(0,2), A21(2,0), A22(2,2).
    # Split B into 2 2x2 blocks: B1(0,4), B2(2,4). (Rows 0-1, 2-3; Cols 4-5)
    # C1 (rows 0-1, cols 4-5) = A11*B1 + A12*B2
    # C2 (rows 2-3, cols 4-5) = A21*B1 + A22*B2
    
    # We need 4 Strassen 2x2x2 decompositions.
    # 1. A11 * B1 -> C1
    # 2. A12 * B2 -> C1
    # 3. A21 * B1 -> C2
    # 4. A22 * B2 -> C2
    
    # Helper to add Strassen 2x2x2 for specific blocks
    def add_strassen_2x2_specific(A_ri, A_cj, B_rj, B_ck, C_rk, C_ci, offset):
        # A_ri, A_cj: top-left of A block
        # B_rj, B_ck: top-left of B block
        # C_rk, C_ci: top-left of C block
        # Note: C is indexed by (k, i) in W.
        # C_rk is the row index in C (which is i in MM).
        # C_ci is the col index in C (which is k in MM).
        # So W index is C_ci * n + C_rk.
        
        # Term 1: M1
        r = offset
        U[A_ri*m + A_cj, r] += 1
        U[(A_ri+1)*m + (A_cj+1), r] += 1
        V[B_rj*p + B_ck, r] += 1
        V[(B_rj+1)*p + (B_ck+1), r] += 1
        W[C_ci*n + C_rk, r] += 1
        W[(C_ci+1)*n + (C_rk+1), r] += 1
        
        # Term 2: M2
        r = offset + 1
        U[(A_ri+1)*m + A_cj, r] += 1
        U[(A_ri+1)*m + (A_cj+1), r] += 1
        V[B_rj*p + B_ck, r] += 1
        W[C_ci*n + (C_rk+1), r] += 1
        W[(C_ci+1)*n + (C_rk+1), r] += -1
        
        # Term 3: M3
        r = offset + 2
        U[A_ri*m + A_cj, r] += 1
        V[B_rj*p + (B_ck+1), r] += 1
        V[(B_rj+1)*p + (B_ck+1), r] += -1
        W[C_ci*n + (C_rk+1), r] += 1
        W[(C_ci+1)*n + (C_rk+1), r] += 1
        
        # Term 4: M4
        r = offset + 3
        U[(A_ri+1)*m + (A_cj+1), r] += 1
        V[(B_rj+1)*p + B_ck, r] += 1
        V[B_rj*p + B_ck, r] += -1
        W[C_ci*n + C_rk, r] += 1
        W[(C_ci+1)*n + C_rk, r] += 1
        
        # Term 5: M5
        r = offset + 4
        U[A_ri*m + A_cj, r] += 1
        U[A_ri*m + (A_cj+1), r] += 1
        V[(B_rj+1)*p + (B_ck+1), r] += 1
        W[C_ci*n + C_rk, r] += -1
        W[C_ci*n + (C_rk+1), r] += 1
        
        # Term 6: M6
        r = offset + 5
        U[(A_ri+1)*m + A_cj, r] += 1
        U[A_ri*m + A_cj, r] += -1
        V[B_rj*p + B_ck, r] += 1
        V[B_rj*p + (B_ck+1), r] += 1
        W[(C_ci+1)*n + (C_rk+1), r] += 1
        
        # Term 7: M7
        r = offset + 6
        U[A_ri*m + (A_cj+1), r] += 1
        U[(A_ri+1)*m + (A_cj+1), r] += -1
        V[(B_rj+1)*p + B_ck, r] += 1
        V[(B_rj+1)*p + (B_ck+1), r] += 1
        W[C_ci*n + C_rk, r] += 1

    # 1. A11 (0,0) * B1 (0,4) -> C1 (0,4)
    add_strassen_2x2_specific(0, 0, 0, 4, 0, 4, col_offset)
    col_offset += 7
    
    # 2. A12 (0,2) * B2 (2,4) -> C1 (0,4)
    add_strassen_2x2_specific(0, 2, 2, 4, 0, 4, col_offset)
    col_offset += 7
    
    # 3. A21 (2,0) * B1 (0,4) -> C2 (2,4)
    add_strassen_2x2_specific(2, 0, 0, 4, 2, 4, col_offset)
    col_offset += 7
    
    # 4. A22 (2,2) * B2 (2,4) -> C2 (2,4)
    add_strassen_2x2_specific(2, 2, 2, 4, 2, 4, col_offset)
    col_offset += 7
    
    # Verify
    A = np.random.rand(4, 4)
    B = np.random.rand(4, 6)
    C_expected = A @ B
    
    # Reconstruct C
    # C[i, k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_k V[j*p+k, r] B[j,k])?
    # No, the decomposition is:
    # T_{u,v,w} = sum_r U[u,r] V[v,r] W[w,r]
    # This means:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_{u,v,w} T_{u,v,w} A[u] B[v] delta(w, i,k)?
    # Actually, the standard tensor decomposition for MM is:
    # C = sum_r (A U_r) (V_r B) W_r? No.
    # The formula is:
    # C_{i,k} = sum_{r=1}^R W_{k*n+i, r} * (sum_{j} U_{i*m+j, r} A_{i,j}) * (sum_{j} V_{j*p+k, r} B_{j,k})?
    # No, that's not right.
    
    # The correct interpretation of U, V, W for MM:
    # We want C = A B.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor T_{i,j,k} = 1.
    # Decomposition: T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r}?
    # No, the factors are vectors.
    # U is (n*m, R). V is (m*p, R). W is (p*n, R).
    # The reconstruction is:
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    # = sum_{j} (sum_r U_{i*m+j, r} x_r) (sum_s V_{j*p+k, s} y_s) ... No.
    
    # The standard way to verify U, V, W:
    # Let A_mat be (n, m). B_mat be (m, p).
    # We form the "linear forms" for each r:
    # L_r(A) = sum_{i,j} U[i*m+j, r] A[i,j]
    # M_r(B) = sum_{j,k} V[j*p+k, r] B[j,k]
    # Then C[i,k] = sum_r W[k*n+i, r] L_r(A) M_r(B)?
    # No, that would be rank 1 in A and B, but C is a matrix.
    
    # Actually, the decomposition is:
    # C = sum_r (A U_r_col) (V_r_row B) W_r?
    
    # Let's check the indices.
    # U[i*m+j, r] is the coefficient of A[i,j] in the r-th product.
    # V[j*p+k, r] is the coefficient of B[j,k] in the r-th product.
    # W[k*n+i, r] is the coefficient of C[i,k] in the r-th product.
    
    # The product term is:
    # P_r = (sum_{i,j} U[i*m+j, r] A[i,j]) * (sum_{j,k} V[j*p+k, r] B[j,k])
    # This P_r is a scalar.
    # But C[i,k] is a matrix entry.
    # So C[i,k] = sum_r W[k*n+i, r] P_r?
    # This implies that P_r is the same for all i,k, which is wrong.
    
    # The correct formula is:
    # C_{i,k} = sum_{r} W_{k*n+i, r} * (sum_{j} U_{i*m+j, r} A_{i,j}) * (sum_{j} V_{j*p+k, r} B_{j,k})?
    # No, the sums over j are coupled.
    
    # The tensor T_{u,v,w} = sum_r U[u,r] V[v,r] W[w,r].
    # u = i*m+j, v = j*p+k, w = k*n+i.
    # So for a fixed i, k:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # = sum_{j} A_{i,j} B_{j,k} * 1
    # = sum_{j} A_{i,j} B_{j,k} * (sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r])
    # = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A_{i,j}) * (sum_j V[j*p+k, r] B_{j,k})?
    # No, you can't pull the sum over r out like that if U and V depend on j.
    # But U[i*m+j, r] and V[j*p+k, r] DO depend on j.
    # So:
    # C_{i,k} = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] A_{i,j} B_{j,k}
    # = sum_r W[k*n+i, r] (sum_j U[i*m+j, r] A_{i,j} V[j*p+k, r] B_{j,k})
    
    # This is not a simple product of two scalars.
    # It is a bilinear form in A and B for each r.
    
    # To verify, we can just compute the full tensor contraction.
    # T_recon[u, v, w] = sum_r U[u,r] V[v,r] W[w,r]
    # Check if T_recon matches the MM tensor.
    
    T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
    
    # Build expected T
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T_expected[u, v, w] = 1
                
    if not np.allclose(T_recon, T_expected):
        # Fallback to Schoolbook
        rank_schoolbook = n * m * p
        U = np.zeros((n * m, rank_schoolbook), dtype=float)
        V = np.zeros((m * p, rank_schoolbook), dtype=float)
        W = np.zeros((p * n, rank_schoolbook), dtype=float)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
