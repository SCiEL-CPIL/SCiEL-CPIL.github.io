import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook decomposition (Rank 48)
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Strategy: Hybrid approach using Strassen for sub-blocks where possible.
    # Specifically, we can treat the 3x4x4 multiplication as:
    # C (3x4) = A (3x4) * B (4x4)
    #
    # We can split A into rows or B into columns/blocks.
    # Let's try splitting the output C into two parts:
    # C_top (2x4) = A_top (2x4) * B (4x4)
    # C_bottom (1x4) = A_bottom (1x4) * B (4x4)
    #
    # For C_top, we have a 2x4x4 problem.
    # For C_bottom, we have a 1x4x4 problem.
    #
    # Alternatively, split B into two 4x2 blocks:
    # B = [B1, B2] where B1, B2 are 4x2.
    # C = [A*B1, A*B2]
    # Each A*Bi is 3x2.
    # So we have two 3x4x2 problems.
    #
    # Let's try the 3x4x2 problem.
    # Rank of 3x4x2?
    # Schoolbook: 3*4*2 = 24.
    # Can we do better?
    # If we use Strassen on a 2x2x2 subpart...
    #
    # Actually, the knowledge says: "Replaces the trivial rank-48 Kronecker decomposition with a hybrid approach using Strassen's algorithm (rank 7) for the 2x2x2 sub-blocks of the top two output rows and trivial multiplication (rank 4) for the bottom row, reducing the total rank to 44."
    #
    # Let's construct this hybrid explicitly.
    #
    # We want to compute C = A * B, where A is 3x4, B is 4x4.
    # Let A = [A0; A1; A2] where A0, A1, A2 are 1x4 vectors.
    # Let B be 4x4.
    #
    # The knowledge suggests:
    # "Strassen's algorithm (rank 7) for the 2x2x2 sub-blocks of the top two output rows"
    # This implies we are grouping the top two rows of A (A0, A1) into a 2x4 matrix A_top.
    # And we are multiplying A_top (2x4) by B (4x4).
    # This is a 2x4x4 problem.
    #
    # How to get rank 7 for a 2x2x2 sub-block?
    # Maybe they split B into 2x2 blocks?
    # B = [[B00, B01], [B10, B11]] where each Bi is 2x2.
    # A_top = [A0_top, A1_top] where A0_top, A1_top are 2x2? No, A_top is 2x4.
    # Let A_top = [A0, A1] where A0, A1 are 1x4.
    # Wait, if we split B into 2x2 blocks, and A_top into 2x2 blocks?
    # Let A_top = [A00, A01] where A00, A01 are 2x2? No, A_top is 2x4, so it can be split into two 2x2 blocks horizontally: A_top = [A_top_left, A_top_right].
    #
    # C_top = A_top * B = [A_top_left * B00 + A_top_right * B10, A_top_left * B01 + A_top_right * B11]
    # This seems complicated.
    #
    # Let's look at the specific rank 44 claim.
    # 44 = 40 + 4? Or 37 + 7?
    # If we use Strassen (rank 7) for 2x2x2, and we have multiple such blocks...
    #
    # Let's try a different decomposition:
    # Split the 4x4 matrix B into four 2x2 blocks: B00, B01, B10, B11.
    # Split the 3x4 matrix A into two parts: A_top (2x4) and A_bottom (1x4).
    #
    # C_top = A_top * B
    # C_bottom = A_bottom * B
    #
    # For C_bottom (1x4 * 4x4 -> 1x4):
    # A_bottom is 1x4. Let A_bottom = [a0, a1, a2, a3].
    # B is 4x4.
    # C_bottom = a0*B_row0 + a1*B_row1 + a2*B_row2 + a3*B_row3.
    # This is a linear combination of the rows of B.
    # Rank 4 is sufficient (schoolbook for 1x4x4 is 4? No, 1*4*4=16? No, the tensor is (1*4, 4*4, 4*1) = (4, 16, 4).
    # Wait, the rank of the multiplication tensor M_{1,4,4} is 4.
    # Proof: C_k = sum_j A_j B_jk.
    # A is 1x4, B is 4x4.
    # C is 1x4.
    # C_k = sum_{j=0}^3 A_j B_jk.
    # We can define 4 terms: for each j, term_j = A_j * B_jk.
    # U_j = A_j (scalar), V_j = B_j (row vector), W_j = e_k (basis vector for output k)?
    # No, the decomposition is sum_r U_r outer V_r outer W_r.
    # For M_{1,4,4}:
    # T[a, b, c] = delta_{a,b} * delta_{b,c}? No.
    # Indices: i (1), j (4), k (4).
    # T[i, j, k] = 1 if i=j=k? No.
    # T[i, j, k] = 1 if the index in A is i, index in B is j, index in C is k.
    # C_k = sum_j A_i B_jk. Here i is fixed (0).
    # So T[0, j, k] = 1.
    # This is a rank 4 tensor?
    # T = sum_{j=0}^3 e_0 outer e_j outer e_j?
    # (e_0 outer e_j outer e_j)[i, j', k'] = delta_{i,0} delta_{j,j'} delta_{k,j'}.
    # Sum over j: delta_{i,0} sum_j delta_{j,j'} delta_{k,j'} = delta_{i,0} delta_{j',k'}.
    # But T[0, j, k] = 1 for all j,k? No.
    # C_k = sum_j A_j B_jk.
    # The tensor T[i, j, k] is 1 if the input indices i,j and output index k are linked.
    # i is the row of A (only 0). j is the column of A / row of B. k is the column of B / column of C.
    # So T[0, j, k] = 1 for all j,k?
    # No, the tensor is T_{ijk} = 1 if i=j=k? No.
    # T_{ijk} = 1 if the product A_i B_j contributes to C_k.
    # A is 1x4, B is 4x4.
    # A has indices (i=0, j). B has indices (j, k).
    # So T[0, j, k] = 1.
    # This tensor is rank 4?
    # T = sum_{j=0}^3 e_0 outer e_j outer (sum_{k} e_k)? No.
    # T[0, j, k] = 1.
    # Let u_0 = e_0.
    # Let v_j = e_j.
    # Let w_j = sum_{k=0}^3 e_k = [1,1,1,1]^T.
    # sum_j u_0 outer v_j outer w_j = sum_j e_0 outer e_j outer [1,1,1,1]^T.
    # Element (0, j', k') = sum_j delta_{0,0} delta_{j,j'} 1 = delta_{j,j'} * 1 = 1 if j'=j.
    # This gives T[0, j', k'] = 1 for all k'.
    # Yes, this works. Rank 4.
    #
    # So C_bottom can be done with rank 4.
    #
    # Now for C_top (2x4 * 4x4 -> 2x4).
    # A_top is 2x4. B is 4x4.
    # T_top[i, j, k] = 1 for i in {0,1}, j in {0..3}, k in {0..3}.
    # Can we do this with rank < 16?
    # The knowledge says "Strassen's algorithm (rank 7) for the 2x2x2 sub-blocks".
    # Maybe they split A_top into two 2x2 blocks?
    # A_top = [A00, A01] where A00, A01 are 2x2.
    # B = [[B00, B01], [B10, B11]] where Bi are 2x2.
    # C_top = [A00*B00 + A01*B10, A00*B01 + A01*B11].
    #
    # This is 4 matrix multiplications of 2x2 matrices.
    # Each 2x2x2 multiplication has rank 7 (Strassen).
    # So total rank 4 * 7 = 28.
    # Plus C_bottom rank 4.
    # Total rank 32.
    #
    # But the knowledge says rank 44. Why?
    # Maybe the blocks are not independent? Or maybe they used a different split?
    # Or maybe the "2x2x2 sub-blocks of the top two output rows" implies something else.
    #
    # Let's check the rank of M_{2,4,4}.
    # It is known that M_{2,4,4} has rank at most 24?
    # Actually, M_{2,2,2} is 7.
    # M_{2,4,4} can be viewed as 4 multiplications of M_{2,2,2}?
    # A (2x4) = [A0, A1] (2x2 blocks).
    # B (4x4) = [[B00, B01], [B10, B11]].
    # C0 = A0 B00 + A1 B10.
    # C1 = A0 B01 + A1 B11.
    #
    # If we use Strassen for each 2x2 multiplication, we need 7 terms per multiplication.
    # Total 28 terms.
    #
    # Can we share terms between C0 and C1?
    # The terms in Strassen for A0 B00 and A0 B01 share the A0 part.
    # But the V and W parts are different.
    #
    # Let's assume the rank 32 construction is valid.
    # 32 < 44.
    #
    # Let's try to construct the rank 32 decomposition.
    #
    # We need to map the Strassen terms for each 2x2 block multiplication to the global indices.
    #
    # Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M4
    # C21 = M2 + M5
    # C22 = M1 - M3 + M6 + M7
    #
    # We have 4 such multiplications:
    # 1. A00 * B00 -> C00
    # 2. A01 * B10 -> C01_part1
    # 3. A00 * B01 -> C02_part1
    # 4. A01 * B11 -> C02_part2
    #
    # Wait, C_top is 2x4.
    # C_top = [C00, C01, C02, C03]
    # C00 = A00 B00
    # C01 = A01 B10
    # C02 = A00 B01
    # C03 = A01 B11
    #
    # So we need 4 separate 2x2x2 multiplications.
    # Each has rank 7.
    # Total rank 28.
    #
    # Plus C_bottom rank 4.
    # Total rank 32.
    #
    # Let's implement this.
    
    R_strassen = 7
    R_bottom = 4
    R_total = 4 * R_strassen + R_bottom
    
    if R_total < best_rank:
        best_rank = R_total
        U_new = np.zeros((n*m, R_total))
        V_new = np.zeros((m*p, R_total))
        W_new = np.zeros((p*n, R_total))
        
        col = 0
        
        # Helper to add Strassen terms for a 2x2 multiplication
        # a, b, c, d are the indices for A, B, C in the global 2x2 block
        # A_block is at rows (r1, r2) and cols (c1, c2) of A_top
        # B_block is at rows (r1, r2) and cols (c1, c2) of B
        # C_block is at rows (r1, r2) and cols (c1, c2) of C_top
        
        def add_strassen(r1, r2, c1, c2):
            nonlocal col
            # Strassen terms
            # M1 = (A11+A22)(B11+B22)
            # Indices in local 2x2:
            # A11 is (r1, c1), A22 is (r2, c2)
            # B11 is (r1, c1), B22 is (r2, c2)
            # C11 is (r1, c1)
            
            # M1
            U_new[r1*m + c1, col] = 1
            U_new[r2*m + c2, col] = 1
            V_new[r1*p + c1, col] = 1
            V_new[r2*p + c2, col] = 1
            W_new[c1*n + r1, col] = 1
            W_new[c1*n + r2, col] = 1
            W_new[c2*n + r1, col] = 1
            W_new[c2*n + r2, col] = 1
            # C11 += M1, C22 += M1
            # Wait, Strassen output:
            # C11 = M1 - M4 + M5 + M7
            # C12 = M3 + M4
            # C21 = M2 + M5
            # C22 = M1 - M3 + M6 + M7
            
            # Let's define the contributions to C for each M.
            # M1 contributes to C11 and C22.
            # W index is k*n + i.
            # C11 is at (r1, c1) -> W[c1*n + r1]
            # C22 is at (r2, c2) -> W[c2*n + r2]
            # So W[c1*n + r1, col] += 1, W[c2*n + r2, col] += 1.
            # But we also need to handle the other terms.
            
            # This manual construction is error-prone.
            # Let's use the known rank 44 construction from the knowledge if I can't verify rank 32.
            # Or just use rank 32 and verify it with numpy.
            
            col += 1
            
            # M2 = (A21+A22)B11
            U_new[r2*m + c1, col] = 1
            U_new[r2*m + c2, col] = 1
            V_new[r1*p + c1, col] = 1
            W_new[c1*n + r2, col] = 1 # C21
            col += 1
            
            # M3 = A11(B21-B22)
            U_new[r1*m + c1, col] = 1
            V_new[r2*p + c1, col] = 1
            V_new[r2*p + c2, col] = -1
            W_new[c1*n + r1, col] = 1 # C12
            col += 1
            
            # M4 = A22(B11-B21)
            U_new[r2*m + c2, col] = 1
            V_new[r1*p + c1, col] = 1
            V_new[r2*p + c1, col] = -1
            W_new[c1*n + r1, col] = 1 # C12
            W_new[c2*n + r2, col] = 1 # C22
            col += 1
            
            # M5 = (A11+A12)B22
            U_new[r1*m + c1, col] = 1
            U_new[r1*m + c2, col] = 1
            V_new[r2*p + c2, col] = 1
            W_new[c1*n + r1, col] = 1 # C11
            W_new[c2*n + r2, col] = 1 # C21
            col += 1
            
            # M6 = (A21-A11)(B11+B12)
            U_new[r2*m + c1, col] = 1
            U_new[r1*m + c1, col] = -1
            V_new[r1*p + c1, col] = 1
            V_new[r1*p + c2, col] = 1
            W_new[c2*n + r2, col] = 1 # C22
            col += 1
            
            # M7 = (A12-A22)(B21+B22)
            U_new[r1*m + c2, col] = 1
            U_new[r2*m + c2, col] = -1
            V_new[r2*p + c1, col] = 1
            V_new[r2*p + c2, col] = 1
            W_new[c1*n + r1, col] = 1 # C11
            W_new[c2*n + r2, col] = 1 # C22
            col += 1
            
        # We need to be careful with the W indices.
        # The W matrix accumulates contributions.
        # My add_strassen function above is incorrect because it assumes W is zero-initialized and just sets values, but multiple M's contribute to the same C element.
        # I should accumulate.
        
        # Let's restart the construction with accumulation.
        U_new = np.zeros((n*m, R_total))
        V_new = np.zeros((m*p, R_total))
        W_new = np.zeros((p*n, R_total))
        col = 0
        
        # Strassen for 2x2 block at A(r1,r2; c1,c2), B(r1,r2; c1,c2), C(r1,r2; c1,c2)
        # Note: A is 3x4, so r1,r2 are 0,1 for top rows. c1,c2 are 0,1 for left cols, 2,3 for right cols.
        # B is 4x4.
        # C is 3x4.
        
        # Block 1: A(0,1; 0,1) * B(0,1; 0,1) -> C(0,1; 0,1)
        # A indices: (0,0), (0,1), (1,0), (1,1)
        # B indices: (0,0), (0,1), (1,0), (1,1)
        # C indices: (0,0), (0,1), (1,0), (1,1)
        
        def add_strassen_block(ar1, ar2, ac1, ac2, br1, br2, bc1, bc2, cr1, cr2, cc1, cc2):
            nonlocal col
            # M1 = (A11+A22)(B11+B22)
            U_new[ar1*m + ac1, col] += 1
            U_new[ar2*m + ac2, col] += 1
            V_new[br1*p + bc1, col] += 1
            V_new[br2*p + bc2, col] += 1
            W_new[cc1*n + cr1, col] += 1
            W_new[cc2*n + cr2, col] += 1
            col += 1
            
            # M2 = (A21+A22)B11
            U_new[ar2*m + ac1, col] += 1
            U_new[ar2*m + ac2, col] += 1
            V_new[br1*p + bc1, col] += 1
            W_new[cc1*n + cr2, col] += 1
            col += 1
            
            # M3 = A11(B21-B22)
            U_new[ar1*m + ac1, col] += 1
            V_new[br2*p + bc1, col] += 1
            V_new[br2*p + bc2, col] -= 1
            W_new[cc2*n + cr1, col] += 1
            col += 1
            
            # M4 = A22(B11-B21)
            U_new[ar2*m + ac2, col] += 1
            V_new[br1*p + bc1, col] += 1
            V_new[br2*p + bc1, col] -= 1
            W_new[cc2*n + cr1, col] += 1
            W_new[cc2*n + cr2, col] += 1
            col += 1
            
            # M5 = (A11+A12)B22
            U_new[ar1*m + ac1, col] += 1
            U_new[ar1*m + ac2, col] += 1
            V_new[br2*p + bc2, col] += 1
            W_new[cc1*n + cr1, col] += 1
            W_new[cc2*n + cr2, col] += 1
            col += 1
            
            # M6 = (A21-A11)(B11+B12)
            U_new[ar2*m + ac1, col] += 1
            U_new[ar1*m + ac1, col] -= 1
            V_new[br1*p + bc1, col] += 1
            V_new[br1*p + bc2, col] += 1
            W_new[cc2*n + cr2, col] += 1
            col += 1
            
            # M7 = (A12-A22)(B21+B22)
            U_new[ar1*m + ac2, col] += 1
            U_new[ar2*m + ac2, col] -= 1
            V_new[br2*p + bc1, col] += 1
            V_new[br2*p + bc2, col] += 1
            W_new[cc1*n + cr1, col] += 1
            W_new[cc2*n + cr2, col] += 1
            col += 1

        # Block 1: C00
        add_strassen_block(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
        # Block 2: C01 (A01 * B10)
        # A01 is A(0,1; 2,3)
        # B10 is B(2,3; 0,1)
        # C01 is C(0,1; 2,3) ? No.
        # C_top = [C00, C01, C02, C03]
        # C01 corresponds to A01 * B10.
        # A01 is columns 2,3 of A_top.
        # B10 is rows 2,3 of B.
        # So A indices: (0,2), (0,3), (1,2), (1,3)
        # B indices: (2,0), (2,1), (3,0), (3,1)
        # C indices: (0,2), (0,3), (1,2), (1,3)
        add_strassen_block(0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3)
        
        # Block 3: C02 (A00 * B01)
        # A00 is A(0,1; 0,1)
        # B01 is B(0,1; 2,3)
        # C02 is C(0,1; 0,1) ? No, C02 is columns 0,1 of the second block?
        # C_top is 2x4.
        # C_top = [C_col0, C_col1, C_col2, C_col3]
        # C_col0 = A00 B00 + A01 B10
        # C_col1 = A00 B01 + A01 B11
        # Wait, my block decomposition was:
        # A_top = [A00, A01] (2x2, 2x2)
        # B = [[B00, B01], [B10, B11]]
        # C_top = [A00 B00 + A01 B10, A00 B01 + A01 B11]
        # So C_top_left (2x2) = A00 B00 + A01 B10
        # C_top_right (2x2) = A00 B01 + A01 B11
        
        # So we need 4 Strassen multiplications:
        # 1. A00 B00
        # 2. A01 B10
        # 3. A00 B01
        # 4. A01 B11
        
        # I already did 1 and 2.
        # Block 3: A00 B01
        # A indices: (0,0), (0,1), (1,0), (1,1)
        # B indices: (0,2), (0,3), (1,2), (1,3)
        # C indices: (0,0), (0,1), (1,0), (1,1)
        add_strassen_block(0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 0, 1)
        
        # Block 4: A01 B11
        # A indices: (0,2), (0,3), (1,2), (1,3)
        # B indices: (2,2), (2,3), (3,2), (3,3)
        # C indices: (0,2), (0,3), (1,2), (1,3)
        add_strassen_block(0, 1, 2, 3, 2, 3, 2, 3, 0, 1, 2, 3)
        
        # Now C_bottom (row 2)
        # A_bottom is row 2 of A (1x4).
        # B is 4x4.
        # C_bottom is row 2 of C (1x4).
        # Rank 4 decomposition.
        # C_k = sum_{j=0}^3 A_j B_jk
        # U_j = A_j (scalar) -> U[2*m + j, col] = 1
        # V_j = B_j (row) -> V[j*p + k, col] = 1? No, V is (m*p, R).
        # V index is j*p + k.
        # For fixed j, V[j*p + k, col] should be 1 for all k?
        # No, the term is A_j * B_jk.
        # U[2*m + j, col] = 1
        # V[j*p + k, col] = 1
        # W[k*n + 2, col] = 1
        # Sum over j.
        # For each j, we have a term.
        for j in range(4):
            U_new[2*m + j, col] = 1
            for k in range(4):
                V_new[j*p + k, col] = 1
                W_new[k*n + 2, col] = 1
            col += 1
        
        # Verify the decomposition
        # T_reconstructed = sum_r U[:,r] outer V[:,r] outer W[:,r]
        # Check if T_reconstructed == T
        
        # For speed, we can check a few random entries or use np.einsum
        # T_reconstructed[i, j, k] = sum_r U[i,r] V[j,r] W[k,r]
        
        # Use einsum: 'ir,jr,kr->ijk'
        T_recon = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
        
        if np.allclose(T_recon, T):
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = R_total
        else:
            # Fallback to schoolbook if verification fails
            best_U, best_V, best_W = U_school, V_school, W_school
            best_rank = rank_school

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
