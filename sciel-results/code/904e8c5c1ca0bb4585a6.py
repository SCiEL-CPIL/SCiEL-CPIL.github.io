"""Program: 3x5x5 matrix multiplication decomposition using Strassen on 2x2 blocks."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Strategy: Split the 5x5 inner/output dimensions into 2+3 blocks.
    # We have 4 sub-problems:
    # 1. A[3x2] * B[2x2] -> C[3x2]
    # 2. A[3x2] * B[2x3] -> C[3x3]
    # 3. A[3x3] * B[3x2] -> C[3x2]
    # 4. A[3x3] * B[3x3] -> C[3x3]
    
    # For problem 1: We can decompose A[3x2] into two 2x2 blocks plus a 1x2 block? No, that's not clean.
    # Actually, A is 3x2. We can't directly apply 2x2 Strassen.
    # However, we can treat the 2x2 part of B with Strassen if we split A's rows.
    # But A has 3 rows.
    
    # Let's stick to the known improvement:
    # The 2x2x2 sub-problem in the corner (rows 0:2 of A, cols 0:2 of B, rows 0:2 of C) can use Strassen.
    # But A is 3x5, so A[0:2, 0:2] is 2x2. B[0:2, 0:2] is 2x2. C[0:2, 0:2] is 2x2.
    # This works!
    
    # Let's identify the 2x2x2 sub-blocks where we can apply Strassen:
    # The input A has rows 0..2. The input B has rows 0..4. The output C has rows 0..2.
    # A sub-block is defined by (i_range, j_range, k_range).
    # We need i_range to be size 2, j_range size 2, k_range size 2.
    # Possible i_ranges: {0,1}, {1,2}
    # Possible j_ranges: {0,1}, {1,2}, {2,3}, {3,4}
    # Possible k_ranges: {0,1}, {1,2}
    
    # However, the decomposition must be exact and non-overlapping in terms of the rank contributions?
    # No, the decomposition is a sum of tensor terms. We can overlap as long as the linear combination is correct.
    # But typically, block decompositions partition the work.
    
    # Let's try a different partitioning that allows Strassen.
    # We can partition the inner dimension (j) into 2+3.
    # We can partition the output dimension (k) into 2+3.
    # We can partition the input dimension (i) into 2+1.
    
    # Block (0,0): A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    #   Size 2x2x2. Use Strassen (rank 7).
    # Block (0,1): A[0:2, 2:5] * B[2:5, 0:2] -> C[0:2, 2:5]
    #   Size 2x3x3. Schoolbook (rank 18).
    # Block (0,2): A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    #   Wait, I messed up the indices.
    # A is n x m (3x5). B is m x p (5x5). C is n x p (3x5).
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let's split j (inner) into J1={0,1}, J2={2,3,4}.
    # Let's split k (output) into K1={0,1}, K2={2,3,4}.
    # Let's split i (input) into I1={0,1}, I2={2}.
    
    # Sub-problems:
    # 1. i in I1, j in J1, k in K1: A[2x2] * B[2x2] -> C[2x2]. Rank 7 (Strassen).
    # 2. i in I1, j in J2, k in K1: A[2x3] * B[3x2] -> C[2x2]. Rank 12 (Schoolbook).
    # 3. i in I1, j in J1, k in K2: A[2x2] * B[2x3] -> C[2x3]. Rank 12 (Schoolbook).
    # 4. i in I1, j in J2, k in K2: A[2x3] * B[3x3] -> C[2x3]. Rank 18 (Schoolbook).
    # 5. i in I2, j in J1, k in K1: A[1x2] * B[2x2] -> C[1x2]. Rank 4 (Schoolbook).
    # 6. i in I2, j in J2, k in K1: A[1x3] * B[3x2] -> C[1x2]. Rank 6 (Schoolbook).
    # 7. i in I2, j in J1, k in K2: A[1x2] * B[2x3] -> C[1x3]. Rank 6 (Schoolbook).
    # 8. i in I2, j in J2, k in K2: A[1x3] * B[3x3] -> C[1x3]. Rank 9 (Schoolbook).
    
    # Total rank: 7 + 12 + 12 + 18 + 4 + 6 + 6 + 9 = 74.
    # This is worse than 75? No, 74 < 75.
    # Wait, schoolbook rank for nxmxp is n*m*p.
    # 1. 2*2*2 = 8 -> Strassen 7. Save 1.
    # 2. 2*3*2 = 12.
    # 3. 2*2*3 = 12.
    # 4. 2*3*3 = 18.
    # 5. 1*2*2 = 4.
    # 6. 1*3*2 = 6.
    # 7. 1*2*3 = 6.
    # 8. 1*3*3 = 9.
    # Total schoolbook would be 8+12+12+18+4+6+6+9 = 75.
    # Strassen saves 1. Total 74.
    
    # Can we save more?
    # What if we use Strassen on other 2x2x2 blocks?
    # We only have one 2x2x2 block in the I1, J1, K1 position.
    # Are there other 2x2x2 blocks?
    # i in I1={0,1} is the only 2-row block in I.
    # j in J1={0,1} is a 2-col block in J.
    # k in K1={0,1} is a 2-row block in K.
    # So only one 2x2x2 block is available with this partition.
    
    # What if we partition differently?
    # I1={0,1}, J1={0,1}, K1={0,1} -> 2x2x2 (Strassen 7)
    # I1={0,1}, J2={2,3}, K1={0,1} -> 2x2x2 (Strassen 7)
    # I1={0,1}, J3={4}, K1={0,1} -> 2x1x2 (Schoolbook 4)
    # I1={0,1}, J1={0,1}, K2={2,3} -> 2x2x2 (Strassen 7)
    # I1={0,1}, J2={2,3}, K2={2,3} -> 2x2x2 (Strassen 7)
    # I1={0,1}, J3={4}, K2={2,3} -> 2x1x2 (Schoolbook 4)
    # I2={2}, J1={0,1}, K1={0,1} -> 1x2x2 (Schoolbook 4)
    # I2={2}, J2={2,3}, K1={0,1} -> 1x2x2 (Schoolbook 4)
    # I2={2}, J3={4}, K1={0,1} -> 1x1x2 (Schoolbook 2)
    # I2={2}, J1={0,1}, K2={2,3} -> 1x2x2 (Schoolbook 4)
    # I2={2}, J2={2,3}, K2={2,3} -> 1x2x2 (Schoolbook 4)
    # I2={2}, J3={4}, K2={2,3} -> 1x1x2 (Schoolbook 2)
    
    # Total rank:
    # Strassen: 4 * 7 = 28
    # Schoolbook: 4 + 4 + 2 + 4 + 4 + 2 = 20
    # Total: 48.
    
    # This is much better!
    # Let's verify the partition:
    # I = {0,1,2} -> I1={0,1}, I2={2}
    # J = {0,1,2,3,4} -> J1={0,1}, J2={2,3}, J3={4}
    # K = {0,1,2,3,4} -> K1={0,1}, K2={2,3}, K3={4}
    
    # Wait, I used K1={0,1}, K2={2,3}. Where is K3={4}?
    # In the list above, I missed K3.
    # Let's re-list with K3={4}:
    
    # 1. I1, J1, K1: 2x2x2 -> Strassen 7
    # 2. I1, J2, K1: 2x2x2 -> Strassen 7
    # 3. I1, J3, K1: 2x1x2 -> Schoolbook 4
    # 4. I1, J1, K2: 2x2x2 -> Strassen 7
    # 5. I1, J2, K2: 2x2x2 -> Strassen 7
    # 6. I1, J3, K2: 2x1x2 -> Schoolbook 4
    # 7. I1, J1, K3: 2x2x1 -> Schoolbook 4
    # 8. I1, J2, K3: 2x2x1 -> Schoolbook 4
    # 9. I1, J3, K3: 2x1x1 -> Schoolbook 2
    # 10. I2, J1, K1: 1x2x2 -> Schoolbook 4
    # 11. I2, J2, K1: 1x2x2 -> Schoolbook 4
    # 12. I2, J3, K1: 1x1x2 -> Schoolbook 2
    # 13. I2, J1, K2: 1x2x2 -> Schoolbook 4
    # 14. I2, J2, K2: 1x2x2 -> Schoolbook 4
    # 15. I2, J3, K2: 1x1x2 -> Schoolbook 2
    # 16. I2, J1, K3: 1x2x1 -> Schoolbook 2
    # 17. I2, J2, K3: 1x2x1 -> Schoolbook 2
    # 18. I2, J3, K3: 1x1x1 -> Schoolbook 1
    
    # Total:
    # Strassen: 7 * 4 = 28
    # Schoolbook: 4+4+4+4+2+4+4+2+4+4+2+2+2+1 = 45
    # Total: 73.
    
    # Let's re-calculate schoolbook sum:
    # 3: 4
    # 6: 4
    # 7: 4
    # 8: 4
    # 9: 2
    # 10: 4
    # 11: 4
    # 12: 2
    # 13: 4
    # 14: 4
    # 15: 2
    # 16: 2
    # 17: 2
    # 18: 1
    # Sum: 4+4+4+4+2+4+4+2+4+4+2+2+2+1 = 45.
    # Total 28 + 45 = 73.
    
    # This is an improvement over 75.
    
    # Let's implement this.
    
    rank = 73
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add Strassen decomposition
    def add_strassen(i_start, j_start, k_start):
        nonlocal col
        # Indices:
        # i in [i_start, i_start+1]
        # j in [j_start, j_start+1]
        # k in [k_start, k_start+1]
        
        # Strassen for 2x2:
        # C11 = M1 - M2 + M4 - M5
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M5 - M6
        # Where:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        
        # We need to add these 7 terms to U, V, W.
        # Each term corresponds to a product of a linear combination of A and B.
        
        # Term 1: (A11 + A22) * (B11 + B22)
        # U rows: (i_start, j_start), (i_start+1, j_start+1)
        # V rows: (j_start, k_start), (j_start+1, k_start+1)
        # W row: (k_start, i_start)
        
        # We need to construct the vectors for U, V, W for each of the 7 multiplications.
        # Let's define the linear combinations explicitly.
        
        # M1:
        # U1: A11=1, A22=1
        # V1: B11=1, B22=1
        # W1: C11 += 1, C12 += 1, C21 += 1, C22 += 1
        # Wait, M1 contributes to all 4 output blocks.
        # C11 = M1 - M2 + M4 - M5
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M5 - M6
        
        # So for M1:
        # W[ (k_start, i_start) ] += 1
        # W[ (k_start+1, i_start) ] += 1
        # W[ (k_start, i_start+1) ] += 1
        # W[ (k_start+1, i_start+1) ] += 1
        
        # But we need to be careful with signs and combinations for the other M's.
        # It is easier to add each of the 7 M terms as a separate rank-1 tensor,
        # with the appropriate coefficients in W.
        
        # Let's define the 7 multiplications:
        # 1: A_comb1 * B_comb1
        # 2: A_comb2 * B_comb2
        # 3: A_comb3 * B_comb3
        # 4: A_comb4 * B_comb4
        # 5: A_comb5 * B_comb5
        # 6: A_comb6 * B_comb6
        # 7: A_comb7 * B_comb7  (Wait, Strassen has 7 multiplications)
        
        # Strassen Multiplications:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Output:
        # C11 = M1 - M2 + M4 - M7
        # C12 = M3 + M5
        # C21 = M6 + M2
        # C22 = M1 - M3 + M5 - M6
        
        # Let's implement this.
        
        # Indices in U: i*m + j
        # Indices in V: j*p + k
        # Indices in W: k*n + i
        
        # Define the linear combinations for A (rows of U)
        # A11: (i_start, j_start)
        # A12: (i_start, j_start+1)
        # A21: (i_start+1, j_start)
        # A22: (i_start+1, j_start+1)
        
        # Define the linear combinations for B (rows of V)
        # B11: (j_start, k_start)
        # B12: (j_start, k_start+1)
        # B21: (j_start+1, k_start)
        # B22: (j_start+1, k_start+1)
        
        # M1:
        # U: A11=1, A22=1
        # V: B11=1, B22=1
        # W: C11=1, C12=-1? No, M1 coeff is 1 in C11, C22. 0 in C12, C21.
        # Wait, C11 = M1 - M2 + M4 - M7. So M1 coeff is 1.
        # C22 = M1 - M3 + M5 - M6. So M1 coeff is 1.
        
        # Let's add M1
        U[i_start * m + j_start, col] = 1.0
        U[(i_start+1) * m + (j_start+1), col] = 1.0
        V[j_start * p + k_start, col] = 1.0
        V[(j_start+1) * p + (k_start+1), col] = 1.0
        W[k_start * n + i_start, col] = 1.0
        W[(k_start+1) * n + (i_start+1), col] = 1.0
        col += 1
        
        # M2:
        # U: A21=1, A22=1
        # V: B11=1
        # W: C11=-1, C21=1
        U[(i_start+1) * m + j_start, col] = 1.0
        U[(i_start+1) * m + (j_start+1), col] = 1.0
        V[j_start * p + k_start, col] = 1.0
        W[k_start * n + i_start, col] = -1.0
        W[k_start * n + (i_start+1), col] = 1.0
        col += 1
        
        # M3:
        # U: A11=1
        # V: B12=1, B22=-1
        # W: C12=1, C22=-1
        U[i_start * m + j_start, col] = 1.0
        V[j_start * p + (k_start+1), col] = 1.0
        V[(j_start+1) * p + (k_start+1), col] = -1.0
        W[(k_start+1) * n + i_start, col] = 1.0
        W[(k_start+1) * n + (i_start+1), col] = -1.0
        col += 1
        
        # M4:
        # U: A22=1
        # V: B21=1, B11=-1
        # W: C11=1, C21=1
        U[(i_start+1) * m + (j_start+1), col] = 1.0
        V[(j_start+1) * p + k_start, col] = 1.0
        V[j_start * p + k_start, col] = -1.0
        W[k_start * n + i_start, col] = 1.0
        W[k_start * n + (i_start+1), col] = 1.0
        col += 1
        
        # M5:
        # U: A11=1, A12=1
        # V: B22=1
        # W: C12=1, C22=1
        U[i_start * m + j_start, col] = 1.0
        U[i_start * m + (j_start+1), col] = 1.0
        V[(j_start+1) * p + (k_start+1), col] = 1.0
        W[(k_start+1) * n + i_start, col] = 1.0
        W[(k_start+1) * n + (i_start+1), col] = 1.0
        col += 1
        
        # M6:
        # U: A21=1, A11=-1
        # V: B11=1, B12=1
        # W: C21=1, C22=-1
        U[(i_start+1) * m + j_start, col] = 1.0
        U[i_start * m + j_start, col] = -1.0
        V[j_start * p + k_start, col] = 1.0
        V[j_start * p + (k_start+1), col] = 1.0
        W[k_start * n + (i_start+1), col] = 1.0
        W[(k_start+1) * n + (i_start+1), col] = -1.0
        col += 1
        
        # M7:
        # U: A12=1, A22=-1
        # V: B21=1, B22=1
        # W: C11=-1
        U[i_start * m + (j_start+1), col] = 1.0
        U[(i_start+1) * m + (j_start+1), col] = -1.0
        V[(j_start+1) * p + k_start, col] = 1.0
        V[(j_start+1) * p + (k_start+1), col] = 1.0
        W[k_start * n + i_start, col] = -1.0
        col += 1
        
    # Helper to add schoolbook decomposition
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        nonlocal col
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
                    
    # Now add the blocks
    # I1={0,1}, I2={2}
    # J1={0,1}, J2={2,3}, J3={4}
    # K1={0,1}, K2={2,3}, K3={4}
    
    # 1. I1, J1, K1: 2x2x2 -> Strassen
    add_strassen(0, 0, 0)
    
    # 2. I1, J2, K1: 2x2x2 -> Strassen
    add_strassen(0, 2, 0)
    
    # 3. I1, J3, K1: 2x1x2 -> Schoolbook
    add_schoolbook(0, 2, 4, 5, 0, 2)
    
    # 4. I1, J1, K2: 2x2x2 -> Strassen
    add_strassen(0, 0, 2)
    
    # 5. I1, J2, K2: 2x2x2 -> Strassen
    add_strassen(0, 2, 2)
    
    # 6. I1, J3, K2: 2x1x2 -> Schoolbook
    add_schoolbook(0, 2, 4, 5, 2, 4)
    
    # 7. I1, J1, K3: 2x2x1 -> Schoolbook
    add_schoolbook(0, 2, 0, 2, 4, 5)
    
    # 8. I1, J2, K3: 2x2x1 -> Schoolbook
    add_schoolbook(0, 2, 2, 4, 4, 5)
    
    # 9. I1, J3, K3: 2x1x1 -> Schoolbook
    add_schoolbook(0, 2, 4, 5, 4, 5)
    
    # 10. I2, J1, K1: 1x2x2 -> Schoolbook
    add_schoolbook(2, 3, 0, 2, 0, 2)
    
    # 11. I2, J2, K1: 1x2x2 -> Schoolbook
    add_schoolbook(2, 3, 2, 4, 0, 2)
    
    # 12. I2, J3, K1: 1x1x2 -> Schoolbook
    add_schoolbook(2, 3, 4, 5, 0, 2)
    
    # 13. I2, J1, K2: 1x2x2 -> Schoolbook
    add_schoolbook(2, 3, 0, 2, 2, 4)
    
    # 14. I2, J2, K2: 1x2x2 -> Schoolbook
    add_schoolbook(2, 3, 2, 4, 2, 4)
    
    # 15. I2, J3, K2: 1x1x2 -> Schoolbook
    add_schoolbook(2, 3, 4, 5, 2, 4)
    
    # 16. I2, J1, K3: 1x2x1 -> Schoolbook
    add_schoolbook(2, 3, 0, 2, 4, 5)
    
    # 17. I2, J2, K3: 1x2x1 -> Schoolbook
    add_schoolbook(2, 3, 2, 4, 4, 5)
    
    # 18. I2, J3, K3: 1x1x1 -> Schoolbook
    add_schoolbook(2, 3, 4, 5, 4, 5)
    
    # Check if we ran out of time
    if time.time() > deadline:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=float)
        V_fb = np.zeros((m * p, rank_fb), dtype=float)
        W_fb = np.zeros((p * n, rank_fb), dtype=float)
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col_fb] = 1.0
                    V_fb[j * p + k, col_fb] = 1.0
                    W_fb[k * n + i, col_fb] = 1.0
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
