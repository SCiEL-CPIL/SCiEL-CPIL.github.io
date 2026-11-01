"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-32 decomposition for 2x2 * 2x9 matrix multiplication.
    
    Strategy:
    Split the 9 output columns into:
    1. Columns 0-3: Two 2x2 blocks. Each uses Strassen (rank 7). Total 14.
    2. Columns 4-8: One 2x3 block and one 2x2 block.
       - 2x3 block uses rank 11 decomposition.
       - 2x2 block uses Strassen (rank 7).
       Total for this part: 18.
    Total rank: 32.
    """
    n, m, p = 2, 2, 9
    rank = 32
    
    # Initialize matrices
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Helper to add a Strassen decomposition for a 2x2 * 2x2 block
    # A is 2x2 (indices i,j in 0..1), B is 2x2 (indices j,k in 0..1)
    # Output C is 2x2 (indices k,i in 0..1)
    # We map local indices to global indices via offsets
    def add_strassen_2x2(u_row_offset, v_row_offset, w_row_offset, col_start):
        # Strassen factors for 2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # U factors (rows correspond to A elements: A11, A12, A21, A22)
        # Indices: 0:11, 1:12, 2:21, 3:22
        u_11 = u_row_offset + 0
        u_12 = u_row_offset + 1
        u_21 = u_row_offset + 2
        u_22 = u_row_offset + 3
        
        # V factors (rows correspond to B elements: B11, B12, B21, B22)
        # Indices: 0:11, 1:12, 2:21, 3:22
        v_11 = v_row_offset + 0
        v_12 = v_row_offset + 1
        v_21 = v_row_offset + 2
        v_22 = v_row_offset + 3
        
        # W factors (rows correspond to C elements: C11, C12, C21, C22)
        # Note: W index is k*n + i.
        # C11: k=0, i=0 -> 0
        # C12: k=0, i=1 -> 1 (Wait, C is 2x2. Output is C[k, i].)
        # Let's check the convention: W[k*n+i, r].
        # For 2x2 output C:
        # C[0,0] is k=0, i=0 -> index 0
        # C[0,1] is k=0, i=1 -> index 1
        # C[1,0] is k=1, i=0 -> index 2
        # C[1,1] is k=1, i=1 -> index 3
        
        w_11 = w_row_offset + 0 # C[0,0]
        w_12 = w_row_offset + 1 # C[0,1]
        w_21 = w_row_offset + 2 # C[1,0]
        w_22 = w_row_offset + 3 # C[1,1]
        
        # Add 7 terms
        # Term 1: (A11+A22)(B11+B22)
        c = col_start + 0
        U[u_11, c] += 1; U[u_22, c] += 1
        V[v_11, c] += 1; V[v_22, c] += 1
        W[w_11, c] += 1; W[w_22, c] += 1; W[w_12, c] -= 1; W[w_21, c] -= 1
        
        # Term 2: (A21+A22)B11
        c = col_start + 1
        U[u_21, c] += 1; U[u_22, c] += 1
        V[v_11, c] += 1
        W[w_21, c] += 1; W[w_22, c] -= 1
        
        # Term 3: A11(B21-B22)
        c = col_start + 2
        U[u_11, c] += 1
        V[v_21, c] += 1; V[v_22, c] -= 1
        W[w_12, c] += 1; W[w_22, c] += 1
        
        # Term 4: A22(B12-B11)
        c = col_start + 3
        U[u_22, c] += 1
        V[v_12, c] += 1; V[v_11, c] -= 1
        W[w_11, c] -= 1; W[w_21, c] += 1
        
        # Term 5: (A11+A12)B22
        c = col_start + 4
        U[u_11, c] += 1; U[u_12, c] += 1
        V[v_22, c] += 1
        W[w_11, c] += 1; W[w_12, c] += 1
        
        # Term 6: (A21-A11)(B11+B12)
        c = col_start + 5
        U[u_21, c] += 1; U[u_11, c] -= 1
        V[v_11, c] += 1; V[v_12, c] += 1
        W[w_22, c] += 1
        
        # Term 7: (A12-A22)(B21+B22)
        c = col_start + 6
        U[u_12, c] += 1; U[u_22, c] -= 1
        V[v_21, c] += 1; V[v_22, c] += 1
        W[w_11, c] -= 1

    # Helper to add a rank-11 decomposition for 2x2 * 2x3 block
    # A is 2x2, B is 2x3. Output C is 2x3.
    # This is equivalent to multiplying 2x2 by [2x2, 2x1].
    # We can use a known rank-11 algorithm for M_{2,2,3}.
    # One such algorithm (from literature, e.g. "Fast rectangular matrix multiplication"):
    # We split B into B_left (2x2) and B_right (2x1).
    # Instead of standard block, we use specific linear combinations.
    
    def add_rank11_2x2x3(u_row_offset, v_row_offset, w_row_offset, col_start):
        # v_row_offset points to B[0,0], B[0,1], B[0,2], B[1,0], B[1,1], B[1,2]
        # v indices: 0:00, 1:01, 2:02, 3:10, 4:11, 5:12
        
        u_11 = u_row_offset + 0
        u_12 = u_row_offset + 1
        u_21 = u_row_offset + 2
        u_22 = u_row_offset + 3
        
        v_00 = v_row_offset + 0
        v_01 = v_row_offset + 1
        v_02 = v_row_offset + 2
        v_10 = v_row_offset + 3
        v_11 = v_row_offset + 4
        v_12 = v_row_offset + 5
        
        # W indices for 2x3 output C (k=0..1, i=0..1)
        # C[0,0] -> 0
        # C[0,1] -> 1
        # C[1,0] -> 2
        # C[1,1] -> 3
        # C[0,2] -> 4
        # C[1,2] -> 5
        
        w_00 = w_row_offset + 0
        w_01 = w_row_offset + 1
        w_10 = w_row_offset + 2
        w_11 = w_row_offset + 3
        w_02 = w_row_offset + 4
        w_12 = w_row_offset + 5
        
        # We need a valid rank-11 decomposition.
        # Let's use the "3x3x3" trick or a specific 2x2x3 construction.
        # Actually, a simple construction:
        # Use Strassen for the 2x2 part (cols 0,1) and schoolbook for col 2?
        # Strassen (7) + Schoolbook for 2x1 (2) = 9? No, they are not disjoint in U/V if we just append.
        # But we CAN just use schoolbook for the whole 2x2x3? Rank 12.
        # Can we do better than 12?
        # Known bounds: R(M_{2,2,3}) <= 11.
        # Let's try to construct rank 11.
        # If we fail to find a simple one, we can use rank 12 (schoolbook) for this block.
        # Rank 12 for 2x2x3 + Rank 7 for 2x2x2 = 19. Total 14+19=33.
        # The directive asks for "significantly lower than 32". If we use 11, total is 32.
        # If we use 12, total is 33.
        # Let's stick to a safe Rank 12 (Schoolbook) for the 2x2x3 block to ensure correctness, 
        # OR try a specific rank 11.
        
        # Let's try a Rank 11 construction for 2x2x3.
        # Reference: "Fast rectangular matrix multiplication" 
        # One method: 
        # M1 = (A11+A22)(B11+B21)
        # M2 = (A11+A22)(B12+B22)
        # ... this is getting complicated to recall exactly.
        
        # Safer approach: Use Schoolbook for 2x2x3 (Rank 12).
        # Total Rank = 7 (2x2x2) + 7 (2x2x2) + 12 (2x2x3) + 7 (2x2x2) = 33.
        # This is worse than 32.
        
        # Let's try to use Rank 11.
        # There is a known rank 11 algorithm for 2x2x3.
        # It can be derived from 3x3x3 rank 23? No.
        
        # Alternative: Decompose 2x2x3 into 2x2x2 (Strassen 7) and 2x2x1 (Schoolbook 2).
        # Total 9? No, the U and V matrices must be consistent.
        # If we just append the terms, the rank is the sum of the ranks IF the supports are disjoint or we just sum them.
        # Summing decompositions: Rank(A*B) <= Rank(A*B1) + Rank(A*B2).
        # So Rank(2x2x3) <= Rank(2x2x2) + Rank(2x2x1) = 7 + 2 = 9.
        # Wait, is Rank(2x2x1) = 2?
        # A(2x2) * B(2x1) -> C(2x1).
        # C1 = A11*B11 + A12*B21
        # C2 = A21*B11 + A22*B21
        # Schoolbook: 4 terms.
        # Can we do it in 2? No, bilinear rank of 2x2x1 is 4?
        # Actually, matrix-vector multiplication 2x2 * 2x1 has rank 4.
        # So 7 + 4 = 11.
        # Yes! We can decompose the 2x2x3 block into a 2x2x2 block (Strassen, rank 7) and a 2x2x1 block (Schoolbook, rank 4).
        # Total rank for 2x2x3 part = 11.
        
        # So, we implement:
        # 1. Strassen for B[:, 0:2] (cols 0,1 of the 2x3 block)
        # 2. Schoolbook for B[:, 2] (col 2 of the 2x3 block)
        
        # Part 1: Strassen for cols 0,1
        # v_00, v_01, v_10, v_11 are the 2x2 matrix.
        # We call add_strassen_2x2 but we need to map V indices correctly.
        # My add_strassen_2x2 assumes V is 2x2 contiguous.
        # Here V is 2x3. The 2x2 part is v_00, v_01, v_10, v_11.
        # Indices in local V: 0, 1, 3, 4.
        # This doesn't fit the "contiguous" assumption of my helper easily without remapping.
        
        # Let's manually add the Strassen terms for the 2x2 part (cols 0,1)
        # and then add Schoolbook terms for the 2x1 part (col 2).
        
        # --- Strassen for cols 0,1 ---
        # Uses v_00, v_01, v_10, v_11
        # Output w_00, w_01, w_10, w_11
        
        c = col_start + 0
        # M1 = (A11+A22)(B11+B21) -> (v_00 + v_10)
        U[u_11, c] += 1; U[u_22, c] += 1
        V[v_00, c] += 1; V[v_10, c] += 1
        W[w_00, c] += 1; W[w_11, c] += 1; W[w_01, c] -= 1; W[w_10, c] -= 1
        
        c = col_start + 1
        # M2 = (A21+A22)B11 -> v_00
        U[u_21, c] += 1; U[u_22, c] += 1
        V[v_00, c] += 1
        W[w_10, c] += 1; W[w_11, c] -= 1
        
        c = col_start + 2
        # M3 = A11(B21-B22) -> (v_10 - v_11)
        U[u_11, c] += 1
        V[v_10, c] += 1; V[v_11, c] -= 1
        W[w_01, c] += 1; W[w_11, c] += 1
        
        c = col_start + 3
        # M4 = A22(B12-B11) -> (v_01 - v_00)
        U[u_22, c] += 1
        V[v_01, c] += 1; V[v_00, c] -= 1
        W[w_00, c] -= 1; W[w_10, c] += 1
        
        c = col_start + 4
        # M5 = (A11+A12)B22 -> v_11
        U[u_11, c] += 1; U[u_12, c] += 1
        V[v_11, c] += 1
        W[w_00, c] += 1; W[w_01, c] += 1
        
        c = col_start + 5
        # M6 = (A21-A11)(B11+B12) -> (v_00 + v_01)
        U[u_21, c] += 1; U[u_11, c] -= 1
        V[v_00, c] += 1; V[v_01, c] += 1
        W[w_11, c] += 1
        
        c = col_start + 6
        # M7 = (A12-A22)(B21+B22) -> (v_10 + v_11)
        U[u_12, c] += 1; U[u_22, c] -= 1
        V[v_10, c] += 1; V[v_11, c] += 1
        W[w_00, c] -= 1
        
        # --- Schoolbook for col 2 ---
        # Uses v_02, v_12
        # Output w_02, w_12
        
        c = col_start + 7
        # A11 * B12 (v_02)
        U[u_11, c] += 1
        V[v_02, c] += 1
        W[w_02, c] += 1
        
        c = col_start + 8
        # A12 * B22 (v_12)
        U[u_12, c] += 1
        V[v_12, c] += 1
        W[w_02, c] += 1
        
        c = col_start + 9
        # A21 * B12 (v_02)
        U[u_21, c] += 1
        V[v_02, c] += 1
        W[w_12, c] += 1
        
        c = col_start + 10
        # A22 * B22 (v_12)
        U[u_22, c] += 1
        V[v_12, c] += 1
        W[w_12, c] += 1

    # --- Assembly ---
    
    # Block 1: 2x2 * 2x2 (Cols 0-1 of A, Cols 0-1 of B, Output Cols 0-1)
    # U rows: 0-3 (A is full 2x2, so u_row_offset=0)
    # V rows: 0-3 (B cols 0-1)
    # W rows: 0-3 (Output cols 0-1, k=0,1, i=0,1)
    # Note: W index is k*n + i.
    # For output col 0: k=0, i=0 -> 0; k=1, i=0 -> 2.
    # For output col 1: k=0, i=1 -> 1; k=1, i=1 -> 3.
    # So W rows 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1).
    # This matches my w_11, w_12, w_21, w_22 mapping in add_strassen_2x2.
    add_strassen_2x2(u_row_offset=0, v_row_offset=0, w_row_offset=0, col_start=0)
    
    # Block 2: 2x2 * 2x2 (Cols 0-1 of A, Cols 2-3 of B, Output Cols 2-3)
    # U rows: 0-3
    # V rows: 4-7 (B cols 2-3)
    # W rows: 4-7 (Output cols 2-3)
    # Output col 2: k=0, i=0 -> 4; k=1, i=0 -> 6.
    # Output col 3: k=0, i=1 -> 5; k=1, i=1 -> 7.
    add_strassen_2x2(u_row_offset=0, v_row_offset=4, w_row_offset=4, col_start=7)
    
    # Block 3: 2x2 * 2x3 (Cols 0-1 of A, Cols 4-6 of B, Output Cols 4-6)
    # U rows: 0-3
    # V rows: 8-13 (B cols 4-6)
    # W rows: 8-13 (Output cols 4-6)
    # Output col 4: k=0, i=0 -> 8; k=1, i=0 -> 10.
    # Output col 5: k=0, i=1 -> 9; k=1, i=1 -> 11.
    # Output col 6: k=0, i=2 -> 12; k=1, i=2 -> 13.
    # Wait, my W mapping in add_rank11_2x2x3 assumed:
    # w_00 -> 0 (k=0, i=0)
    # w_01 -> 1 (k=0, i=1)
    # w_10 -> 2 (k=1, i=0)
    # w_11 -> 3 (k=1, i=1)
    # w_02 -> 4 (k=0, i=2)
    # w_12 -> 5 (k=1, i=2)
    # 
    # Global W indices for cols 4,5,6:
    # Col 4: k=0,i=0 -> 4*2+0 = 8. k=1,i=0 -> 4*2+2 = 10.
    # Col 5: k=0,i=1 -> 5*2+0 = 10? No.
    # W[k*n+i, r]. n=2.
    # Col 4: i=4? No, i is the column index of the OUTPUT matrix.
    # Output is 2x9. Rows are k (0,1), Cols are i (0..8).
    # W index = k*2 + i.
    # Col 4: i=4.
    #   k=0 -> 0*2+4 = 4.
    #   k=1 -> 1*2+4 = 6.
    # Col 5: i=5.
    #   k=0 -> 5.
    #   k=1 -> 7.
    # Col 6: i=6.
    #   k=0 -> 6.
    #   k=1 -> 8.
    
    # My previous offset logic for W was slightly off for the 2x9 case.
    # Let's re-verify the W mapping.
    # W has shape (p*n, rank) = (18, rank).
    # Index r in W corresponds to output element C[k, i] where r = k*n + i.
    
    # Block 1 (Output cols 0,1):
    # i=0: r=0 (k=0), r=2 (k=1).
    # i=1: r=1 (k=0), r=3 (k=1).
    # So W rows 0,1,2,3 are used.
    # My add_strassen_2x2 uses w_11 (0), w_12 (1), w_21 (2), w_22 (3).
    # w_11 is C[0,0] -> r=0. Correct.
    # w_12 is C[0,1] -> r=1. Correct.
    # w_21 is C[1,0] -> r=2. Correct.
    # w_22 is C[1,1] -> r=3. Correct.
    
    # Block 2 (Output cols 2,3):
    # i=2: r=2 (k=0), r=4 (k=1).
    # i=3: r=3 (k=0), r=5 (k=1).
    # So W rows 2,3,4,5 are used.
    # My add_strassen_2x2 with w_row_offset=4 would use 4,5,6,7.
    # This is INCORRECT. The W indices are interleaved!
    # W[k*n+i].
    # For i=2: 0*2+2=2, 1*2+2=4.
    # For i=3: 0*2+3=3, 1*2+3=5.
    # So the rows are 2,3,4,5.
    # My helper assumes contiguous rows w_offset, w_offset+1, w_offset+2, w_offset+3.
    # This works for i=0,1 (0,1,2,3).
    # It does NOT work for i=2,3 (2,3,4,5).
    
    # I need a more flexible helper or manual construction.
    # Given the complexity and risk of indexing errors in a generic helper for interleaved W,
    # I will implement the construction manually for each block to ensure correctness.
    
    # Let's restart the assembly with manual index calculation.
    U = np.zeros((4, 32), dtype=float)
    V = np.zeros((18, 32), dtype=float)
    W = np.zeros((18, 32), dtype=float)
    
    # --- Block 1: Output Cols 0,1 ---
    # U: 0-3
    # V: 0-3 (B cols 0,1)
    # W: 0,1,2,3
    add_strassen_2x2(0, 0, 0, 0)
    
    # --- Block 2: Output Cols 2,3 ---
    # U: 0-3
    # V: 4-7 (B cols 2,3)
    # W: 2,3,4,5
    # My helper assumes W rows are w_off, w_off+1, w_off+2, w_off+3.
    # Here they are 2,3,4,5.
    # I can call the helper with w_row_offset=2.
    # w_11 -> 2, w_12 -> 3, w_21 -> 4, w_22 -> 5.
    # This matches!
    add_strassen_2x2(0, 4, 2, 7)
    
    # --- Block 3: Output Cols 4,5,6 ---
    # U: 0-3
    # V: 8-13 (B cols 4,5,6)
    # W: 
    # i=4: 4,6
    # i=5: 5,7
    # i=6: 6,8
    # Rows: 4,5,6,7,8,9? No.
    # i=4: k=0->4, k=1->6.
    # i=5: k=0->5, k=1->7.
    # i=6: k=0->6, k=1->8.
    # Rows: 4,5,6,7,8. (5 is used, 6 is used twice? No, 6 is k=1 for i=4 and k=0 for i=6? No.)
    # W index = k*2 + i.
    # i=4: 4, 6.
    # i=5: 5, 7.
    # i=6: 6, 8.
    # So rows are 4,5,6,7,8.
    # My add_rank11_2x2x3 assumes w_00..w_12 are w_off..w_off+5.
    # Here they are 4,5,6,7,8. (Wait, 6 is shared? No, W is a matrix. Each row is a distinct output element.)
    # Row 4: C[0,4]
    # Row 5: C[0,5]
    # Row 6: C[0,6] AND C[1,4]?
    # C[1,4] is k=1, i=4 -> 1*2+4 = 6.
    # C[0,6] is k=0, i=6 -> 0*2+6 = 6.
    # CONFLICT! The W matrix rows are NOT unique for different (k,i) if n=2?
    # No, W[k*n+i] is unique for each (k,i) pair.
    # k=0, i=4 -> 4.
    # k=1, i=4 -> 6.
    # k=0, i=6 -> 6.
    # Wait. 1*2+4 = 6. 0*2+6 = 6.
    # This implies that the output matrix C is stored in a way that C[1,4] and C[0,6] share the same index?
    # No, the output is 2x9.
    # Rows k=0,1. Cols i=0..8.
    # Total 18 elements.
    # Index = k*9 + i? Or k*n + i?
    # The task says: "output factor follows the official trace convention W[k*n+i,r]".
    # If n=2, index = k*2 + i.
    # Max i is 8. Max index = 1*2 + 8 = 10.
    # But W has shape (p*n, rank) = (18, rank).
    # This implies the index is NOT k*n+i if p=9.
    # Let's re-read: "W[k*n+i,r]".
    # Usually, for M_{n,m,p}, the output is n x p.
    # Here n=2, p=9. Output is 2x9.
    # If the formula is k*n+i, and k ranges 0..1, i ranges 0..8.
    # Then indices are 0..10.
    # But W has 18 rows.
    # This suggests the formula might be k*p + i? Or i*n + k?
    # "W[k*n+i,r]".
    # If n=2, p=9.
    # Maybe the dimensions are defined differently?
    # "multiplying a 2 by 2 matrix by a 2 by 9 matrix".
    # A is 2x2 (n=2, m=2).
    # B is 2x9 (m=2, p=9).
    # C is 2x9 (n=2, p=9).
    # W shape (p*n, rank) = 18.
    # If index is k*p + i, max is 1*9+8=17. Fits.
    # If index is i*n + k, max is 8*2+1=17. Fits.
    # The prompt says "W[k*n+i,r]".
    # If n=2, k=1, i=8 -> 1*2+8 = 10.
    # This leaves rows 11-17 unused?
    # Or maybe k ranges 0..p-1 and i ranges 0..n-1?
    # "k*n+i". If k is the row of C (0..1) and i is the col (0..8), it doesn't fill 18.
    # If k is the col (0..8) and i is the row (0..1), index = k*2+i. Max 17.
    # This fits the shape (18).
    # So likely: Index = col * n + row.
    # i.e., W[col*n + row, r].
    # Let's assume this convention: W index = i*n + k.
    # Where i is output col (0..8), k is output row (0..1).
    
    # Let's re-verify Block 1 with this convention.
    # Output cols 0,1.
    # i=0: k=0->0, k=1->1.
    # i=1: k=0->2, k=1->3.
    # Rows 0,1,2,3. Contiguous.
    
    # Block 2: Output cols 2,3.
    # i=2: k=0->4, k=1->5.
    # i=3: k=0->6, k=1->7.
    # Rows 4,5,6,7. Contiguous.
    
    # Block 3: Output cols 4,5,6.
    # i=4: 8,9.
    # i=5: 10,11.
    # i=6: 12,13.
    # Rows 8-13. Contiguous.
    
    # This convention makes the blocks contiguous in W.
    # So my helper functions work with w_row_offset = i_start * n.
    
    # Block 3: add_rank11_2x2x3(0, 8, 8, 14)
    add_rank11_2x2x3(0, 8, 8, 14)
    
    # Block 4: Output Cols 7,8.
    # 2x2 block.
    # U: 0-3
    # V: 14-17 (B cols 7,8)
    # W: 14-17 (i=7,8 -> 14,15,16,17)
    add_strassen_2x2(0, 14, 14, 25)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
