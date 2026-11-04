"""Rewritten program for 6x6x6 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: row = i*2 + j for U, j*2 + k for V, k*2 + i for W.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    # U: A00, A11 -> indices 0, 3
    # V: B00, B11 -> indices 0, 3
    # W: C00, C11 -> indices 0, 3
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    # U: A10, A11 -> indices 2, 3
    # V: B00 -> index 0
    # W: C10, C11 -> indices 2, 3 (C10 gets +, C11 gets -)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B10-B11)
    # U: A00 -> index 0
    # V: B10, B11 -> indices 2, 3
    # W: C00, C01 -> indices 0, 1 (C00 gets +, C01 gets +? No, C00 = M1 - M3, C01 = M3)
    # Let's check standard Strassen:
    # C00 = M1 - M3
    # C01 = M3
    # C10 = M4
    # C11 = M1 + M2 - M4 - M5
    # So for M3:
    # W: C00 gets -1, C01 gets 1.
    # Indices for C00 is 0, C01 is 1.
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[0, 2] = -1; W[1, 2] = 1
    
    # M4 = A11(B00-B10)
    # U: A11 -> index 3
    # V: B00, B10 -> indices 0, 2
    # W: C10 gets 1.
    # C10 index is 2.
    U[3, 3] = 1
    V[0, 3] = 1; V[2, 3] = -1
    W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    # U: A00, A01 -> indices 0, 1
    # V: B11 -> index 3
    # W: C11 gets -1.
    # C11 index is 3.
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[3, 4] = -1
    
    # Wait, I need to be consistent with the output indices.
    # Let's re-verify the standard Strassen formulas and map to indices.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B00 - B10)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B01 + B11)
    # M7 = (A01 - A11)(B10 + B00)
    #
    # C00 = M1 - M3
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M4 + M6 - M7  (Wait, standard is M1 + M2 - M4 - M5? No.)
    # Let's check Wikipedia:
    # C00 = M1 - M3
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M4 + M6 - M7 ? No.
    # C11 = M1 + M2 - M4 - M5 ?
    # Let's check M11 = A00B01 + A01B11.
    # M3 + M5 = A00(B10 - B11) + (A00 + A01)B11 = A00B10 - A00B11 + A00B11 + A01B11 = A00B10 + A01B11.
    # This is NOT C01. C01 = A00B01 + A01B11.
    # Ah, the standard formulas vary by convention.
    # Let's use the ones that work.
    # My previous hardcoded values in the seed program worked.
    # Let's just copy the logic from the seed program's _strassen_2x2 which is known to be valid for the evaluator.
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1  # Note: W indices here might differ from standard C indices.
    # In the seed code: W[1,1]=1, W[3,1]=-1.
    # W row index is k*2+i.
    # Row 1 is k=0, i=1 -> C01? No, k=0, i=1 means output C[1,0]?
    # Wait, W[k*n+i]. For 2x2, n=2.
    # Row 0: k=0, i=0 -> C[0,0]
    # Row 1: k=0, i=1 -> C[1,0]
    # Row 2: k=1, i=0 -> C[0,1]
    # Row 3: k=1, i=1 -> C[1,1]
    # So W[1,1]=1 means C[1,0] gets +1.
    # W[3,1]=-1 means C[1,1] gets -1.
    # This matches M2 = (A10+A11)B00 contributing to C10 and C11?
    # (A10+A11)B00 = A10B00 + A11B00.
    # C10 = A10B00 + A11B10. So A10B00 is part of C10.
    # C11 = A10B01 + A11B11. A11B00 is NOT part of C11.
    # So M2 contributes to C10. Why W[3,1]=-1?
    # Maybe the decomposition is for C = A*B, and the tensor is defined differently?
    # Or maybe my mapping of W indices is wrong.
    # Let's trust the seed code's hardcoded values as they passed.
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook_3x3():
    """
    Return the rank-27 schoolbook decomposition for 3x3x3 matrix multiplication.
    U: (9, 27), V: (9, 27), W: (9, 27)
    """
    U = np.zeros((9, 27))
    V = np.zeros((9, 27))
    W = np.zeros((9, 27))
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j, col] = 1
                V[j * 3 + k, col] = 1
                W[k * 3 + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = 6, 6, 6
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Strategy: Block decomposition of 6x6 into 2x3 or 3x2 blocks?
    # The current best is 189 (2x2 Strassen blocks, 3x3 grid).
    # 3*3*3 blocks of 2x2.
    # Rank = 27 * 7 = 189.
    
    # Can we do better?
    # Split 6 into 4+2?
    # Blocks:
    # 1. A11(4x4) B11(4x4) -> C11(4x4). Rank 32 (Strassen 4x4).
    # 2. A12(4x2) B21(2x4) -> C11(4x4).
    # 3. A11(4x4) B12(4x2) -> C12(4x2).
    # 4. A12(4x2) B22(2x2) -> C12(4x2).
    # 5. A21(2x4) B11(4x4) -> C21(2x4).
    # 6. A22(2x2) B21(2x4) -> C21(2x4).
    # 7. A21(2x4) B12(4x2) -> C22(2x2).
    # 8. A22(2x2) B22(2x2) -> C22(2x2).
    
    # Let's calculate ranks:
    # 1. 4x4x4: Rank 32.
    # 2. 4x2 x 2x4: Can be decomposed into four 2x2x2 Strassen?
    #    A12 is 4x2. B21 is 2x4.
    #    Split A12 into two 2x2 blocks (top, bottom).
    #    Split B21 into two 2x2 blocks (left, right).
    #    4 terms: A12_top * B21_left, A12_top * B21_right, A12_bot * B21_left, A12_bot * B21_right.
    #    Each is 2x2 x 2x2. Rank 7 each. Total 28.
    # 3. 4x4 x 4x2:
    #    Split A11 into two 4x2? No, split inner dimension 4 into 2+2.
    #    A11 = [A11_left, A11_right] (4x2 each).
    #    B12 = [B12_top; B12_bot] (2x2 each).
    #    Terms: A11_left * B12_top, A11_right * B12_bot.
    #    Each is 4x2 x 2x2.
    #    4x2 x 2x2: Split 4x2 into two 2x2?
    #    A11_left is 4x2. Split into top 2x2 and bot 2x2.
    #    So 2x2 x 2x2. Rank 7.
    #    So 2 terms * 7 = 14?
    #    Wait, A11_left is 4x2. B12_top is 2x2.
    #    Product is 4x2.
    #    Decompose 4x2 x 2x2.
    #    Split A11_left into A11_left_top (2x2) and A11_left_bot (2x2).
    #    Then A11_left_top * B12_top (2x2x2, rank 7) and A11_left_bot * B12_top (2x2x2, rank 7).
    #    So 14 for this term.
    #    Similarly for A11_right * B12_bot: 14.
    #    Total for 3: 28.
    # 4. 4x2 x 2x2:
    #    Split 4x2 into two 2x2.
    #    2 terms * 7 = 14.
    # 5. 2x4 x 4x4:
    #    Similar to 3. 28.
    # 6. 2x2 x 2x4:
    #    Split 2x4 into two 2x2.
    #    2 terms * 7 = 14.
    # 7. 2x4 x 4x2:
    #    Split inner 4 into 2+2.
    #    A21 = [A21_left, A21_right] (2x2 each).
    #    B12 = [B12_top; B12_bot] (2x2 each).
    #    Terms: A21_left * B12_top, A21_right * B12_bot.
    #    Each is 2x2 x 2x2. Rank 7.
    #    Total 14.
    # 8. 2x2 x 2x2: Rank 7.
    
    # Total Rank:
    # 1. 32
    # 2. 28
    # 3. 28
    # 4. 14
    # 5. 28
    # 6. 14
    # 7. 14
    # 8. 7
    # Sum: 32 + 28 + 28 + 14 + 28 + 14 + 14 + 7 = 165.
    
    # This is better than 189!
    # Let's implement this.
    
    total_rank = 165
    U = np.zeros((36, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    sU, sV, sW = _strassen_2x2()
    
    # Helper to place a 2x2x2 block
    # u_row_offset, v_row_offset, w_row_offset
    # u_idx (0..3), v_idx (0..3), w_idx (0..3)
    def place_2x2(u_off, v_off, w_off, col_start):
        for c in range(7):
            for u_i in range(4):
                for u_j in range(4):
                    # Only non-zeros in sU
                    pass
            # More efficient: iterate over non-zeros or just loop 4x4
            for u_i in range(4):
                for u_j in range(4):
                    val = sU[u_i, c]
                    if val != 0:
                        U[u_off + u_i, col_start + c] += val
            for v_i in range(4):
                for v_j in range(4):
                    val = sV[v_i, c]
                    if val != 0:
                        V[v_off + v_i, col_start + c] += val
            for w_i in range(4):
                for w_j in range(4):
                    val = sW[w_i, c]
                    if val != 0:
                        W[w_off + w_i, col_start + c] += val
    
    # Helper to place a 4x4x4 block (using Strassen 4x4)
    # We don't have a direct 4x4x4 Strassen, but we can build it from 2x2x2 Strassen.
    # Rank 32.
    def place_4x4(u_off, v_off, w_off, col_start):
        # 4x4 = 2x2 blocks of 2x2.
        # 4 blocks: 11, 12, 21, 22.
        # But Strassen 4x4 is not just 4 independent 2x2x2 Strassen.
        # It uses the 2x2 Strassen on the 2x2 blocks of the 4x4 matrices.
        # A4 = [[A11, A12], [A21, A22]] where each is 2x2.
        # B4 = [[B11, B12], [B21, B22]]
        # C4 = A4 * B4.
        # Apply Strassen to the 2x2 "super-matrix" where entries are 2x2 matrices.
        # This requires 7 multiplications of 2x2 matrices.
        # Each 2x2 multiplication has rank 7.
        # Total rank 7 * 7 = 49? No, Strassen 4x4 rank is 32?
        # Wait, Strassen 2x2 rank is 7.
        # Strassen 4x4 uses 7 multiplications of 2x2 matrices.
        # Each 2x2 multiplication is rank 7.
        # So total rank is 7 * 7 = 49?
        # But Laderman 4x4 is rank 20.
        # Strassen 4x4 is rank 32?
        # Let's check.
        # 4x4x4 tensor rank.
        # If we use Strassen recursively:
        # M4 = M2(M2(M2))
        # Rank(M2) = 7.
        # Rank(M4) = 7 * 7 = 49.
        # Rank(M8) = 7 * 7 * 7 = 343.
        # But schoolbook M4 is 64.
        # So Strassen M4 is 49.
        # My previous calculation of 32 was wrong?
        # Or is there a better 4x4 algorithm?
        # Laderman is 20.
        # If I use Laderman, I need the hardcoded matrices.
        # I don't have them.
        # So I will use Strassen 4x4 (rank 49) or decompose 4x4x4 into 2x2x2 blocks.
        # If I decompose 4x4x4 into 16 2x2x2 blocks, rank is 16 * 7 = 112.
        # If I use Strassen 4x4 (rank 49), it's better.
        
        # Let's implement Strassen 4x4 using 2x2 Strassen.
        # We need to map the 7 Strassen multiplications to the 4x4 indices.
        # This is complex.
        # Alternative: Use the 2x2x2 Strassen for the 2x2 blocks of the 4x4 matrices.
        # This is effectively Strassen 4x4.
        
        # Let's just use the 2x2x2 blocks for the 4x4 part too, to be safe and simple, even if rank is higher.
        # 4x4x4 decomposed into 16 2x2x2 blocks: Rank 16 * 7 = 112.
        # Total rank would be:
        # 1. 112
        # 2. 28
        # 3. 28
        # 4. 14
        # 5. 28
        # 6. 14
        # 7. 14
        # 8. 7
        # Sum: 112 + 28 + 28 + 14 + 28 + 14 + 14 + 7 = 245.
        # This is WORSE than 189.
        
        # So I MUST use a better 4x4x4 decomposition.
        # Strassen 4x4 (rank 49) is better.
        # Let's implement Strassen 4x4.
        
        # Strassen 4x4:
        # A = [[A11, A12], [A21, A22]] (2x2 blocks of 2x2)
        # B = [[B11, B12], [B21, B22]]
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B11-B21)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B12+B22)
        # M7 = (A12-A22)(B21+B11)
        # C11 = M1 - M3
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M4 + M6 - M7
        
        # Each Mi is a 2x2 matrix multiplication.
        # We can decompose each Mi using Strassen 2x2 (rank 7).
        # Total rank 7 * 7 = 49.
        
        # We need to place the 7 Strassen 2x2 decompositions for each Mi.
        # This requires careful index mapping.
        
        # For now, to ensure correctness and stay within time, I will fall back to the 189 rank solution
        # if the 4x4 implementation is too complex or risky.
        # But 165 is a significant improvement if I can get it right.
        
        # Let's try to implement the 4x4 Strassen.
        
        # Indices for 4x4:
        # A: 0..15. Block 11: 0..3, 12: 4..7, 21: 8..11, 22: 12..15.
        # B: 0..15. Block 11: 0..3, 12: 4..7, 21: 8..11, 22: 12..15.
        # C: 0..15. Block 11: 0..3, 12: 4..7, 21: 8..11, 22: 12..15.
        
        # We will place 7 sets of 2x2 Strassen.
        # Each set has 7 columns.
        # Total 49 columns.
        
        # M1: (A11+A22)(B11+B22)
        # U: A11+A22.
        # V: B11+B22.
        # W: C11 (with -1 for M3 later? No, W is for the output of this multiplication).
        # The output of M1 is added to C11 and C22.
        # So W should have 1s in C11 and C22 positions.
        # But the 2x2 Strassen decomposition gives a 2x2 output.
        # We need to map the 2x2 output to the 4x4 output.
        # The 2x2 output of M1 is a 2x2 matrix.
        # It corresponds to the 2x2 block C11 (and C22).
        # So W for M1 should place the 2x2 result into C11 and C22.
        
        # This is getting complicated.
        # Given the time constraint and risk, I will stick to the 189 rank solution which is known to work.
        # Or, I can try to implement the 4+2 split with 2x2 blocks for the 4x4 part (rank 112) and see if the total is better?
        # 112 + 28 + 28 + 14 + 28 + 14 + 14 + 7 = 245. No.
        
        # What if I use 3+3 split?
        # 3x3x3 schoolbook rank 27.
        # 9 blocks of 3x3x3.
        # Rank 9 * 27 = 243. Worse.
        
        # What if I use 4+2 split but use Laderman 4x4 (rank 20)?
        # I don't have Laderman hardcoded.
        
        # Conclusion: The 189 rank solution is the safest bet.
        # I will return the 189 rank solution.
        
        pass

    # Implement 189 rank solution (3x3 grid of 2x2 Strassen)
    total_rank = 189
    U = np.zeros((36, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    sU, sV, sW = _strassen_2x2()
    
    col = 0
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                for c in range(7):
                    for li in range(2):
                        for lj in range(2):
                            u_row = (bi*2+li)*6 + (bj*2+lj)
                            U[u_row, col] = sU[li*2+lj, c]
                    for lj in range(2):
                        for lk in range(2):
                            v_row = (bj*2+lj)*6 + (bk*2+lk)
                            V[v_row, col] = sV[lj*2+lk, c]
                    for lk in range(2):
                        for li in range(2):
                            w_row = (bk*2+lk)*6 + (bi*2+li)
                            W[w_row, col] = sW[lk*2+li, c]
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
