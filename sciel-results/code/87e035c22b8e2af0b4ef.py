import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = 2, 4, 8
    
    # Rank 56 decomposition
    # Based on decomposing into 8 independent 2x2x2 Strassen multiplications?
    # Or 2x(2x4x4) where 2x4x4 is 14 rank?
    # 2 * 14 = 28? No, 2x4x4 rank is likely 14?
    # Let's check: R(2,4,4).
    # A(2x4) * B(4x4) -> C(2x4).
    # Split A into two 2x2. Split B into two 2x4.
    # C = A1 B1 + A2 B2.
    # R(2,2,4) = 14.
    # So R(2,4,4) <= 28.
    # Then R(2,4,8) <= 2 * 28 = 56.
    # So 56 is consistent.
    
    # We will construct the 56 rank solution.
    # The current code uses a specific pattern. I will replicate the logic but cleaner.
    
    # Strassen 2x2x2 decomposition
    # M1 = (A11 - A22)(B22 + B11)
    # M2 = (A21 + A22)(B11)
    # M3 = (A11)(B12 - B22)
    # M4 = (A22)(B21 - B11)
    # M5 = (A11 + A22)(B22)
    # M6 = (A12 - A11)(B11 + B21)
    # M7 = (A21 - A22)(B12 + B22)
    
    # This decomposes 2x2 * 2x2.
    # We need to map 2x4 * 4x8 to this.
    
    # Let's use the block structure:
    # A = [A00 A01] (2x4) -> A00, A01 are 2x2
    # B = [B00; B01; B10; B11] (4x8) -> B00, B01, B10, B11 are 2x4? No.
    # B is 4x8. Split rows into 2 groups of 2.
    # B = [B_top; B_bot] where B_top, B_bot are 2x8.
    # Split B_top, B_bot into 4 columns of 2?
    # B_top = [B00 B01] (2x4)
    # B_bot = [B10 B11] (2x4)
    # Then C = A00 B00 + A01 B10 + A00 B01 + A01 B11 ?
    # C is 2x8.
    # C = [C00 C01] where C00, C01 are 2x4.
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    
    # This is 4 terms of 2x2 * 2x4.
    # R(2,2,4) = 14.
    # So total rank = 4 * 14 = 56.
    
    # We need a rank-14 decomposition for 2x2 * 2x4.
    # Let X be 2x2, Y be 2x4. Z = X Y (2x4).
    # Split Y into Y = [Y_L | Y_R] (2x2 each).
    # Z = [X Y_L | X Y_R].
    # This is two 2x2 * 2x2 multiplications.
    # Rank = 7 + 7 = 14.
    
    # So the total rank is 4 * 14 = 56.
    # The structure is:
    # 4 groups (corresponding to the 4 terms in C00, C01)
    # Each group uses a 2x2x2 Strassen decomposition (rank 7) for the left part and right part?
    # Wait, X Y_L is 2x2 * 2x2. Rank 7.
    # X Y_R is 2x2 * 2x2. Rank 7.
    # So for one X (2x2) and Y (2x4), we need 14 rank.
    # We have 4 such pairs (A00,B00), (A01,B10), (A00,B01), (A01,B11).
    # Total 4 * 14 = 56.
    
    # Implementation:
    # U, V, W shapes: (8, 56), (32, 56), (16, 56).
    # n=2, m=4, p=8.
    # U indices: i*m + j. i in 0..1, j in 0..3.
    # V indices: j*p + k. j in 0..3, k in 0..7.
    # W indices: k*n + i. k in 0..7, i in 0..1.
    
    rank = 56
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00 - A11)(B11 + B00)
    # M2 = (A10 + A11)(B00)
    # M3 = (A00)(B01 - B11)
    # M4 = (A11)(B10 - B00)
    # M5 = (A00 + A11)(B11)
    # M6 = (A01 - A00)(B00 + B10)
    # M7 = (A10 - A11)(B01 + B11)
    
    # We will iterate over the 4 blocks of the product.
    # Block 0: A00 * B00
    # Block 1: A01 * B10
    # Block 2: A00 * B01
    # Block 3: A01 * B11
    
    # For each block, we perform a 2x2 * 2x4 multiplication.
    # Which is two 2x2 * 2x2 multiplications.
    # So 2 * 7 = 14 ranks per block.
    # 4 blocks * 14 = 56.
    
    r = 0
    for block in range(4):
        # Determine which A block and which B block
        # block 0: A00, B00
        # block 1: A01, B10
        # block 2: A00, B01
        # block 3: A01, B11
        
        if block == 0:
            a_row_offset = 0 # A00 is rows 0-1, cols 0-1 of A
            a_col_offset = 0
            b_row_offset = 0 # B00 is rows 0-1, cols 0-1 of B
            b_col_offset = 0
        elif block == 1:
            a_row_offset = 0
            a_col_offset = 2 # A01 is rows 0-1, cols 2-3 of A
            b_row_offset = 2 # B10 is rows 2-3, cols 0-1 of B
            b_col_offset = 0
        elif block == 2:
            a_row_offset = 0
            a_col_offset = 0
            b_row_offset = 0
            b_col_offset = 2 # B01 is rows 0-1, cols 2-3 of B
        else: # block 3
            a_row_offset = 0
            a_col_offset = 2
            b_row_offset = 2
            b_col_offset = 2 # B11 is rows 2-3, cols 2-3 of B
            
        # For this block, we compute X * Y where X is 2x2, Y is 2x4.
        # Y is split into Y_L (2x2) and Y_R (2x2).
        # Y_L corresponds to cols 0-1 of the 2x4 block Y.
        # Y_R corresponds to cols 2-3 of the 2x4 block Y.
        
        # Part 1: X * Y_L
        # This is a 2x2 * 2x2 multiplication.
        # We use Strassen.
        # The output is 2x2.
        # The output indices in C correspond to:
        # C rows: 0-1 (since A rows are 0-1)
        # C cols: b_col_offset to b_col_offset+1
        
        # Part 2: X * Y_R
        # Output is 2x2.
        # C cols: b_col_offset+2 to b_col_offset+3
        
        for part in range(2):
            # part 0: Y_L
            # part 1: Y_R
            
            # Strassen 7 products
            for s in range(7):
                # U entries
                # X is A block.
                # X indices in U: i*m + j
                # i in 0..1, j in a_col_offset .. a_col_offset+1
                
                # V entries
                # Y is B block.
                # Y indices in V: j*p + k
                # j in b_row_offset .. b_row_offset+1
                # k in b_col_offset + part*2 .. b_col_offset + part*2 + 1
                
                # W entries
                # C indices in W: k*n + i
                # k in b_col_offset + part*2 .. b_col_offset + part*2 + 1
                # i in 0..1
                
                # Strassen formulas for M_s
                # We need to map the 2x2 indices (0,1) to local indices.
                
                # Local A indices: (0,0), (0,1), (1,0), (1,1)
                # Local B indices: (0,0), (0,1), (1,0), (1,1)
                # Local C indices: (0,0), (0,1), (1,0), (1,1)
                
                # U global index for local (li, lj) is li*m + (a_col_offset + lj)
                # V global index for local (lj, lk) is (b_row_offset + lj)*p + (b_col_offset + part*2 + lk)
                # W global index for local (lk, li) is (b_col_offset + part*2 + lk)*n + li
                
                # Define Strassen coefficients
                # s=0: M1 = (A00 - A11)(B11 + B00)
                # s=1: M2 = (A10 + A11)(B00)
                # s=2: M3 = (A00)(B01 - B11)
                # s=3: M4 = (A11)(B10 - B00)
                # s=4: M5 = (A00 + A11)(B11)
                # s=5: M6 = (A01 - A00)(B00 + B10)
                # s=6: M7 = (A10 - A11)(B01 + B11)
                
                # We'll just fill U, V, W based on these.
                
                # Helper to add to U
                def add_U(li, lj, val):
                    U[li*m + (a_col_offset + lj), r] += val
                    
                def add_V(lj, lk, val):
                    V[(b_row_offset + lj)*p + (b_col_offset + part*2 + lk), r] += val
                    
                def add_W(lk, li, val):
                    W[(b_col_offset + part*2 + lk)*n + li, r] += val

                if s == 0:
                    # (A00 - A11)
                    add_U(0, 0, 1)
                    add_U(1, 1, -1)
                    # (B11 + B00)
                    add_V(1, 1, 1)
                    add_V(0, 0, 1)
                    # C00 += M1, C11 += M1
                    add_W(0, 0, 1)
                    add_W(1, 1, 1)
                elif s == 1:
                    # (A10 + A11)
                    add_U(1, 0, 1)
                    add_U(1, 1, 1)
                    # B00
                    add_V(0, 0, 1)
                    # C10 += M2, C11 += M2
                    add_W(0, 1, 1)
                    add_W(1, 1, 1)
                elif s == 2:
                    # A00
                    add_U(0, 0, 1)
                    # (B01 - B11)
                    add_V(0, 1, 1)
                    add_V(1, 1, -1)
                    # C01 += M3, C11 -= M3
                    add_W(1, 0, 1)
                    add_W(1, 1, -1)
                elif s == 3:
                    # A11
                    add_U(1, 1, 1)
                    # (B10 - B00)
                    add_V(1, 0, 1)
                    add_V(0, 0, -1)
                    # C10 -= M4, C11 += M4
                    add_W(0, 1, -1)
                    add_W(1, 1, 1)
                elif s == 4:
                    # (A00 + A11)
                    add_U(0, 0, 1)
                    add_U(1, 1, 1)
                    # B11
                    add_V(1, 1, 1)
                    # C11 += M5
                    add_W(1, 1, 1)
                elif s == 5:
                    # (A01 - A00)
                    add_U(0, 1, 1)
                    add_U(0, 0, -1)
                    # (B00 + B10)
                    add_V(0, 0, 1)
                    add_V(1, 0, 1)
                    # C00 += M6, C01 += M6
                    add_W(0, 0, 1)
                    add_W(1, 0, 1)
                elif s == 6:
                    # (A10 - A11)
                    add_U(1, 0, 1)
                    add_U(1, 1, -1)
                    # (B01 + B11)
                    add_V(0, 1, 1)
                    add_V(1, 1, 1)
                    # C10 += M7, C11 += M7
                    add_W(0, 1, 1)
                    add_W(1, 1, 1)
                
                r += 1
                
    # Verify
    U_arr = U
    V_arr = V
    W_arr = W
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r_idx in range(rank):
                    val += U_arr[i*m+j, r_idx] * V_arr[j*p+k, r_idx] * W_arr[k*n+i, r_idx]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-10:
        # Fallback
        rank_naive = n * m * p
        U_naive = np.zeros((n*m, rank_naive))
        V_naive = np.zeros((m*p, rank_naive))
        W_naive = np.zeros((p*n, rank_naive))
        r_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_naive[i*m+j, r_idx] = 1
                    V_naive[j*p+k, r_idx] = 1
                    W_naive[k*n+i, r_idx] = 1
                    r_idx += 1
        return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
