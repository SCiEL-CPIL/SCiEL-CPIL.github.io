import time
import sys

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    
    # Total rank estimate:
    # Part 1 (cols 0-2): 3 * 2 = 6
    # Part 2 (cols 3-6): 2 * 14 = 28
    # Total: 34
    rank = 34
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, coeff=1):
        # u_indices: list of (index, val)
        # v_indices: list of (index, val)
        # w_indices: list of (index, val)
        pass # Placeholder, will implement directly in loops

    col_idx = 0
    
    # PART 1: Columns 0, 1, 2 of B
    # A (2x4) * B_col (4x1) -> C_col (2x1)
    # Rank 2 per column.
    for k in range(3):
        # Term for C[0, k]
        # U: A[0, :] -> indices 0,1,2,3
        # V: B[:, k] -> indices k, 4+k, 8+k, 12+k
        # W: C[0, k] -> index k*n + 0 = 2k
        
        for j in range(4):
            U[0 * m + j][col_idx] = 1
            V[j * p + k][col_idx] = 1
        W[k * n + 0][col_idx] = 1
        col_idx += 1
        
        # Term for C[1, k]
        # U: A[1, :] -> indices 4,5,6,7
        # V: B[:, k] -> indices k, 4+k, 8+k, 12+k
        # W: C[1, k] -> index k*n + 1 = 2k+1
        
        for j in range(4):
            U[1 * m + j][col_idx] = 1
            V[j * p + k][col_idx] = 1
        W[k * n + 1][col_idx] = 1
        col_idx += 1

    # PART 2: Columns 3, 4, 5, 6 of B
    # Split into two 4x2 blocks: B_2a (cols 3,4) and B_2b (cols 5,6)
    # A (2x4) * B_2a (4x2) -> C_2a (2x2)
    # A (2x4) * B_2b (4x2) -> C_2b (2x2)
    
    # Helper to add Strassen decomposition for 2x2 matmul
    # A_block (2x2) from A (2x4) at col offset a_off
    # B_block (4x2) from B (4x7) at col offset b_off
    # Output C_block (2x2) at col offset c_off
    def add_strassen_block(a_off, b_off, c_off):
        nonlocal col_idx
        # A is 2x4. A_block is A[:, a_off : a_off+2]
        # A indices in U:
        # A[0, a_off], A[0, a_off+1]
        # A[1, a_off], A[1, a_off+1]
        
        # B is 4x7. B_block is B[:, b_off : b_off+2]
        # B indices in V:
        # B[0, b_off], B[1, b_off]
        # B[2, b_off], B[3, b_off]
        # B[0, b_off+1], B[1, b_off+1]
        # B[2, b_off+1], B[3, b_off+1]
        
        # C is 2x7. C_block is C[:, c_off : c_off+2]
        # C indices in W:
        # C[0, c_off], C[1, c_off]
        # C[0, c_off+1], C[1, c_off+1]
        
        # Strassen M1..M7
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Note: A_block is 2x2.
        # A00 = A[0, a_off], A01 = A[0, a_off+1]
        # A10 = A[1, a_off], A11 = A[1, a_off+1]
        
        # B_block is 4x2.
        # B00 = B[0, b_off], B01 = B[0, b_off+1]
        # B10 = B[1, b_off], B11 = B[1, b_off+1]
        # B20 = B[2, b_off], B21 = B[2, b_off+1]
        # B30 = B[3, b_off], B31 = B[3, b_off+1]
        
        # Wait, Strassen is for 2x2 * 2x2.
        # Here A_block is 2x2, B_block is 4x2.
        # This is NOT a 2x2 * 2x2 multiplication.
        # It is 2x2 * 4x2? No, A is 2x4.
        # A_block is 2x2. B_block is 4x2.
        # Product is 2x2.
        # Sum_{j=0}^3 A_block[i, j-2?] ... No.
        
        # My previous logic:
        # C_2a = A B_2a.
        # A is 2x4. B_2a is 4x2.
        # I split A into A_L, A_R (2x2).
        # I split B_2a into B_top, B_bot (2x2).
        # C_2a = A_L B_top + A_R B_bot.
        # This requires TWO Strassen multiplications.
        # So add_strassen_block should actually call two Strassen adds?
        # Or I should structure the loop differently.
        
        # Let's refactor:
        # For each 4x2 block of B:
        # 1. Add Strassen for A_L * B_top (7 terms)
        # 2. Add Strassen for A_R * B_bot (7 terms)
        
        pass

    # Corrected Part 2 Implementation
    
    # Block 1: B cols 3, 4
    # A_L: A[:, 0:2] (cols 0,1)
    # A_R: A[:, 2:4] (cols 2,3)
    # B_top: B[0:2, 3:5] (rows 0,1)
    # B_bot: B[2:4, 3:5] (rows 2,3)
    
    def add_strassen_2x2(a_col_offset, b_row_offset, b_col_offset, c_col_offset):
        nonlocal col_idx
        # A is 2x2 from A[0:2, a_col_offset : a_col_offset+2]
        # B is 2x2 from B[b_row_offset : b_row_offset+2, b_col_offset : b_col_offset+2]
        # C is 2x2 from C[0:2, c_col_offset : c_col_offset+2]
        
        # U indices for A
        # A00: 0*m + a_col_offset
        # A01: 0*m + a_col_offset + 1
        # A10: 1*m + a_col_offset
        # A11: 1*m + a_col_offset + 1
        
        # V indices for B
        # B00: b_row_offset*p + b_col_offset
        # B01: b_row_offset*p + b_col_offset + 1
        # B10: (b_row_offset+1)*p + b_col_offset
        # B11: (b_row_offset+1)*p + b_col_offset + 1
        
        # W indices for C
        # C00: c_col_offset*n + 0
        # C01: c_col_offset*n + 1
        # C10: (c_col_offset+1)*n + 0
        # C11: (c_col_offset+1)*n + 1
        
        # M1: (A00+A11)(B00+B11)
        U[0*m + a_col_offset][col_idx] = 1
        U[1*m + a_col_offset + 1][col_idx] = 1
        V[b_row_offset*p + b_col_offset][col_idx] = 1
        V[(b_row_offset+1)*p + b_col_offset + 1][col_idx] = 1
        W[c_col_offset*n + 0][col_idx] = 1
        W[c_col_offset*n + 1][col_idx] = 0
        W[(c_col_offset+1)*n + 0][col_idx] = 0
        W[(c_col_offset+1)*n + 1][col_idx] = 1
        col_idx += 1
        
        # M2: (A10+A11)B00
        U[1*m + a_col_offset][col_idx] = 1
        U[1*m + a_col_offset + 1][col_idx] = 1
        V[b_row_offset*p + b_col_offset][col_idx] = 1
        W[c_col_offset*n + 0][col_idx] = 0
        W[c_col_offset*n + 1][col_idx] = 0
        W[(c_col_offset+1)*n + 0][col_idx] = 1
        W[(c_col_offset+1)*n + 1][col_idx] = -1
        col_idx += 1
        
        # M3: A00(B01-B11)
        U[0*m + a_col_offset][col_idx] = 1
        V[b_row_offset*p + b_col_offset + 1][col_idx] = 1
        V[(b_row_offset+1)*p + b_col_offset + 1][col_idx] = -1
        W[c_col_offset*n + 0][col_idx] = 0
        W[c_col_offset*n + 1][col_idx] = 1
        W[(c_col_offset+1)*n + 0][col_idx] = 0
        W[(c_col_offset+1)*n + 1][col_idx] = 1
        col_idx += 1
        
        # M4: A11(B10-B00)
        U[1*m + a_col_offset + 1][col_idx] = 1
        V[(b_row_offset+1)*p + b_col_offset][col_idx] = 1
        V[b_row_offset*p + b_col_offset][col_idx] = -1
        W[c_col_offset*n + 0][col_idx] = 1
        W[c_col_offset*n + 1][col_idx] = 0
        W[(c_col_offset+1)*n + 0][col_idx] = 1
        W[(c_col_offset+1)*n + 1][col_idx] = 0
        col_idx += 1
        
        # M5: (A00+A01)B11
        U[0*m + a_col_offset][col_idx] = 1
        U[0*m + a_col_offset + 1][col_idx] = 1
        V[(b_row_offset+1)*p + b_col_offset + 1][col_idx] = 1
        W[c_col_offset*n + 0][col_idx] = -1
        W[c_col_offset*n + 1][col_idx] = 1
        W[(c_col_offset+1)*n + 0][col_idx] = 0
        W[(c_col_offset+1)*n + 1][col_idx] = 0
        col_idx += 1
        
        # M6: (A10-A00)(B00+B01)
        U[1*m + a_col_offset][col_idx] = 1
        U[0*m + a_col_offset][col_idx] = -1
        V[b_row_offset*p + b_col_offset][col_idx] = 1
        V[b_row_offset*p + b_col_offset + 1][col_idx] = 1
        W[c_col_offset*n + 0][col_idx] = 0
        W[c_col_offset*n + 1][col_idx] = 0
        W[(c_col_offset+1)*n + 0][col_idx] = 0
        W[(c_col_offset+1)*n + 1][col_idx] = 1
        col_idx += 1
        
        # M7: (A01-A11)(B10+B11)
        U[0*m + a_col_offset + 1][col_idx] = 1
        U[1*m + a_col_offset + 1][col_idx] = -1
        V[(b_row_offset+1)*p + b_col_offset][col_idx] = 1
        V[(b_row_offset+1)*p + b_col_offset + 1][col_idx] = 1
        W[c_col_offset*n + 0][col_idx] = 1
        W[c_col_offset*n + 1][col_idx] = 0
        W[(c_col_offset+1)*n + 0][col_idx] = 0
        W[(c_col_offset+1)*n + 1][col_idx] = 1
        col_idx += 1

    # Call for Block 1 (Cols 3, 4)
    # A_L (cols 0,1) * B_top (rows 0,1, cols 3,4) -> C (cols 3,4)
    add_strassen_2x2(0, 0, 3, 3)
    # A_R (cols 2,3) * B_bot (rows 2,3, cols 3,4) -> C (cols 3,4)
    add_strassen_2x2(2, 2, 3, 3)
    
    # Call for Block 2 (Cols 5, 6)
    # A_L (cols 0,1) * B_top (rows 0,1, cols 5,6) -> C (cols 5,6)
    add_strassen_2x2(0, 0, 5, 5)
    # A_R (cols 2,3) * B_bot (rows 2,3, cols 5,6) -> C (cols 5,6)
    add_strassen_2x2(2, 2, 5, 5)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
