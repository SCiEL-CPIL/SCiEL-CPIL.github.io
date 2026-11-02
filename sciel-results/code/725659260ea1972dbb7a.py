"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_sb
    
    # Try to find lower rank via structured decomposition
    # Strategy: split into sub-problems and use known low-rank decompositions
    
    # For 4x4x6, try splitting the 6 into 3+3, giving two 4x4x3 problems
    # Or split 4 into 2+2, giving four 2x4x6 problems
    # Or use Strassen-like block decomposition
    
    # Let's try: split A (4x4) into 2x2 blocks, B (4x6) into 2x6 blocks
    # C = A*B where A = [[A11,A12],[A21,A22]] (each 2x2), B = [[B1],[B2]] (each 2x6)
    # C1 = A11*B1 + A12*B2 (2x6)
    # C2 = A21*B1 + A22*B2 (2x6)
    
    # Each 2x2x6 multiplication can use Strassen-like decomposition
    # 2x2x2 has rank 7, so 2x2x6 = 2x2x(2+2+2) could use 7*3=21 per block
    # Total: 4 blocks * 21 = 84, worse than 96? No, 84 < 96!
    
    # Actually, let's think more carefully.
    # A is 4x4, B is 4x6.
    # Split A into four 2x2 blocks: A11, A12, A21, A22
    # Split B into two 2x6 blocks: B_top (rows 0-1), B_bot (rows 2-3)
    # C_top = A11*B_top + A12*B_bot (2x6)
    # C_bot = A21*B_top + A22*B_bot (2x6)
    
    # For each 2x2x6 multiplication, we can use Strassen on the 2x2 part
    # 2x2x2 Strassen has rank 7
    # 2x2x6 = doing 2x2x2 three times (splitting 6 into 2+2+2)
    # Each 2x2x2 needs rank 7, so 2x2x6 needs rank 21
    # Two such multiplications: 42 total? But they share B_top and B_bot
    
    # Actually, let's use a different approach:
    # Use the fact that 2x2x6 can be decomposed with rank 14 (using 2 Strassen steps)
    # Or just use schoolbook for 2x2x6: rank 24
    
    # Let me try a cleaner approach: use known low-rank decompositions
    
    # Rank of 2x2x2 is 7 (Strassen)
    # For 4x4x4, Strassen gives rank 7^2 = 49
    # For 4x4x6, we could do 4x4x4 (rank 49) + 4x4x2 (rank ?)
    # 4x4x2: split 4x4 into Strassen (rank 49 for 4x4x4), but for 4x4x2...
    # 4x4x2 = (2x2+2x2)x(2x2+2x2)x(2x2) ... complex
    
    # Let me just try to find a rank < 96 decomposition through search
    # with integer/half-integer entries
    
    # Try rank 84: split into 4 blocks of 2x2x6, each with rank 21
    # 2x2x6: use Strassen for each 2x2x2 sub-block
    # 2x2x6 = multiply 2x2 by 2x6 = three 2x2x2 multiplications
    # Each 2x2x2 has rank 7, so 2x2x6 has rank 21
    # Four such: 4 * 21 = 84
    
    # Let me construct this explicitly
    rank_84 = 84
    U84 = [[0] * rank_84 for _ in range(n * m)]  # 16 x 84
    V84 = [[0] * rank_84 for _ in range(m * p)]  # 24 x 84
    W84 = [[0] * rank_84 for _ in range(p * n)]  # 24 x 84
    
    # Strassen decomposition for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 2x2x6, split B into three 2x2 blocks: B = [B1, B2, B3]
    # C = A*B = A*B1 + A*B2 + A*B3
    # Each A*Bi is 2x2x2, rank 7
    # Total rank for 2x2x6: 21
    
    # Now for 4x4x6:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B_top], [B_bot]] where B_top, B_bot are 2x6
    # C_top = A11*B_top + A12*B_bot
    # C_bot = A21*B_top + A22*B_bot
    
    # For A11*B_top: 2x2x6, rank 21
    # For A12*B_bot: 2x2x6, rank 21
    # For A21*B_top: 2x2x6, rank 21
    # For A22*B_bot: 2x2x6, rank 21
    # Total: 84
    
    # Let me build the Strassen-based 2x2x6 decomposition
    # Strassen for 2x2x2 with A (2x2), B (2x2):
    # U1 = A11 + A22, V1 = B11 + B22, W1 = C11 + C22 (contribution)
    # U2 = A21 + A22, V2 = B11, W2 = C21
    # U3 = A11, V3 = B21 - B22, W3 = C12
    # U4 = A22, V4 = B11 - B21, W4 = C11
    # U5 = A11 + A12, V5 = B22, W5 = C12
    # U6 = A21 - A11, V6 = B11 + B12, W6 = C22
    # U7 = A12 - A22, V7 = B21 + B22, W7 = C11
    
    # More precisely, for 2x2x2:
    # The 7 rank-1 terms are:
    # (A11+A22, B11+B22) -> contributes to C11 and C22
    # (A21+A22, B11) -> contributes to C21
    # (A11, B21-B22) -> contributes to C12
    # (A22, B11-B21) -> contributes to C11
    # (A11+A12, B22) -> contributes to C12
    # (A21-A11, B11+B12) -> contributes to C22
    # (A12-A22, B21+B22) -> contributes to C11
    
    # For 2x2x6, A is 2x2, B is 2x6 = [B1, B2, B3] where each Bi is 2x2
    # C = A*B1 + A*B2 + A*B3, each Ai*Bj is 2x2
    # Use Strassen for each pair: 7 terms per pair, 21 total
    
    # Let me index:
    # A indices: A[0]=A11, A[1]=A12, A[2]=A21, A[3]=A22
    # B indices for 2x6: B[0..5] = B11,B12,B21,B22 for B1, then B11,B12,B21,B22 for B2, etc.
    # Actually B is 2x6, so B[row,col] for row in {0,1}, col in {0..5}
    # B_top_half: B[0,0:2], B[1,0:2] for first 2x2, etc.
    
    # This is getting complex. Let me just implement it carefully.
    
    def strassen_2x2x6_decomp(base_rank):
        """
        Returns (U, V, W) for 2x2x6 matrix multiplication with rank 21.
        U: 4 x 21 (for A: 2x2 matrix, indexed as [A11, A12, A21, A22])
        V: 12 x 21 (for B: 2x6 matrix, indexed as [B00,B01,B02,B03,B04,B05, B10,B11,B12,B13,B14,B15])
        W: 12 x 21 (for C: 2x6 matrix, indexed as [C00,C01,C02,C03,C04,C05, C10,C11,C12,C13,C14,C15])
        """
        U = [[0] * 21 for _ in range(4)]
        V = [[0] * 21 for _ in range(12)]
        W = [[0] * 21 for _ in range(12)]
        
        # For each of the 3 2x2 blocks of B, apply Strassen
        for block in range(3):
            b_offset = block * 2  # column offset in B
            c_offset = block * 2  # column offset in C
            rank_offset = block * 7  # rank offset
            
            # Strassen 7 terms for this 2x2x2 block
            # Term 0: (A11+A22) * (B11+B22) -> C11 += 1, C22 += 1
            U[0][rank_offset + 0] = 1  # A11
            U[3][rank_offset + 0] = 1  # A22
            V[0 + b_offset][rank_offset + 0] = 1  # B11 (row 0, col b_offset)
            V[1 + b_offset][rank_offset + 0] = 1  # B22 (row 1, col b_offset+1)
            W[0 + c_offset][rank_offset + 0] = 1  # C11
            W[3 + c_offset][rank_offset + 0] = 1  # C22
            
            # Term 1: (A21+A22) * B11 -> C21 += 1
            U[2][rank_offset + 1] = 1  # A21
            U[3][rank_offset + 1] = 1  # A22
            V[0 + b_offset][rank_offset + 1] = 1  # B11
            W[2 + c_offset][rank_offset + 1] = 1  # C21
            
            # Term 2: A11 * (B21-B22) -> C12 += 1
            U[0][rank_offset + 2] = 1  # A11
            V[1 + b_offset][rank_offset + 2] = 1  # B21 (row 1, col b_offset)
            V[1 + b_offset + 1][rank_offset + 2] = -1  # -B22 (row 1, col b_offset+1)
            W[1 + c_offset][rank_offset + 2] = 1  # C12
            
            # Term 3: A22 * (B11-B21) -> C11 += 1
            U[3][rank_offset + 3] = 1  # A22
            V[0 + b_offset][rank_offset + 3] = 1  # B11
            V[1 + b_offset][rank_offset + 3] = -1  # -B21
            W[0 + c_offset][rank_offset + 3] = 1  # C11
            
            # Term 4: (A11+A12) * B22 -> C12 += 1
            U[0][rank_offset + 4] = 1  # A11
            U[1][rank_offset + 4] = 1  # A12
            V[1 + b_offset + 1][rank_offset + 4] = 1  # B22
            W[1 + c_offset][rank_offset + 4] = 1  # C12
            
            # Term 5: (A21-A11) * (B11+B12) -> C22 += 1
            U[2][rank_offset + 5] = 1  # A21
            U[0][rank_offset + 5] = -1  # -A11
            V[0 + b_offset][rank_offset + 5] = 1  # B11
            V[0 + b_offset + 1][rank_offset + 5] = 1  # B12
            W[3 + c_offset][rank_offset + 5] = 1  # C22
            
            # Term 6: (A12-A22) * (B21+B22) -> C11 += 1
            U[1][rank_offset + 6] = 1  # A12
            U[3][rank_offset + 6] = -1  # -A22
            V[1 + b_offset][rank_offset + 6] = 1  # B21
            V[1 + b_offset + 1][rank_offset + 6] = 1  # B22
            W[0 + c_offset][rank_offset + 6] = 1  # C11
        
        return U, V, W
    
    # Now assemble 4x4x6 from four 2x2x6 decompositions
    # A (4x4) indexed as A[i,j] for i,j in 0..3
    # A11 = A[0,0], A[0,1], A[1,0], A[1,1] -> local indices 0,1,2,3
    # A12 = A[0,2], A[0,3], A[1,2], A[1,3] -> local indices 0,1,2,3
    # A21 = A[2,0], A[2,1], A[3,0], A[3,1] -> local indices 0,1,2,3
    # A22 = A[2,2], A[2,3], A[3,2], A[3,3] -> local indices 0,1,2,3
    
    # B (4x6) indexed as B[i,j] for i in 0..3, j in 0..5
    # B_top = B[0,0:6], B[1,0:6] -> 2x6
    # B_bot = B[2,0:6], B[3,0:6] -> 2x6
    
    # C (4x6) indexed as C[i,j] for i in 0..3, j in 0..5
    # C_top = C[0,0:6], C[1,0:6] -> 2x6
    # C_bot = C[2,0:6], C[3,0:6] -> 2x6
    
    # C_top = A11 * B_top + A12 * B_bot
    # C_bot = A21 * B_top + A22 * B_bot
    
    # Four 2x2x6 decompositions:
    # 1. A11 * B_top -> C_top (rank 21)
    # 2. A12 * B_bot -> C_top (rank 21)
    # 3. A21 * B_top -> C_bot (rank 21)
    # 4. A22 * B_bot -> C_bot (rank 21)
    
    U_full = [[0] * 84 for _ in range(16)]  # 4x4 = 16 entries
    V_full = [[0] * 84 for _ in range(24)]  # 4x6 = 24 entries
    W_full = [[0] * 84 for _ in range(24)]  # 6x4 = 24 entries (W[k*n+i])
    
    # Helper to get 2x2x6 decomp and place it
    def place_2x2x6(a_row_offset, a_col_offset, b_row_offset, c_row_offset, rank_offset):
        """
        Place a 2x2x6 decomposition.
        a_row_offset, a_col_offset: top-left corner of A's 2x2 block in the 4x4 matrix
        b_row_offset: top row of B's 2x6 block in the 4x6 matrix
        c_row_offset: top row of C's 2x6 block in the 4x6 matrix
        rank_offset: starting rank index
        """
        U_sub, V_sub, W_sub = strassen_2x2x6_decomp(rank_offset)
        
        # Map A local indices to global
        # Local A: [A11, A12, A21, A22] = indices 0,1,2,3
        # Global A index: (a_row_offset + dr) * 4 + (a_col_offset + dc)
        a_map = {0: (a_row_offset, a_col_offset),       # A11
                 1: (a_row_offset, a_col_offset + 1),   # A12
                 2: (a_row_offset + 1, a_col_offset),   # A21
                 3: (a_row_offset + 1, a_col_offset + 1)} # A22
        
        for local_idx in range(4):
            for r in range(21):
                if U_sub[local_idx][r] != 0:
                    gr, gc = a_map[local_idx]
                    global_a_idx = gr * 4 + gc
                    U_full[global_a_idx][rank_offset + r] = U_sub[local_idx][r]
        
        # Map B local indices to global
        # Local B: 2x6, index = row * 6 + col, row in 0,1, col in 0..5
        for local_idx in range(12):
            for r in range(21):
                if V_sub[local_idx][r] != 0:
                    local_row = local_idx // 6
                    local_col = local_idx % 6
                    global_b_idx = (b_row_offset + local_row) * 6 + local_col
                    V_full[global_b_idx][rank_offset + r] = V_sub[local_idx][r]
        
        # Map C local indices to global W
        # Local C: 2x6, but W is indexed as W[k*n+i] = W[col*4 + row]
        # C[row, col] corresponds to W[col * 4 + row]
        for local_idx in range(12):
            for r in range(21):
                if W_sub[local_idx][r] != 0:
                    local_row = local_idx // 6
                    local_col = local_idx % 6
                    global_row = c_row_offset + local_row
                    global_col = local_col
                    # W index: col * n + row = global_col * 4 + global_row
                    global_w_idx = global_col * 4 + global_row
                    W_full[global_w_idx][rank_offset + r] = W_sub[local_idx][r]
    
    # Place the four decompositions
    # 1. A11 * B_top -> C_top: A rows 0-1, cols 0-1; B rows 0-1; C rows 0-1
    place_2x2x6(0, 0, 0, 0, 0)
    
    # 2. A12 * B_bot -> C_top: A rows 0-1, cols 2-3; B rows 2-3; C rows 0-1
    place_2x2x6(0, 2, 2, 0, 21)
    
    # 3. A21 * B_top -> C_bot: A rows 2-3, cols 0-1; B rows 0-1; C rows 2-3
    place_2x2x6(2, 0, 0, 2, 42)
    
    # 4. A22 * B_bot -> C_bot: A rows 2-3, cols 2-3; B rows 2-3; C rows 2-3
    place_2x2x6(2, 2, 2, 2, 63)
    
    # Verify the decomposition
    U_arr = np.array(U_full, dtype=float)
    V_arr = np.array(V_full, dtype=float)
    W_arr = np.array(W_full, dtype=float)
    
    # Check: sum_c U[a,c] * V[b,c] * W[c_idx,c] should give C
    # Where a = i*4+j, b = j*6+k, c_idx = k*4+i
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    error = 0
    for i in range(4):
        for k in range(6):
            c_val = 0
            for c in range(84):
                for j in range(4):
                    c_val += U_arr[i*4+j, c] * V_arr[j*6+k, c]
            # c_val should be C[i,k] = sum_j A[i,j]*B[j,k]
            # But we're checking the tensor decomposition, so:
            # sum_c U[a,c]*V[b,c]*W[wc,c] = 1 if a,b,wc correspond to valid (i,j,k)
            # Actually, the tensor T[a,b,wc] = 1 if a=i*m+j, b=j*p+k, wc=k*n+i
            # And sum_c U[a,c]*V[b,c]*W[wc,c] = T[a,b,wc]
            
            # Let me verify differently
            pass
    
    # Better verification:
    # T[i,j,k] = 1, where a = i*4+j, b = j*6+k, wc = k*4+i
    # sum_c U[a,c] * V[b,c] * W[wc,c] should equal 1 for valid (i,j,k) and 0 otherwise
    
    max_err = 0
    for i in range(4):
        for j in range(4):
            for k in range(6):
                a = i * 4 + j
                b = j * 6 + k
                wc = k * 4 + i
                val = 0
                for c in range(84):
                    val += U_arr[a, c] * V_arr[b, c] * W_arr[wc, c]
                if abs(val - 1) > 1e-9:
                    max_err = max(max_err, abs(val - 1))
    
    # Check invalid entries are 0
    for a in range(16):
        for b in range(24):
            for wc in range(24):
                # Check if this is a valid (i,j,k) triple
                i = a // 4
                j = a % 4
                j2 = b // 6
                k = b % 6
                k2 = wc // 4
                i2 = wc % 4
                if j != j2 or k != k2 or i != i2:
                    val = 0
                    for c in range(84):
                        val += U_arr[a, c] * V_arr[b, c] * W_arr[wc, c]
                    if abs(val) > 1e-9:
                        max_err = max(max_err, abs(val))
    
    if max_err < 1e-9:
        # Check entries are integer or half-integer
        valid_entries = True
        for mat in [U_arr, V_arr, W_arr]:
            for val in mat.flat:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    valid_entries = False
                    break
            if not valid_entries:
                break
        
        if valid_entries:
            # Convert to proper format
            def to_list(arr):
                result = []
                for row in arr:
                    row_list = []
                    for val in row:
                        if abs(val - round(val)) < 1e-9:
                            row_list.append(int(round(val)))
                        elif abs(val * 2 - round(val * 2)) < 1e-9:
                            row_list.append(round(val * 2) / 2)
                        else:
                            row_list.append(val)
                    result.append(row_list)
                return result
            
            return {"U": to_list(U_arr), "V": to_list(V_arr), "W": to_list(W_arr)}
    
    # If verification failed, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
