import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """
    Returns U, V, W for multiplying a 2x2 matrix by a 2x2 matrix.
    Rank 7.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indices in U: (i, j) -> i*2+j. i in {0,1}, j in {0,1}.
    Indices in V: (j, k) -> j*2+k. j in {0,1}, k in {0,1}.
    Indices in W: (k, i) -> k*2+i. k in {0,1}, i in {0,1}.
    """
    # Strassen 7 terms
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
    
    # U (A side): rows are A00, A01, A10, A11 (indices 0,1,2,3)
    # V (B side): rows are B00, B01, B10, B11 (indices 0,1,2,3)
    # W (C side): rows are C00, C01, C10, C11 (indices 0,1,2,3) but indexed as k*n+i.
    # Here n=2, so C is 2x2.
    # k is column index of B / row index of C.
    # i is row index of A / column index of C.
    # W index is k*2 + i.
    # C00 -> k=0, i=0 -> 0
    # C01 -> k=1, i=0 -> 2
    # C10 -> k=0, i=1 -> 1
    # C11 -> k=1, i=1 -> 3
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # Term 0: M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C00, C11
    
    # Term 1: M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1 # C10, C11
    
    # Term 2: M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1 # C01, C11
    
    # Term 3: M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1 # C00, C10
    
    # Term 4: M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1 # C00, C01
    
    # Term 5: M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C11
    
    # Term 6: M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C00
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    # We want to multiply A (2x5) and B (5x6).
    # We can view A as a row of 2x2 blocks and one 2x1 block.
    # A = [A00 (2x2), A01 (2x2), A02 (2x1)]
    # B = [B00 (2x2), B10 (2x2), B20 (2x2)] -- wait, B is 5x6.
    # B rows are 0..4.
    # Block 0 of B: rows 0,1 (2x2).
    # Block 1 of B: rows 2,3 (2x2).
    # Block 2 of B: row 4 (1x2).
    # So B = [[B0 (2x2), B1 (2x2), B2 (2x2)]]? No, B is 5x6.
    # Let's split B into vertical strips: B_left (5x2), B_mid (5x2), B_right (5x2).
    # This doesn't help with Strassen on A.
    
    # Let's stick to the hint: "Strassen to all 2x2 sub-blocks and schoolbook to the remaining 2x1x2 strips".
    # A is 2x5. Split into A0 (2x2), A1 (2x2), A2 (2x1).
    # B is 5x6. Split into B0 (2x6), B1 (2x6), B2 (1x6).
    # Product C = A0*B0 + A1*B1 + A2*B2.
    
    # Term 1: A0 (2x2) * B0 (2x6).
    # B0 is 2x6. Split B0 into B00 (2x2), B01 (2x2), B02 (2x2).
    # C1 = A0*B00 + A0*B01 + A0*B02.
    # We can compute A0*B00 using Strassen (rank 7).
    # A0*B01 using Strassen (rank 7).
    # A0*B02 using Strassen (rank 7).
    # Total for A0 part: 21.
    
    # Term 2: A1 (2x2) * B1 (2x6).
    # Similarly, 21.
    
    # Term 3: A2 (2x1) * B2 (1x6).
    # This is outer product of A2 (2x1) and B2 (1x6).
    # Rank 2? No, rank 2*1*6 = 12 schoolbook.
    # But it's just A2[i] * B2[j].
    # C[i, j] = sum_k A2[i,k] * B2[k,j]. Here k is only 0.
    # C[i, j] = A2[i,0] * B2[0,j].
    # This is a rank 2 tensor? No, it's a matrix.
    # The decomposition is sum over c.
    # C[i,j] = U[i,c] V[c,j] ? No, V is from B.
    # A2 is 2x1. B2 is 1x6.
    # U2 is 2x? V2 is 6x?
    # We can use rank 2?
    # C = A2 * B2.
    # This is an outer product of a vector (2x1) and a vector (1x6).
    # Rank 1? No, rank is 2*1*6 = 12 if we treat it as general matmul.
    # But since it's 2x1 and 1x6, we can do:
    # c=0: U[0,0]=1, V[0,0]=1, W[0,0]=1, W[1,0]=1...
    # Let's just use schoolbook for this part. Rank 12.
    
    # Total rank: 21 + 21 + 12 = 54.
    
    # Let's construct this.
    # A rows: 0,1.
    # A cols: 0,1 (A0), 2,3 (A1), 4 (A2).
    # B rows: 0,1 (B0), 2,3 (B1), 4 (B2).
    # B cols: 0,1 (B00), 2,3 (B01), 4,5 (B02) for B0 part.
    #        0,1 (B10), 2,3 (B11), 4,5 (B12) for B1 part.
    #        0,1 (B20), 2,3 (B21), 4,5 (B22) for B2 part.
    
    # Wait, B is 5x6.
    # B0 is rows 0,1. Cols 0..5.
    # B1 is rows 2,3. Cols 0..5.
    # B2 is row 4. Cols 0..5.
    
    # A0 is rows 0,1, cols 0,1.
    # A1 is rows 0,1, cols 2,3.
    # A2 is rows 0,1, col 4.
    
    # C is 2x6.
    # C = A0 B0 + A1 B1 + A2 B2.
    
    # Let's build U, V, W.
    # U shape (2*5, rank) = (10, rank)
    # V shape (5*6, rank) = (30, rank)
    # W shape (6*2, rank) = (12, rank)
    
    U = []
    V = []
    W = []
    
    sU, sV, sW = _strassen_2x2()
    
    def add_term(U_s, V_s, W_s, u_offset, v_offset, w_offset):
        # U_s: 4x7, V_s: 4x7, W_s: 4x7
        for c in range(7):
            u_col = np.zeros(10, dtype=int)
            v_col = np.zeros(30, dtype=int)
            w_col = np.zeros(12, dtype=int)
            
            for i in range(4):
                if U_s[i, c] != 0:
                    u_col[u_offset + i] = U_s[i, c]
                if V_s[i, c] != 0:
                    v_col[v_offset + i] = V_s[i, c]
                if W_s[i, c] != 0:
                    w_col[w_offset + i] = W_s[i, c]
            
            U.append(u_col)
            V.append(v_col)
            W.append(w_col)
            
    # Block A0 (cols 0,1) * B0 (rows 0,1)
    # B0 splits into 3 2x2 blocks: cols 0-1, 2-3, 4-5.
    # B00: rows 0,1, cols 0,1. Offset in V: 0.
    # B01: rows 0,1, cols 2,3. Offset in V: 2. (cols 2,3 are indices 2,3 in row, so 2*6+2, 2*6+3? No.)
    # V index is j*p+k. j is row of B. k is col of B.
    # B0 rows are 0,1.
    # B00: j in {0,1}, k in {0,1}. V indices: 0*6+0, 0*6+1, 1*6+0, 1*6+1.
    # The Strassen V matrix assumes V[0] is B00, V[1] is B01, V[2] is B10, V[3] is B11.
    # Here, for a 2x2 block at B rows r, r+1 and cols c, c+1:
    # V[0] -> B[r, c] -> V_idx = r*6 + c
    # V[1] -> B[r, c+1] -> V_idx = r*6 + c + 1
    # V[2] -> B[r+1, c] -> V_idx = (r+1)*6 + c
    # V[3] -> B[r+1, c+1] -> V_idx = (r+1)*6 + c + 1
    
    # Similarly for U:
    # U[0] -> A[i, c] -> U_idx = i*5 + c
    # U[1] -> A[i, c+1] -> U_idx = i*5 + c + 1
    # U[2] -> A[i+1, c] -> U_idx = (i+1)*5 + c
    # U[3] -> A[i+1, c+1] -> U_idx = (i+1)*5 + c + 1
    
    # And W:
    # W[0] -> C[c, i] -> W_idx = c*2 + i
    # W[1] -> C[c+1, i] -> W_idx = (c+1)*2 + i
    # W[2] -> C[c, i+1] -> W_idx = c*2 + i + 1
    # W[3] -> C[c+1, i+1] -> W_idx = (c+1)*2 + i + 1
    
    def add_strassen_block(a_row, a_col, b_row, b_col, c_col):
        # a_row, a_col: top-left of A 2x2 block. (a_row is 0)
        # b_row, b_col: top-left of B 2x2 block.
        # c_col: left col of C 2x2 block. (c_row is 0)
        
        for c_term in range(7):
            u_col = np.zeros(10, dtype=int)
            v_col = np.zeros(30, dtype=int)
            w_col = np.zeros(12, dtype=int)
            
            # Map Strassen indices to global indices
            # Strassen U: 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1) of the block.
            # Global U index: i*5 + j.
            # Block starts at row a_row, col a_col.
            # (0,0) -> a_row*5 + a_col
            # (0,1) -> a_row*5 + a_col + 1
            # (1,0) -> (a_row+1)*5 + a_col
            # (1,1) -> (a_row+1)*5 + a_col + 1
            
            # Strassen V: 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1) of the block.
            # Global V index: j*6 + k.
            # Block starts at row b_row, col b_col.
            # (0,0) -> b_row*6 + b_col
            # (0,1) -> b_row*6 + b_col + 1
            # (1,0) -> (b_row+1)*6 + b_col
            # (1,1) -> (b_row+1)*6 + b_col + 1
            
            # Strassen W: 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1) of the block.
            # Wait, Strassen W in _strassen_2x2() was defined with indices:
            # 0: C00 (k=0, i=0)
            # 2: C01 (k=1, i=0) -- WAIT.
            # In _strassen_2x2, W indices were:
            # 0: C00
            # 2: C01 (k=1, i=0) -> W_idx = 1*2+0 = 2. Correct.
            # 1: C10 (k=0, i=1) -> W_idx = 0*2+1 = 1. Correct.
            # 3: C11 (k=1, i=1) -> W_idx = 1*2+1 = 3. Correct.
            # So Strassen W index 0 -> local (0,0), 1 -> local (0,1)? No.
            # Local C matrix:
            # C[0,0] is k=0, i=0.
            # C[0,1] is k=0, i=1.
            # C[1,0] is k=1, i=0.
            # C[1,1] is k=1, i=1.
            # My W mapping in _strassen_2x2:
            # W[0] is C00. (k=0, i=0)
            # W[2] is C01. (k=1, i=0) -- This is C[0,1] in math notation?
            # Math C[i, k] or C[k, i]?
            # Task: W[k*n+i, r].
            # C[i, k] is standard. i is row, k is col.
            # W index is k*2 + i.
            # So W[0] is k=0, i=0 -> C[0,0].
            # W[1] is k=0, i=1 -> C[1,0].
            # W[2] is k=1, i=0 -> C[0,1].
            # W[3] is k=1, i=1 -> C[1,1].
            
            # In _strassen_2x2:
            # W[0, 0] = 1 (C00)
            # W[3, 0] = 1 (C11)
            # W[1, 1] = 1 (C10)
            # W[3, 1] = -1 (C11)
            # W[2, 2] = 1 (C01)
            # W[3, 2] = 1 (C11)
            # W[0, 3] = 1 (C00)
            # W[1, 3] = 1 (C10)
            # W[0, 4] = -1 (C00)
            # W[2, 4] = 1 (C01)
            # W[3, 5] = 1 (C11)
            # W[0, 6] = 1 (C00)
            
            # This matches the Strassen formulas:
            # C00 = M1 + M4 - M5 + M7 -> W[0] gets 1, 1, -1, 1.
            # C10 = M2 + M4 -> W[1] gets 1, 1.
            # C01 = M3 + M5 -> W[2] gets 1, 1.
            # C11 = M1 - M2 + M3 + M6 -> W[3] gets 1, -1, 1, 1.
            
            # So Strassen W index 0 is local C[0,0].
            # Strassen W index 1 is local C[1,0].
            # Strassen W index 2 is local C[0,1].
            # Strassen W index 3 is local C[1,1].
            
            # Local C[0,0] is global C[c_row, c_col].
            # Local C[1,0] is global C[c_row+1, c_col].
            # Local C[0,1] is global C[c_row, c_col+1].
            # Local C[1,1] is global C[c_row+1, c_col+1].
            
            # Global W index: k*2 + i.
            # C[i, k] -> W[k*2 + i].
            # Local C[0,0] -> i=c_row, k=c_col. W_idx = c_col*2 + c_row.
            # Local C[1,0] -> i=c_row+1, k=c_col. W_idx = c_col*2 + c_row + 1.
            # Local C[0,1] -> i=c_row, k=c_col+1. W_idx = (c_col+1)*2 + c_row.
            # Local C[1,1] -> i=c_row+1, k=c_col+1. W_idx = (c_col+1)*2 + c_row + 1.
            
            # Mapping:
            # sW[0] -> w_col[c_col*2 + c_row]
            # sW[1] -> w_col[c_col*2 + c_row + 1]
            # sW[2] -> w_col[(c_col+1)*2 + c_row]
            # sW[3] -> w_col[(c_col+1)*2 + c_row + 1]
            
            for i in range(4):
                val = sU[i, c_term]
                if val != 0:
                    # sU[0] -> (0,0), sU[1] -> (0,1), sU[2] -> (1,0), sU[3] -> (1,1)
                    if i == 0:
                        u_col[a_row*5 + a_col] += val
                    elif i == 1:
                        u_col[a_row*5 + a_col + 1] += val
                    elif i == 2:
                        u_col[(a_row+1)*5 + a_col] += val
                    elif i == 3:
                        u_col[(a_row+1)*5 + a_col + 1] += val
                        
                val = sV[i, c_term]
                if val != 0:
                    if i == 0:
                        v_col[b_row*6 + b_col] += val
                    elif i == 1:
                        v_col[b_row*6 + b_col + 1] += val
                    elif i == 2:
                        v_col[(b_row+1)*6 + b_col] += val
                    elif i == 3:
                        v_col[(b_row+1)*6 + b_col + 1] += val
                        
                val = sW[i, c_term]
                if val != 0:
                    if i == 0:
                        w_col[c_col*2 + c_row] += val
                    elif i == 1:
                        w_col[c_col*2 + c_row + 1] += val
                    elif i == 2:
                        w_col[(c_col+1)*2 + c_row] += val
                    elif i == 3:
                        w_col[(c_col+1)*2 + c_row + 1] += val
                        
            U.append(u_col)
            V.append(v_col)
            W.append(w_col)
            
    c_row = 0
    
    # A0 (rows 0, cols 0,1) * B0 (rows 0,1)
    # B0 cols 0,1
    add_strassen_block(0, 0, 0, 0, 0)
    # B0 cols 2,3
    add_strassen_block(0, 0, 0, 2, 2)
    # B0 cols 4,5
    add_strassen_block(0, 0, 0, 4, 4)
    
    # A1 (rows 0, cols 2,3) * B1 (rows 2,3)
    # B1 cols 0,1
    add_strassen_block(0, 2, 2, 0, 0)
    # B1 cols 2,3
    add_strassen_block(0, 2, 2, 2, 2)
    # B1 cols 4,5
    add_strassen_block(0, 2, 2, 4, 4)
    
    # A2 (rows 0,1, col 4) * B2 (row 4)
    # A2 is 2x1. B2 is 1x6.
    # C2[i, k] = A2[i, 0] * B2[0, k].
    # We can do this with 2*1*6 = 12 terms? Or less?
    # Rank 2? No.
    # U[i, c], V[k, c], W[k, i].
    # We need W[k, i] = A2[i, 0] * B2[0, k].
    # Let's use schoolbook.
    for i in range(2):
        for k in range(6):
            u_col = np.zeros(10, dtype=int)
            v_col = np.zeros(30, dtype=int)
            w_col = np.zeros(12, dtype=int)
            
            u_col[i*5 + 4] = 1
            v_col[4*6 + k] = 1
            w_col[k*2 + i] = 1
            
            U.append(u_col)
            V.append(v_col)
            W.append(w_col)
            
    U = np.array(U).T
    V = np.array(V).T
    W = np.array(W).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
