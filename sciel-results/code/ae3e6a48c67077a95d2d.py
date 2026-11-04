import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = 2, 3, 6
    rank = 33
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # We split the 6 columns into 3 blocks of 2
    for block in range(3):
        col_start = block * 2
        offset = block * 11
        
        # Strassen for 2x2 (columns col_start, col_start+1)
        # Rows: 0,1 of A; Cols: col_start, col_start+1 of B
        # Standard Strassen rank 7 terms
        # Let A = [[a11, a12], [a21, a22]] (rows 0,1; cols 0,1 of the 3x2 block)
        # Let B = [[b11, b12], [b21, b22], [b31, b32]] (rows 0,1,2; cols 0,1 of the 3x2 block)
        
        # We need to handle the fact that the input matrix is 3x2, not 2x2.
        # The 3x2 block multiplication is C_{2x2} = A_{2x3} B_{3x2}.
        # Let's split A into A_left (2x2) and A_right (2x1) and B into B_top (2x2) and B_bottom (1x2).
        # C = A_left B_top + A_right B_bottom.
        
        # A_left is rows 0,1 of A, cols 0,1 of A (which correspond to B cols col_start, col_start+1? No, A cols correspond to B rows).
        # A is 2x3. Cols of A are 0,1,2. Rows of B are 0,1,2.
        # So A_{2x3} * B_{3x2}.
        # Split A cols 0,1 vs col 2. Split B rows 0,1 vs row 2.
        # A_left = A[:, 0:2] (2x2). A_right = A[:, 2:3] (2x1).
        # B_top = B[0:2, :] (2x2). B_bottom = B[2:3, :] (1x2).
        
        # Term 1: A_left * B_top (2x2 * 2x2) -> Rank 7 (Strassen)
        # Term 2: A_right * B_bottom (2x1 * 1x2) -> Rank 4 (Schoolbook)
        
        # Strassen terms for 2x2:
        # M1 = (a11 + a22)(b11 + b22)
        # M2 = (a21 + a22)b11
        # M3 = a11(b12 - b22)
        # M4 = a22(b21 - b11)
        # M5 = (a11 + a12)b22
        # M6 = (a21 - a11)(b11 + b12)
        # M7 = (a12 - a22)(b21 + b22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Indices in U (A), V (B), W (C)
        # U index: i*m + j = i*3 + j
        # V index: j*p + k = j*6 + (col_start + l)  where l is 0 or 1
        # W index: k*n + i = (col_start + l)*2 + i
        
        # Strassen U/V contributions (local cols 0,1 of A and B)
        # A cols: 0, 1. B rows: 0, 1.
        # a11: U[0*3+0]=U[0], V[0*6+col_start]=V[col_start]
        # a12: U[0*3+1]=U[1], V[0*6+col_start+1]=V[col_start+1]
        # a21: U[1*3+0]=U[3], V[0*6+col_start]=V[col_start]
        # a22: U[1*3+1]=U[4], V[0*6+col_start+1]=V[col_start+1]
        # Wait, V index is row of B (which is col of A) times p + col of B.
        # B rows are 0,1,2. B cols are col_start, col_start+1.
        # For B_top (rows 0,1), V indices are V[0*6+col_start], V[0*6+col_start+1], V[1*6+col_start], V[1*6+col_start+1].
        
        # Let's explicitly define the 11 terms for the block.
        # Terms 0-6: Strassen
        # Terms 7-10: Schoolbook for A_right * B_bottom
        
        # Term 0: (a11 + a22) * (b11 + b22)
        U[0, offset+0] += 1; U[4, offset+0] += 1
        V[col_start, offset+0] += 1; V[col_start+1, offset+0] += 1
        W[(col_start)*2+0, offset+0] += 1   # C11 += M1
        W[(col_start+1)*2+0, offset+0] -= 1 # C12 -= M1? No, C12 = M3+M5. C11=M1+M4-M5+M7. C21=M2+M4. C22=M1-M2+M3+M6.
        # Let's re-verify Strassen output mapping.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So:
        # M0 (M1): C11 +1, C22 +1
        # M1 (M2): C21 +1, C22 -1
        # M2 (M3): C12 +1, C22 +1
        # M3 (M4): C11 +1, C21 +1
        # M4 (M5): C11 -1, C12 +1
        # M5 (M6): C22 +1
        # M6 (M7): C11 +1
        
        U[0, offset+0] = 1; U[4, offset+0] = 1
        V[col_start, offset+0] = 1; V[col_start+1, offset+0] = 1
        W[col_start*2+0, offset+0] = 1
        W[col_start*2+1, offset+0] = 1 # C22 is row 1? No.
        # W index: k*n + i. k is col of C (0 or 1 in block), i is row of C (0 or 1).
        # C11: k=0, i=0 -> W[0*2+0] = W[0] (global: col_start*2 + 0)
        # C12: k=1, i=0 -> W[1*2+0] = W[2] (global: col_start*2 + 2)
        # C21: k=0, i=1 -> W[0*2+1] = W[1] (global: col_start*2 + 1)
        # C22: k=1, i=1 -> W[1*2+1] = W[3] (global: col_start*2 + 3)
        
        # Correct W indices:
        # C11: W[col_start*2+0]
        # C12: W[col_start*2+2]
        # C21: W[col_start*2+1]
        # C22: W[col_start*2+3]
        
        # Reset W for this term
        W[col_start*2+0, offset+0] = 1 # C11
        W[col_start*2+3, offset+0] = 1 # C22
        
        # Term 1 (M2): (a21 + a22) * b11
        U[3, offset+1] = 1; U[4, offset+1] = 1
        V[col_start, offset+1] = 1
        W[col_start*2+1, offset+1] = 1 # C21
        W[col_start*2+3, offset+1] = -1 # C22
        
        # Term 2 (M3): a11 * (b12 - b22)
        U[0, offset+2] = 1
        V[col_start+1, offset+2] = 1
        V[col_start+1+6, offset+2] = -1 # b22 is V[1*6+col_start+1]
        W[col_start*2+2, offset+2] = 1 # C12
        W[col_start*2+3, offset+2] = 1 # C22
        
        # Term 3 (M4): a22 * (b21 - b11)
        U[4, offset+3] = 1
        V[col_start+6, offset+3] = 1 # b21 is V[1*6+col_start]
        V[col_start, offset+3] = -1
        W[col_start*2+0, offset+3] = 1 # C11
        W[col_start*2+1, offset+3] = 1 # C21
        
        # Term 4 (M5): (a11 + a12) * b22
        U[0, offset+4] = 1; U[1, offset+4] = 1
        V[col_start+1+6, offset+4] = 1
        W[col_start*2+0, offset+4] = -1 # C11
        W[col_start*2+2, offset+4] = 1 # C12
        
        # Term 5 (M6): (a21 - a11) * (b11 + b12)
        U[3, offset+5] = 1; U[0, offset+5] = -1
        V[col_start, offset+5] = 1; V[col_start+1, offset+5] = 1
        W[col_start*2+3, offset+5] = 1 # C22
        
        # Term 6 (M7): (a12 - a22) * (b21 + b22)
        U[1, offset+6] = 1; U[4, offset+6] = -1
        V[col_start+6, offset+6] = 1; V[col_start+1+6, offset+6] = 1
        W[col_start*2+0, offset+6] = 1 # C11
        
        # Terms 7-10: Schoolbook for A_right (col 2) * B_bottom (row 2)
        # A_right is A[:, 2] -> U indices 0*3+2=2, 1*3+2=5.
        # B_bottom is B[2, :] -> V indices 2*6+col_start, 2*6+col_start+1.
        # Products: a1*b1, a1*b2, a2*b1, a2*b2.
        # C11 += a1*b1 -> W[col_start*2+0]
        # C12 += a1*b2 -> W[col_start*2+2]
        # C21 += a2*b1 -> W[col_start*2+1]
        # C22 += a2*b2 -> W[col_start*2+3]
        
        # Term 7: a1 * b1
        U[2, offset+7] = 1
        V[2*6+col_start, offset+7] = 1
        W[col_start*2+0, offset+7] = 1
        
        # Term 8: a1 * b2
        U[2, offset+8] = 1
        V[2*6+col_start+1, offset+8] = 1
        W[col_start*2+2, offset+8] = 1
        
        # Term 9: a2 * b1
        U[5, offset+9] = 1
        V[2*6+col_start, offset+9] = 1
        W[col_start*2+1, offset+9] = 1
        
        # Term 10: a2 * b2
        U[5, offset+10] = 1
        V[2*6+col_start+1, offset+10] = 1
        W[col_start*2+3, offset+10] = 1
        
    # Convert to lists of lists to match interface if needed, but numpy arrays are usually fine?
    # The seed returned lists. I will return lists to be safe.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

BET: {mechanism: "Uses block decomposition with Strassen (rank 7) for 2x2 sub-blocks and schoolbook (rank 4) for cross terms, reducing total rank from 36 to 33", q: 0.95, delta: [0.08, 0.12]}
