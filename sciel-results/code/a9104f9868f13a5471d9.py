"""Matrix multiplication 2,4,6 - Strassen-based rank-42 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    start_time = time.time()
    deadline = start_time + 115

    # We'll build the Strassen decomposition for 2x4 * 4x6
    # View A as 2x2 blocks of 2x2: A = [[A11, A12], [A21, A22]]
    # View B as 2x2 blocks of 2x2: B = [[B11, B12], [B21, B22]]
    # C = A*B gives 2x2 blocks of 2x2: C = [[C11, C12], [C21, C22]]
    #
    # Strassen's 7 multiplications:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mx is a 2x2 matrix, and each entry is a scalar product.
    # So we have 7 * 4 = 28 scalar multiplications.
    # But wait - for the tensor decomposition, each "multiplication" Mx[r,c]
    # is itself a bilinear form, and we need to decompose it into rank-1 terms.
    #
    # Actually, the standard approach: each of the 7 matrix multiplications
    # (which are 2x2 * 2x2) can be done with rank 7 each (by Strassen again),
    # but that would give 7*7=49. Instead, since each 2x2*2x2 has rank 7,
    # and we have 7 such multiplications, the total rank is 7*7=49? No...
    #
    # Let me reconsider. The rank of 2x4*4x6 matrix multiplication:
    # The schoolbook rank is 2*4*6 = 48.
    # 
    # Actually, let me think about this more carefully.
    # The tensor for n x m * m x p matrix mult has rank at most n*m*p (schoolbook).
    # For 2x4*4x6, schoolbook is 48.
    #
    # The Kronecker product structure: if we write 2x4*4x6 as (2*2)x(4) * (4)x(6),
    # we can view it as a 2x2 block matrix multiplication where each block is 2x6... 
    # no, that doesn't work cleanly.
    #
    # Let me just directly construct the rank-42 decomposition using Strassen on 2x2 blocks.
    # 
    # A is 2x4, B is 4x6, C is 2x6.
    # Split columns of A into two groups of 2: A = [A_left, A_right] where each is 2x2.
    # Split rows of B into two groups of 2: B = [[B_top], [B_bottom]] where each is 2x6.
    #
    # C = A_left * B_top + A_right * B_bottom
    #
    # This is just 2 matrix multiplications of 2x2 * 2x6, which has rank 2*2*6=24 each,
    # total 48. Not helpful.
    #
    # Better: Split A into 2x2 blocks and B into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] (2x2 blocks of 2x2)
    # B = [[B11, B12], [B21, B22]] (2x2 blocks of 2x2)  -- but B is 4x6, so blocks are 2x3
    #
    # Hmm, B is 4x6, so if we split into 2x2 blocks we get 2x3 blocks (2 rows, 3 cols).
    # Let me use 2x3 blocks for B.
    #
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12, B13], [B21, B22, B23]] where each is 2x2
    # C = A*B is 2x6, split into 2x2 blocks: C = [[C11, C12, C13], [C21, C22, C23]]
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    #
    # Using Strassen on the 2x2 block structure:
    # M1 = (A11+A22)*(B11+B21)  -- 2x2 * 2x2
    # M2 = (A21+A22)*B11        -- 2x2 * 2x2
    # M3 = A11*(B12-B22)        -- 2x2 * 2x2
    # M4 = A22*(B21-B11)        -- 2x2 * 2x2
    # M5 = (A11+A12)*B22        -- 2x2 * 2x2
    # M6 = (A21-A11)*(B11+B12)  -- 2x2 * 2x2
    # M7 = (A12-A22)*(B21+B22)  -- 2x2 * 2x2
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # But we also need C13 and C23:
    # C13 = A11*B13 + A12*B23
    # C23 = A21*B13 + A22*B23
    #
    # These require additional multiplications. We can do them with schoolbook:
    # C13 = A11*B13 + A12*B23: this is 2x2*2x2 + 2x2*2x2, rank 4+4=8
    # C23 = A21*B13 + A22*B23: same, rank 8
    #
    # Total rank: 7 (Strassen) + 8 + 8 = 23? No, that's not right either.
    # Each "multiplication" Mx is a 2x2*2x2, which has rank 7 by Strassen.
    # So 7 Strassen mults * 7 rank each = 49, plus the extra terms.
    #
    # I think the cleanest approach is:
    # Rank of 2x4*4x6 <= rank of (2x2*2x2) Kronecker something...
    #
    # Actually, let me just use the known fact: rank(2x4*4x6) <= 42 via Strassen.
    # The construction: 2x4*4x6 can be seen as (2*2)x(2*2) * (2*2)x(2*3)
    # which is NOT a clean Kronecker product.
    #
    # Let me try a different approach: just use the schoolbook rank 48 and see if I can
    # do better. Actually, for the purpose of this task, let me construct the rank-48
    # schoolbook decomposition first (which is guaranteed correct), then try to reduce.
    #
    # Actually, re-reading the task: "score is the rank" and "lower is better".
    # The seed gives rank 48 (score 1.0). I need to find a lower rank.
    #
    # Let me construct the rank-42 Strassen decomposition properly.
    #
    # The key insight: 2x4 * 4x6 matrix multiplication.
    # Think of it as: A (2x4) times B (4x6) = C (2x6).
    # 
    # Partition: A = [A1 | A2] where A1, A2 are 2x2
    #            B = [B1; B2] where B1, B2 are 2x6
    # C = A1*B1 + A2*B2
    #
    # Each Ai*Bi is 2x2 * 2x6, which has rank 2*2*6 = 24 by schoolbook.
    # Total: 48.
    #
    # Now, 2x2 * 2x6 can be done with rank... well, it's not a square multiplication.
    # But we can note that 2x2*2x6 has the same rank as 2x2*2x2 times identity on the 6 side...
    # No, that's not quite right.
    #
    # Alternative: 2x2 * 2x6 can be viewed as 2x2*2x2 block mult where B has 3 column-blocks.
    # Rank(2x2*2x2) = 7 (Strassen). So 2x2*2x6 has rank at most 7*3 = 21? No...
    # Actually, if B = [B1 B2 B3] where each Bi is 2x2, then A*B = [A*B1, A*B2, A*B3].
    # Each A*Bi has rank 7 (Strassen). Total rank = 7*3 = 21 for 2x2*2x6.
    # Then total for 2x4*4x6: 21 + 21 = 42.
    #
    # So the construction is:
    # A = [A1 | A2], B = [B1; B2] where B1 = [B11 B12 B13], B2 = [B21 B22 B23]
    # C1 = A1 * B1 = [A1*B11, A1*B12, A1*B13] (each 2x2*2x2, rank 7)
    # C2 = A2 * B2 = [A2*B21, A2*B22, A2*B23] (each 2x2*2x2, rank 7)
    # C = [C1 | C2] (row concatenation)
    #
    # Total rank: 3*7 + 3*7 = 42.
    #
    # Now I need to explicitly construct the U, V, W matrices.
    #
    # For each 2x2*2x2 Strassen multiplication, I need to specify the 7 rank-1 terms.
    #
    # Let me define a helper that given the row/col offsets for A, B, C blocks,
    # adds 7 Strassen terms to U, V, W.
    
    def strassen_2x2(U, V, W, rank_offset, a_row, a_col, b_row, b_col, c_row, c_col):
        """
        Add Strassen rank-7 decomposition for 2x2*2x2 multiplication.
        
        A block: rows a_row:a_row+2, cols a_col:a_col+2
        B block: rows b_row:b_row+2, cols b_col:b_col+2  
        C block: rows c_row:c_row+2, cols c_col:c_col+2
        
        U has shape (n*m, rank) where index is i*m+j for A[i,j]
        V has shape (m*p, rank) where index is j*p+k for B[j,k]
        W has shape (p*n, rank) where index is k*n+i for C[k,i]
        
        Strassen:
        M1 = (A00+A11)*(B00+B11)
        M2 = (A10+A11)*B00
        M3 = A00*(B01-B11)
        M4 = A11*(B10-B00)
        M5 = (A00+A01)*B11
        M6 = (A10-A00)*(B00+B01)
        M7 = (A01-A11)*(B10+B11)
        
        C00 = M1+M4-M5+M7
        C01 = M3+M5
        C10 = M2+M4
        C11 = M1-M2+M3+M6
        """
        # For each of the 7 multiplications, we add a rank-1 term.
        # Each rank-1 term has:
        # U entries: which A elements (with coefficients)
        # V entries: which B elements (with coefficients)
        # W entries: which C elements (with coefficients)
        
        # M1 = (A00+A11)*(B00+B11)
        # U: A00->1, A11->1  (indices: a_row+a_col, (a_row+1)+(a_col+1))
        # V: B00->1, B11->1  (indices: b_row+b_col, (b_row+1)+(b_col+1))
        # W: C00->1, C11->1  (indices: c_row+c_col, (c_row+1)+(c_col+1))
        r = rank_offset
        U[a_row + a_col, r] += 1  # A[0,0]
        U[a_row+1 + a_col+1, r] += 1  # A[1,1]
        V[b_row + b_col, r] += 1  # B[0,0]
        V[b_row+1 + b_col+1, r] += 1  # B[1,1]
        W[c_row + c_col, r] += 1  # C[0,0]
        W[c_row+1 + c_col+1, r] += 1  # C[1,1]
        
        # M2 = (A10+A11)*B00
        r = rank_offset + 1
        U[a_row+1 + a_col, r] += 1  # A[1,0]
        U[a_row+1 + a_col+1, r] += 1  # A[1,1]
        V[b_row + b_col, r] += 1  # B[0,0]
        W[c_row+1 + c_col, r] += 1  # C[1,0]
        
        # M3 = A00*(B01-B11)
        r = rank_offset + 2
        U[a_row + a_col, r] += 1  # A[0,0]
        V[b_row + b_col+1, r] += 1  # B[0,1]
        V[b_row+1 + b_col+1, r] += -1  # B[1,1]
        W[c_row + c_col+1, r] += 1  # C[0,1]
        
        # M4 = A11*(B10-B00)
        r = rank_offset + 3
        U[a_row+1 + a_col+1, r] += 1  # A[1,1]
        V[b_row+1 + b_col, r] += 1  # B[1,0]
        V[b_row + b_col, r] += -1  # B[0,0]
        W[c_row+1 + c_col, r] += 1  # C[1,0]
        
        # M5 = (A00+A01)*B11
        r = rank_offset + 4
        U[a_row + a_col, r] += 1  # A[0,0]
        U[a_row + a_col+1, r] += 1  # A[0,1]
        V[b_row+1 + b_col+1, r] += 1  # B[1,1]
        W[c_row + c_col+1, r] += 1  # C[0,1]
        
        # M6 = (A10-A00)*(B00+B01)
        r = rank_offset + 5
        U[a_row+1 + a_col, r] += 1  # A[1,0]
        U[a_row + a_col, r] += -1  # A[0,0]
        V[b_row + b_col, r] += 1  # B[0,0]
        V[b_row + b_col+1, r] += 1  # B[0,1]
        W[c_row+1 + c_col+1, r] += 1  # C[1,1]
        
        # M7 = (A01-A11)*(B10+B11)
        r = rank_offset + 6
        U[a_row + a_col+1, r] += 1  # A[0,1]
        U[a_row+1 + a_col+1, r] += -1  # A[1,1]
        V[b_row+1 + b_col, r] += 1  # B[1,0]
        V[b_row+1 + b_col+1, r] += 1  # B[1,1]
        W[c_row + c_col, r] += 1  # C[0,0]
        
        return rank_offset + 7
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if the decomposition is valid."""
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Build the rank-42 decomposition
    rank = 42
    U = np.zeros((n*m, rank))  # (8, 42)
    V = np.zeros((m*p, rank))  # (24, 42)
    W = np.zeros((p*n, rank))  # (12, 42)
    
    # A is 2x4: A[i,j] where i=0,1 (row), j=0,1,2,3 (col)
    # U index: i*m + j = i*4 + j
    # 
    # B is 4x6: B[j,k] where j=0,1,2,3 (row), k=0,1,2,3,4,5 (col)
    # V index: j*p + k = j*6 + k
    #
    # C is 2x6: C[i,k] where i=0,1 (row), k=0,1,2,3,4,5 (col)
    # W index: k*n + i = k*2 + i
    #
    # Partition:
    # A = [A1 | A2] where A1 = A[:, 0:2], A2 = A[:, 2:4]
    # B = [B1; B2] where B1 = B[0:2, :], B2 = B[2:4, :]
    # B1 = [B11 B12 B13] where B11 = B1[:,0:2], B12 = B1[:,2:4], B13 = B1[:,4:6]
    # B2 = [B21 B22 B23] where B21 = B2[:,0:2], B22 = B2[:,2:4], B23 = B2[:,4:6]
    #
    # C = [C1 | C2] where C1 = C[:, 0:3*2]... no wait.
    # C1 = A1 * B1 is 2x6, C2 = A2 * B2 is 2x6
    # C = C1 + C2 (element-wise addition, not concatenation!)
    #
    # Wait, I made an error. Let me re-derive.
    # A (2x4) = [A1 | A2] where A1, A2 are 2x2
    # B (4x6) = [B1; B2] where B1, B2 are 2x6
    # C (2x6) = A1*B1 + A2*B2
    #
    # So C1 = A1*B1 and C2 = A2*B2 are both 2x6, and C = C1 + C2.
    #
    # For C1 = A1*B1: A1 is 2x2, B1 is 2x6 = [B11 B12 B13] (three 2x2 blocks)
    # C1 = [A1*B11, A1*B12, A1*B13] (each 2x2)
    #
    # For C2 = A2*B2: A2 is 2x2, B2 is 2x6 = [B21 B22 B23] (three 2x2 blocks)
    # C2 = [A2*B21, A2*B22, A2*B23] (each 2x2)
    #
    # C = C1 + C2, so C[:, 0:2] = A1*B11 + A2*B21
    #                  C[:, 2:4] = A1*B12 + A2*B22
    #                  C[:, 4:6] = A1*B13 + A2*B23
    #
    # Each C[:, 2t:2t+2] = A1*B1t + A2*B2t for t=0,1,2
    #
    # For each t, we need to compute A1*B1t + A2*B2t.
    # This is the sum of two 2x2*2x2 multiplications.
    # We can do each with Strassen (rank 7), total rank 14 per t, total 42.
    
    # A1: rows 0-1, cols 0-1 => a_row=0, a_col=0
    # A2: rows 0-1, cols 2-3 => a_row=0, a_col=2
    # B1t: rows 0-1, cols 2t to 2t+1 => b_row=0, b_col=2*t
    # B2t: rows 2-3, cols 2t to 2t+1 => b_row=2, b_col=2*t
    # C_t: rows 0-1, cols 2t to 2t+1 => c_row=0, c_col=2*t
    
    rank_offset = 0
    for t in range(3):
        # C_t = A1*B1t + A2*B2t
        # A1*B1t: Strassen with a_row=0, a_col=0, b_row=0, b_col=2*t, c_row=0, c_col=2*t
        rank_offset = strassen_2x2(U, V, W, rank_offset,
                                    a_row=0, a_col=0,
                                    b_row=0, b_col=2*t,
                                    c_row=0, c_col=2*t)
        
        # A2*B2t: Strassen with a_row=0, a_col=2, b_row=2, b_col=2*t, c_row=0, c_col=2*t
        rank_offset = strassen_2x2(U, V, W, rank_offset,
                                    a_row=0, a_col=2,
                                    b_row=2, b_col=2*t,
                                    c_row=0, c_col=2*t)
    
    assert rank_offset == 42, f"Expected rank 42, got {rank_offset}"
    
    # Verify
    if check_decomposition(U, V, W, tol=1e-6):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If verification fails, return schoolbook as fallback
    rank_fb = n * m * p
    U_fb = np.zeros((n*m, rank_fb))
    V_fb = np.zeros((m*p, rank_fb))
    W_fb = np.zeros((p*n, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
