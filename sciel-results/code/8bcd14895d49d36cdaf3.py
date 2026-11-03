import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def _build_strassen_2x2x2():
    """
    Returns (U, V, W) for 2x2 * 2x2 = 2x2 using 7 rank-1 terms.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Index convention: A is 2x2 (4 elements), B is 2x2 (4 elements), C is 2x2 (4 elements).
    We use row-major flattening: element (i,j) -> index 2*i + j.
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # M3 = A00(B10-B11) -> C01 += M3, C11 += M3
    # M4 = A11(B01-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    
    # Term 0: M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 1: M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # Term 2: M3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Term 3: M4
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Term 4: M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Term 5: M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 6: M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Target: C[i,k] = sum_j A[i,j] B[j,k]
    # U shape: (n*m, R) = (8, R)
    # V shape: (m*p, R) = (32, R)
    # W shape: (p*n, R) = (16, R)
    # Indexing:
    # U row: i*m + j  (i in 0..n-1, j in 0..m-1)
    # V row: j*p + k  (j in 0..m-1, k in 0..p-1)
    # W row: k*n + i  (k in 0..p-1, i in 0..n-1)
    # Note: The task says "output factor follows the official trace convention W[k*n+i,r]".
    # This matches my W row indexing: k*p + i where p is the number of columns of B (which is n? No, p is columns of B, n is rows of A).
    # Let's re-verify dimensions.
    # A: n x m
    # B: m x p
    # C: n x p
    # U: (n*m, R) -> A is n*m elements
    # V: (m*p, R) -> B is m*p elements
    # W: (p*n, R) -> C is n*p elements. The index is k*n + i.
    # Yes, W[k*n+i] corresponds to C[i,k].
    
    # Strategy:
    # We want to compute C = A * B.
    # n=2, m=4, p=8.
    
    # The standard Strassen approach for 2x4 * 4x8:
    # Split A into two 2x2 blocks horizontally: A = [A1, A2]
    # Split B into two 2x8 blocks vertically: B = [B1; B2]
    # C = A1*B1 + A2*B2.
    # This reduces to two 2x2 * 2x8 multiplications.
    
    # Now, how to do 2x2 * 2x8 efficiently?
    # Let X be 2x2, Y be 2x8. Result Z = X*Y is 2x8.
    # We can view Y as four 2x2 blocks: Y = [Y1 Y2 Y3 Y4] where Yj is 2x2.
    # Then Z = [X*Y1 X*Y2 X*Y3 X*Y4].
    # Each X*Yj is a 2x2 * 2x2 multiplication.
    # If we use Strassen for each, it costs 7 * 4 = 28 terms for one 2x2 * 2x8 product.
    # Since we have two such products (A1*B1 and A2*B2), total rank = 28 + 28 = 56.
    
    # Is there a better way?
    # Consider computing all four Zj = X*Yj simultaneously.
    # The input Y has 16 elements. X has 4. Output Z has 16.
    # The tensor for 2x2 * 2x8 is a block diagonal tensor consisting of four 2x2*2x2 tensors?
    # No, it's not block diagonal in the rank-1 sense because the same X is used for all Yj.
    # However, the structure is:
    # Z[i, k] = sum_{j=0}^1 X[i, j] * Y[j, k].
    # This is equivalent to 8 separate 2x1 vector multiplications (for each column k of Y),
    # where the vector is Y[:, k] and the matrix is X.
    # Or, it's 4 separate 2x2 matrix multiplications sharing X.
    
    # Let's look at the rank of the 2x2 * 2x8 multiplication tensor.
    # Let T_{2,2,8} be the tensor.
    # T[i, j, k] = 1 if we are computing Z[i,k] from X[i,j] and Y[j,k].
    # Actually, the bilinear form is sum_{j=0}^1 sum_{l=0}^1 X[i, j] Y[j, l] delta_{k, l}? No.
    # For a fixed column pair of Y (indices k, k+1), we compute a 2x2 block of Z.
    # So Z[:, k:k+2] = X * Y[:, k:k+2].
    # Since X is shared, can we exploit this?
    # In the standard decomposition, we just concatenate the four Strassen decompositions.
    # Rank = 4 * 7 = 28.
    # Is it possible to do better?
    # The border rank of 2x2 * 2x8 is likely 28 or close.
    # Lower bound for rank of MM_{n,m,p} is max(n*p, m*p, n*m).
    # For 2x2 * 2x8: max(2*8, 2*8, 2*2) = max(16, 16, 4) = 16.
    # But Strassen gives 28.
    # For 2x2 * 2x2, rank is 7.
    # For 2x2 * 2x4, rank is likely 14 (two Strassen blocks).
    # For 2x2 * 2x8, rank is likely 28.
    
    # So the total rank for 2x4 * 4x8 using this decomposition is 56.
    # This matches the current program's score.
    
    # Can we do better than 56?
    # Let's check if we can mix the two A blocks.
    # C = A1 B1 + A2 B2.
    # This is a sum of two products.
    # Is there a decomposition that shares terms between A1 B1 and A2 B2?
    # The inputs are A1, A2, B1, B2.
    # This is essentially a 2x4 * 4x8 multiplication.
    # Known results for small matrix multiplication ranks:
    # MM_{2,2,2} rank 7.
    # MM_{2,4,4} rank? 
    # MM_{2,4,8} rank?
    
    # Let's try to construct a rank 56 solution properly to ensure it's valid.
    # If we can't find a lower rank, we stick with 56 but ensure the code is robust and fast.
    
    # The previous code seems to implement the 56 rank solution.
    # Let's verify the previous code's logic for 2x2 * 2x8.
    # It loops j in 0..3 (rows of B / columns of A split).
    # Wait, the previous code:
    # for j in range(4): # j is column index of A / row index of B
    #   for s in range(2): # s splits the 8 columns of B into two 4-column groups?
    #   No, the previous code comments say:
    #   "A = [A_left | A_right] where A_left is 2x2, A_right is 2x2"
    #   "B = [B_top; B_bottom] where B_top is 2x8, B_bottom is 2x8"
    #   "C = A_left * B_top + A_right * B_bottom"
    #   This implies A is 2x4, split into two 2x2. B is 4x8, split into two 2x8.
    #   This gives C = A1 B1 + A2 B2.
    #   Each term is 2x2 * 2x8.
    #   The code then processes 2x2 * 2x8.
    #   Inside, it seems to split the 8 columns of B into four 2-column blocks?
    #   "for j in range(4): ... base_r = (j * 2 + s) * 7"
    #   This looks like it's handling 4 pairs of columns.
    #   Total rank 56.
    
    # Is there a rank 48 or lower solution?
    # Some sources suggest MM_{2,4,8} might have rank 52 or 56.
    # Without a specific known lower-rank algorithm, 56 is a safe bet.
    # However, we should check if there are any optimizations or different decompositions.
    
    # Let's implement the rank 56 solution cleanly.
    
    # 1. Define Strassen 2x2*2x2
    U_ss, V_ss, W_ss = _build_strassen_2x2x2()
    
    # 2. Build 2x2 * 2x8 decomposition
    # We need U_2228 (4, R), V_2228 (16, R), W_2228 (16, R)
    # R should be 28.
    R_2228 = 28
    U_2228 = np.zeros((4, R_2228), dtype=int)
    V_2228 = np.zeros((16, R_2228), dtype=int)
    W_2228 = np.zeros((16, R_2228), dtype=int)
    
    # Y is 2x8. Flatten row-major: Y[j, k] -> index j*8 + k.
    # Z is 2x8. Flatten row-major: Z[i, k] -> index i*8 + k.
    # X is 2x2. Flatten row-major: X[i, j] -> index i*2 + j.
    
    # We process 4 blocks of 2 columns of Y.
    # Block b (0..3) uses columns 2b, 2b+1 of Y.
    # For each block, we use one Strassen instance.
    for b in range(4):
        r_offset = b * 7
        for l in range(7):
            r = r_offset + l
            
            # U part (X) is the same for all blocks
            for i in range(4):
                U_2228[i, r] += U_ss[i, l]
            
            # V part (Y block b)
            # Y indices for block b:
            # Y[0, 2b], Y[0, 2b+1], Y[1, 2b], Y[1, 2b+1]
            # Flat indices: 0*8 + 2b, 0*8 + 2b+1, 1*8 + 2b, 1*8 + 2b+1
            y_idx = [2*b, 2*b+1, 8+2*b, 8+2*b+1]
            for i in range(4):
                V_2228[y_idx[i], r] += V_ss[i, l]
            
            # W part (Z block b)
            # Z indices for block b:
            # Z[0, 2b], Z[0, 2b+1], Z[1, 2b], Z[1, 2b+1]
            # Flat indices: 0*8 + 2b, 0*8 + 2b+1, 1*8 + 2b, 1*8 + 2b+1
            z_idx = [2*b, 2*b+1, 8+2*b, 8+2*b+1]
            for i in range(4):
                W_2228[z_idx[i], r] += W_ss[i, l]
                
    # 3. Combine two 2x2*2x8 decompositions for 2x4*4x8
    # A = [A1, A2] (2x4). A1 is cols 0,1. A2 is cols 2,3.
    # B = [B1; B2] (4x8). B1 is rows 0,1. B2 is rows 2,3.
    # C = A1 B1 + A2 B2.
    
    # Term 1: A1 * B1
    # A1 indices in A (2x4): (0,0), (0,1), (1,0), (1,1) -> 0, 1, 4, 5
    # B1 indices in B (4x8): rows 0,1.
    # Row 0: 0..7. Row 1: 8..15.
    # B1 flat indices: 0..7, 8..15.
    # Specifically, B1[j, k] where j in {0,1}, k in 0..7.
    # Flat index: j*8 + k.
    # So B1 flat indices are 0..7 (j=0) and 8..15 (j=1).
    # This matches the V_2228 indices 0..15.
    
    # C1 = A1 B1 (2x8).
    # C1 indices in C (2x8): 0..7, 8..15.
    # W_2228 indices 0..15 match C1.
    
    # Term 2: A2 * B2
    # A2 indices in A: (0,2), (0,3), (1,2), (1,3) -> 2, 3, 6, 7
    # B2 indices in B: rows 2,3.
    # Row 2: 16..23. Row 3: 24..31.
    # B2 flat indices: 16..23, 24..31.
    # We need to map V_2228 indices 0..15 to 16..31.
    # V_2228[j, k] where j in {0,1}, k in 0..7.
    # For B2, j=0 corresponds to row 2 of B, j=1 to row 3.
    # So V_2228 index i (0..15) maps to B2 index i + 16.
    
    # C2 = A2 B2 (2x8).
    # C2 adds to the same C as C1.
    # So W_2228 indices 0..15 add to C indices 0..15.
    
    R_total = 2 * R_2228 # 56
    U = np.zeros((8, R_total), dtype=int)
    V = np.zeros((32, R_total), dtype=int)
    W = np.zeros((16, R_total), dtype=int)
    
    # Term 1: A1 * B1
    a1_map = [0, 1, 4, 5] # X[0,0], X[0,1], X[1,0], X[1,1] -> A[0,0], A[0,1], A[1,0], A[1,1]
    # Wait, A is 2x4. A[i, j].
    # A1 is A[:, 0:2].
    # A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5. Correct.
    # V_2228 maps to B1 (rows 0,1 of B).
    # B1[j, k] -> B[j, k]. Flat index j*8 + k.
    # V_2228 index is j*8 + k. So it maps directly to V indices 0..15.
    
    for r in range(R_2228):
        # U
        for i in range(4):
            U[a1_map[i], r] += U_2228[i, r]
        # V
        for i in range(16):
            V[i, r] += V_2228[i, r]
        # W
        for i in range(16):
            W[i, r] += W_2228[i, r]
            
    # Term 2: A2 * B2
    a2_map = [2, 3, 6, 7] # A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
    # B2 is B[2:4, :].
    # B2[j, k] -> B[j+2, k]. Flat index (j+2)*8 + k = j*8 + k + 16.
    # V_2228 index i corresponds to B2 index i.
    # So V_2228[i] maps to V[i+16].
    r_offset = R_2228
    for r in range(R_2228):
        rr = r + r_offset
        # U
        for i in range(4):
            U[a2_map[i], rr] += U_2228[i, r]
        # V
        for i in range(16):
            V[i+16, rr] += V_2228[i, r]
        # W
        for i in range(16):
            W[i, rr] += W_2228[i, r]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
