import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Uses a block decomposition strategy embedding into 4x4x4 (Strassen rank 49) 
    and 3x4x1 (schoolbook rank 12).
    Total rank = 49 + 12 = 61.
    """
    n, m, p = DIMENSIONS
    
    # We split the multiplication A(3x4) * B(4x5) into:
    # C = A * B[:, :4] + A * B[:, 4:]
    # Part 1: A(3x4) * B1(4x4) -> C1(3x4)
    # Part 2: A(3x4) * B2(4x1) -> C2(3x1)
    
    # Construct Part 1: 3x4x4 decomposition using 4x4x4 Strassen (Rank 49)
    # The 4x4x4 Strassen decomposition is the tensor square of the 2x2x2 Strassen decomposition.
    
    # Strassen 2x2x2 decomposition (Rank 7)
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to build the 4x4x4 decomposition from two levels of 2x2x2.
    # This results in a rank 49 decomposition.
    # Instead of deriving the 49 terms manually, we can use the property that 
    # the 4x4x4 multiplication tensor is the Kronecker product of two 2x2x2 tensors.
    
    # Let's generate the 4x4x4 Strassen decomposition.
    # U_4, V_4, W_4 of shape (16, 49), (16, 49), (16, 49).
    
    # Base 2x2x2 Strassen matrices
    # U2: (4, 7), V2: (4, 7), W2: (4, 7)
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Indices for 2x2: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # M1 = (A00 + A11)(B00 + B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A10 + A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1 # C10, C11 parts? 
    # Wait, W maps to C indices.
    # C10 gets M2. C11 gets -M2.
    # So W2[2, 1] = 1, W2[3, 1] = -1.
    W2[2, 1] = 1
    W2[3, 1] = -1
    
    # M3 = A00(B01 - B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1 # C01 gets M3
    # C00 gets M3? No.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So M3 is in C01 and C11.
    W2[1, 2] = 1
    W2[3, 2] = 1
    
    # M4 = A11(B10 - B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1 # C00
    W2[2, 3] = 1 # C10
    
    # M5 = (A00 + A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1 # C00
    W2[1, 4] = 1 # C01
    
    # M6 = (A10 - A00)(B00 + B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C11
    
    # M7 = (A01 - A11)(B10 + B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C00
    
    # Now construct 4x4x4 by Kronecker product.
    # U4 = U2 kron U2? 
    # The indices are (i1, i0). 
    # U4[i1*4+i0, r1*7+r0] = U2[i1, r1] * U2[i0, r0]
    
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # U4 is 16x49. V4 is 16x49. W4 is 16x49.
    
    # We need to restrict this to 3x4x4.
    # A is 3x4. Rows 0,1,2. (Row 3 is 0).
    # B is 4x4. All rows/cols used.
    # C is 3x4. Rows 0,1,2. (Row 3 is 0).
    
    # U_part1: Take rows 0..11 (0..2 in first dim, 0..3 in second dim).
    # Indices: i in {0,1,2}, j in {0,1,2,3}.
    # Row index in U4: i*4 + j.
    rows_u = [i*4 + j for i in range(3) for j in range(4)]
    U1 = U4[rows_u, :]
    
    # V_part1: B is 4x4. All 16 rows.
    V1 = V4
    
    # W_part1: C is 3x4. Rows 0,1,2.
    rows_w = [i*4 + j for i in range(3) for j in range(4)]
    W1 = W4[rows_w, :]
    
    # Part 2: A(3x4) * B2(4x1) -> C2(3x1)
    # Rank 12.
    # U2_part: 3x4 = 12 rows.
    # V2_part: 4x1 = 4 rows.
    # W2_part: 1x3 = 3 rows? 
    # Trace convention: W[k*n+i, r]. Here n=3 (output rows), p=1 (output cols? No, p is B cols).
    # Wait, the problem is 3x4x5.
    # Part 2 is 3x4x1.
    # U: 3x4 (12 rows).
    # V: 4x1 (4 rows).
    # W: 1x3 (3 rows). W[k*n+i] with k=0..0, i=0..2.
    
    U2_part = np.zeros((12, 12))
    V2_part = np.zeros((4, 12))
    W2_part = np.zeros((3, 12))
    
    for i in range(3):
        for j in range(4):
            idx = i*4 + j
            U2_part[idx, idx] = 1.0
            V2_part[j, idx] = 1.0
            W2_part[i, idx] = 1.0
            
    # Combine
    rank = 49 + 12
    U = np.zeros((12, rank))
    V = np.zeros((20, rank)) # 4x5 = 20 rows
    W = np.zeros((15, rank)) # 5x3 = 15 rows? 
    # Wait, W size is p*n = 5*3 = 15.
    # Part 1 W is 3x4 = 12 rows.
    # Part 2 W is 1x3 = 3 rows.
    # Total 15 rows.
    
    # Fill Part 1
    U[:, :49] = U1
    V[:16, :49] = V1 # B1 is 4x4, rows 0..15
    W[:12, :49] = W1 # C1 is 3x4, rows 0..11 (k=0..0? No, k is B col index? 
    # W index is k*n + i.
    # In 4x4x4, W index is k*4 + i.
    # We restricted i to 0..2.
    # So W1 rows are k*4 + i.
    # k=0..3, i=0..2.
    # Rows: 0,1,2, 4,5,6, 8,9,10, 12,13,14.
    # This matches the first 12 rows of the 15-row W?
    # W total size 15.
    # Part 1 uses k=0..3, i=0..2.
    # Part 2 uses k=4, i=0..2.
    
    # Fill Part 2
    U[:, 49:] = U2_part
    V[16:, 49:] = V2_part # B2 is rows 16..19 (j=0..3, k=4? No, B is 4x5.
    # B indices: j (0..3) * 5 + k (0..4).
    # B1 is k=0..3. Rows j*5 + k.
    # B2 is k=4. Rows j*5 + 4.
    
    # We need to map V4 (16x49) to V (20x49).
    # V4 indices: j*4 + k (k=0..3).
    # V1 indices: j*5 + k (k=0..3).
    
    V1_mapped = np.zeros((20, 49))
    for j in range(4):
        for k in range(4):
            V1_mapped[j*5 + k, :] = V4[j*4 + k, :]
            
    V[:20, :49] = V1_mapped
    
    # W1 indices: k*4 + i (i=0..2).
    # W total indices: k*3 + i (k=0..4, i=0..2).
    
    W1_mapped = np.zeros((15, 49))
    for k in range(4):
        for i in range(3):
            W1_mapped[k*3 + i, :] = W4[k*4 + i, :]
            
    W[:15, :49] = W1_mapped
    
    # Part 2: B2 is k=4.
    # V2_part indices: j (0..3).
    # V total indices: j*5 + 4.
    V2_mapped = np.zeros((20, 12))
    for j in range(4):
        V2_mapped[j*5 + 4, :] = V2_part[j, :]
        
    V[:20, 49:] = V2_mapped
    
    # W2_part indices: i (0..2).
    # W total indices: 4*3 + i = 12 + i.
    W2_mapped = np.zeros((15, 12))
    for i in range(3):
        W2_mapped[12 + i, :] = W2_part[i, :]
        
    W[:15, 49:] = W2_mapped
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
