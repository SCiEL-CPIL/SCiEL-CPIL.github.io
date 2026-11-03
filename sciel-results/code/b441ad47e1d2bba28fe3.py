import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Base case: 2x2x2 Strassen decomposition
    # U, V, W shapes (4, 7)
    # Indices:
    # U: 00=0, 01=1, 10=2, 11=3
    # V: 00=0, 01=1, 10=2, 11=3
    # W: k*n+i. n=2.
    # C00 (i=0,k=0) -> 0
    # C10 (i=1,k=0) -> 1
    # C01 (i=0,k=1) -> 2
    # C11 (i=1,k=1) -> 3
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0], # a00
        [0, 0, 0, 0, 1, 0, 1],  # a01
        [0, 1, 0, 0, 0, 1, 0],  # a10
        [1, 1, 0, 1, 0, 0, -1]  # a11
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # b00
        [0, 0, 1, 0, 0, 1, 0],  # b01
        [0, 0, 0, 1, 0, 0, 1],  # b10
        [1, 0, -1, 0, 1, 0, 1]  # b11
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ], dtype=float)
    
    # 4x4x4 Decomposition (Rank 49)
    # We use 7 block multiplications.
    # Each block multiplication is 2x2, so we expand each into 7 scalar multiplications.
    # Total 49 columns.
    
    # The 7 Strassen products for 4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to construct U4, V4, W4 of size (16, 49), (16, 49), (16, 49).
    # Note: V4 for 4x4x4 will be part of the final V.
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49)) # For 4x4x4, V is 4x4 so 16 rows.
    W4 = np.zeros((16, 49)) # For 4x4x4, W is 4x4 so 16 rows.
    
    # Helper to add a scaled 2x2 decomposition to a block
    # block_u: indices for A blocks (4x4 -> 4 blocks of 2x2)
    # block_v: indices for B blocks
    # block_w: indices for C blocks
    # scale: multiplier
    # col_offset: starting column in U4, V4, W4
    
    def add_2x2_block(u_blocks, v_blocks, w_blocks, scale, col_offset):
        # u_blocks: dict {block_idx: coeff}
        # v_blocks: dict {block_idx: coeff}
        # w_blocks: dict {block_idx: coeff}
        for c in range(7):
            # Add U2[:, c] * scale to the appropriate rows of U4
            for b_idx, coeff in u_blocks.items():
                # Block b_idx corresponds to rows b_idx*2 .. b_idx*2+1 in A (flattened 4x4)
                # A is 4x4. Block 00 is rows 0-1, cols 0-1.
                # Flattened index: i*4 + j.
                # Block 00: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 4, 5
                # Block 01: (0,2), (0,3), (1,2), (1,3) -> 2, 3, 6, 7
                # Block 10: (2,0), (2,1), (3,0), (3,1) -> 8, 9, 12, 13
                # Block 11: (2,2), (2,3), (3,2), (3,3) -> 10, 11, 14, 15
                
                # Map block index to row indices in flattened 4x4
                # b_idx 0 -> 00, 1 -> 01, 2 -> 10, 3 -> 11
                if b_idx == 0:
                    rows = [0, 1, 4, 5]
                elif b_idx == 1:
                    rows = [2, 3, 6, 7]
                elif b_idx == 2:
                    rows = [8, 9, 12, 13]
                else:
                    rows = [10, 11, 14, 15]
                
                # The 2x2 decomposition U2 has rows 0,1,2,3 corresponding to a00, a01, a10, a11
                # We need to map these to the 4 rows of the block.
                # Block rows:
                # 0: a00
                # 1: a01
                # 2: a10
                # 3: a11
                # This matches U2 row order.
                
                for r_local, r_global in enumerate(rows):
                    U4[r_global, col_offset + c] += coeff * scale * U2[r_local, c]
            
            for b_idx, coeff in v_blocks.items():
                if b_idx == 0:
                    rows = [0, 1, 4, 5]
                elif b_idx == 1:
                    rows = [2, 3, 6, 7]
                elif b_idx == 2:
                    rows = [8, 9, 12, 13]
                else:
                    rows = [10, 11, 14, 15]
                
                for r_local, r_global in enumerate(rows):
                    V4[r_global, col_offset + c] += coeff * scale * V2[r_local, c]
            
            for b_idx, coeff in w_blocks.items():
                # W indices for 4x4: k*4 + i
                # Block 00 (i=0..1, k=0..1):
                # C00 (i=0,k=0) -> 0
                # C10 (i=1,k=0) -> 1
                # C01 (i=0,k=1) -> 4
                # C11 (i=1,k=1) -> 5
                # Block 01 (i=0..1, k=2..3):
                # C02 (i=0,k=2) -> 8
                # C12 (i=1,k=2) -> 9
                # C03 (i=0,k=3) -> 12
                # C13 (i=1,k=3) -> 13
                # Block 10 (i=2..3, k=0..1):
                # C20 (i=2,k=0) -> 2
                # C30 (i=3,k=0) -> 3
                # C21 (i=2,k=1) -> 6
                # C31 (i=3,k=1) -> 7
                # Block 11 (i=2..3, k=2..3):
                # C22 (i=2,k=2) -> 10
                # C32 (i=3,k=2) -> 11
                # C23 (i=2,k=3) -> 14
                # C33 (i=3,k=3) -> 15
                
                if b_idx == 0:
                    rows = [0, 1, 4, 5]
                elif b_idx == 1:
                    rows = [8, 9, 12, 13]
                elif b_idx == 2:
                    rows = [2, 3, 6, 7]
                else:
                    rows = [10, 11, 14, 15]
                
                for r_local, r_global in enumerate(rows):
                    W4[r_global, col_offset + c] += coeff * scale * W2[r_local, c]
                    
    # Define the 7 Strassen steps
    # Each step has U_blocks, V_blocks, W_blocks
    # U_blocks: {block_idx: coeff}
    
    steps = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    for i, (u_b, v_b, w_b) in enumerate(steps):
        add_2x2_block(u_b, v_b, w_b, 1, i * 7)
        
    # 4x4x2 Decomposition (Rank 28)
    # A is 4x4. B is 4x2.
    # B has 2 block rows, 1 block col.
    # B00 is top-left 2x2 of B. B10 is bottom-left 2x2 of B.
    # C is 4x2.
    # C00 = A00*B00 + A01*B10
    # C10 = A10*B00 + A11*B10
    
    U42 = np.zeros((16, 28))
    V42 = np.zeros((16, 28)) # V is 4x2, so 8 rows? No, V shape is (m*p, rank) = (4*2, 28) = (8, 28).
    # Wait, in the final output, V has shape (m*p, rank) = (4*6, 77) = (24, 77).
    # For the 4x4x2 part, V corresponds to B which is 4x2.
    # So V42 should have 8 rows (4*2).
    # But my add_2x2_block assumes 16 rows for V (4x4).
    # I need to adjust for V42.
    
    # Let's create a separate function or adapt.
    # B is 4x2.
    # Block 0 (top): rows 0-1 of B.
    # Block 1 (bottom): rows 2-3 of B.
    # Flattened B indices:
    # B00: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
    # B10: (2,0), (2,1), (3,0), (3,1) -> 4, 5, 6, 7
    
    # C is 4x2.
    # C00: (0,0), (0,1), (1,0), (1,1) -> k*4+i
    # C00: k=0, i=0 -> 0; k=1, i=0 -> 4; k=0, i=1 -> 1; k=1, i=1 -> 5
    # C10: k=0, i=2 -> 2; k=1, i=2 -> 6; k=0, i=3 -> 3; k=1, i=3 -> 7
    
    def add_2x2_block_4x2(u_blocks, v_blocks, w_blocks, scale, col_offset):
        for c in range(7):
            for b_idx, coeff in u_blocks.items():
                if b_idx == 0:
                    rows = [0, 1, 4, 5]
                elif b_idx == 1:
                    rows = [2, 3, 6, 7]
                elif b_idx == 2:
                    rows = [8, 9, 12, 13]
                else:
                    rows = [10, 11, 14, 15]
                
                for r_local, r_global in enumerate(rows):
                    U42[r_global, col_offset + c] += coeff * scale * U2[r_local, c]
            
            for b_idx, coeff in v_blocks.items():
                # v_blocks indices: 0 for B00, 1 for B10
                if b_idx == 0:
                    rows = [0, 1, 2, 3]
                else:
                    rows = [4, 5, 6, 7]
                
                for r_local, r_global in enumerate(rows):
                    V42[r_global, col_offset + c] += coeff * scale * V2[r_local, c]
            
            for b_idx, coeff in w_blocks.items():
                # w_blocks indices: 0 for C00, 1 for C10
                if b_idx == 0:
                    rows = [0, 1, 4, 5]
                else:
                    rows = [2, 3, 6, 7]
                
                for r_local, r_global in enumerate(rows):
                    W42[r_global, col_offset + c] += coeff * scale * W2[r_local, c]

    # Steps for 4x4x2
    # C00 = A00*B00 + A01*B10
    # C10 = A10*B00 + A11*B10
    
    steps_4x2 = [
        # A00*B00 -> C00
        ({0: 1}, {0: 1}, {0: 1}),
        # A01*B10 -> C00
        ({1: 1}, {1: 1}, {0: 1}),
        # A10*B00 -> C10
        ({2: 1}, {0: 1}, {1: 1}),
        # A11*B10 -> C10
        ({3: 1}, {1: 1}, {1: 1})
    ]
    
    for i, (u_b, v_b, w_b) in enumerate(steps_4x2):
        add_2x2_block_4x2(u_b, v_b, w_b, 1, i * 7)
        
    # Combine
    # U: 16 x 77
    # V: 24 x 77
    # W: 24 x 77
    
    U_final = np.zeros((16, 77))
    V_final = np.zeros((24, 77))
    W_final = np.zeros((24, 77))
    
    # First 49 columns: 4x4x4 part
    U_final[:, :49] = U4
    # V4 is 16x49. V_final first 16 rows correspond to B's first 4 cols.
    V_final[:16, :49] = V4
    # V_final rows 16-23 (B's last 2 cols) are 0 for this part.
    
    # W4 is 16x49. W_final first 16 rows?
    # W index is k*4+i.
    # For 4x4x4, k is 0..3.
    # So W4 rows 0..15 correspond to k=0..3, i=0..3.
    # This covers all of W_final rows 0..15?
    # W_final has 24 rows (k=0..5, i=0..3).
    # For 4x4x4, k is 0..3. So rows 0..15 are used.
    # Rows 16..23 (k=4,5) are 0.
    W_final[:16, :49] = W4
    
    # Next 28 columns: 4x4x2 part
    U_final[:, 49:] = U42
    # V42 is 8x28. V_final rows 16..23 correspond to B's last 2 cols.
    V_final[16:, 49:] = V42
    # V_final rows 0..15 are 0 for this part.
    
    # W42 is 8x28.
    # W42 rows 0..7 correspond to k=0..1, i=0..3.
    # But in the final W, the 4x4x2 part contributes to C's last 2 cols (k=4,5).
    # So we need to shift the rows of W42.
    # W index for k=4, i=0 is 4*4+0 = 16.
    # W index for k=5, i=3 is 5*4+3 = 23.
    # So W42 row 0 (k=0, i=0 in local 4x2) should go to W_final row 16 (k=4, i=0).
    # W42 row 1 (k=0, i=1) -> W_final row 17.
    # ...
    # W42 row 7 (k=1, i=3) -> W_final row 23.
    
    W_final[16:, 49:] = W42
    
    # Convert to lists
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    # Simple test
    res = construct()
    print("Rank:", len(res["U"][0]))
