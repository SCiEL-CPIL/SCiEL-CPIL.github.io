import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 140.0

    # We use a block-Strassen approach.
    # Split A (4x4) into four 2x2 blocks.
    # Split B (4x8) into eight 2x4 blocks (4 rows, 8 cols -> 2x2 block grid of 2x4 blocks).
    # Actually, let's be more precise.
    # A is 4x4. Blocks A_00, A_01, A_10, A_11 are 2x2.
    # B is 4x8. Blocks B_00, B_01, B_10, B_11 are 2x4? No.
    # B has 4 rows. So 2 blocks of 2 rows.
    # B has 8 cols. So 4 blocks of 2 cols.
    # So B is a 2x4 grid of 2x2 blocks.
    # Let's call them B_{i,j} where i in {0,1} (row block), j in {0,1,2,3} (col block).
    
    # C is 4x8. Blocks C_{i,j} where i in {0,1}, j in {0,1,2,3}. Each is 2x2.
    
    # We can apply Strassen on the "row block" dimension (size 2).
    # For each column block j (0..3), we compute C_{0,j} and C_{1,j} using the Strassen formula
    # applied to the 2x2 block matrix multiplication:
    # [C_{0,j}]   [A_00 A_01]   [B_{0,j}]
    # [C_{1,j}] = [A_10 A_11] * [B_{1,j}]
    
    # Strassen for 2x2 blocks (each block is 2x2 for A, 2x2 for B_{i,j}, 2x2 for C_{i,j}):
    # P_0 = (A_00 + A_11) * (B_{0,j} + B_{1,j})
    # P_1 = (A_10 + A_11) * B_{0,j}
    # P_2 = A_00 * (B_{1,j} - B_{0,j})
    # P_3 = A_11 * (B_{0,j} - B_{1,j})
    # P_4 = (A_00 + A_01) * B_{1,j}
    # P_5 = (A_10 - A_00) * (B_{0,j} + B_{1,j})
    # P_6 = (A_01 - A_11) * (B_{1,j} - B_{0,j})
    
    # C_{0,j} = P_0 + P_3 - P_4 + P_6
    # C_{1,j} = P_1 + P_2 + P_3 + P_5
    
    # This uses 7 products per column block j.
    # There are 4 column blocks.
    # Total products = 4 * 7 = 28 products of 2x2 matrices.
    # Each 2x2 matrix multiplication has rank 7 (Strassen) or 4 (schoolbook).
    # If we use Strassen for the inner 2x2 products, total rank = 28 * 7 = 196.
    # If we use schoolbook for the inner 2x2 products, total rank = 28 * 4 = 112.
    # Wait, the current program has rank 98. How?
    # Ah, the current program uses a specific Strassen-based decomposition with rank 98.
    # Let's check: 4 * 4 * 8 = 128 (schoolbook).
    # 112 is the standard block Strassen rank.
    # 98 is better. How?
    # Maybe by sharing some of the sums/differences?
    
    # Let's stick with the known 112 rank decomposition first, and see if we can optimize.
    # Actually, the current program claims rank 98. Let's trust that and try to improve or match it.
    # The current program's rank is 98.
    
    # Let's implement the 112 rank decomposition cleanly.
    rank = 112
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2 decomposition (rank 7)
    # U_2, V_2, W_2 are 4x7
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # c=0: (A00+A11)(B00+B11)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    # c=1: (A10+A11)B00
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    # c=2: A00(B11-B00)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    # c=3: A11(B00-B11)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    # c=4: (A00+A01)B11
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    # c=5: (A10-A00)(B00+B11)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    # c=6: (A01-A11)(B11-B00)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1
    
    # Now, for each of the 4 column blocks (j=0..3), we apply the outer Strassen.
    # The outer Strassen uses 7 products, each of which is a 2x2 matrix product.
    # Each 2x2 matrix product uses the inner Strassen (rank 7).
    # So for each j, we have 7 * 7 = 49 terms.
    # Total terms = 4 * 49 = 196. This is too high.
    
    # Wait, the standard block Strassen for 4x4x8 is:
    # Split A into 2x2 blocks of 2x2.
    # Split B into 2x4 blocks of 2x2.
    # The "outer" multiplication is (2x2 blocks) * (2x4 blocks) -> (2x4 blocks).
    # This is not a square multiplication.
    
    # Let's re-examine.
    # A is 4x4. B is 4x8.
    # We can split A into 2x2 blocks (each 2x2).
    # We can split B into 2x4 blocks (each 2x2).
    # The product C is 4x8, split into 2x4 blocks (each 2x2).
    
    # The multiplication of a 2x2 block matrix A by a 2x4 block matrix B
    # is a block matrix multiplication where the "scalar" multiplication is replaced by 2x2 matrix multiplication.
    # Let A_blocks be a 2x2 matrix of 2x2 blocks.
    # Let B_blocks be a 2x4 matrix of 2x2 blocks.
    # C_blocks = A_blocks * B_blocks (2x4 matrix of 2x2 blocks).
    
    # We can use Strassen for the 2x2 * 2x4 multiplication?
    # Strassen works for square matrices.
    # We can pad B to be 2x2 (of 2x4 blocks) by adding a zero block?
    # Or we can use a different approach.
    
    # Let's use the fact that C_{i,j} = sum_k A_{i,k} B_{k,j}.
    # For fixed j, C_{0,j} = A_00 B_{0,j} + A_01 B_{1,j}
    #          C_{1,j} = A_10 B_{0,j} + A_11 B_{1,j}
    
    # We can compute these using Strassen on the 2x2 "system":
    # [C_{0,j}]   [A_00 A_01]   [B_{0,j}]
    # [C_{1,j}] = [A_10 A_11] * [B_{1,j}]
    
    # This is exactly the 2x2 Strassen case, where the "elements" are 2x2 matrices.
    # So we need 7 products of 2x2 matrices for each j.
    # Each 2x2 matrix product has rank 7 (Strassen) or 4 (schoolbook).
    # If we use schoolbook for the inner products, rank = 4 * 7 * 4 = 112.
    # If we use Strassen for the inner products, rank = 4 * 7 * 7 = 196.
    
    # So the 112 rank decomposition uses schoolbook for the inner 2x2 products.
    # Let's implement this.
    
    rank = 112
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Schoolbook 2x2x2 decomposition (rank 4)
    U_2 = np.zeros((4, 4))
    V_2 = np.zeros((4, 4))
    W_2 = np.zeros((4, 4))
    
    # c=0: A00*B00
    U_2[0, 0] = 1; V_2[0, 0] = 1; W_2[0, 0] = 1
    # c=1: A01*B10
    U_2[1, 1] = 1; V_2[2, 1] = 1; W_2[0, 1] = 1
    # c=2: A10*B01
    U_2[2, 2] = 1; V_2[1, 2] = 1; W_2[2, 2] = 1
    # c=3: A11*B11
    U_2[3, 3] = 1; V_2[3, 3] = 1; W_2[3, 3] = 1
    
    # Outer Strassen specs
    A_specs = [
        [(1, 0), (1, 3)],   # P0: A00 + A11
        [(1, 2), (1, 3)],   # P1: A10 + A11
        [(1, 0)],           # P2: A00
        [(1, 3)],           # P3: A11
        [(1, 0), (1, 1)],   # P4: A00 + A01
        [(1, 2), (-1, 0)],  # P5: A10 - A00
        [(1, 1), (-1, 3)],  # P6: A01 - A11
    ]
    
    B_specs = [
        [(1, 0), (1, 2)],   # P0: B_{0,j} + B_{1,j} (j=0 for B00, j=2 for B11? No.)
        # Wait, B_specs should be for the 2x2 block matrix [B_{0,j}, B_{1,j}]^T.
        # Let's index B blocks as B_{0,j} and B_{1,j}.
        # In the 2x2 Strassen formula, the "B matrix" is [[B_{0,j}], [B_{1,j}]].
        # So B_specs should refer to B_{0,j} and B_{1,j}.
        [(1, 0), (1, 1)],   # P0: B_{0,j} + B_{1,j}
        [(1, 0)],           # P1: B_{0,j}
        [(1, 1), (-1, 0)],  # P2: B_{1,j} - B_{0,j}
        [(1, 0), (-1, 1)],  # P3: B_{0,j} - B_{1,j}
        [(1, 1)],           # P4: B_{1,j}
        [(1, 0), (1, 1)],   # P5: B_{0,j} + B_{1,j}
        [(1, 1), (-1, 0)],  # P6: B_{1,j} - B_{0,j}
    ]
    
    C_specs = [
        [(1, 0), (1, 1)],   # P0: C_{0,j} + C_{1,j}
        [(1, 1)],           # P1: C_{1,j}
        [(1, 0)],           # P2: C_{0,j}
        [(1, 0), (1, 1)],   # P3: C_{0,j} + C_{1,j}
        [(-1, 0), (1, 1)],  # P4: -C_{0,j} + C_{1,j}
        [(1, 1)],           # P5: C_{1,j}
        [(1, 0)],           # P6: C_{0,j}
    ]
    
    # Let's verify the C_specs.
    # C_{0,j} = P0 + P3 - P4 + P6
    # C_{1,j} = P1 + P2 + P3 + P5
    
    # P0 contributes to C_{0,j} and C_{1,j}
    # P1 contributes to C_{1,j}
    # P2 contributes to C_{0,j}
    # P3 contributes to C_{0,j} and C_{1,j}
    # P4 contributes to -C_{0,j} and C_{1,j}
    # P5 contributes to C_{1,j}
    # P6 contributes to C_{0,j}
    
    # So:
    # C_{0,j} = 1*P0 + 0*P1 + 1*P2 + 1*P3 - 1*P4 + 0*P5 + 1*P6
    # C_{1,j} = 1*P0 + 1*P1 + 0*P2 + 1*P3 + 1*P4 + 1*P5 + 0*P6
    
    # Let's check:
    # C_{0,j} = (A00+A11)(B0j+B1j) + A11(B0j-B1j) - (A00+A01)B1j + (A01-A11)(B1j-B0j)
    # = A00B0j + A00B1j + A11B0j + A11B1j + A11B0j - A11B1j - A00B1j - A01B1j + A01B1j - A01B0j - A11B1j + A11B0j
    # = A00B0j + 2*A11B0j - A01B0j - A11B1j
    # This is not C_{0,j} = A00B0j + A01B1j.
    
    # I made a mistake in the Strassen formula.
    # The correct Strassen formula for 2x2 is:
    # P0 = (A00+A11)(B00+B11)
    # P1 = (A10+A11)B00
    # P2 = A00(B11-B00)
    # P3 = A11(B00-B11)
    # P4 = (A00+A01)B11
    # P5 = (A10-A00)(B00+B11)
    # P6 = (A01-A11)(B11-B00)
    
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    
    # In our case, the "B matrix" is [[B_{0,j}], [B_{1,j}]].
    # So B00 = B_{0,j}, B01 = 0, B10 = B_{1,j}, B11 = 0.
    # Wait, that's not right. The B matrix is 2x2, but we only have two blocks B_{0,j} and B_{1,j}.
    # The multiplication is [A_00 A_01; A_10 A_11] * [B_{0,j}; B_{1,j}].
    # This is a 2x2 matrix times a 2x1 matrix.
    # We can pad the 2x1 matrix to a 2x2 matrix by adding a zero column.
    # Let B' = [B_{0,j} 0; B_{1,j} 0].
    # Then A * B' = [A_00B_{0,j} + A_01B_{1,j}, 0; A_10B_{0,j} + A_11B_{1,j}, 0].
    # So C_{0,j} = (A * B')_{0,0} and C_{1,j} = (A * B')_{1,0}.
    
    # Now apply Strassen to A * B':
    # P0 = (A00+A11)(B'00+B'11) = (A00+A11)(B_{0,j}+0) = (A00+A11)B_{0,j}
    # P1 = (A10+A11)B'00 = (A10+A11)B_{0,j}
    # P2 = A00(B'11-B'00) = A00(0-B_{0,j}) = -A00B_{0,j}
    # P3 = A11(B'00-B'11) = A11(B_{0,j}-0) = A11B_{0,j}
    # P4 = (A00+A01)B'11 = (A00+A01)*0 = 0
    # P5 = (A10-A00)(B'00+B'11) = (A10-A00)B_{0,j}
    # P6 = (A01-A11)(B'11-B'00) = (A01-A11)(0-B_{0,j}) = -(A01-A11)B_{0,j}
    
    # C'00 = P0 + P3 - P4 + P6 = (A00+A11)B_{0,j} + A11B_{0,j} - 0 - (A01-A11)B_{0,j}
    # = (A00+A11+A11-A01+A11)B_{0,j} = (A00+3A11-A01)B_{0,j}
    # This is not C_{0,j} = A00B_{0,j} + A01B_{1,j}.
    
    # The issue is that B'11 is 0, but in the Strassen formula, B'11 is the bottom-right block.
    # In our case, B' = [B_{0,j} 0; B_{1,j} 0].
    # So B'00 = B_{0,j}, B'01 = 0, B'10 = B_{1,j}, B'11 = 0.
    
    # Let's re-evaluate the Strassen products:
    # P0 = (A00+A11)(B'00+B'11) = (A00+A11)(B_{0,j}+0) = (A00+A11)B_{0,j}
    # P1 = (A10+A11)B'00 = (A10+A11)B_{0,j}
    # P2 = A00(B'11-B'00) = A00(0-B_{0,j}) = -A00B_{0,j}
    # P3 = A11(B'00-B'11) = A11(B_{0,j}-0) = A11B_{0,j}
    # P4 = (A00+A01)B'11 = (A00+A01)*0 = 0
    # P5 = (A10-A00)(B'00+B'11) = (A10-A00)B_{0,j}
    # P6 = (A01-A11)(B'11-B'00) = (A01-A11)(0-B_{0,j}) = -(A01-A11)B_{0,j}
    
    # C'00 = P0 + P3 - P4 + P6 = (A00+A11)B_{0,j} + A11B_{0,j} - 0 - (A01-A11)B_{0,j}
    # = (A00 + A11 + A11 - A01 + A11)B_{0,j} = (A00 + 3A11 - A01)B_{0,j}
    
    # This is still wrong. The problem is that the Strassen formula assumes a full 2x2 B matrix.
    # In our case, B'01 and B'11 are zero.
    # So we are wasting some of the Strassen products.
    
    # Let's use a different approach.
    # We can directly compute C_{0,j} and C_{1,j} using 4 products:
    # C_{0,j} = A00B_{0,j} + A01B_{1,j}
    # C_{1,j} = A10B_{0,j} + A11B_{1,j}
    
    # This uses 4 products per j, total 16 products.
    # Each product is 2x2, rank 4 (schoolbook) or 7 (Strassen).
    # Total rank = 16 * 4 = 64 (schoolbook) or 16 * 7 = 112 (Strassen).
    
    # Wait, the schoolbook rank for 4x4x8 is 128.
    # If we use 16 products of 2x2 matrices, and each has rank 4, total rank is 64.
    # This is better than 128.
    
    # Let's implement this.
    rank = 64
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Schoolbook 2x2x2 decomposition (rank 4)
    U_2 = np.zeros((4, 4))
    V_2 = np.zeros((4, 4))
    W_2 = np.zeros((4, 4))
    
    U_2[0, 0] = 1; V_2[0, 0] = 1; W_2[0, 0] = 1
    U_2[1, 1] = 1; V_2[2, 1] = 1; W_2[0, 1] = 1
    U_2[2, 2] = 1; V_2[1, 2] = 1; W_2[2, 2] = 1
    U_2[3, 3] = 1; V_2[3, 3] = 1; W_2[3, 3] = 1
    
    # For each j (0..3), we compute 4 products.
    # Product 0: A00 * B_{0,j}
    # Product 1: A01 * B_{1,j}
    # Product 2: A10 * B_{0,j}
    # Product 3: A11 * B_{1,j}
    
    for j in range(4):
        for prod_idx in range(4):
            col = j * 4 + prod_idx
            
            # Determine which A block and which B block
            if prod_idx == 0:
                a_block = 0  # A00
                b_block = 0  # B_{0,j}
            elif prod_idx == 1:
                a_block = 1  # A01
                b_block = 1  # B_{1,j}
            elif prod_idx == 2:
                a_block = 2  # A10
                b_block = 0  # B_{0,j}
            else:
                a_block = 3  # A11
                b_block = 1  # B_{1,j}
            
            # A block indices
            a_row_start = (a_block // 2) * 2
            a_col_start = (a_block % 2) * 2
            
            # B block indices
            b_row_start = (b_block // 2) * 2
            b_col_start = j * 2  # Since B has 4 col blocks of size 2
            
            # For each element in the 2x2 A block and 2x2 B block
            for r in range(2):
                for c in range(2):
                    a_global_r = a_row_start + r
                    a_global_c = a_col_start + c
                    a_idx = a_global_r * 4 + a_global_c
                    
                    b_global_r = b_row_start + r
                    b_global_c = b_col_start + c
                    b_idx = b_global_r * 8 + b_global_c
                    
                    # The product A_{a_global_r, a_global_c} * B_{b_global_r, b_global_c}
                    # contributes to C_{a_global_r, b_global_c}
                    c_global_r = a_global_r
                    c_global_c = b_global_c
                    c_idx = c_global_c * 4 + c_global_r
                    
                    # The 2x2 schoolbook product has 4 terms.
                    # We need to map (r, c) to the schoolbook index.
                    # Schoolbook index: 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                    # But in our U_2, V_2, W_2, the indices are:
                    # U_2[0,0]=1, V_2[0,0]=1, W_2[0,0]=1 -> (0,0)
                    # U_2[1,1]=1, V_2[2,1]=1, W_2[0,1]=1 -> (0,1)
                    # U_2[2,2]=1, V_2[1,2]=1, W_2[2,2]=1 -> (1,0)
                    # U_2[3,3]=1, V_2[3,3]=1, W_2[3,3]=1 -> (1,1)
                    
                    # So the schoolbook index is r*2 + c.
                    sb_idx = r * 2 + c
                    
                    # U_2[a_eff_idx, sb_idx] where a_eff_idx is the index in the 2x2 A block.
                    # a_eff_idx = r*2 + c
                    u_val = U_2[a_eff_idx, sb_idx]
                    if u_val != 0:
                        U[a_idx, col] += u_val
                    
                    v_val = V_2[b_eff_idx, sb_idx]
                    if v_val != 0:
                        V[b_idx, col] += v_val
                    
                    w_val = W_2[p_eff_idx, sb_idx]
                    if w_val != 0:
                        W[c_idx, col] += w_val
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
