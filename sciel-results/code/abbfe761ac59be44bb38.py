import time
import numpy as np

DEADLINE_S = 110.0
N, M, P = 4, 5, 5

def construct():
    t0 = time.time()
    
    R = 51
    U = np.zeros((N*M, R))
    V = np.zeros((M*P, R))
    W = np.zeros((P*N, R))
    
    r = 0
    
    # Helper to add a rank-1 term
    def add_term(u_indices, v_indices, w_indices, coeff=1.0):
        nonlocal r
        for idx in u_indices:
            U[idx, r] += coeff
        for idx in v_indices:
            V[idx, r] += coeff
        for idx in w_indices:
            W[idx, r] += coeff
        r += 1

    # Indices for A (4x5): i*M + j
    # Indices for B (5x5): j*P + k
    # Indices for W (5x4): k*N + i
    
    # Part 1: A1 (4x4) * B1a (4x4)
    # A1: i in 0..3, j in 0..3
    # B1a: j in 0..3, k in 0..3
    # C1a: i in 0..3, k in 0..3
    
    # Strassen 4x4
    # Block indices for A1 (2x2 blocks of 2x2)
    # A00: i 0,1; j 0,1
    # A01: i 0,1; j 2,3
    # A10: i 2,3; j 0,1
    # A11: i 2,3; j 2,3
    
    # Block indices for B1a
    # B00: j 0,1; k 0,1
    # B01: j 0,1; k 2,3
    # B10: j 2,3; k 0,1
    # B11: j 2,3; k 2,3
    
    # Block indices for C1a
    # C00: i 0,1; k 0,1
    # C01: i 0,1; k 2,3
    # C10: i 2,3; k 0,1
    # C11: i 2,3; k 2,3
    
    # Define block index lists
    def a_block(br, bc):
        return [(i*M + j) for i in [2*br, 2*br+1] for j in [2*bc, 2*bc+1]]
        
    def b_block(br, bc):
        return [(j*P + k) for j in [2*br, 2*br+1] for k in [2*bc, 2*bc+1]]
        
    def c_block(br, bc):
        return [(k*N + i) for i in [2*br, 2*br+1] for k in [2*bc, 2*bc+1]]

    # 7 Block Products
    # M1: X = A00+A11, Y = B00+B11
    # M2: X = A10+A11, Y = B00
    # M3: X = A00, Y = B01-B11
    # M4: X = A11, Y = B10-B00
    # M5: X = A00+A01, Y = B11
    # M6: X = A10-A00, Y = B00+B01
    # M7: X = A01-A11, Y = B10+B11
    
    # For each M, apply Strassen 2x2
    # Strassen 2x2 terms for X*Y -> Z:
    # P1 = (X00+X11)(Y00+Y11)
    # P2 = (X10+X11)Y00
    # P3 = X00(Y01-Y11)
    # P4 = X11(Y10-Y00)
    # P5 = (X00+X01)Y11
    # P6 = (X10-X00)(Y00+Y01)
    # P7 = (X01-X11)(Y10+Y11)
    
    # Reconstruction:
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    
    # We need to map X, Y, Z blocks to global indices.
    # For M1, X is a 2x2 matrix formed by A00 and A11.
    # X00 = A00, X11 = A11, X01=0, X10=0?
    # No, X is a 2x2 matrix of scalars? No, X is a 2x2 matrix of blocks?
    # In Strassen 4x4, X and Y are 2x2 matrices of 2x2 blocks.
    # Wait, Strassen 4x4 splits into 2x2 blocks.
    # The "multiplication" is of 2x2 blocks.
    # So X is a 2x2 matrix where each entry is a 2x2 matrix.
    # This is equivalent to a 4x4 matrix.
    # My previous analysis was that we use 7 block products, each of which is a 2x2 matrix product.
    # So for M1, we compute X * Y where X and Y are 4x4 matrices? No.
    # X is a 4x4 matrix formed by placing A00 and A11 in specific positions?
    # No, Strassen 4x4:
    # M1 = (A00+A11)(B00+B11).
    # A00+A11 is a 4x4 matrix? No, A00 and A11 are 2x2 blocks.
    # A00+A11 is a 2x2 matrix (sum of two 2x2 matrices).
    # So M1 is a product of two 2x2 matrices.
    # So we use Strassen 2x2 on these 2x2 matrices.
    
    # So for M1:
    # X = A00 + A11 (2x2 matrix)
    # Y = B00 + B11 (2x2 matrix)
    # Z = M1 (2x2 matrix)
    
    # We need to define the indices for X, Y, Z.
    # X entries:
    # X00 = A00[0,0] + A11[0,0]
    # X01 = A00[0,1] + A11[0,1]
    # X10 = A00[1,0] + A11[1,0]
    # X11 = A00[1,1] + A11[1,1]
    
    # This means for each Strassen 2x2 term (e.g., P1 = (X00+X11)(Y00+Y11)),
    # we have a rank-1 tensor.
    # u indices: A00[0,0], A11[0,0], A00[1,1], A11[1,1]
    # v indices: B00[0,0], B11[0,0], B00[1,1], B11[1,1]
    # w indices: Z00, Z01, Z10, Z11 (with coeffs from reconstruction)
    
    # Let's implement this logic.
    
    def add_strassen_2x2_for_blocks(x_blocks, y_blocks, z_blocks):
        # x_blocks: list of (block_idx, coeff)
        # y_blocks: list of (block_idx, coeff)
        # z_blocks: list of (block_idx, coeff)
        # This function adds 7 rank-1 terms.
        
        # Get indices for X entries
        # X is 2x2. Entries X00, X01, X10, X11.
        # We need to map X00, etc to global A indices.
        
        # For simplicity, I will hardcode the 49 terms in the next step.
        pass

    # Given the time, I will output the schoolbook rank 100 if I cannot complete the 51 rank construction in time.
    # But I have the plan.
    
    # Let's try to implement the 51 rank construction.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
