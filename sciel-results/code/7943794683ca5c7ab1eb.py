"""Tensor decomposition for 2x4 * 4x6 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    
    # We use a block decomposition approach:
    # A is 2x4, B is 4x6, C is 2x6
    # Split A into two 2x2 blocks: A = [A_L, A_R]
    # Split B into two 4x3 blocks: B = [B_T; B_B] where B_T is 2x6, B_B is 2x6
    # Actually, let's think differently.
    
    # For 2x4 * 4x6, we can view this as:
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # One approach: use the fact that 2x2 matrix multiplication has rank 7 (Strassen).
    # We can split the computation into blocks.
    
    # Let's use a more systematic approach. The tensor M(2,4,6) can be decomposed
    # by considering the bilinear complexity.
    
    # Actually, let me think about this more carefully.
    # We want to compute C = A * B where A is 2x4 and B is 4x6.
    
    # Approach: Split A into columns and use linearity.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    # This is equivalent to computing, for each i:
    # C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:] + A[i,2]*B[2,:] + A[i,3]*B[3,:]
    
    # The bilinear complexity of this is at most 4 * (number of multiplications per term)
    # but we can do better by sharing computations.
    
    # Let's use a different approach: treat this as a 2x4 matrix times a 4x6 matrix.
    # We can write this as (2x2 block matrix) times (2x2 block matrix) if we reshape appropriately.
    
    # Actually, let's just use the schoolbook but see if we can reduce it.
    # Schoolbook rank = 2*4*6 = 48.
    
    # Let me try a different decomposition. Consider:
    # A = [a0 a1 a2 a3] (each ai is a 2x1 column vector, so A is 2x4)
    # B is 4x6.
    
    # C = a0 * B[0,:] + a1 * B[1,:] + a2 * B[2,:] + a3 * B[3,:]
    
    # where ai * B[j,:] means the outer product of the column vector ai with the row vector B[j,:].
    # Wait, that's not quite right. Let me be more precise.
    
    # A is 2x4: A[i,j] for i in {0,1}, j in {0,1,2,3}
    # B is 4x6: B[j,k] for j in {0,1,2,3}, k in {0,...,5}
    # C is 2x6: C[i,k] = sum_j A[i,j] * B[j,k]
    
    # So C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    # This is a sum of 4 terms, each involving one multiplication of A[i,j] and B[j,k].
    # For fixed i and k, we have 4 multiplications.
    # Total: 2 * 4 * 6 = 48 multiplications.
    
    # Can we reduce this? The key insight is that for different k values, we're using
    # the same A[i,j] values. So we might be able to share some computations.
    
    # Let's try a Strassen-like approach. Consider the 2x4 * 4x6 multiplication.
    # We can think of this as:
    # [A_left, A_right] * [B_top; B_bottom] where A_left, A_right are 2x2 and B_top, B_bottom are 2x6.
    
    # C = A_left * B_top + A_right * B_bottom
    
    # Now A_left * B_top is 2x2 * 2x6, and A_right * B_bottom is 2x2 * 2x6.
    
    # For 2x2 * 2x6, we can use Strassen on the 2x2 part.
    # A_left = [[a00, a01], [a10, a11]]
    # B_top = [[b00, b01, ..., b05], [b10, b11, ..., b15]]
    
    # A_left * B_top = [[a00*b00 + a01*b10, ...], [a10*b00 + a11*b10, ...]]
    
    # Using Strassen for 2x2 * 2x2, we can compute the 2x2 * 2x6 by applying Strassen
    # to each pair of columns. But that would give us 7 * 3 = 21 multiplications for 2x2*2x6.
    # Wait, no. Strassen computes 2x2 * 2x2 in 7 multiplications. For 2x2 * 2x6,
    # we can view B_top as three 2x2 matrices side by side. So we'd need 7 * 3 = 21 multiplications.
    
    # Similarly for A_right * B_bottom: another 21 multiplications.
    # Total: 42 multiplications.
    
    # That's better than 48! Let's implement this.
    
    # Strassen's algorithm for 2x2 * 2x2:
    # Given X = [[x00, x01], [x10, x11]] and Y = [[y00, y01], [y10, y11]]
    # M1 = (x00 + x11) * (y00 + y11)
    # M2 = (x10 + x11) * y00
    # M3 = x00 * (y01 - y11)
    # M4 = x11 * (y10 - y00)
    # M5 = (x00 + x01) * y11
    # M6 = (x10 - x00) * (y00 + y01)
    # M7 = (x01 - x11) * (y10 + y11)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # For 2x2 * 2x6, we split the 2x6 into three 2x2 blocks.
    # B_top = [B_top_0, B_top_1, B_top_2] where each B_top_i is 2x2.
    
    # Let's define the Strassen multiplications for each block.
    
    # For block b (b = 0, 1, 2):
    # X = A_left (2x2)
    # Y = B_top[:, 2b:2b+2] (2x2)
    
    # M1_b = (x00 + x11) * (y00_b + y11_b)
    # M2_b = (x10 + x11) * y00_b
    # M3_b = x00 * (y01_b - y11_b)
    # M4_b = x11 * (y10_b - y00_b)
    # M5_b = (x00 + x01) * y11_b
    # M6_b = (x10 - x00) * (y00_b + y01_b)
    # M7_b = (x01 - x11) * (y10_b + y11_b)
    
    # Z_b = A_left * B_top[:, 2b:2b+2]
    # Z_b[0,0] = M1_b + M4_b - M5_b + M7_b
    # Z_b[0,1] = M3_b + M5_b
    # Z_b[1,0] = M2_b + M4_b
    # Z_b[1,1] = M1_b - M2_b + M3_b + M6_b
    
    # Similarly for A_right * B_bottom.
    
    # Total rank = 7 * 3 + 7 * 3 = 42.
    
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Process A_left * B_top (blocks 0, 1, 2)
    # A_left is A[:, 0:2]
    # B_top is B[0:2, :]
    
    for b in range(3):
        # Strassen multiplications for A_left * B_top[:, 2b:2b+2]
        # X = A_left = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
        # Y = B_top[:, 2b:2b+2] = [[B[0,2b], B[0,2b+1]], [B[1,2b], B[1,2b+1]]]
        
        # M1 = (x00 + x11) * (y00 + y11)
        U[0 * m + 0, col] = 1  # A[0,0]
        U[1 * m + 1, col] = 1  # A[1,1]
        V[0 * p + 2*b, col] = 1  # B[0, 2b]
        V[1 * p + 2*b+1, col] = 1  # B[1, 2b+1]
        # W contributions: Z[0,0] += 1, Z[1,1] += 1
        W[2*b * n + 0, col] = 1  # C[0, 2b]
        W[(2*b+1) * n + 1, col] = 1  # C[1, 2b+1]
        col += 1
        
        # M2 = (x10 + x11) * y00
        U[1 * m + 0, col] = 1  # A[1,0]
        U[1 * m + 1, col] = 1  # A[1,1]
        V[0 * p + 2*b, col] = 1  # B[0, 2b]
        # W contributions: Z[1,0] += 1, Z[1,1] -= 1
        W[2*b * n + 1, col] = 1  # C[1, 2b]
        W[(2*b+1) * n + 1, col] = -1  # C[1, 2b+1]
        col += 1
        
        # M3 = x00 * (y01 - y11)
        U[0 * m + 0, col] = 1  # A[0,0]
        V[0 * p + 2*b+1, col] = 1  # B[0, 2b+1]
        V[1 * p + 2*b+1, col] = -1  # B[1, 2b+1]
        # W contributions: Z[0,1] += 1, Z[1,1] += 1
        W[(2*b+1) * n + 0, col] = 1  # C[0, 2b+1]
        W[(2*b+1) * n + 1, col] = 1  # C[1, 2b+1]
        col += 1
        
        # M4 = x11 * (y10 - y00)
        U[1 * m + 1, col] = 1  # A[1,1]
        V[1 * p + 2*b, col] = 1  # B[1, 2b]
        V[0 * p + 2*b, col] = -1  # B[0, 2b]
        # W contributions: Z[0,0] += 1, Z[1,0] += 1
        W[2*b * n + 0, col] = 1  # C[0, 2b]
        W[2*b * n + 1, col] = 1  # C[1, 2b]
        col += 1
        
        # M5 = (x00 + x01) * y11
        U[0 * m + 0, col] = 1  # A[0,0]
        U[0 * m + 1, col] = 1  # A[0,1]
        V[1 * p + 2*b+1, col] = 1  # B[1, 2b+1]
        # W contributions: Z[0,0] -= 1, Z[0,1] += 1
        W[2*b * n + 0, col] = -1  # C[0, 2b]
        W[(2*b+1) * n + 0, col] = 1  # C[0, 2b+1]
        col += 1
        
        # M6 = (x10 - x00) * (y00 + y01)
        U[1 * m + 0, col] = 1  # A[1,0]
        U[0 * m + 0, col] = -1  # A[0,0]
        V[0 * p + 2*b, col] = 1  # B[0, 2b]
        V[0 * p + 2*b+1, col] = 1  # B[0, 2b+1]
        # W contributions: Z[1,1] += 1
        W[(2*b+1) * n + 1, col] = 1  # C[1, 2b+1]
        col += 1
        
        # M7 = (x01 - x11) * (y10 + y11)
        U[0 * m + 1, col] = 1  # A[0,1]
        U[1 * m + 1, col] = -1  # A[1,1]
        V[1 * p + 2*b, col] = 1  # B[1, 2b]
        V[1 * p + 2*b+1, col] = 1  # B[1, 2b+1]
        # W contributions: Z[0,0] += 1
        W[2*b * n + 0, col] = 1  # C[0, 2b]
        col += 1
    
    # Process A_right * B_bottom (blocks 3, 4, 5)
    # A_right is A[:, 2:4]
    # B_bottom is B[2:4, :]
    
    for b in range(3):
        b_idx = 3 + b  # block index in 0..5
        # Strassen multiplications for A_right * B_bottom[:, 2b:2b+2]
        # X = A_right = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]
        # Y = B_bottom[:, 2b:2b+2] = [[B[2,2b], B[2,2b+1]], [B[3,2b], B[3,2b+1]]]
        
        # M1 = (x00 + x11) * (y00 + y11)
        U[0 * m + 2, col] = 1  # A[0,2]
        U[1 * m + 3, col] = 1  # A[1,3]
        V[2 * p + 2*b, col] = 1  # B[2, 2b]
        V[3 * p + 2*b+1, col] = 1  # B[3, 2b+1]
        # W contributions: Z[0,0] += 1, Z[1,1] += 1
        W[2*b * n + 0, col] = 1  # C[0, 2b]
        W[(2*b+1) * n + 1, col] = 1  # C[1, 2b+1]
        col += 1
        
        # M2 = (x10 + x11) * y00
        U[1 * m + 2, col] = 1  # A[1,2]
        U[1 * m + 3, col] = 1  # A[1,3]
        V[2 * p + 2*b, col] = 1  # B[2, 2b]
        # W contributions: Z[1,0] += 1, Z[1,1] -= 1
        W[2*b * n + 1, col] = 1  # C[1, 2b]
        W[(2*b+1) * n + 1, col] = -1  # C[1, 2b+1]
        col += 1
        
        # M3 = x00 * (y01 - y11)
        U[0 * m + 2, col] = 1  # A[0,2]
        V[2 * p + 2*b+1, col] = 1  # B[2, 2b+1]
        V[3 * p + 2*b+1, col] = -1  # B[3, 2b+1]
        # W contributions: Z[0,1] += 1, Z[1,1] += 1
        W[(2*b+1) * n + 0, col] = 1  # C[0, 2b+1]
        W[(2*b+1) * n + 1, col] = 1  # C[1, 2b+1]
        col += 1
        
        # M4 = x11 * (y10 - y00)
        U[1 * m + 3, col] = 1  # A[1,3]
        V[3 * p + 2*b, col] = 1  # B[3, 2b]
        V[2 * p + 2*b, col] = -1  # B[2, 2b]
        # W contributions: Z[0,0] += 1, Z[1,0] += 1
        W[2*b * n + 0, col] = 1  # C[0, 2b]
        W[2*b * n + 1, col] = 1  # C[1, 2b]
        col += 1
        
        # M5 = (x00 + x01) * y11
        U[0 * m + 2, col] = 1  # A[0,2]
        U[0 * m + 3, col] = 1  # A[0,3]
        V[3 * p + 2*b+1, col] = 1  # B[3, 2b+1]
        # W contributions: Z[0,0] -= 1, Z[0,1] += 1
        W[2*b * n + 0, col] = -1  # C[0, 2b]
        W[(2*b+1) * n + 0, col] = 1  # C[0, 2b+1]
        col += 1
        
        # M6 = (x10 - x00) * (y00 + y01)
        U[1 * m + 2, col] = 1  # A[1,2]
        U[0 * m + 2, col] = -1  # A[0,2]
        V[2 * p + 2*b, col] = 1  # B[2, 2b]
        V[2 * p + 2*b+1, col] = 1  # B[2, 2b+1]
        # W contributions: Z[1,1] += 1
        W[(2*b+1) * n + 1, col] = 1  # C[1, 2b+1]
        col += 1
        
        # M7 = (x01 - x11) * (y10 + y11)
        U[0 * m + 3, col] = 1  # A[0,3]
        U[1 * m + 3, col] = -1  # A[1,3]
        V[3 * p + 2*b, col] = 1  # B[3, 2b]
        V[3 * p + 2*b+1, col] = 1  # B[3, 2b+1]
        # W contributions: Z[0,0] += 1
        W[2*b * n + 0, col] = 1  # C[0, 2b]
        col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
