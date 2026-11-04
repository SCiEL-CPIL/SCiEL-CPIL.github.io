"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    Strategy:
    - Use block-Winograd (Strassen) for the 4x4x4 part: rank 7 (actually 7 for 2x2, 
      but for 4x4x4 via 2x2 blocks it's 7*4=28? No, let me think carefully).
      
    Actually, for 4x4x4, the standard Strassen recursion gives:
    - 4x4x4 can be decomposed into 7 blocks of 2x2x2, each of rank 7, total rank 49? 
    No, that's not right either.
    
    Let me reconsider. The Strassen algorithm for 2x2x2 has rank 7.
    For 4x4x4, we can partition into 2x2 blocks, giving 4 subproblems of 2x2x2,
    but Strassen reduces the number of multiplications from 8 to 7.
    
    Actually, the standard way: 4x4x4 tensor rank is at most 7 * 4 = 28? No.
    
    Let me think about this differently. The matrix multiplication tensor for n x m x p
    has rank at most n*m*p (schoolbook).
    
    For 4x4x4, the known upper bound on tensor rank is 28 (using Strassen-based methods).
    Actually, the exact rank of the 4x4x4 matrix multiplication tensor is known to be 
    between 26 and 28. The best known upper bound is 28.
    
    A construction achieving rank 28 for 4x4x4:
    - Partition the 4x4x4 into four 2x2x2 blocks
    - Use Strassen for each, but with shared structure
    
    Actually, a simpler approach: use the known rank-28 decomposition for 4x4x4.
    
    For the 4x4x5 problem:
    - The 4x4x4 part has rank at most 28
    - The 4x4x1 part (the 5th column of the second matrix) has rank at most 16
    - Total rank at most 28 + 16 = 44
    
    Let me implement the rank-28 decomposition for 4x4x4 and rank-16 for 4x4x1.
    
    For 4x4x4 with rank 28, I'll use a block Strassen approach:
    Partition A (4x4), B (4x4), C (4x4) into 2x2 blocks:
    A = [[A11, A12], [A21, A22]]
    B = [[B11, B12], [B21, B22]]
    C = [[C11, C12], [C21, C22]]
    
    Using Strassen's 7 products for each pair of 2x2 blocks... 
    Actually, this gets complex. Let me use a known explicit construction.
    
    Alternative: I'll use a randomized NMTF-like approach to find a low-rank decomposition,
    but given the time constraint, let me first try the constructive approach.
    
    For 4x4x1: rank 16 is the schoolbook, which is optimal since it's just 16 scalar products.
    
    For 4x4x4: Let me use the fact that rank(4x4x4) <= 28.
    One way: view 4x4x4 as two 2x4x4 problems or similar.
    
    Actually, let me just implement a careful Strassen-based decomposition.
    
    Strassen for 2x2:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    For 4x4x4, partition into 2x2 blocks of size 2:
    We have 4 "block positions" for the output, each requiring 7 products of 2x2 matrices.
    But the inputs overlap, so we can't just do 4*7=28 independent Strassen calls.
    
    Actually, we CAN do this if we treat each output 2x2 block independently:
    C11 (top-left 2x2 of output) depends on all 4 input 2x2 blocks.
    Similarly for C12, C21, C22.
    
    So we need 4 separate Strassen computations, each with rank 7, total rank 28.
    But wait, each Strassen computation uses the same input blocks, so the U, V, W 
    matrices would have repeated entries. The rank is still 28 because each Strassen 
    product is a separate rank-1 tensor.
    
    Let me implement this.
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # We'll build the decomposition in two parts:
    # Part 1: 4x4x4 (first 4 columns of B), rank 28
    # Part 2: 4x4x1 (5th column of B), rank 16
    # Total rank: 44
    
    U_rows = n * m  # 16
    V_rows = m * p  # 20
    W_rows = p * n  # 20
    
    rank = 44
    U = np.zeros((U_rows, rank))
    V = np.zeros((V_rows, rank))
    W = np.zeros((W_rows, rank))
    
    # Part 1: 4x4x4 decomposition using 4 independent Strassen 2x2x2 decompositions
    # Each output 2x2 block (of the 4x4 output) is computed via Strassen on the 2x2 input blocks
    
    # Define the 2x2 block indices
    # A is 4x4, partitioned as:
    # A[0:2, 0:2] = A11, A[0:2, 2:4] = A12
    # A[2:4, 0:2] = A21, A[2:4, 2:4] = A22
    # B is 4x4 (first 4 cols), partitioned similarly
    # C is 4x4, partitioned similarly
    
    # For each output block position (oi, oj) in {0,1}x{0,1}:
    # C[2*oi:2*oi+2, 2*oj:2*oj+2] = Strassen(A_blocks, B_blocks)
    
    # Strassen formulas for computing C_block from A_block and B_block:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For a given output block (oi, oj), the A blocks are:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # These are the SAME for all output blocks!
    # The B blocks are:
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
    # B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    # Also the SAME for all output blocks!
    
    # Wait, that means all 4 output blocks use the same A and B blocks,
    # so we only need ONE Strassen computation, giving rank 7, not 28!
    # But that can't be right because the output blocks are different.
    
    # I made an error. The Strassen algorithm computes the entire 2x2 output from 
    # 2x2 inputs. For 4x4x4, we partition into 2x2 blocks, and each 2x2 output block
    # is computed from the 2x2 input blocks. But the input blocks are shared.
    
    # Actually, no. Let me reconsider. In the 4x4x4 case:
    # A is 4x4, B is 4x4, C = A*B is 4x4.
    # Partition A, B, C into 2x2 blocks.
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each C_ij is a sum of two 2x2x2 products.
    # We can use Strassen to reduce the number of 2x2x2 products.
    
    # The standard Strassen for the whole 4x4x4 treats it as a 2x2 "super" multiplication
    # where each element is a 2x2 matrix. This gives 7 "super" products, each of which
    # is a 2x2x2 tensor of rank 7, so total rank 7*7 = 49. That's worse than schoolbook 64?
    # No, 49 < 64, so it's better.
    
    # But I want rank 28. Let me look up the correct construction.
    
    # Actually, I recall that the rank of the 4x4x4 matrix multiplication tensor is 
    # known to be at most 28, achieved by a specific construction. Let me try a 
    # different approach.
    
    # Alternative: use 2x2x2 Strassen blocks more cleverly.
    
    # For now, let me just use schoolbook for everything to ensure correctness,
    # then try to optimize. Schoolbook rank = 4*4*5 = 80.
    
    # Let me reset and use schoolbook first.
    rank = n * m * p
    U = np.zeros((U_rows, rank))
    V = np.zeros((V_rows, rank))
    W = np.zeros((W_rows, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
