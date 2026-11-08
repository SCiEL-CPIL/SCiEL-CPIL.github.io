"""Rewritten program for matrix multiplication 4,4,8 rank minimization."""
import numpy as np
import time
import itertools

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying a 4x4 matrix by a 4x8 matrix.
    
    We use the known result that 4x4*4x8 can be done with rank 98 using two independent
    4x4*4x4 multiplications (each rank 49 via Strassen-like decomposition).
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # We'll split the 8 columns into two groups of 4
    # C = A * B where A is 4x4, B is 4x8
    # B = [B1 | B2] where B1 and B2 are 4x4
    # C = A*B1 | A*B2
    # So we need two 4x4*4x4 multiplications
    
    # Use a rank-49 decomposition for 4x4*4x4 (Strassen-based)
    # The standard Strassen algorithm for 2x2*2x2 has rank 7
    # For 4x4*4x4, we can recursively apply Strassen: 
    # 4x4 = 2x2 blocks, so we need 7 multiplications of 2x2*2x2, each of rank 7
    # Total rank = 7 * 7 = 49
    
    # Let's construct the rank-49 decomposition for 4x4*4x4
    # Using recursive Strassen
    
    def strassen_decomposition_2x2():
        """
        Rank-7 decomposition for 2x2 * 2x2 matrix multiplication.
        Returns U (4x7), V (4x7), W (4x7) where:
        - U[i*2+j, c] corresponds to A[i,j]
        - V[j*2+k, c] corresponds to B[j,k]  
        - W[k*2+i, c] corresponds to C[i,k]
        """
        # Strassen's 7 products for 2x2:
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
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Indexing: A[i,j] -> U[i*2+j], B[j,k] -> V[j*2+k], C[i,k] -> W[k*2+i]
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1  # A00
        U[3, 0] = 1  # A11
        V[0, 0] = 1  # B00
        V[3, 0] = 1  # B11
        W[0, 0] = 1  # C00
        W[1, 0] = 1  # C01
        W[2, 0] = -1 # C10
        W[3, 0] = 1  # C11
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1  # A10
        U[3, 1] = 1  # A11
        V[0, 1] = 1  # B00
        W[2, 1] = 1  # C10
        W[3, 1] = -1 # C11
        
        # M3 = A00(B01-B11)
        U[0, 2] = 1  # A00
        V[1, 2] = 1  # B01
        V[3, 2] = -1 # B11
        W[1, 2] = 1  # C01
        W[3, 2] = 1  # C11
        
        # M4 = A11(B10-B00)
        U[3, 3] = 1  # A11
        V[2, 3] = 1  # B10
        V[0, 3] = -1 # B00
        W[0, 3] = 1  # C00
        W[2, 3] = 1  # C10
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1  # A00
        U[1, 4] = 1  # A01
        V[3, 4] = 1  # B11
        W[0, 4] = -1 # C00
        W[1, 4] = 1  # C01
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1  # A10
        U[0, 5] = -1 # A00
        V[0, 5] = 1  # B00
        V[1, 5] = 1  # B01
        W[3, 5] = 1  # C11
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1  # A01
        U[3, 6] = -1 # A11
        V[2, 6] = 1  # B10
        V[3, 6] = 1  # B11
        W[0, 6] = 1  # C00
        
        return U, V, W
    
    def strassen_decomposition_4x4():
        """
        Rank-49 decomposition for 4x4 * 4x4 matrix multiplication.
        Uses recursive Strassen: 7 multiplications of 2x2*2x2, each rank 7.
        """
        U_2x2, V_2x2, W_2x2 = strassen_decomposition_2x2()
        
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        # 4x4 matrix divided into 2x2 blocks:
        # A = [[A00, A01], [A10, A11]] where each block is 2x2
        # B = [[B00, B01], [B10, B11]]
        # C = [[C00, C01], [C10, C11]]
        #
        # Using Strassen at the 4x4 level:
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
        
        # Each Mi is a 2x2*2x2 multiplication with rank 7
        # So total rank = 7 * 7 = 49
        
        # Block indexing for 4x4:
        # A[i,j] where i,j in 0..3
        # Block A00: rows 0-1, cols 0-1 -> indices 0,1,2,3
        # Block A01: rows 0-1, cols 2-3 -> indices 4,5,6,7
        # Block A10: rows 2-3, cols 0-1 -> indices 8,9,10,11
        # Block A11: rows 2-3, cols 2-3 -> indices 12,13,14,15
        
        # For each of the 7 Strassen products, we have a 2x2*2x2 problem
        # The 2x2 matrices are the blocks themselves
        
        # Define which blocks go into each Strassen product
        # Strassen product s uses:
        # U side: combination of A blocks
        # V side: combination of B blocks
        
        # Let's define the block-level Strassen coefficients
        # For product s, U_block[s, block] and V_block[s, block]
        
        # Block indices: 0=A00, 1=A01, 2=A10, 3=A11
        #               4=B00, 5=B01, 6=B10, 7=B11
        
        U_blocks = np.zeros((7, 4))  # 7 products, 4 A-blocks
        V_blocks = np.zeros((7, 4))  # 7 products, 4 B-blocks
        W_blocks = np.zeros((7, 4))  # 7 products, 4 C-blocks (with signs)
        
        # M1 = (A00+A11)(B00+B11)
        U_blocks[0, 0] = 1  # A00
        U_blocks[0, 3] = 1  # A11
        V_blocks[0, 0] = 1  # B00
        V_blocks[0, 3] = 1  # B11
        W_blocks[0, 0] = 1  # C00
        W_blocks[0, 1] = 1  # C01
        W_blocks[0, 2] = -1 # C10
        W_blocks[0, 3] = 1  # C11
        
        # M2 = (A10+A11)B00
        U_blocks[1, 2] = 1  # A10
        U_blocks[1, 3] = 1  # A11
        V_blocks[1, 0] = 1  # B00
        W_blocks[1, 2] = 1  # C10
        W_blocks[1, 3] = -1 # C11
        
        # M3 = A00(B01-B11)
        U_blocks[2, 0] = 1  # A00
        V_blocks[2, 1] = 1  # B01
        V_blocks[2, 3] = -1 # B11
        W_blocks[2, 1] = 1  # C01
        W_blocks[2, 3] = 1  # C11
        
        # M4 = A11(B10-B00)
        U_blocks[3, 3] = 1  # A11
        V_blocks[3, 2] = 1  # B10
        V_blocks[3, 0] = -1 # B00
        W_blocks[3, 0] = 1  # C00
        W_blocks[3, 2] = 1  # C10
        
        # M5 = (A00+A01)B11
        U_blocks[4, 0] = 1  # A00
        U_blocks[4, 1] = 1  # A01
        V_blocks[4, 3] = 1  # B11
        W_blocks[4, 0] = -1 # C00
        W_blocks[4, 1] = 1  # C01
        
        # M6 = (A10-A00)(B00+B01)
        U_blocks[5, 2] = 1  # A10
        U_blocks[5, 0] = -1 # A00
        V_blocks[5, 0] = 1  # B00
        V_blocks[5, 1] = 1  # B01
        W_blocks[5, 3] = 1  # C11
        
        # M7 = (A01-A11)(B10+B11)
        U_blocks[6, 1] = 1  # A01
        U_blocks[6, 3] = -1 # A11
        V_blocks[6, 2] = 1  # B10
        V_blocks[6, 3] = 1  # B11
        W_blocks[6, 0] = 1  # C00
        
        # Now expand each block-level operation into element-level
        # For each Strassen product s (0..6), we have a 2x2*2x2 problem
        # with U_2x2, V_2x2, W_2x2 (each 4x7)
        
        # Block A00 corresponds to elements [0,1,2,3] in the 4x4 matrix
        # Block A01 corresponds to elements [4,5,6,7]
        # Block A10 corresponds to elements [8,9,10,11]
        # Block A11 corresponds to elements [12,13,14,15]
        
        # Similarly for B and C
        
        a_block_offsets = [0, 4, 8, 12]  # A00, A01, A10, A11
        b_block_offsets = [0, 4, 8, 12]  # B00, B01, B10, B11
        c_block_offsets = [0, 4, 8, 12]  # C00, C01, C10, C11
        
        for s in range(7):
            # For this Strassen product, we need to combine
            # the 2x2*2x2 decomposition with the block coefficients
            
            # The 2x2*2x2 decomposition has 7 terms
            # Each term t (0..6) has:
            # U_2x2[local_a_idx, t] for local 2x2 A indices
            # V_2x2[local_b_idx, t] for local 2x2 B indices
            # W_2x2[local_c_idx, t] for local 2x2 C indices
            
            for t in range(7):
                col = s * 7 + t
                
                # U side: sum over A blocks
                for ab in range(4):
                    if U_blocks[s, ab] != 0:
                        # For this block, add U_2x2 values
                        for local_a in range(4):
                            if U_2x2[local_a, t] != 0:
                                global_a = a_block_offsets[ab] + local_a
                                U[global_a, col] += U_blocks[s, ab] * U_2x2[local_a, t]
                
                # V side: sum over B blocks
                for bb in range(4):
                    if V_blocks[s, bb] != 0:
                        for local_b in range(4):
                            if V_2x2[local_b, t] != 0:
                                global_b = b_block_offsets[bb] + local_b
                                V[global_b, col] += V_blocks[s, bb] * V_2x2[local_b, t]
                
                # W side: sum over C blocks
                for cb in range(4):
                    if W_blocks[s, cb] != 0:
                        for local_c in range(4):
                            if W_2x2[local_c, t] != 0:
                                global_c = c_block_offsets[cb] + local_c
                                W[global_c, col] += W_blocks[s, cb] * W_2x2[local_c, t]
        
        return U, V, W
    
    # Get the 4x4*4x4 decomposition
    U_44, V_44, W_44 = strassen_decomposition_4x4()
    
    # Now build the 4x4*4x8 decomposition
    # Split B into B1 (cols 0-3) and B2 (cols 4-7)
    # C = A*B1 | A*B2
    
    # U is the same for both halves (A is 4x4)
    # V1 uses B1, V2 uses B2
    # W1 gives C1, W2 gives C2
    
    # U shape: (16, 49)
    # V1 shape: (16, 49), V2 shape: (16, 49)
    # W1 shape: (16, 49), W2 shape: (16, 49)
    
    # Total rank = 49 + 49 = 98
    rank = 98
    
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))  # m*p = 4*8 = 32
    W = np.zeros((32, rank))  # p*n = 8*4 = 32
    
    # First half: A * B1 -> C1
    # U[:, 0:49] = U_44
    U[:, 0:49] = U_44
    
    # V1: B1 is the first 4 columns of B (4x4)
    # B1[j,k] corresponds to V_44[j*4+k, :]
    # In the full 4x8 B, B1 is at rows 0-3, cols 0-3
    # V index: j*p + k where j in 0..3, k in 0..7
    # For B1: j in 0..3, k in 0..3 -> V[j*8+k]
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V_44[j*4+k, :]
    
    # W1: C1 is the first 4 rows of C (4x4)
    # C1[i,k] corresponds to W_44[k*4+i, :]
    # In the full 4x8 C, C1 is at rows 0-3, cols 0-3
    # W index: k*n + i where k in 0..7, i in 0..3
    # For C1: k in 0..3, i in 0..3 -> W[k*4+i]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W_44[k*4+i, :]
    
    # Second half: A * B2 -> C2
    # U[:, 49:98] = U_44
    U[:, 49:98] = U_44
    
    # V2: B2 is the last 4 columns of B (4x4)
    # B2[j,k] where k is 4..7 in the full B
    # In V_44, B2[j,k-4] -> V_44[j*4+(k-4)]
    # In full V: B2[j,k] -> V[j*8+k] where k in 4..7
    for j in range(4):
        for k in range(4):
            V[j*8+(k+4), 49:98] = V_44[j*4+k, :]
    
    # W2: C2 is the last 4 rows of C (4x4)
    # C2[i,k] where k is 4..7 in the full C
    # In W_44, C2[i,k-4] -> W_44[(k-4)*4+i]
    # In full W: C2[i,k] -> W[k*4+i] where k in 4..7
    for k in range(4):
        for i in range(4):
            W[(k+4)*4+i, 49:98] = W_44[k*4+i, :]
    
    # Convert to lists of integers (entries are already integers)
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Ensure all entries are integers
    U_list = [[int(round(x)) for x in row] for row in U_list]
    V_list = [[int(round(x)) for x in row] for row in V_list]
    W_list = [[int(round(x)) for x in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
