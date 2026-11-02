"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 4x4 matrix by a 4x9 matrix.
    
    Strategy:
    - Use Strassen's algorithm for 4x4 * 4x4 matrix multiplication (rank 49).
    - Split the 4x9 matrix B into two 4x4 blocks (B1, B2) and one 4x1 block (B3).
    - Use Strassen for 4x4 * 4x4 (rank 49 each) and schoolbook for 4x4 * 4x1 (rank 4).
    - Total rank: 49 + 49 + 4 = 102.
    
    The output factor W follows the convention W[k*n+i, r] where:
    - k is the column index of C (0..8)
    - i is the row index of C (0..3)
    - r is the rank index
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strassen decomposition for 4x4 * 4x4
    # This gives rank 49 for the 4x4*4x4 case
    def strassen_4x4_decomposition():
        """
        Return U, V, W for 4x4 * 4x4 matrix multiplication using Strassen.
        U has shape (4*4, 49) = (16, 49)
        V has shape (4*4, 49) = (16, 49)
        W has shape (4*4, 49) = (16, 49)
        
        Convention:
        U[i*m+j, r] where i is row of A, j is col of A
        V[j*p+k, r] where j is row of B, k is col of B
        W[k*n+i, r] where k is col of C, i is row of C
        """
        # Build U2, V2, W2 for 2x2*2x2 Strassen (rank 7)
        U2 = np.zeros((4, 7))
        V2 = np.zeros((4, 7))
        W2 = np.zeros((4, 7))
        
        # M1 = (A11+A22)*(B11+B22)
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        W2[0, 0] = 1; W2[3, 0] = 1
        
        # M2 = (A21+A22)*B11
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[2, 1] = 1; W2[3, 1] = 1
        
        # M3 = A11*(B12-B22)
        U2[0, 2] = 1
        V2[1, 2] = 1; V2[3, 2] = -1
        W2[1, 2] = 1; W2[3, 2] = 1
        
        # M4 = A22*(B21-B11)
        U2[3, 3] = 1
        V2[2, 3] = 1; V2[0, 3] = -1
        W2[2, 3] = 1; W2[0, 3] = 1
        
        # M5 = (A11+A12)*B22
        U2[0, 4] = 1; U2[1, 4] = 1
        V2[3, 4] = 1
        W2[1, 4] = 1; W2[3, 4] = 1
        
        # M6 = (A21-A11)*(B11+B12)
        U2[2, 5] = 1; U2[0, 5] = -1
        V2[0, 5] = 1; V2[1, 5] = 1
        W2[3, 5] = 1
        
        # M7 = (A12-A22)*(B21+B22)
        U2[1, 6] = 1; U2[3, 6] = -1
        V2[2, 6] = 1; V2[3, 6] = 1
        W2[0, 6] = 1; W2[2, 6] = 1
        
        # Now build the 4x4*4x4 decomposition using 7 products of 2x2*2x2
        U4 = np.zeros((16, 49))
        V4 = np.zeros((16, 49))
        W4 = np.zeros((16, 49))
        
        # Define block index mappings
        # For 4x4 matrix, blocks are 2x2:
        # Block (bi, bj) has rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
        # Flattened index: row*4 + col
        
        # Top-level Strassen products and their block contributions
        # Each product has coefficients for A blocks, B blocks, and C blocks
        # Format: (a_blocks: list of (block_row, block_col, coeff), 
        #          b_blocks: list of (block_row, block_col, coeff),
        #          c_blocks: list of (block_row, block_col, coeff))
        
        products = [
            # M1 = (A11+A22)*(B11+B22) -> C11, C22
            {
                'a': [((0,0), 1), ((1,1), 1)],
                'b': [((0,0), 1), ((1,1), 1)],
                'c': [((0,0), 1), ((1,1), 1)]
            },
            # M2 = (A21+A22)*B11 -> C21, C22
            {
                'a': [((1,0), 1), ((1,1), 1)],
                'b': [((0,0), 1)],
                'c': [((1,0), 1), ((1,1), -1)]
            },
            # M3 = A11*(B12-B22) -> C12, C22
            {
                'a': [((0,0), 1)],
                'b': [((0,1), 1), ((1,1), -1)],
                'c': [((0,1), 1), ((1,1), 1)]
            },
            # M4 = A22*(B21-B11) -> C11, C21
            {
                'a': [((1,1), 1)],
                'b': [((1,0), 1), ((0,0), -1)],
                'c': [((0,0), 1), ((1,0), 1)]
            },
            # M5 = (A11+A12)*B22 -> C12, C11
            {
                'a': [((0,0), 1), ((0,1), 1)],
                'b': [((1,1), 1)],
                'c': [((0,1), 1), ((0,0), -1)]
            },
            # M6 = (A21-A11)*(B11+B12) -> C22
            {
                'a': [((1,0), 1), ((0,0), -1)],
                'b': [((0,0), 1), ((0,1), 1)],
                'c': [((1,1), 1)]
            },
            # M7 = (A12-A22)*(B21+B22) -> C11
            {
                'a': [((0,1), 1), ((1,1), -1)],
                'b': [((1,0), 1), ((1,1), 1)],
                'c': [((0,0), 1)]
            },
        ]
        
        for top_idx, prod in enumerate(products):
            base_r = top_idx * 7
            
            # For each inner Strassen term (0..6)
            for inner_idx in range(7):
                r = base_r + inner_idx
                
                # A part: sum over block contributions
                for (ar, ac, coeff) in prod['a']:
                    for local_i in range(2):
                        for local_j in range(2):
                            global_i = 2*ar + local_i
                            global_j = 2*ac + local_j
                            U4[global_i*4 + global_j, r] += coeff * U2[local_i*2 + local_j, inner_idx]
                
                # B part: sum over block contributions
                for (br, bc, coeff) in prod['b']:
                    for local_j in range(2):
                        for local_k in range(2):
                            global_j = 2*br + local_j
                            global_k = 2*bc + local_k
                            V4[global_j*4 + global_k, r] += coeff * V2[local_j*2 + local_k, inner_idx]
                
                # C part: sum over block contributions
                for (cr, cc, coeff) in prod['c']:
                    for local_k in range(2):
                        for local_i in range(2):
                            global_k = 2*cr + local_k
                            global_i = 2*cc + local_i
                            W4[global_k*4 + global_i, r] += coeff * W2[local_k*2 + local_i, inner_idx]
        
        return U4, V4, W4
    
    # For 4x4 * 4x1, use schoolbook (rank 4)
    def schoolbook_4x4_4x1_decomposition():
        """
        Return U, V, W for 4x4 * 4x1 matrix multiplication (rank 4).
        U has shape (16, 4), V has shape (4, 4), W has shape (4, 4)
        """
        U = np.zeros((16, 4))
        V = np.zeros((4, 4))
        W = np.zeros((4, 4))
        
        # C[i,0] = sum_{j=0}^{3} A[i,j] * B[j,0]
        for j in range(4):
            for i in range(4):
                U[i * 4 + j, j] = 1
                V[j, j] = 1
                W[i, j] = 1
        
        return U, V, W
    
    # Build the full 4x4 * 4x9 decomposition
    # Split B into B1 (4x4), B2 (4x4), B3 (4x1)
    # C = A*B1 + A*B2 + A*B3
    
    U_strassen, V_strassen, W_strassen = strassen_4x4_decomposition()
    U_sb, V_sb, W_sb = schoolbook_4x4_4x1_decomposition()
    
    # Total rank: 49 + 49 + 4 = 102
    total_rank = 49 + 49 + 4
    
    U_full = np.zeros((16, total_rank))
    V_full = np.zeros((36, total_rank))  # m*p = 4*9 = 36
    W_full = np.zeros((36, total_rank))  # p*n = 9*4 = 36
    
    # First 49 terms: A*B1 (B columns 0-3)
    U_full[:, 0:49] = U_strassen
    V_full[0:16, 0:49] = V_strassen  # B1 is rows 0-3, cols 0-3 -> indices 0-15
    W_full[0:16, 0:49] = W_strassen  # C1 is rows 0-3, cols 0-3 -> indices 0-15
    
    # Next 49 terms: A*B2 (B columns 4-7)
    U_full[:, 49:98] = U_strassen
    V_full[16:32, 49:98] = V_strassen  # B2 is rows 0-3, cols 4-7 -> indices 16-31
    W_full[16:32, 49:98] = W_strassen  # C2 is rows 0-3, cols 4-7 -> indices 16-31
    
    # Last 4 terms: A*B3 (B column 8)
    U_full[:, 98:102] = U_sb
    V_full[32:36, 98:102] = V_sb  # B3 is rows 0-3, col 8 -> indices 32-35
    W_full[32:36, 98:102] = W_sb  # C3 is rows 0-3, col 8 -> indices 32-35
    
    # Verify
    np.random.seed(42)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 9)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((4, 9))
    for i in range(4):
        for k in range(9):
            total = 0
            for j in range(4):
                for r in range(total_rank):
                    total += A_test[i, j] * U_full[i * 4 + j, r] * B_test[j, k] * V_full[j * 9 + k, r] * W_full[k * 4 + i, r]
            C_computed[i, k] = total
    
    error = np.max(np.abs(C_computed - C_expected))
    if error > 1e-10:
        raise ValueError(f"Verification failed: error = {error}")
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

# EVOLVE-BLOCK-END
