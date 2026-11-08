"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _is_half_int(x):
    """Check if x is an integer or half-integer (multiple of 0.5)."""
    return x == int(x) or x == (int(x) + 0.5) or x == (int(x) - 0.5)

def _rank_of_matrix(mat):
    """Compute numerical rank of a matrix."""
    try:
        U, s, Vh = np.linalg.svd(mat, full_matrices=False)
        return np.sum(s > 1e-10)
    except:
        return 0

def _build_strassen_2x2_2x2():
    """
    Build Strassen decomposition for 2x2 * 2x2.
    Returns (U, V, W) where U is 4x7, V is 4x7, W is 4x7.
    Indexing: U[i*2+j, k], V[j*2+k, k], W[k*2+i, k] for i,j,k in {0,1}
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    # C = A * B where A is 2x2, B is 2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B10 - B00)
    # M4 = A11 (B01 - B11)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00) (B00 + B01)
    # M7 = (A01 - A11) (B10 + B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    W[0, 0] = 1  # C00 (k=0, i=0)
    W[3, 0] = 1  # C11 (k=1, i=1)
    
    # M2 = (A10 + A11) B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10 (k=0, i=1)
    W[3, 1] = -1 # C11 (k=1, i=1)
    
    # M3 = A00 (B10 - B00)
    U[0, 2] = 1  # A00
    V[2, 2] = 1  # B10
    V[0, 2] = -1 # B00
    W[2, 2] = 1  # C10 (k=0, i=1)
    W[1, 2] = 1  # C01 (k=1, i=0)
    
    # M4 = A11 (B01 - B11)
    U[3, 3] = 1  # A11
    V[1, 3] = 1  # B01
    V[3, 3] = -1 # B11
    W[0, 3] = 1  # C00 (k=0, i=0)
    W[2, 3] = 1  # C10 (k=0, i=1)
    
    # M5 = (A00 + A01) B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1 # C00 (k=0, i=0)
    W[1, 4] = 1  # C01 (k=1, i=0)
    
    # M6 = (A10 - A00) (B00 + B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1 # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    W[2, 5] = 1  # C10 (k=0, i=1)
    
    # M7 = (A01 - A11) (B10 + B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1 # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    W[1, 6] = 1  # C01 (k=1, i=0)
    W[3, 6] = 1  # C11 (k=1, i=1)
    
    return U, V, W

def _build_strassen_4x4_4x4():
    """
    Build Strassen decomposition for 4x4 * 4x4 by applying Strassen to 2x2 blocks.
    Each 2x2 block multiplication uses rank 7.
    We have 4 block positions, but Strassen shares computations.
    
    Actually, let's use the standard approach: view 4x4 as 2x2 blocks of 2x2.
    This gives us a decomposition of rank 7 * 4 = 28? No, Strassen for 4x4 is rank 7^2 = 49?
    
    Let me think more carefully. The standard Strassen for n x n where n is power of 2:
    T(n) = 7 T(n/2), so T(4) = 7 * T(2) = 7 * 7 = 49.
    
    But we need to be careful about the tensor structure.
    
    For 4x4 * 4x4, we can partition into 2x2 blocks:
    A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    
    Then C = A*B has blocks Cij = sum Aik * Bkj
    
    Using Strassen at the block level:
    M1 = (A00+A11)(B00+B11) -> 2x2 * 2x2, rank 7
    M2 = (A10+A11) B00 -> 2x2 * 2x2, rank 7
    M3 = A00 (B10-B00) -> 2x2 * 2x2, rank 7
    M4 = A11 (B01-B11) -> 2x2 * 2x2, rank 7
    M5 = (A00+A01) B11 -> 2x2 * 2x2, rank 7
    M6 = (A10-A00)(B00+B01) -> 2x2 * 2x2, rank 7
    M7 = (A01-A11)(B10+B11) -> 2x2 * 2x2, rank 7
    
    Total rank = 7*7 = 49.
    
    Let's build this by composing the 2x2 Strassen at block level and element level.
    """
    # First, get the 2x2 Strassen decomposition
    U2, V2, W2 = _build_strassen_2x2_2x2()
    
    # For 4x4, we need to combine 7 block-level Strassen steps, each of which
    # involves 2x2 block multiplications, each of which uses rank 7.
    
    # Let's build the full 4x4 * 4x4 decomposition.
    # U has shape (4*4, 49) = (16, 49)
    # V has shape (4*4, 49) = (16, 49)
    # W has shape (4*4, 49) = (16, 49)
    
    n = m = p = 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Block-level Strassen: 7 products
    # Each product k has an A_block and B_block definition
    # A_block is a linear combination of 4 2x2 blocks of A
    # B_block is a linear combination of 4 2x2 blocks of B
    
    # Let's define the block-level coefficients
    # For each Strassen step k (0-6), and each block position (i,j) in {0,1}x{0,1}:
    # a_k[i,j] = coefficient of A_ij in the k-th Strassen product
    # b_k[i,j] = coefficient of B_ij in the k-th Strassen product
    
    # M1 = (A00+A11)(B00+B11): a=[1,0;0,1], b=[1,0;0,1]
    # M2 = (A10+A11)B00: a=[0,0;1,1], b=[1,0;0,0]
    # M3 = A00(B10-B00): a=[1,0;0,0], b=[0,-1;1,0] -> wait, B10-B00 means b[1,0]=1, b[0,0]=-1
    # M4 = A11(B01-B11): a=[0,0;0,1], b=[0,1;0,-1]
    # M5 = (A00+A01)B11: a=[1,1;0,0], b=[0,0;0,1]
    # M6 = (A10-A00)(B00+B01): a=[-1,0;1,0], b=[1,1;0,0]
    # M7 = (A01-A11)(B10+B11): a=[0,1;0,-1], b=[0,0;1,1]
    
    a_blocks = [
        [[1,0],[0,1]],  # M1
        [[0,0],[1,1]],  # M2
        [[1,0],[0,0]],  # M3
        [[0,0],[0,1]],  # M4
        [[1,1],[0,0]],  # M5
        [[-1,0],[1,0]], # M6
        [[0,1],[0,-1]], # M7
    ]
    b_blocks = [
        [[1,0],[0,1]],  # M1
        [[1,0],[0,0]],  # M2
        [[0,-1],[1,0]], # M3
        [[0,1],[0,-1]], # M4
        [[0,0],[0,1]],  # M5
        [[1,1],[0,0]],  # M6
        [[0,0],[1,1]],  # M7
    ]
    
    # For each block-level Strassen step k, we do a 2x2 * 2x2 multiplication
    # which uses rank 7. So total rank = 7*7 = 49.
    
    for k in range(7):
        for s in range(7):  # s indexes the 2x2 Strassen decomposition
            col = k * 7 + s
            
            # Determine which elements of A and B are involved
            # U[k*7+s, i*4+j] gets contributions from a_blocks[k][bi,bj] * U2[s, bi*2+ei, bj*2+ej]
            # where (bi,bj) is the block position and (ei,ej) is the element within the block
            
            for bi in range(2):
                for bj in range(2):
                    for ei in range(2):
                        for ej in range(2):
                            # Element (bi*2+ei, bj*2+ej) in A
                            a_row = bi * 2 + ei
                            a_col = bj * 2 + ej
                            a_idx = a_row * m + a_col  # i*m + j where i=a_row, j=a_col
                            
                            # The coefficient is a_blocks[k][bi,bj] * U2[s, a_row, a_col]
                            coeff_a = a_blocks[k][bi][bj] * U2[s, a_row, a_col]
                            if coeff_a != 0:
                                U[a_idx, col] += coeff_a
            
            for bi in range(2):
                for bj in range(2):
                    for ei in range(2):
                        for ej in range(2):
                            # Element (bi*2+ei, bj*2+ej) in B
                            b_row = bi * 2 + ei
                            b_col = bj * 2 + ej
                            b_idx = b_row * p + b_col  # j*p + k where j=b_row, k=b_col
                            
                            # The coefficient is b_blocks[k][bi,bj] * V2[s, b_row, b_col]
                            coeff_b = b_blocks[k][bi][bj] * V2[s, b_row, b_col]
                            if coeff_b != 0:
                                V[b_idx, col] += coeff_b
            
            # For W: the output C has blocks C = sum M_k
            # Each M_k is a 2x2 matrix, and we need to add it to the appropriate C block
            # M1 contributes to C00 and C11
            # M2 contributes to C10 and C11 (with -1 for C11)
            # etc.
            
            # Let's define the block-level W coefficients
            # For each Strassen step k, which C blocks does it contribute to, and with what sign?
            
            # C00 = M1 + M4 - M5 - M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M7
            
            c_blocks = [
                [[1,0],[0,0]],   # M1: C00+=1
                [[0,0],[1,-1]],  # M2: C10+=1, C11+=-1
                [[0,0],[1,1]],   # M3: C10+=1, C11+=1 -> wait, let me recheck
                [[0,0],[0,1]],   # M4: C11+=1 -> wait
                [[-1,1],[0,0]],  # M5: C00+=-1, C01+=1
                [[0,0],[1,0]],   # M6: C10+=1
                [[0,1],[0,1]],   # M7: C01+=1, C11+=1
            ]
            
            # Let me re-derive carefully:
            # C00 = M1 + M4 - M5 - M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M7
            
            c_blocks = [
                [[1,0],[0,0]],   # M1: C00+=1
                [[0,0],[1,-1]],  # M2: C10+=1, C11+=-1
                [[0,1],[0,0]],   # M3: C01+=1
                [[1,0],[0,1]],   # M4: C00+=1, C11+=1
                [[-1,1],[0,0]],  # M5: C00+=-1, C01+=1
                [[0,0],[1,0]],   # M6: C10+=1
                [[0,1],[0,1]],   # M7: C01+=1, C11+=1
            ]
            
            for bi in range(2):
                for bj in range(2):
                    for ei in range(2):
                        for ej in range(2):
                            # Element (bi*2+ei, bj*2+ej) in C
                            c_row = bi * 2 + ei
                            c_col = bj * 2 + ej
                            c_idx = c_col * n + c_row  # k*n + i where k=c_col, i=c_row
                            
                            # The coefficient is c_blocks[k][bi,bj] * W2[s, c_row, c_col]
                            coeff_c = c_blocks[k][bi][bj] * W2[s, c_row, c_col]
                            if coeff_c != 0:
                                W[c_idx, col] += coeff_c
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # Start with schoolbook rank = 4*4*9 = 144
    # Try to use Strassen-like reductions
    
    # For 4x4 * 4x9, we can split the 4x9 into a 4x4 and a 4x5 part.
    # The 4x4 * 4x4 part can use rank 49 (Strassen).
    # The 4x4 * 4x5 part can be done with schoolbook rank 4*4*5 = 80.
    # Total: 49 + 80 = 129.
    
    # Alternatively, split 4x9 into three 4x3 parts, each schoolbook rank 4*4*3 = 48, total 144.
    # Or split into 4x4 + 4x5 as above.
    
    # Let's try: 4x4 * 4x4 (rank 49) + 4x4 * 4x5 (rank 80) = 129.
    
    # Build the 4x4 * 4x4 part
    U4, V4, W4 = _build_strassen_4x4_4x4()
    # U4: (16, 49), V4: (16, 49), W4: (16, 49)
    
    # Build the 4x4 * 4x5 schoolbook part
    # A is 4x4, B is 4x5, C is 4x5
    # Schoolbook rank = 4*4*5 = 80
    n2, m2, p2 = 4, 4, 5
    rank2 = n2 * m2 * p2
    U5 = np.zeros((n2 * m2, rank2))
    V5 = np.zeros((m2 * p2, rank2))
    W5 = np.zeros((p2 * n2, rank2))
    
    col = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U5[i * m2 + j, col] = 1
                V5[j * p2 + k, col] = 1
                W5[k * n2 + i, col] = 1
                col += 1
    
    # Now combine:
    # The full A is 4x4, B is 4x9 = [B4x4 | B4x5]
    # C = A * B = [A*B4x4 | A*B4x5]
    
    # For the U matrix: same for both parts since A is the same 4x4
    # U_full has shape (16, 49+80) = (16, 129)
    U_full = np.zeros((n * m, 49 + rank2))
    U_full[:, :49] = U4
    U_full[:, 49:] = U5
    
    # V matrix: 
    # For the first 49 columns: V4 has shape (16, 49), corresponding to B4x4
    # For the next 80 columns: V5 has shape (20, 80), corresponding to B4x5
    # But V_full needs shape (m*p, rank) = (4*9, 129) = (36, 129)
    # V4 uses rows 0-15 (for j=0..3, k=0..3)
    # V5 uses rows 0-19 (for j=0..3, k=0..4)
    # In the full B, the first 4 columns are B4x4 (k=0..3), the next 5 are B4x5 (k=4..8)
    # So V5's columns should be shifted by 4 in the k dimension
    
    V_full = np.zeros((m * p, 49 + rank2))
    V_full[:16, :49] = V4  # B4x4 part
    
    # For V5: it's indexed by j*p2 + k where j in 0..3, k in 0..4
    # In the full matrix, the k index should be offset by 4 (since B4x4 uses k=0..3)
    for j in range(m2):
        for k in range(p2):
            full_k = k + 4  # offset by 4
            V_full[j * p + full_k, 49 + j * p2 + k] = V5[j * p2 + k, j * p2 + k]  # this is wrong, let me redo
    
    # Actually, V5[j * p2 + k, col] for col in 0..79
    # We need to place this in V_full[j * p + (k+4), 49 + col]
    for j in range(m2):
        for k in range(p2):
            for col in range(rank2):
                V_full[j * p + (k + 4), 49 + col] = V5[j * p2 + k, col]
    
    # W matrix:
    # W4 has shape (16, 49), corresponding to C4x4
    # W5 has shape (20, 80), corresponding to C4x5
    # W_full needs shape (p*n, rank) = (9*4, 129) = (36, 129)
    # W4 uses k*n+i where k in 0..3, i in 0..3
    # W5 uses k*n+i where k in 0..4, i in 0..3
    # In the full C, k=0..3 corresponds to C4x4, k=4..8 corresponds to C4x5
    
    W_full = np.zeros((p * n, 49 + rank2))
    W_full[:16, :49] = W4  # C4x4 part
    
    for k in range(p2):
        for i in range(n2):
            full_k = k + 4
            for col in range(rank2):
                W_full[full_k * n + i, 49 + col] = W5[k * n2 + i, col]
    
    # Convert to lists and return
    U_list = [[float(x) for x in row] for row in U_full]
    V_list = [[float(x) for x in row] for row in V_full]
    W_list = [[float(x) for x in row] for row in W_full]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
