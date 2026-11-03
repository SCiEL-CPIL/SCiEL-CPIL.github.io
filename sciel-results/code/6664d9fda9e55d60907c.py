"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    """
    Return a rank-98 tensor decomposition for 4x4 * 4x8 matrix multiplication.
    
    Uses Strassen's algorithm at the block level (2x2 blocks) applied to
    a 4x4 * 4x8 product.
    
    Structure:
    - Split A (4x4) into 2x2 blocks of 2x2 matrices.
    - Split B (4x8) into 2x2 blocks of 2x4 matrices.
    - C (4x8) is 2x2 blocks of 2x4 matrices.
    
    Strassen block factors (7 total):
    M1 = (A11+A22)*(B11+B22)
    M2 = (A21+A22)*B11
    M3 = A11*(B12-B22)
    M4 = A22*(B21-B11)
    M5 = (A11+A12)*B22
    M6 = (A21-A11)*(B11+B12)
    M7 = (A12-A22)*(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Each Mi is a product of a 2x2 effective A and a 2x4 effective B.
    We split each 2x4 B into two 2x2 blocks (left and right columns).
    Each 2x2 * 2x2 product uses Strassen rank 7.
    
    Total rank = 7 (block) * 2 (left/right) * 7 (scalar) = 98.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # Define Strassen for 2x2*2x2.
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Each factor is (A_coeffs, B_coeffs, C_coeffs) as dicts {local_idx: coeff}
    
    s22 = [
        # M1: (a00+a11)(b00+b11) -> C00+=1, C11-=-1
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:-1}),
        # M2: (a10+a11)b00 -> C10+=1, C11-=-1
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        # M3: a00(b01-b11) -> C01+=1, C11+=1
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        # M4: a11(b10-b00) -> C10+=1, C00+=1
        ({3:1}, {2:1, 0:-1}, {2:1, 0:1}),
        # M5: (a00+a01)b11 -> C01+=1, C00-=-1
        ({0:1, 1:1}, {3:1}, {1:1, 0:-1}),
        # M6: (a10-a00)(b00+b01) -> C11+=1
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        # M7: (a01-a11)(b10+b11) -> C00+=1
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),
    ]
    
    # Block-level Strassen factors.
    # Each is (A_blocks, B_blocks, C_blocks) as dicts {block_id: coeff}
    # block_id: 11, 12, 21, 22
    
    block_factors = [
        # M1 = (A11+A22)(B11+B22)
        ({11:1, 22:1}, {11:1, 22:1}, {11:1, 22:-1}),
        # M2 = (A21+A22)B11
        ({21:1, 22:1}, {11:1}, {21:1, 22:-1}),
        # M3 = A11(B12-B22)
        ({11:1}, {12:1, 22:-1}, {12:1, 22:1}),
        # M4 = A22(B21-B11)
        ({22:1}, {21:1, 11:-1}, {21:1, 11:1}),
        # M5 = (A11+A12)B22
        ({11:1, 12:1}, {22:1}, {12:1, 11:-1}),
        # M6 = (A21-A11)(B11+B12)
        ({21:1, 11:-1}, {11:1, 12:1}, {22:1}),
        # M7 = (A12-A22)(B21+B22)
        ({12:1, 22:-1}, {21:1, 22:1}, {11:1}),
    ]
    
    # Mapping functions: local (r, c) in a block -> global index
    # A blocks (2x2 each):
    # 11: A[0:2, 0:2] -> (r, c)
    # 12: A[0:2, 2:4] -> (r, c+2)
    # 21: A[2:4, 0:2] -> (r+2, c)
    # 22: A[2:4, 2:4] -> (r+2, c+2)
    
    def a_map(block_id, r, c):
        if block_id == 11: return (r, c)
        if block_id == 12: return (r, c+2)
        if block_id == 21: return (r+2, c)
        if block_id == 22: return (r+2, c+2)
    
    # B blocks (2x4 each):
    # 11: B[0:2, 0:4] -> (r, c)
    # 12: B[0:2, 4:8] -> (r, c+4)
    # 21: B[2:4, 0:4] -> (r+2, c)
    # 22: B[2:4, 4:8] -> (r+2, c+4)
    
    def b_map(block_id, r, c):
        if block_id == 11: return (r, c)
        if block_id == 12: return (r, c+4)
        if block_id == 21: return (r+2, c)
        if block_id == 22: return (r+2, c+4)
    
    # C blocks (2x4 each):
    # 11: C[0:2, 0:4] -> (r, c)
    # 12: C[0:2, 4:8] -> (r, c+4)
    # 21: C[2:4, 0:4] -> (r+2, c)
    # 22: C[2:4, 4:8] -> (r+2, c+4)
    
    def c_map(block_id, r, c):
        if block_id == 11: return (r, c)
        if block_id == 12: return (r, c+4)
        if block_id == 21: return (r+2, c)
        if block_id == 22: return (r+2, c+4)
    
    # Build the 98 factors.
    factors = []
    
    for bf_idx in range(7):
        A_blocks, B_blocks, C_blocks = block_factors[bf_idx]
        
        for sub in range(2):  # 0=left half of 2x4, 1=right half
            col_offset = sub * 2
            
            for s_idx in range(7):
                s22_A, s22_B, s22_C = s22[s_idx]
                
                A_entries = {}  # (i, j) -> coeff
                B_entries = {}  # (j, k) -> coeff
                C_entries = {}  # (i, k) -> coeff
                
                # A part
                for local_idx, a_coeff in s22_A.items():
                    lr = local_idx // 2
                    lc = local_idx % 2
                    for block_id, block_coeff in A_blocks.items():
                        gi, gj = a_map(block_id, lr, lc)
                        key = (gi, gj)
                        A_entries[key] = A_entries.get(key, 0) + a_coeff * block_coeff
                
                # B part
                for local_idx, b_coeff in s22_B.items():
                    lr = local_idx // 2
                    lc = local_idx % 2
                    g_c = lc + col_offset  # 0,1 for sub=0; 2,3 for sub=1
                    for block_id, block_coeff in B_blocks.items():
                        gj, gk = b_map(block_id, lr, g_c)
                        key = (gj, gk)
                        B_entries[key] = B_entries.get(key, 0) + b_coeff * block_coeff
                
                # C part
                for local_idx, c_coeff in s22_C.items():
                    lr = local_idx // 2
                    lc = local_idx % 2
                    g_c = lc + col_offset
                    for block_id, block_coeff in C_blocks.items():
                        gi, gk = c_map(block_id, lr, g_c)
                        key = (gi, gk)
                        C_entries[key] = C_entries.get(key, 0) + c_coeff * block_coeff
                
                # Convert to index lists
                # U index: i*m + j = i*4 + j
                # V index: j*p + k = j*8 + k
                # W index: k*n + i = k*4 + i (Trace convention)
                
                A_list = [(i * m + j, coeff) for (i, j), coeff in A_entries.items() if coeff != 0]
                B_list = [(j * p + k, coeff) for (j, k), coeff in B_entries.items() if coeff != 0]
                C_list = [(k * n + i, coeff) for (i, k), coeff in C_entries.items() if coeff != 0]
                
                if A_list and B_list and C_list:
                    factors.append((A_list, B_list, C_list))
    
    rank = len(factors)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r, (A_list, B_list, C_list) in enumerate(factors):
        for idx, coeff in A_list:
            U[idx, r] = coeff
        for idx, coeff in B_list:
            V[idx, r] = coeff
        for idx, coeff in C_list:
            W[idx, r] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
