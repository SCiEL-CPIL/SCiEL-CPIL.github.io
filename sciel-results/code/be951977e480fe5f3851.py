import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x9 matrix.
    
    Strategy:
    - Split B (4x9) into B1 (4x4) and B2 (4x5)
    - For 4x4x4: Use Strassen's algorithm recursively (rank 49)
    - For 4x4x5: Use schoolbook (rank 80)
    - Total: 49 + 80 = 129 terms
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Strassen 2x2x2 decomposition
    # A indices: 0=A11, 1=A12, 2=A21, 3=A22
    # B indices: 0=B11, 1=B12, 2=B21, 3=B22
    # C indices: 0=C11, 1=C12, 2=C21, 3=C22
    #
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B12-B22)
    # M4 = (A22)(B21-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    strassen_222 = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),  # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),  # M2
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]), # M3
        ([0,0,0,1], [0,-1,1,0], [1,0,1,0]), # M4
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]), # M5
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]), # M6
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]), # M7
    ]
    
    # For 4x4x4 using Strassen:
    # Split A, B, C into 2x2 blocks
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bj is 2x2
    # C = [[C11, C12], [C21, C22]] where each Ck is 2x2
    #
    # For each Strassen multiplication Mi, we need to:
    # 1. Form linear combinations of A and B blocks
    # 2. Decompose the 2x2x2 multiplication into 7 rank-1 terms
    #
    # The 4x4 matrices are indexed in row-major order:
    # A[i*4+j] for i,j in 0..3
    # Block A11: indices [0,1,4,5] (rows 0-1, cols 0-1)
    # Block A12: indices [2,3,6,7] (rows 0-1, cols 2-3)
    # Block A21: indices [8,9,12,13] (rows 2-3, cols 0-1)
    # Block A22: indices [10,11,14,15] (rows 2-3, cols 2-3)
    
    # Define block index mappings for 2x2 blocks within 4x4
    # For a 2x2 block starting at (row_offset, col_offset):
    # Block indices: [row_offset*4+col_offset, row_offset*4+col_offset+1,
    #                 (row_offset+1)*4+col_offset, (row_offset+1)*4+col_offset+1]
    
    def get_block_indices(row_offset, col_offset, size=4):
        """Get the 4 indices of a 2x2 block in a size x size matrix."""
        return [
            row_offset * size + col_offset,
            row_offset * size + col_offset + 1,
            (row_offset + 1) * size + col_offset,
            (row_offset + 1) * size + col_offset + 1
        ]
    
    # Block indices for 4x4 matrices
    A11_idx = get_block_indices(0, 0)  # [0,1,4,5]
    A12_idx = get_block_indices(0, 2)  # [2,3,6,7]
    A21_idx = get_block_indices(2, 0)  # [8,9,12,13]
    A22_idx = get_block_indices(2, 2)  # [10,11,14,15]
    
    # For B1 (first 4 columns of B, which is 4x9):
    # B1 is 4x4, same block structure
    B11_idx = get_block_indices(0, 0)  # [0,1,4,5]
    B12_idx = get_block_indices(0, 2)  # [2,3,6,7]
    B21_idx = get_block_indices(2, 0)  # [8,9,12,13]
    B22_idx = get_block_indices(2, 2)  # [10,11,14,15]
    
    # For C1 (first 4 columns of C, which is 9x4):
    # C is 9x4, so C1 is 9x4... wait, no.
    # C = A * B where A is 4x4, B is 4x9, so C is 4x9
    # C1 = first 4 columns of C, so C1 is 4x4
    # C1 indices: k*n+i where k=col, i=row
    # C11: rows 0-1, cols 0-1: indices [0*4+0, 0*4+1, 1*4+0, 1*4+1] = [0,1,4,5]
    # C12: rows 0-1, cols 2-3: indices [2*4+0, 2*4+1, 3*4+0, 3*4+1] = [8,9,12,13]
    # C21: rows 2-3, cols 0-1: indices [0*4+2, 0*4+3, 1*4+2, 1*4+3] = [2,3,6,7]
    # C22: rows 2-3, cols 2-3: indices [2*4+2, 2*4+3, 3*4+2, 3*4+3] = [10,11,14,15]
    
    # Wait, the W convention is W[k*n+i, r] where k is the column index of C and i is the row index
    # So C[k, i] = sum_j A[i,j] * B[j,k]
    # C11 (rows 0-1, cols 0-1): C[0,0]=W[0*4+0], C[0,1]=W[1*4+0], C[1,0]=W[0*4+1], C[1,1]=W[1*4+1]
    # So C11 indices in W: [0, 4, 1, 5]
    # C12 (rows 0-1, cols 2-3): C[0,2]=W[2*4+0], C[0,3]=W[3*4+0], C[1,2]=W[2*4+1], C[1,3]=W[3*4+1]
    # So C12 indices in W: [8, 12, 9, 13]
    # C21 (rows 2-3, cols 0-1): C[2,0]=W[0*4+2], C[2,1]=W[1*4+2], C[3,0]=W[0*4+3], C[3,1]=W[1*4+3]
    # So C21 indices in W: [2, 6, 3, 7]
    # C22 (rows 2-3, cols 2-3): C[2,2]=W[2*4+2], C[2,3]=W[3*4+2], C[3,2]=W[2*4+3], C[3,3]=W[3*4+3]
    # So C22 indices in W: [10, 14, 11, 15]
    
    C11_idx = [0, 4, 1, 5]
    C12_idx = [8, 12, 9, 13]
    C21_idx = [2, 6, 3, 7]
    C22_idx = [10, 14, 11, 15]
    
    # Now build the decomposition
    # Part 1: 4x4x4 (A * B1 -> C1)
    # Use Strassen recursively
    
    # For each of the 7 Strassen multiplications, we form linear combinations
    # and then decompose the 2x2x2 product
    
    # Strassen multiplications:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B12-B22)
    # M4 = (A22)(B21-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For each Mi, we have:
    # u_vec: linear combination of A blocks (4 elements, one per 2x2 block position)
    # v_vec: linear combination of B blocks (4 elements)
    # w_vec: linear combination of C blocks (4 elements)
    
    # The 7 Strassen terms for 4x4x4:
    strassen_444 = [
        # M1 = (A11+A22)(B11+B22)
        # u: A11 + A22, v: B11 + B22, w: C11 + C22 (approximately, need to check)
        # Actually, M1 contributes to C11, C22
        # C11 = M1 + M4 - M5 + M7
        # C22 = M1 - M2 + M3 + M6
        # So M1 contributes +1 to C11 and +1 to C22
        # w_vec for C blocks: [1, 0, 0, 1] (C11=1, C12=0, C21=0, C22=1)
        
        # u_vec: coefficients for [A11, A12, A21, A22]
        # v_vec: coefficients for [B11, B12, B21, B22]
        # w_vec: coefficients for [C11, C12, C21, C22]
        
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),  # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]), # M3
        ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
    ]
    
    # Block index mappings:
    # A blocks: [A11_idx, A12_idx, A21_idx, A22_idx]
    # B blocks: [B11_idx, B12_idx, B21_idx, B22_idx]
    # C blocks: [C11_idx, C12_idx, C21_idx, C22_idx]
    
    A_blocks = [A11_idx, A12_idx, A21_idx, A22_idx]
    B_blocks = [B11_idx, B12_idx, B21_idx, B22_idx]
    C_blocks = [C11_idx, C12_idx, C21_idx, C22_idx]
    
    # Build U, V, W matrices
    # U: 16 x rank (A is 4x4, so 16 entries)
    # V: 36 x rank (B is 4x9, so 36 entries)
    # W: 36 x rank (C is 9x4, so 36 entries)
    
    # We'll build the decomposition term by term
    terms = []  # List of (u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs)
    
    # Part 1: 4x4x4 using Strassen
    for u_coeff, v_coeff, w_coeff in strassen_444:
        # For each Strassen multiplication, we need to decompose the 2x2x2 product
        # into 7 rank-1 terms using strassen_222
        for u2, v2, w2 in strassen_222:
            # u2, v2, w2 are 4-element vectors for the 2x2 blocks
            # We need to map these to the actual indices in U, V, W
            
            # For U: u2[i] * u_coeff[j] for block i, coefficient j
            # Actually, u_coeff is the coefficient for each A block in the linear combination
            # u2 is the coefficient for each element within the 2x2 block
            
            # The total coefficient for U[index] is:
            # sum over blocks j: u_coeff[j] * u2[block_element_position]
            
            # Let's build the u vector for this term
            u_vec = {}
            for j in range(4):  # For each A block
                if u_coeff[j] != 0:
                    for pos in range(4):  # For each element in the 2x2 block
                        if u2[pos] != 0:
                            idx = A_blocks[j][pos]
                            u_vec[idx] = u_vec.get(idx, 0) + u_coeff[j] * u2[pos]
            
            v_vec = {}
            for j in range(4):  # For each B block
                if v_coeff[j] != 0:
                    for pos in range(4):
                        if v2[pos] != 0:
                            idx = B_blocks[j][pos]
                            v_vec[idx] = v_vec.get(idx, 0) + v_coeff[j] * v2[pos]
            
            w_vec = {}
            for j in range(4):  # For each C block
                if w_coeff[j] != 0:
                    for pos in range(4):
                        if w2[pos] != 0:
                            idx = C_blocks[j][pos]
                            w_vec[idx] = w_vec.get(idx, 0) + w_coeff[j] * w2[pos]
            
            if u_vec and v_vec and w_vec:
                terms.append((u_vec, v_vec, w_vec))
    
    # Part 2: 4x4x5 (A * B2 -> C2) using schoolbook
    # B2 is columns 4-8 of B (4x5 matrix)
    # C2 is columns 4-8 of C (4x5 matrix)
    # For schoolbook: C[i,k] = sum_j A[i,j] * B[j,k]
    # U[i*4+j] = 1, V[j*5+k] = 1, W[(k+4)*4+i] = 1
    
    for i in range(4):
        for j in range(4):
            for k in range(5):
                u_vec = {i * 4 + j: 1}
                v_vec = {j * 9 + (k + 4): 1}  # B is 4x9, so B[j, k+4] is at index j*9 + (k+4)
                w_vec = {(k + 4) * 4 + i: 1}  # C is 9x4, so C[k+4, i] is at index (k+4)*4 + i
                terms.append((u_vec, v_vec, w_vec))
    
    rank = len(terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    for col, (u_vec, v_vec, w_vec) in enumerate(terms):
        for idx, val in u_vec.items():
            U[idx][col] = val
        for idx, val in v_vec.items():
            V[idx][col] = val
        for idx, val in w_vec.items():
            W[idx][col] = val
    
    # Verify the decomposition
    # Check: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta for valid (i,j,k)
    # Actually, the tensor T[i,j,k] = delta_{i,j} * delta_{j,k} ... no.
    # The matrix multiplication tensor: T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] should be 1 for all valid i,j,k (it's the standard matrix multiplication tensor)
    
    # Let's verify by checking a few entries
    # T[i,j,k] = sum_r U[i*4+j, r] * V[j*9+k, r] * W[k*4+i, r]
    # This should equal 1 for all i in 0..3, j in 0..3, k in 0..8
    
    # But this verification could be expensive, so let's do a quick check
    # and return the result if time allows
    
    deadline = start_time + 140  # Leave 10s margin
    
    # Quick verification of a few terms
    errors = 0
    for i in range(4):
        for j in range(4):
            for k in range(9):
                total = 0
                for r in range(rank):
                    total += U[i*4+j][r] * V[j*9+k][r] * W[k*4+i][r]
                if abs(total - 1.0) > 1e-10:
                    errors += 1
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if errors > 0:
        # Fall back to schoolbook if verification fails
        rank = 144
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(36)]
        W = [[0] * rank for _ in range(36)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    U[i*4+j][col] = 1
                    V[j*9+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
