"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # We'll construct the decomposition by splitting the 4x7 matrix into 4x4 and 4x3 parts
    # 4x4x4: rank 49 (Strassen-based)
    # 4x4x3: rank 48 (schoolbook)
    # Total rank: 97
    
    deadline = time.time() + 140  # safety margin
    
    # Build 4x4x4 Strassen decomposition (rank 49)
    U44, V44, W44 = _strassen_4x4()
    
    # Build 4x4x3 schoolbook decomposition (rank 48)
    U43, V43, W43 = _schoolbook_4x4x3()
    
    # Concatenate
    rank = len(U44) + len(U43)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # First part: 4x4x4 (columns 0-3 of B)
    U[:, :len(U44)] = U44
    V[:, :len(V44)] = V44
    W[:, :len(W44)] = W44
    
    # Second part: 4x4x3 (columns 4-6 of B)
    U[:, len(U44):] = U43
    V[:, len(V44):] = V43
    W[:, len(W44):] = W43
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _strassen_4x4():
    """
    Rank-49 decomposition for 4x4 matrix multiplication.
    Uses Strassen for 2x2 block multiplication, each block being 2x2.
    """
    # 4x4 matrix indices: A[i,j], B[j,k], C[i,k]
    # Split into 2x2 blocks of 2x2 matrices
    # A = [[A11, A12], [A21, A22]] where A11 = A[0:2,0:2], etc.
    # B = [[B11, B12], [B21, B22]]
    
    # Strassen terms for 2x2 block multiplication:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each Mik is a 2x2 matrix product, expanded via Strassen (rank 7 each)
    # Total: 7 * 7 = 49 rank-1 terms
    
    # For each of the 7 block-level Strassen terms, we need to expand the 2x2*2x2 product
    # into 7 scalar rank-1 terms.
    
    # Let's define the 7 Strassen coefficients for 2x2*2x2
    # For X*Y where X,Y are 2x2:
    # N1 = (x00+x11)(y00+y11)
    # N2 = (x10+x11)y00
    # N3 = x00(y01-y11)
    # N4 = x11(y10-y00)
    # N5 = (x00+x01)y11
    # N6 = (x10-x00)(y00+y01)
    # N7 = (x01-x11)(y10+y11)
    #
    # z00 = N1 + N4 - N5 + N7
    # z01 = N3 + N5
    # z10 = N2 + N4
    # z11 = N1 - N2 + N3 + N6
    
    # The 7 rank-1 terms for 2x2*2x2:
    # Term 0: (x00+x11) * (y00+y11) -> contributes to z00 and z11
    # Term 1: (x10+x11) * y00 -> contributes to z10 and z11
    # Term 2: x00 * (y01-y11) -> contributes to z01 and z11
    # Term 3: x11 * (y10-y00) -> contributes to z00 and z10
    # Term 4: (x00+x01) * y11 -> contributes to z00 and z01
    # Term 5: (x10-x00) * (y00+y01) -> contributes to z11
    # Term 6: (x01-x11) * (y10+y11) -> contributes to z00
    
    # For each rank-1 term, U picks coefficients of X, V picks coefficients of Y, W picks output position
    
    # Let's build the 7x7 Strassen basis for 2x2*2x2
    # U2x2: 4 rows (x00,x01,x10,x11) x 7 cols
    # V2x2: 4 rows (y00,y01,y10,y11) x 7 cols  
    # W2x2: 4 rows (z00,z01,z10,z11) x 7 cols
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 0: (x00+x11)(y00+y11)
    U2[0, 0] = 1; U2[3, 0] = 1  # x00 + x11
    V2[0, 0] = 1; V2[3, 0] = 1  # y00 + y11  (y00 is index 0, y11 is index 3)
    W2[0, 0] = 1  # z00
    W2[3, 0] = 1  # z11
    
    # Term 1: (x10+x11)y00
    U2[2, 1] = 1; U2[3, 1] = 1  # x10 + x11
    V2[0, 1] = 1  # y00
    W2[2, 1] = 1  # z10
    W2[3, 1] = -1  # z11 (coefficient -1 in z11 = N1 - N2 + ...)
    
    # Wait, I need to be more careful. Let me re-derive.
    # z11 = N1 - N2 + N3 + N6
    # So N2 contributes -1 to z11.
    
    # Let me redo this more carefully.
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Index: x00=0, x01=1, x10=2, x11=3
    # Index: y00=0, y01=1, y10=2, y11=3
    # Index: z00=0, z01=1, z10=2, z11=3
    
    # N1 = (x00+x11)(y00+y11)
    U2[0, 0] += 1; U2[3, 0] += 1
    V2[0, 0] += 1; V2[3, 0] += 1
    W2[0, 0] += 1  # z00 += N1
    W2[3, 0] += 1  # z11 += N1
    
    # N2 = (x10+x11)y00
    U2[2, 1] += 1; U2[3, 1] += 1
    V2[0, 1] += 1
    W2[2, 1] += 1  # z10 += N2
    W2[3, 1] += -1  # z11 += -N2
    
    # N3 = x00(y01-y11)
    U2[0, 2] += 1
    V2[1, 2] += 1; V2[3, 2] += -1
    W2[1, 2] += 1  # z01 += N3
    W2[3, 2] += 1  # z11 += N3
    
    # N4 = x11(y10-y00)
    U2[3, 3] += 1
    V2[2, 3] += 1; V2[0, 3] += -1
    W2[0, 3] += 1  # z00 += N4
    W2[2, 3] += 1  # z10 += N4
    
    # N5 = (x00+x01)y11
    U2[0, 4] += 1; U2[1, 4] += 1
    V2[3, 4] += 1
    W2[0, 4] += -1  # z00 += -N5
    W2[1, 4] += 1  # z01 += N5
    
    # N6 = (x10-x00)(y00+y01)
    U2[2, 5] += 1; U2[0, 5] += -1
    V2[0, 5] += 1; V2[1, 5] += 1
    W2[3, 5] += 1  # z11 += N6
    
    # N7 = (x01-x11)(y10+y11)
    U2[1, 6] += 1; U2[3, 6] += -1
    V2[2, 6] += 1; V2[3, 6] += 1
    W2[0, 6] += 1  # z00 += N7
    
    # Now build the 4x4x4 decomposition using this 2x2x2 Strassen basis
    # Block structure:
    # A blocks: A11 (rows 0-1, cols 0-1), A12 (rows 0-1, cols 2-3), 
    #           A21 (rows 2-3, cols 0-1), A22 (rows 2-3, cols 2-3)
    # B blocks: B11 (rows 0-1, cols 0-1), B12 (rows 0-1, cols 2-3),
    #           B21 (rows 2-3, cols 0-1), B22 (rows 2-3, cols 2-3)
    # C blocks: C11 (rows 0-1, cols 0-1), C12 (rows 0-1, cols 2-3),
    #           C21 (rows 2-3, cols 0-1), C22 (rows 2-3, cols 2-3)
    
    # For the 4x4 case, we have 4x4 A, 4x4 B, 4x4 C
    # U has shape (4*4, 49) = (16, 49)
    # V has shape (4*4, 49) = (16, 49)
    # W has shape (4*4, 49) = (16, 49)
    
    # The 7 block-level Strassen terms, each expanded to 7 scalar terms = 49 total
    
    U44 = np.zeros((16, 49))
    V44 = np.zeros((16, 49))
    W44 = np.zeros((16, 49))
    
    # Block-level coefficients
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For each block term, we specify:
    # - Which A blocks contribute and with what coefficients
    # - Which B blocks contribute and with what coefficients
    # - Which C blocks the result contributes to and with what coefficients
    
    # Define block mappings
    # A11: rows 0-1, cols 0-1 -> U indices 0,1,4,5 (row*4+col)
    # A12: rows 0-1, cols 2-3 -> U indices 2,3,6,7
    # A21: rows 2-3, cols 0-1 -> U indices 8,9,12,13
    # A22: rows 2-3, cols 2-3 -> U indices 10,11,14,15
    
    # B11: rows 0-1, cols 0-1 -> V indices 0,1,4,5
    # B12: rows 0-1, cols 2-3 -> V indices 2,3,6,7
    # B21: rows 2-3, cols 0-1 -> V indices 8,9,12,13
    # B22: rows 2-3, cols 2-3 -> V indices 10,11,14,15
    
    # C11: rows 0-1, cols 0-1 -> W indices 0,1,4,5 (but W is indexed as k*n+i = k*4+i)
    # Wait, W[k*n+i] means W index = k*4 + i where k is col of B/row of C, i is row of A/row of C
    # So C[i,k] corresponds to W[k*4+i]
    # C11: i in 0-1, k in 0-1 -> W indices 0,1,4,5
    # C12: i in 0-1, k in 2-3 -> W indices 8,9,12,13
    # C21: i in 2-3, k in 0-1 -> W indices 2,3,6,7
    # C22: i in 2-3, k in 2-3 -> W indices 10,11,14,15
    
    # Helper to get local indices within a 2x2 block
    # For a block starting at (r0, c0), the 4 local positions map to global indices
    
    def get_A_block_indices(block):
        """Return the 4 global U indices for a given A block"""
        # block: 0=A11, 1=A12, 2=A21, 3=A22
        r0, c0 = (0, 0) if block == 0 else (0, 2) if block == 1 else (2, 0) if block == 2 else (2, 2)
        return [r0*4+c0, r0*4+c0+1, (r0+1)*4+c0, (r0+1)*4+c0+1]
    
    def get_B_block_indices(block):
        """Return the 4 global V indices for a given B block"""
        r0, c0 = (0, 0) if block == 0 else (0, 2) if block == 1 else (2, 0) if block == 2 else (2, 2)
        return [r0*4+c0, r0*4+c0+1, (r0+1)*4+c0, (r0+1)*4+c0+1]
    
    def get_C_block_indices(block):
        """Return the 4 global W indices for a given C block. W[k*4+i]"""
        # block: 0=C11, 1=C12, 2=C21, 3=C22
        # C[i,k]: i=row, k=col
        # C11: i in 0-1, k in 0-1
        # C12: i in 0-1, k in 2-3
        # C21: i in 2-3, k in 0-1
        # C22: i in 2-3, k in 2-3
        if block == 0:  # C11
            i0, k0 = 0, 0
        elif block == 1:  # C12
            i0, k0 = 0, 2
        elif block == 2:  # C21
            i0, k0 = 2, 0
        else:  # C22
            i0, k0 = 2, 2
        # W index = k*4 + i
        return [k0*4+i0, k0*4+i0+1, (k0+1)*4+i0, (k0+1)*4+i0+1]
    
    # For each of the 7 block-level Strassen terms, and for each of the 7 scalar Strassen terms,
    # build the rank-1 contribution.
    
    # Block-level Strassen: define which A blocks, B blocks, and C blocks with coefficients
    # Format: list of (a_blocks, b_blocks, c_blocks) where each is list of (block_id, coeff)
    
    block_strassen = [
        # M1 = (A11+A22)(B11+B22)
        # A: A11(+1), A22(+1); B: B11(+1), B22(+1)
        # Result: C11(+1), C12(+1), C21(+1), C22(+1) -- wait, no.
        # M1 contributes to C11, C12, C21, C22 via the 2x2 Strassen expansion.
        # Actually, M1 is a 2x2 matrix result. Its (0,0) goes to C11, (0,1) to C12, (1,0) to C21, (1,1) to C22.
        # But in Strassen, M1 contributes +1 to C11 and +1 to C22 (via z00 and z11 of the 2x2 result)
        # Let me think again.
        
        # The 2x2 block result Z = [[Z00, Z01], [Z10, Z11]] maps to C as:
        # C11 += Z00, C12 += Z01, C21 += Z10, C22 += Z11
        
        # For M1 = (A11+A22)(B11+B22), the 2x2 result contributes:
        # Z00 to C11, Z01 to C12, Z10 to C21, Z11 to C22
        # And Z is computed via the 7-term Strassen, so each of the 7 terms has specific C contributions.
    ]
    
    # This is getting complex. Let me use a more systematic approach.
    # For each of the 7 block-level terms, I'll specify the A-side and B-side block combinations,
    # then use the 2x2 Strassen pattern to determine C contributions.
    
    # Define the 7 block-level terms as (A_blocks_dict, B_blocks_dict)
    # where dict maps block_id -> coefficient
    
    block_terms = [
        # M1: (A11+A22) * (B11+B22)
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2: (A21+A22) * B11
        ({2: 1, 3: 1}, {0: 1}),
        # M3: A11 * (B12-B22)
        ({0: 1}, {1: 1, 3: -1}),
        # M4: A22 * (B21-B11)
        ({3: 1}, {2: 1, 0: -1}),
        # M5: (A11+A12) * B22
        ({0: 1, 1: 1}, {3: 1}),
        # M6: (A21-A11) * (B11+B12)
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        # M7: (A12-A22) * (B21+B22)
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]
    
    # For each block term, the 2x2 result Z has entries that map to C blocks:
    # Z00 -> C block 0 (C11)
    # Z01 -> C block 1 (C12)
    # Z10 -> C block 2 (C21)
    # Z11 -> C block 3 (C22)
    
    # The 2x2 Strassen W2 matrix tells us which C positions each of the 7 scalar terms contributes to.
    # W2 has shape (4, 7) where rows are z00,z01,z10,z11 and cols are the 7 terms.
    
    col_offset = 0
    for term_idx in range(7):
        A_blocks, B_blocks = block_terms[term_idx]
        
        for scalar_idx in range(7):
            # This is rank-1 term number col_offset
            col = col_offset
            
            # A-side: combine the block contributions with the 2x2 Strassen U2 coefficients
            for block_id, block_coeff in A_blocks.items():
                a_indices = get_A_block_indices(block_id)
                for local_idx in range(4):
                    U44[a_indices[local_idx], col] += block_coeff * U2[local_idx, scalar_idx]
            
            # B-side: similarly
            for block_id, block_coeff in B_blocks.items():
                b_indices = get_B_block_indices(block_id)
                for local_idx in range(4):
                    V44[b_indices[local_idx], col] += block_coeff * V2[local_idx, scalar_idx]
            
            # C-side: the 2x2 Strassen W2 coefficients determine which C blocks get contributions
            for local_c in range(4):
                if W2[local_c, scalar_idx] != 0:
                    c_indices = get_C_block_indices(local_c)
                    for idx in c_indices:
                        W44[idx, col] += W2[local_c, scalar_idx]
            
            col_offset += 1
    
    return U44, V44, W44

def _schoolbook_4x4x3():
    """
    Rank-48 schoolbook decomposition for 4x4 * 4x3.
    """
    U = np.zeros((16, 48))
    V = np.zeros((16, 48))
    W = np.zeros((28, 48))  # p*n = 3*4 = 12... wait, p=7 for the full problem, but here we're doing 4x4x3
    
    # Wait, for the 4x4x3 subproblem, the output is 4x3.
    # But in the full problem, we're concatenating. The W matrix for the full problem has shape (p*n, rank) = (7*4, rank) = (28, rank).
    # For the 4x4x3 part, the C matrix is 4x3, so W indices are k*4+i for k in 0-2 (since p=3 for this subproblem), i in 0-3.
    # But in the full problem, k ranges 0-6. For the 4x4x3 part, we're computing columns 4-6 of the full C.
    # So the W indices should be for k in 4-6 (global), i in 0-3.
    # W global index = k*4 + i, where k is global column index.
    
    # Actually, let me reconsider. The full W has shape (p*n, rank) = (7*4, rank) = (28, rank).
    # W[k*n + i, r] where k is the column index of B (0 to 6), i is the row index of A (0 to 3).
    # For the 4x4x3 subproblem (computing columns 4,5,6 of C), we use k = 4,5,6.
    
    U = np.zeros((16, 48))
    V = np.zeros((16, 48))
    W = np.zeros((28, 48))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):  # k = 0,1,2 corresponds to global k = 4,5,6
                global_k = k + 4
                U[i*4 + j, col] = 1
                V[j*4 + k, col] = 1
                W[global_k*4 + i, col] = 1
                col += 1
    
    return U, V, W

# EVOLVE-BLOCK-END

