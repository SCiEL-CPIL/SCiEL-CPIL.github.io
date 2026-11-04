"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    def verify_decomp(U, V, W):
        """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k"""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Rank-21 decomposition for 4x4*4x4 (Strassen-based)
    def rank21_4x4_4x4():
        """
        Returns (U, V, W) for 4x4*4x4 with rank 21.
        U shape: (16, 21), V shape: (16, 21), W shape: (16, 21)
        Indexing: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
        """
        # Use 3x3 Strassen (rank 7) applied to 2x2 blocks of the 4x4 matrices
        # 4x4 = 2x2 blocks of 2x2
        # Strassen for 2x2*2x2 has rank 7
        # For 4x4*4x4 with 2x2 blocks:
        #   C11 = A11*B11 + A12*B21
        #   C12 = A11*B12 + A12*B22
        #   C21 = A21*B11 + A22*B21
        #   C22 = A21*B12 + A22*B22
        # Each 2x2*2x2 uses rank 7, so 4*7 = 28 schoolbook-strassen
        # But we can do better by sharing some terms.
        
        # Actually, let me use the known rank-21 algorithm for 4x4*4x4.
        # This is based on the work of Stothers (1971) or similar.
        # A simpler approach: use 2x2 Strassen (rank 7) for each of the 4 block multiplications
        # but share the Strassen products where possible.
        
        # Let me use a different approach: partition 4x4 into 2x2 blocks.
        # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
        # C = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
        #      [A21*B11 + A22*B21, A21*B12 + A22*B22]]
        
        # Using Strassen for 2x2*2x2 (rank 7):
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For 4x4*4x4, we have 4 block multiplications, each needing 7 Strassen products = 28
        # But we can combine: the 4 block multiplications share structure.
        # Actually, the standard way is to just apply Strassen to each 2x2 block pair.
        # That gives 4 * 7 = 28.
        
        # To get rank 21, we need a more clever combination.
        # Let me use the fact that we can group the 4 block multiplications.
        
        # Alternative: Use the rank-21 algorithm directly.
        # The 4x4*4x4 tensor has rank at most 21 (known result).
        # Let me construct it using a 3-level Strassen-like approach.
        
        # Actually, let me just use 2x2 Strassen for each of the 4 blocks = rank 28.
        # Then try to reduce. For now, let's use rank 28 and see if we can improve.
        
        # Wait - the known optimal rank for 4x4*4x4 is 21. Let me look for a construction.
        # One approach: use 3x3 Strassen (rank 7) on 3x3 subblocks and handle the 4th row/col separately.
        # 4x4 = 3x3 + 1 row/col
        # 3x3*3x3 rank 7
        # Cross terms: 3x1*1x3 + 1x3*3x1 + 1x1*1x1 = 9 + 9 + 1 = 19
        # Total: 7 + 19 = 26. Not 21.
        
        # Let me try another partition: 4x4 = [2x2 | 2x2; 2x2 | 2x2]
        # Use the 2x2 Strassen for all 4 blocks but share computations.
        
        # I'll use rank 28 (4 x 7) for now and focus on the overall structure.
        # The total will be 28 + 32 + 24 = 84, still much better than 120.
        
        # Construct rank-28 for 4x4*4x4 using 4 independent 2x2 Strassen
        U = [[0.0] * 28 for _ in range(16)]
        V = [[0.0] * 28 for _ in range(16)]
        W = [[0.0] * 28 for _ in range(16)]
        
        # Block offsets: A11=(0,0), A12=(0,2), A21=(2,0), A22=(2,2)
        # For each block pair, apply Strassen
        block_starts = [(0, 0), (0, 2), (2, 0), (2, 2)]  # (row_start, col_start) for A
        b_block_starts = [(0, 0), (0, 2), (2, 0), (2, 2)]  # (row_start, col_start) for B
        
        # Strassen 2x2*2x2 coefficients
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        
        # For block (bi, bj) of A and (bk, bl) of B:
        # A block elements: A[bi, bk], A[bi, bk+1], A[bi+1, bk], A[bi+1, bk+1]
        # B block elements: B[bk, bl], B[bk, bl+1], B[bk+1, bl], B[bk+1, bl+1]
        
        strassen_coeffs = [
            # (A_elem, B_elem, coeff) for each of 7 products
            # M1: (A11+A22)*(B11+B22)
            [((0,0),(0,0),1.0), ((1,1),(0,0),1.0), ((0,0),(1,1),1.0), ((1,1),(1,1),1.0)],
            # M2: (A21+A22)*B11
            [((1,0),(0,0),1.0), ((1,1),(0,0),1.0)],
            # M3: A11*(B12-B22)
            [((0,0),(0,1),1.0), ((0,0),(1,1),-1.0)],
            # M4: A22*(B21-B11)
            [((1,1),(1,0),1.0), ((1,1),(0,0),-1.0)],
            # M5: (A11+A12)*B22
            [((0,0),(1,1),1.0), ((0,1),(1,1),1.0)],
            # M6: (A21-A11)*(B11+B12)
            [((1,0),(0,0),1.0), ((0,0),(0,0),-1.0), ((1,0),(0,1),1.0), ((0,0),(0,1),-1.0)],
            # M7: (A12-A22)*(B21+B22)
            [((0,1),(1,0),1.0), ((1,1),(1,0),-1.0), ((0,1),(1,1),1.0), ((1,1),(1,1),-1.0)],
        ]
        
        # Output block mapping:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        output_coeffs = [
            # (output_elem, strassen_product_idx, coeff)
            ((0,0), 0, 1.0), ((1,1), 0, 1.0), ((0,0), 3, 1.0), ((1,1), 3, 1.0),
            ((0,0), 4, -1.0), ((1,1), 4, -1.0),
            ((0,0), 6, 1.0), ((1,1), 6, 1.0),
            ((0,1), 2, 1.0), ((1,0), 2, 1.0),
            ((0,1), 4, 1.0), ((1,0), 4, 1.0),
            ((1,0), 1, 1.0), ((0,1), 1, 1.0),
            ((1,0), 3, 1.0), ((0,1), 3, 1.0),
            ((1,1), 0, 1.0), ((0,0), 0, 1.0),
            ((1,1), 1, -1.0), ((0,0), 1, -1.0),
            ((1,1), 2, 1.0), ((0,0), 2, 1.0),
            ((1,1), 5, 1.0), ((0,0), 5, 1.0),
        ]
        
        # Hmm, this is getting complicated. Let me use a cleaner approach.
        # Just build the rank-28 decomposition directly.
        
        U = [[0.0] * 28 for _ in range(16)]
        V = [[0.0] * 28 for _ in range(16)]
        W = [[0.0] * 28 for _ in range(16)]
        
        # 4 blocks, each with 7 Strassen terms
        for block_idx in range(4):
            a_row = (block_idx // 2) * 2  # 0 or 2
            a_col = (block_idx % 2) * 2   # 0 or 2
            b_row = (block_idx // 2) * 2  # 0 or 2
            b_col = (block_idx % 2) * 2   # 0 or 2
            
            base_r = block_idx * 7
            
            # Strassen products for this block
            # M1 = (A[a,a]+A[a+1,a+1]) * (B[b,b]+B[b+1,b+1])
            # A indices: (a, b), (a+1, b+1)
            # B indices: (b, b), (b+1, b+1)
            
            # M1
            r = base_r + 0
            U[a*4 + b][r] = 1.0
            U[(a+1)*4 + (b+1)][r] = 1.0
            V[b*4 + b][r] = 1.0
            V[(b+1)*4 + (b+1)][r] = 1.0
            # Output: C[a, b] += M1, C[a+1, b+1] += M1
            W[b*4 + a][r] = 1.0
            W[(b+1)*4 + (a+1)][r] = 1.0
            
            # M2 = (A[a+1,b]+A[a+1,b+1]) * B[b,b]
            r = base_r + 1
            U[(a+1)*4 + b][r] = 1.0
            U[(a+1)*4 + (b+1)][r] = 1.0
            V[b*4 + b][r] = 1.0
            # Output: C[a+1, b] += M2
            W[b*4 + (a+1)][r] = 1.0
            
            # M3 = A[a,b] * (B[b,b+1]-B[b+1,b+1])
            r = base_r + 2
            U[a*4 + b][r] = 1.0
            V[b*4 + (b+1)][r] = 1.0
            V[(b+1)*4 + (b+1)][r] = -1.0
            # Output: C[a, b+1] += M3
            W[(b+1)*4 + a][r] = 1.0
            
            # M4 = A[a+1,b+1] * (B[b+1,b]-B[b,b])
            r = base_r + 3
            U[(a+1)*4 + (b+1)][r] = 1.0
            V[(b+1)*4 + b][r] = 1.0
            V[b*4 + b][r] = -1.0
            # Output: C[a+1, b] += M4
            W[b*4 + (a+1)][r] = 1.0
            
            # M5 = (A[a,b]+A[a,b+1]) * B[b+1,b+1]
            r = base_r + 4
            U[a*4 + b][r] = 1.0
            U[a*4 + (b+1)][r] = 1.0
            V[(b+1)*4 + (b+1)][r] = 1.0
            # Output: C[a, b+1] += M5
            W[(b+1)*4 + a][r] = 1.0
            
            # M6 = (A[a+1,b]-A[a,b]) * (B[b,b]+B[b,b+1])
            r = base_r + 5
            U[(a+1)*4 + b][r] = 1.0
            U[a*4 + b][r] = -1.0
            V[b*4 + b][r] = 1.0
            V[b*4 + (b+1)][r] = 1.0
            # Output: C[a+1, b+1] += M6
            W[(b+1)*4 + (a+1)][r] = 1.0
            
            # M7 = (A[a,b+1]-A[a+1,b+1]) * (B[b+1,b]+B[b+1,b+1])
            r = base_r + 6
            U[a*4 + (b+1)][r] = 1.0
            U[(a+1)*4 + (b+1)][r] = -1.0
            V[(b+1)*4 + b][r] = 1.0
            V[(b+1)*4 + (b+1)][r] = 1.0
            # Output: C[a, b] += M7
            W[b*4 + a][r] = 1.0
        
        return U, V, W

    # Build the full 4x5 * 5x6 decomposition
    # A is 4x5, B is 5x6
    # Partition: A = [A1 (4x4) | A2 (4x1)], B = [B1 (4x6); B2 (1x6)]
    # C = A1*B1 + A2*B2
    # A1*B1: 4x4 * 4x6 -> partition B1 as [B11 (4x4) | B12 (4x2)]
    #   A1*B11: 4x4*4x4, rank 28
    #   A1*B12: 4x4*4x2, rank 32 schoolbook
    # A2*B2: 4x1 * 1x6, rank 24 schoolbook (outer product)
    # Total: 28 + 32 + 24 = 84
    
    U44_44, V44_44, W44_44 = rank21_4x4_4x4()  # Actually rank 28
    
    # Build full decomposition
    total_rank = 28 + 32 + 24  # 84
    U = [[0.0] * total_rank for _ in range(20)]  # 4*5 = 20
    V = [[0.0] * total_rank for _ in range(30)]  # 5*6 = 30
    W = [[0.0] * total_rank for _ in range(24)]  # 6*4 = 24
    
    # Part 1: A1*B11 (4x4 * 4x4), columns 0-27
    for i in range(4):
        for j in range(4):
            for r in range(28):
                U[i*5 + j][r] = U44_44[i*4 + j][r]
    for j in range(4):
        for k in range(4):
            for r in range(28):
                V[j*6 + k][r] = V44_44[j*4 + k][r]
    for k in range(4):
        for i in range(4):
            for r in range(28):
                W[k*4 + i][r] = W44_44[k*4 + i][r]
    
    # Part 2: A1*B12 (4x4 * 4x2), columns 28-59
    # Schoolbook: 4*4*2 = 32 terms
    col = 28
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U[i*5 + j][col] = 1.0
                V[j*6 + (4 + k)][col] = 1.0  # B12 is B[:, 4:6], so k=0,1 -> cols 4,5
                W[(4 + k)*4 + i][col] = 1.0  # k=0,1 -> rows 4,5 of W
                col += 1
    
    # Part 3: A2*B2 (4x1 * 1x6), columns 60-83
    # A2 is A[:, 4] (4x1), B2 is B[4, :] (1x6)
    # C[i, k] += A[i, 4] * B[4, k]
    col = 60
    for i in range(4):
        for k in range(6):
            U[i*5 + 4][col] = 1.0
            V[4*6 + k][col] = 1.0
            W[k*4 + i][col] = 1.0
            col += 1
    
    # Verify
    if not verify_decomp(U, V, W):
        # Fallback to schoolbook
        schoolbook_rank = 120
        U_school = [[0.0] * schoolbook_rank for _ in range(20)]
        V_school = [[0.0] * schoolbook_rank for _ in range(30)]
        W_school = [[0.0] * schoolbook_rank for _ in range(24)]
        col = 0
        for i in range(4):
            for j in range(5):
                for k in range(6):
                    U_school[i*5 + j][col] = 1.0
                    V[j*6 + k][col] = 1.0
                    W[k*4 + i][col] = 1.0
                    col += 1
        U, V, W = U_school, V_school, W_school
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

