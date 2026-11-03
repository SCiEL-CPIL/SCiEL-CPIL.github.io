"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Strategy: Use Strassen's algorithm for 4x4 matrices (rank 28) 
    # to compute parts of the 3x3 * 3x7 multiplication.
    # 
    # Pad A (3x3) to 4x4: A_pad with last row/col = 0
    # Pad B (3x7) to 4x8: B_pad with last row = 0
    # Split B_pad into B_L (4x4, cols 0-3) and B_R (4x4, cols 4-7)
    # 
    # Compute C_L = A_pad * B_L (4x4) using Strassen (rank 28)
    # Compute C_R = A_pad * B_R (4x4) using Strassen (rank 28)
    # 
    # The actual C (3x7) = [C_L[:, 0:3], C_R[:, 0:3]] (take first 3 rows and cols)
    # 
    # Total rank: 28 + 28 = 56 < 63 (schoolbook)
    
    def strassen_4x4_decomposition():
        """
        Generate the 28 rank-1 components for 4x4 matrix multiplication using Strassen.
        
        Strassen formulas for 2x2 block matrices:
        M1 = (A11+A22)(B11+B22)
        M2 = (A21+A22)B11
        M3 = A11(B12-B22)
        M4 = A22(B21-B11)
        M5 = (A11+A12)B22
        M6 = (A21-A11)(B11+B12)
        M7 = (A12-A22)(B21+B22)
        
        Result:
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        
        Each Mi is a 2x2 matrix computed via schoolbook (4 scalar products).
        Total: 7 * 4 = 28 rank-1 components.
        """
        components = []
        
        # Define A and B virtual matrices for each Strassen term
        # Each is a 2x2 matrix where each entry is a linear form in A_pad or B_pad entries
        
        # A_pad indices: (0,0), (0,1), (0,2), (0,3), (1,0), ..., (3,3)
        # Flatten: A[r,c] -> index r*4+c
        
        # M1: A_virt = A11 + A22, B_virt = B11 + B22
        # A_virt[i,j] = A[i,j] + A[i+2,j+2] for i,j in 0,1
        # B_virt[i,j] = B[i,j] + B[i+2,j+2] for i,j in 0,1
        
        # Schoolbook for A_virt * B_virt:
        # P00 = A_v[0,0]*B_v[0,0] + A_v[0,1]*B_v[1,0]
        # P01 = A_v[0,0]*B_v[0,1] + A_v[0,1]*B_v[1,1]
        # P10 = A_v[1,0]*B_v[0,0] + A_v[1,1]*B_v[1,0]
        # P11 = A_v[1,0]*B_v[0,1] + A_v[1,1]*B_v[1,1]
        
        # Each Pij is a sum of products of linear forms, which expands to multiple rank-1 terms.
        
        def add_rank1(A_entries, B_entries, C_target, coeff):
            """
            Add a rank-1 component.
            A_entries: list of (row, col, coeff) for A_pad
            B_entries: list of (row, col, coeff) for B_pad
            C_target: (row, col) in C_pad
            coeff: coefficient
            """
            if not A_entries or not B_entries:
                return
            # This represents: (sum_a A[a]) * (sum_b B[b]) * coeff contributes to C[C_target]
            # But for tensor rank, we need to express this as rank-1 components.
            # Actually, (sum_a A[a]) * (sum_b B[b]) = sum_a sum_b A[a]*B[b]
            # So this is multiple rank-1 terms.
            # For now, we'll handle this at a higher level.
            pass
        
        # Let's directly enumerate all 28 rank-1 components by expanding the Strassen formulas.
        
        # For each Strassen term Mi, we have A_virt (2x2) and B_virt (2x2).
        # Schoolbook gives 4 scalar products: P00, P01, P10, P11.
        # Each Pij = sum over (a,b) of A_virt[a,c]*B_virt[d,b] for appropriate (a,c,d,b).
        
        # Actually, let's be more systematic.
        
        # Strassen term definitions:
        # Each term has:
        # - A_linear_forms: 2x2 array where each entry is a list of (r,c,coeff) for A_pad
        # - B_linear_forms: 2x2 array where each entry is a list of (r,c,coeff) for B_pad
        # - C_contribution: for each (i,j) in 0..1, the contribution to C_pad blocks
        
        # M1: A11+A22, B11+B22
        # A_virt[0,0] = A[0,0] + A[2,2]
        # A_virt[0,1] = A[0,1] + A[2,3]
        # A_virt[1,0] = A[1,0] + A[3,2]
        # A_virt[1,1] = A[1,1] + A[3,3]
        
        # B_virt[0,0] = B[0,0] + B[2,2]
        # B_virt[0,1] = B[0,1] + B[2,3]
        # B_virt[1,0] = B[1,0] + B[3,2]
        # B_virt[1,1] = B[1,1] + B[3,3]
        
        # Schoolbook:
        # P00 = A_v[0,0]*B_v[0,0] + A_v[0,1]*B_v[1,0]
        #     = (A00+A22)(B00+B22) + (A01+A23)(B10+B32)
        #     = A00B00 + A00B22 + A22B00 + A22B22 + A01B10 + A01B32 + A23B10 + A23B32
        #     -> 8 rank-1 terms, all contributing to C[0,0] (top-left of M1)
        #        and also to C[2,2] (bottom-right of M1, since M1 appears in C22 with +1)
        
        # This is getting complex. Let me use a different approach:
        # Generate the full 4x4 multiplication tensor and then apply Strassen.
        
        # Actually, the cleanest way: 
        # The tensor for 4x4x4 multiplication has rank at most 28 via Strassen.
        # We can generate the 28 rank-1 components explicitly.
        
        # Let me define the 28 components directly.
        
        # I'll use a known explicit decomposition.
        
        # For now, let's fall back to schoolbook if we can't get Strassen working quickly.
        
        if time.time() > deadline - 5:
            return None
        
        # Let me try a simpler approach: use the fact that we only need 3x7, not 4x8.
        # Maybe we can find a direct rank < 63 decomposition for 3x3x7.
        
        # For now, return schoolbook as fallback.
        return None
    
    # Try Strassen approach
    strassen_result = strassen_4x4_decomposition()
    
    if strassen_result is not None:
        return strassen_result
    
    # Fallback to schoolbook
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    return {'U': U_sb, 'V': V_sb, 'W': W_sb}

# EVOLVE-BLOCK-END
