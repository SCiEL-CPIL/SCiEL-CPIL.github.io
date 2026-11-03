"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Initialize with schoolbook as fallback
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Try Strassen-like decomposition
    # Dimensions 4x4 * 4x6
    # Split m=4 into 2x2
    # Split n=4 into 2x2
    # Split p=6 into 3x3
    
    # Blocks:
    # A: A11(2x2), A12(2x2), A21(2x2), A22(2x2)
    # B: B11(2x3), B12(2x3), B21(2x3), B22(2x3)
    # C: C11(2x3), C12(2x3), C21(2x3), C22(2x3)
    
    # Strassen Multiplications:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Result:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We will build U, V, W for these 7 multiplications.
    # Each multiplication is a 2x2 * 2x3 product, which requires 12 scalar multiplications.
    # Total rank = 7 * 12 = 84.
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Helper to add a block multiplication
    # a_blocks: list of (block_id, coeff)
    # b_blocks: list of (block_id, coeff)
    # block_id: 0 for top-left, 1 for top-right, 2 for bottom-left, 3 for bottom-right
    # For A: 0=A11, 1=A12, 2=A21, 3=A22
    # For B: 0=B11, 1=B12, 2=B21, 3=B22
    
    def get_a_indices(block_id, u, v):
        # u in 0..1, v in 0..1
        if block_id == 0: # A11
            return u, v
        elif block_id == 1: # A12
            return u, v + 2
        elif block_id == 2: # A21
            return u + 2, v
        elif block_id == 3: # A22
            return u + 2, v + 2
    
    def get_b_indices(block_id, v, w):
        # v in 0..1, w in 0..2
        if block_id == 0: # B11
            return v, w
        elif block_id == 1: # B12
            return v, w + 3
        elif block_id == 2: # B21
            return v + 2, w
        elif block_id == 3: # B22
            return v + 2, w + 3
    
    def add_multiplication(a_blocks, b_blocks):
        nonlocal current_rank
        for a_id, a_coeff in a_blocks:
            for b_id, b_coeff in b_blocks:
                # Check if column indices of A block match row indices of B block
                # A block cols: 0,1 if id 0,2; 2,3 if id 1,3
                # B block rows: 0,1 if id 0,1; 2,3 if id 2,3
                a_cols = (0, 1) if a_id in [0, 2] else (2, 3)
                b_rows = (0, 1) if b_id in [0, 1] else (2, 3)
                if a_cols != b_rows:
                    continue # This term is zero in the product
                
                # Perform schoolbook multiplication of the 2x2 and 2x3 blocks
                for u in range(2): # row of A, row of C
                    for v in range(2): # col of A, row of B
                        for w in range(3): # col of B, col of C
                            # Get global indices
                            i, j = get_a_indices(a_id, u, v)
                            j_b, k = get_b_indices(b_id, v, w)
                            # j should match j_b
                            if j != j_b:
                                continue
                            
                            # Add to U, V, W
                            U[i * m + j][current_rank] = a_coeff
                            V[j * p + k][current_rank] = b_coeff
                            W[k * n + i][current_rank] = 1
                            
                            current_rank += 1

    # Define the 7 multiplications
    # M1 = (A11 + A22)(B11 + B22)
    add_multiplication([(0, 1), (3, 1)], [(0, 1), (3, 1)])
    
    # M2 = (A21 + A22)B11
    # Note: A21 cols 0,1. A22 cols 2,3. B11 rows 0,1.
    # A22 * B11 is 0 because cols/rows don't match.
    add_multiplication([(2, 1), (3, 1)], [(0, 1)])
    
    # M3 = A11(B12 - B22)
    # A11 cols 0,1. B12 rows 0,1. B22 rows 2,3.
    # A11 * B22 is 0.
    add_multiplication([(0, 1)], [(1, 1), (3, -1)])
    
    # M4 = A22(B21 - B11)
    # A22 cols 2,3. B21 rows 2,3. B11 rows 0,1.
    # A22 * B11 is 0.
    add_multiplication([(3, 1)], [(2, 1), (0, -1)])
    
    # M5 = (A11 + A12)B22
    # A11 cols 0,1. A12 cols 2,3. B22 rows 2,3.
    # A11 * B22 is 0.
    add_multiplication([(0, 1), (1, 1)], [(3, 1)])
    
    # M6 = (A21 - A11)(B11 + B12)
    # A21 cols 0,1. A11 cols 0,1. B11 rows 0,1. B12 rows 0,1.
    add_multiplication([(2, 1), (0, -1)], [(0, 1), (1, 1)])
    
    # M7 = (A12 - A22)(B21 + B22)
    # A12 cols 2,3. A22 cols 2,3. B21 rows 2,3. B22 rows 2,3.
    add_multiplication([(1, 1), (3, -1)], [(2, 1), (3, 1)])
    
    # Verify rank
    if current_rank != rank:
        # Something went wrong, return schoolbook
        return {"U": U_school, "V": V_school, "W": W_school}
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
