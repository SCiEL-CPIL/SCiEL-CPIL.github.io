import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def generate_strassen_decomposition(n):
    """
    Generates the rank-minimal (known) decomposition for n x n x n matrix multiplication
    using recursive Strassen.
    Returns U, V, W matrices of shape (n*n, rank).
    """
    if n == 1:
        # Rank 1
        U = [[1]]
        V = [[1]]
        W = [[1]]
        return U, V, W
    
    half = n // 2
    # Get decomposition for half size
    U_half, V_half, W_half = generate_strassen_decomposition(half)
    rank_half = len(U_half[0])
    
    # Strassen's 7 multiplications
    # We need to construct the factors for the 7 terms.
    # Each term is a product of two linear combinations of blocks.
    # Let A = [[A11, A12], [A21, A22]] and B = [[B11, B12], [B21, B22]]
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # The U factor for the full size n corresponds to the linear combination of A blocks.
    # The V factor corresponds to the linear combination of B blocks.
    # The W factor corresponds to how the result M_k contributes to C blocks.
    
    # We will build U, V, W by concatenating the contributions from the 7 terms.
    U_full = []
    V_full = []
    W_full = []
    
    # Helper to get block indices
    # A block (i, j) has rows i*half to (i+1)*half-1 and cols j*half to (j+1)*half-1.
    # In the flattened vector of A (size n*n), the index is (i*half + r)*n + (j*half + c)?
    # Wait, the problem statement says U has shape (n*m, rank).
    # For square n x n, U has shape (n*n, rank).
    # The index in U is i*m + j. Here m=n. So index is i*n + j.
    # This corresponds to A[i, j].
    
    # Let's define the 7 terms as lists of (coeff, block_index) for U and V.
    # And for W, we need to map the result of M_k to the C blocks.
    
    # Term 1: (A11 + A22) * (B11 + B22)
    # U part: A11 (coeff 1), A22 (coeff 1)
    # V part: B11 (coeff 1), B22 (coeff 1)
    # W part: C11 += M1, C12 -= M1, C21 -= M1, C22 += M1
    # C11 is at (0,0), C12 at (0,1), C21 at (1,0), C22 at (1,1) in block space.
    # Flattened C index: i*n + k.
    
    terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U_coeffs: list of (block_idx, coeff)
        # block_idx is (i, j) in 2x2 block space
        (([(0,0), (1,1)], [1, 1]), 
         ([(0,0), (1,1)], [1, 1]), 
         ([(0,0), (0,1), (1,0), (1,1)], [1, -1, -1, 1])),
        (([(1,0), (1,1)], [1, 1]), 
         ([(0,0)], [1]), 
         ([(0,1), (1,1)], [1, 1])),
        (([(0,0)], [1]), 
         ([(0,1), (1,1)], [1, -1]), 
         ([(0,1), (1,1)], [1, 1])), # Wait, M3 = A11(B12 - B22). C12 += M3, C22 -= M3?
        # Let's check Strassen formulas carefully.
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M6 + M7
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # So:
        # M1: C11+1, C12-1, C21-1, C22+1
        # M2: C12+1, C22+1  -> Wait, C12 = M3+M5, C21 = M2+M4.
        # Let's re-verify.
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M6 + M7
        
        # M2 contributes to C21 and C22?
        # C21 = M2 + M4. So M2 -> C21 (+1).
        # C22 = M1 - M3 + M6 + M7. M2 is not in C22.
        # Wait, standard Strassen:
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M6 + M7
        # Yes.
        
        # So M2: C21 (+1)
        # M3: C12 (+1), C22 (-1)
        # M4: C11 (-1), C21 (+1)
        # M5: C11 (+1), C12 (+1)
        # M6: C22 (+1)
        # M7: C11 (-1), C22 (+1)
    ]
    
    # Corrected W coefficients:
    terms = [
        # M1
        ([(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1], [(0,0), (0,1), (1,0), (1,1)], [1, -1, -1, 1]),
        # M2
        ([(1,0), (1,1)], [1, 1], [(0,0)], [1], [(1,0)], [1]),
        # M3
        ([(0,0)], [1], [(0,1), (1,1)], [1, -1], [(0,1), (1,1)], [1, -1]),
        # M4
        ([(1,1)], [1], [(1,0), (0,0)], [1, -1], [(0,0), (1,0)], [-1, 1]),
        # M5
        ([(0,0), (0,1)], [1, 1], [(1,1)], [1], [(0,0), (0,1)], [1, 1]),
        # M6
        ([(1,0), (0,0)], [1, -1], [(0,0), (0,1)], [1, 1], [(1,1)], [1]),
        # M7
        ([(0,1), (1,1)], [1, -1], [(1,0), (1,1)], [1, 1], [(0,0), (1,1)], [-1, 1])
    ]
    
    for u_blocks, u_coeffs, v_blocks, v_coeffs, w_blocks, w_coeffs in terms:
        # Expand these into full n x n vectors
        u_vec = [0] * (n*n)
        v_vec = [0] * (n*n)
        w_vec = [0] * (n*n)
        
        for (bi, bj), coeff in zip(u_blocks, u_coeffs):
            for r in range(half):
                for c in range(half):
                    idx = (bi*half + r)*n + (bj*half + c)
                    u_vec[idx] += coeff * U_half[r*half + c] # Wait, U_half is for the sub-problem.
                    # The sub-problem U_half has shape (half*half, rank_half).
                    # We need to iterate over the rank of the sub-problem.
                    # This recursive construction is tricky to implement directly in this loop structure.
                    # Let's restart the recursive construction logic.
                    pass
        pass
    pass

# The above recursive construction is complex. Let's use a simpler approach:
# Generate the rank 49 decomposition for 4x4x4 by explicitly using the 7 Strassen terms,
# where each term is a 2x2x2 multiplication of rank 7 (Strassen base).
# Or even simpler: use the known rank 49 factors if I can find them online or derive them.
# Actually, I can just use the rank 56 decomposition (Strassen with schoolbook 2x2) which is easier to implement?
# Rank 56 * 2 = 112. Still better than 128.
# But rank 49 * 2 = 98 is better.

# Let's try to implement the rank 49 decomposition by hardcoding the structure.
# Or use a library? No, standard library only.

# Let's try a different approach:
# Use the rank 7 decomposition for 2x2x2.
# Apply it to the 4x4x4 case.
# The 4x4x4 case can be seen as 7 multiplications of 2x2x2.
# Each 2x2x2 multiplication has rank 7.
# So total rank is 7 * 7 = 49.

# I will implement a function to generate the U, V, W for a 2x2x2 Strassen decomposition.
# Then I will use that to build the 4x4x4 decomposition.

def get_2x2_strassen_factors():
    # Returns U, V, W for 2x2x2, rank 7.
    # U shape (4, 7), V shape (4, 7), W shape (4, 7).
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U = [
        [1, 0, 1, 0, 1, 0, 0], # A11
        [0, 1, 0, 0, 0, 1, 1], # A12
        [0, 0, 1, 1, 0, 1, 0], # A21
        [1, 0, 0, 1, 0, 0, -1]  # A22
    ]
    V = [
        [1, 1, 0, 1, 0, 1, 0], # B11
        [0, 0, 1, 0, 1, 1, 0], # B12
        [1, 0, 0, 1, 0, 0, 1], # B21
        [1, 0, -1, 0, 1, 0, 1]  # B22
    ]
    W = [
        [1, 0, 0, -1, 1, 0, -1], # C11
        [0, 0, 1, 0, 1, 0, 0],   # C12
        [0, 1, 0, 1, 0, 0, 0],   # C21
        [1, 0, -1, 0, 0, 1, 1]   # C22
    ]
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    # n=4, m=4, p=8
    
    # Get 2x2 Strassen factors
    U2, V2, W2 = get_2x2_strassen_factors()
    
    # Build 4x4x4 decomposition using Strassen on 4x4 blocks of size 2.
    # The 4x4 multiplication is decomposed into 7 multiplications of 2x2 matrices.
    # Let A4 be 4x4, B4 be 4x4.
    # A4 = [[A11, A12], [A21, A22]] where Ai are 2x2.
    # B4 = [[B11, B12], [B21, B22]] where Bi are 2x2.
    
    # The 7 terms are linear combinations of Ai and Bi.
    # Each term is a 2x2 multiplication.
    # We use the rank 7 decomposition for each 2x2 multiplication.
    
    # We need to map the 4x4 indices to the 2x2 block indices.
    # U4 will have shape (16, 49).
    # V4 will have shape (16, 49).
    # W4 will have shape (16, 49).
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    # Define the 7 Strassen terms for 4x4
    # Each term t has:
    # - U terms: list of (block_idx, coeff)
    # - V terms: list of (block_idx, coeff)
    # - W terms: list of (block_idx, coeff)
    # block_idx is (i, j) in 2x2 block space.
    
    terms = [
        # M1
        ([(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1], [(0,0), (0,1), (1,0), (1,1)], [1, -1, -1, 1]),
        # M2
        ([(1,0), (1,1)], [1, 1], [(0,0)], [1], [(1,0)], [1]),
        # M3
        ([(0,0)], [1], [(0,1), (1,1)], [1, -1], [(0,1), (1,1)], [1, -1]),
        # M4
        ([(1,1)], [1], [(1,0), (0,0)], [1, -1], [(0,0), (1,0)], [-1, 1]),
        # M5
        ([(0,0), (0,1)], [1, 1], [(1,1)], [1], [(0,0), (0,1)], [1, 1]),
        # M6
        ([(1,0), (0,0)], [1, -1], [(0,0), (0,1)], [1, 1], [(1,1)], [1]),
        # M7
        ([(0,1), (1,1)], [1, -1], [(1,0), (1,1)], [1, 1], [(0,0), (1,1)], [-1, 1])
    ]
    
    for t_idx, (u_blocks, u_coeffs, v_blocks, v_coeffs, w_blocks, w_coeffs) in enumerate(terms):
        # This term corresponds to a 2x2 multiplication.
        # We use the 7 rank-1 terms from U2, V2, W2.
        for r in range(7):
            col = t_idx * 7 + r
            # Fill U4
            for (bi, bj), coeff in zip(u_blocks, u_coeffs):
                for i in range(2):
                    for j in range(2):
                        # Block (bi, bj) consists of 2x2 elements.
                        # Global row index in U4: (bi*2 + i)*4 + (bj*2 + j)
                        row = (bi*2 + i)*4 + (bj*2 + j)
                        U4[row][col] += coeff * U2[i*2 + j][r]
                        
            # Fill V4
            for (bi, bj), coeff in zip(v_blocks, v_coeffs):
                for i in range(2):
                    for j in range(2):
                        row = (bi*2 + i)*4 + (bj*2 + j)
                        V4[row][col] += coeff * V2[i*2 + j][r]
                        
            # Fill W4
            for (bi, bj), coeff in zip(w_blocks, w_coeffs):
                for i in range(2):
                    for j in range(2):
                        row = (bi*2 + i)*4 + (bj*2 + j)
                        W4[row][col] += coeff * W2[i*2 + j][r]
                        
    # Now we have U4, V4, W4 for 4x4x4. Rank 49.
    # We need 4x4x8.
    # Split B into B1 (cols 0-3) and B2 (cols 4-7).
    # Split C into C1 (cols 0-3) and C2 (cols 4-7).
    
    # Total rank 98.
    U = [[0]*98 for _ in range(16)]
    V = [[0]*98 for _ in range(32)] # 4x8 = 32
    W = [[0]*98 for _ in range(32)] # 8x4 = 32
    
    # First 49 columns: for B1, C1
    for r in range(49):
        for i in range(16):
            U[i][r] = U4[i][r]
            U[i][r+49] = U4[i][r] # Same U for both
            
        for j in range(16):
            # V4 is for 4x4 B.
            # B1 is cols 0-3. B2 is cols 4-7.
            # In V (shape 32x98), rows 0-15 correspond to B1, rows 16-31 to B2.
            # V4 row j corresponds to B1[j].
            V[j][r] = V4[j][r]
            V[j+16][r+49] = V4[j][r]
            
        for k in range(16):
            # W4 is for 4x4 C.
            # C1 is cols 0-3. C2 is cols 4-7.
            # In W (shape 32x98), rows 0-15 correspond to C1, rows 16-31 to C2.
            # Note: W index is k*n + i. Here n=4.
            # C1[k, i] -> row k*4 + i.
            # C2[k, i] -> row (k+4)*4 + i = k*4 + i + 16.
            W[k][r] = W4[k][r]
            W[k+16][r+49] = W4[k][r]
            
    return {"U": U, "V": V, "W": W}
