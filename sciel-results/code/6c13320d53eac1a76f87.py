"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x8 matrix.
    
    Strategy: Use Strassen decomposition on 2x2 blocks.
    - The 2x4 * 4x8 multiplication can be viewed as 4 independent 2x2 * 2x2 multiplications
      (since 2 = 2, 4 = 2*2, 8 = 2*4)
    - Actually, let's think more carefully:
      - A is 2x4, B is 4x8
      - We can partition A into 2 blocks of 2x2 (columns 0-1, 2-3)
      - We can partition B into 4 blocks of 2x2 (columns 0-1, 2-3, 4-5, 6-7)
      - So we have 2x4 = 2 blocks of 2x2 for A, and 4 blocks of 2x2 for B
      - The result is 2x8, which is 4 blocks of 2x2
    
    Wait, let me reconsider. The matrix multiplication is:
      C = A * B where A is 2x4, B is 4x8, C is 2x8
    
    We can partition this as:
      A = [A1, A2] where A1, A2 are 2x2
      B = [B1; B2; B3; B4] where each Bi is 2x2 (stacked vertically)
      
    Actually B is 4x8, so we partition B into 2x2 blocks:
      B has 2 rows of 2x2 blocks and 4 columns of 2x2 blocks
      B = [[B11, B12, B13, B14],
           [B21, B22, B23, B24]]
    
    And A is 2x4, so:
      A = [[A11, A12, A13, A14]]
      But A is 2 rows, 4 columns, so:
      A = [[A11, A12, A13, A14],
           [A21, A22, A23, A24]]
      where each Aij is 1x1... no, that's not right.
    
    Let me use a different approach. Partition into 2x2 blocks:
      A is 2x4: partition into 1 row of 2x2 blocks: A = [A_left, A_right] where each is 2x2
      B is 4x8: partition into 2 rows of 4 columns of 2x2 blocks:
        B = [[B11, B12, B13, B14],
             [B21, B22, B23, B24]]
    
    C = A * B is 2x8:
      C = [C1, C2, C3, C4] where each Ci is 2x2
      
      C1 = A_left * B11 + A_right * B21
      C2 = A_left * B12 + A_right * B22
      C3 = A_left * B13 + A_right * B23
      C4 = A_left * B14 + A_right * B24
    
    So we have 4 independent 2x2 matrix multiplications (each Ci involves products of 2x2 matrices).
    
    Wait, each Ci = A_left * B_i1 + A_right * B_i2 where B_i1 and B_i2 are 2x2.
    This is not a single 2x2 multiplication, but a sum of two 2x2 multiplications.
    
    Actually, let's think of it differently. We have:
      A is 2x4, B is 4x8
      
    Let's partition A into 2 blocks of 2x2 (by columns): A = [A1, A2]
    Let's partition B into 2 blocks of 2x4 (by rows): B = [B1; B2] where B1 is 2x4, B2 is 2x4
    
    Then C = A1*B1 + A2*B2 where A1, A2 are 2x2, B1, B2 are 2x4, and C is 2x4.
    
    Hmm, this is getting complicated. Let me use the standard approach:
    
    For m x n times n x p matrix multiplication, the schoolbook rank is m*n*p.
    Here m=2, n=4, p=8, so schoolbook rank = 2*4*8 = 64.
    
    Using Strassen for 2x2 blocks:
    We can view this as 4 independent 2x2 * 2x2 multiplications (since 2/2=1, 4/2=2, 8/2=4).
    Wait, that doesn't work cleanly.
    
    Let me think again. The standard Strassen decomposition for 2x2 matrix multiplication has rank 7 instead of 8.
    
    For our problem (2x4 * 4x8), we can partition:
    - A into 2x2 blocks: 1 row, 2 columns of blocks
    - B into 2x2 blocks: 2 rows, 4 columns of blocks
    
    The result C is 2x8, which is 1 row, 4 columns of 2x2 blocks.
    
    Each C block (2x2) is computed as:
      C_j = A_1 * B_1j + A_2 * B_2j for j = 1,2,3,4
    
    where A_1, A_2 are 2x2, and B_ij are 2x2.
    
    So we have 4 separate computations, each involving 2 matrix multiplications of 2x2 matrices.
    
    Total naive rank: 4 * 2 * (2*2*2) = 4 * 2 * 8 = 64.
    
    Using Strassen on each 2x2 multiplication:
    Each 2x2 * 2x2 can be done with rank 7.
    So total rank = 4 * 2 * 7 = 56.
    
    This matches the known result that Strassen reduces 64 to 56.
    
    Let me implement this:
    
    For each of the 4 output blocks C_j (j=1,2,3,4):
      C_j = A_1 * B_1j + A_2 * B_2j
    
    For the first product A_1 * B_1j, use Strassen with 7 multiplications.
    For the second product A_2 * B_2j, use Strassen with 7 multiplications.
    
    Total: 4 * 7 + 4 * 7 = 56 multiplications.
    
    Strassen's 7 products for X*Y where X, Y are 2x2:
      M1 = (X11 + X22) * (Y11 + Y22)
      M2 = (X21 + X22) * Y11
      M3 = X11 * (Y12 - Y22)
      M4 = X22 * (Y21 - Y11)
      M5 = (X11 + X12) * Y22
      M6 = (X21 - X11) * (Y11 + Y12)
      M7 = (X12 - X22) * (Y21 + Y22)
    
    Then:
      XY11 = M1 + M4 - M5 + M7
      XY12 = M3 + M5
      XY21 = M2 + M4
      XY22 = M1 - M2 + M3 + M6
    
    Now I need to express this in terms of the tensor decomposition.
    
    The tensor T for 2x4 * 4x8 multiplication has indices (i, j, k) where:
      i in {0,1} (rows of A)
      j in {0,1,2,3} (columns of A / rows of B)
      k in {0,...,7} (columns of B)
    
    T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] * B[j,k]
    
    The output index is (k*n + i, r) where n=2, so output index = k*2 + i.
    
    For the schoolbook decomposition, each term A[i,j]*B[j,k] corresponds to a rank-1 term.
    
    For Strassen, we replace groups of 8 rank-1 terms with 7.
    
    Let me define the Strassen decomposition for the 2x2 blocks.
    
    Block partition:
      A = [A1, A2] where A1 = A[:, 0:2], A2 = A[:, 2:4] (each 2x2)
      B = [B1; B2] where B1 = B[0:2, :], B2 = B[2:4, :] (each 2x4)
      
    Wait, B is 4x8, so B1 = B[0:2, 0:8] is 2x8, B2 = B[2:4, 0:8] is 2x8.
    
    C = A1*B1 + A2*B2
    
    Now partition B1 and B2 into 2x2 blocks:
      B1 = [B11, B12, B13, B14] where each is 2x2
      B2 = [B21, B22, B23, B24] where each is 2x2
    
    C = [C1, C2, C3, C4] where each Ci is 2x2:
      Cj = A1*B1j + A2*B2j
    
    For each j, Cj = A1*B1j + A2*B2j.
    
    Let me define the Strassen products for A1*B1j:
      Let X = A1, Y = B1j
      M1_j = (X11 + X22) * (Y11 + Y22)
      M2_j = (X21 + X22) * Y11
      M3_j = X11 * (Y12 - Y22)
      M4_j = X22 * (Y21 - Y11)
      M5_j = (X11 + X12) * Y22
      M6_j = (X21 - X11) * (Y11 + Y12)
      M7_j = (X12 - X22) * (Y21 + Y22)
    
    And for A2*B2j:
      Let X = A2, Y = B2j
      M8_j = (X11 + X22) * (Y11 + Y22)
      M9_j = (X21 + X22) * Y11
      M10_j = X11 * (Y12 - Y22)
      M11_j = X22 * (Y21 - Y11)
      M12_j = (X11 + X12) * Y22
      M13_j = (X21 - X11) * (Y11 + Y12)
      M14_j = (X12 - X22) * (Y21 + Y22)
    
    Then:
      Cj11 = M1_j + M4_j - M5_j + M7_j + M8_j + M11_j - M12_j + M14_j
      Cj12 = M3_j + M5_j + M10_j + M12_j
      Cj21 = M2_j + M4_j + M9_j + M11_j
      Cj22 = M1_j - M2_j + M3_j + M6_j + M8_j - M9_j + M10_j + M13_j
    
    Now I need to express each of these 14 products per j (56 total) in terms of the original A and B entries.
    
    Let me index A[i,j] where i in {0,1}, j in {0,1,2,3}.
    A1 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    A2 = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]
    
    B[j,k] where j in {0,1,2,3}, k in {0,...,7}.
    B1j = B[0:2, 2*j:2*j+2] = [[B[0,2j], B[0,2j+1]], [B[1,2j], B[1,2j+1]]]
    B2j = B[2:4, 2*j:2*j+2] = [[B[2,2j], B[2,2j+1]], [B[3,2j], B[3,2j+1]]]
    
    For each j (0 to 3), I have 14 products. Let me enumerate them.
    
    For j=0:
      B10 = [[B[0,0], B[0,1]], [B[1,0], B[1,1]]]
      B20 = [[B[2,0], B[2,1]], [B[3,0], B[3,1]]]
      
      M1_0 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
      M2_0 = (A[1,0] + A[1,1]) * B[0,0]
      M3_0 = A[0,0] * (B[0,1] - B[1,1])
      M4_0 = A[1,1] * (B[1,0] - B[0,0])
      M5_0 = (A[0,0] + A[0,1]) * B[1,1]
      M6_0 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
      M7_0 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
      
      M8_0 = (A[0,2] + A[1,3]) * (B[2,0] + B[3,1])
      M9_0 = (A[1,2] + A[1,3]) * B[2,0]
      M10_0 = A[0,2] * (B[2,1] - B[3,1])
      M11_0 = A[1,3] * (B[3,0] - B[2,0])
      M12_0 = (A[0,2] + A[0,3]) * B[3,1]
      M13_0 = (A[1,2] - A[0,2]) * (B[2,0] + B[2,1])
      M14_0 = (A[0,3] - A[1,3]) * (B[3,0] + B[3,1])
    
    And so on for j=1,2,3.
    
    Now, each product is of the form (linear combo of A entries) * (linear combo of B entries).
    In the tensor decomposition, each product corresponds to a rank-1 term where:
      U[i*m+j, c] = coefficient of A[i,j] in the first factor
      V[j*p+k, c] = coefficient of B[j,k] in the second factor
      W[k*n+i, c] = coefficient in the output
    
    Wait, the convention is W[k*n+i, r] where the output index is k*n + i.
    n=2, so output index = k*2 + i.
    
    For a product involving A entries and B entries, the output is a 2x2 block Cj which contributes to C[i, 2j + col] where col in {0,1}.
    
    Let me be more careful. The output C is 2x8, with C[i, k] for i in {0,1}, k in {0,...,7}.
    The output index in the tensor is k*n + i = k*2 + i.
    
    For block Cj (j=0,1,2,3), the entries are C[i, 2j + col] for i, col in {0,1}.
    Output index = (2j + col)*2 + i.
    
    Now, for each of the 14 products per j, I need to determine which A entries and B entries are involved, and how they contribute to the output.
    
    Let me write a helper to build the U, V, W matrices.
    
    Actually, let me just directly construct the rank-56 decomposition.
    
    I'll create a list of 56 terms, each with:
      - A coefficients: dict mapping (i,j) to coefficient
      - B coefficients: dict mapping (j,k) to coefficient
      - C coefficients: dict mapping (i,k) to coefficient
    
    Then I'll fill in U, V, W.
    
    Let me code this up.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # U has shape (n*m, rank) = (8, rank)
    # V has shape (m*p, rank) = (32, rank)
    # W has shape (p*n, rank) = (16, rank)
    
    terms = []
    
    for j in range(4):
        # B1j = B[0:2, 2j:2j+2]
        # B2j = B[2:4, 2j:2j+2]
        
        # A1 = A[:, 0:2], A2 = A[:, 2:4]
        
        # Strassen for A1 * B1j
        # X = A1, Y = B1j
        # X11 = A[0,0], X12 = A[0,1], X21 = A[1,0], X22 = A[1,1]
        # Y11 = B[0,2j], Y12 = B[0,2j+1], Y21 = B[1,2j], Y22 = B[1,2j+1]
        
        # M1 = (X11 + X22) * (Y11 + Y22)
        terms.append({
            'A': {(0,0): 1, (1,1): 1},
            'B': {(0, 2*j): 1, (1, 2*j+1): 1},
            'C': {}  # Will be filled below
        })
        
        # M2 = (X21 + X22) * Y11
        terms.append({
            'A': {(1,0): 1, (1,1): 1},
            'B': {(0, 2*j): 1},
            'C': {}
        })
        
        # M3 = X11 * (Y12 - Y22)
        terms.append({
            'A': {(0,0): 1},
            'B': {(0, 2*j+1): 1, (1, 2*j+1): -1},
            'C': {}
        })
        
        # M4 = X22 * (Y21 - Y11)
        terms.append({
            'A': {(1,1): 1},
            'B': {(1, 2*j): 1, (0, 2*j): -1},
            'C': {}
        })
        
        # M5 = (X11 + X12) * Y22
        terms.append({
            'A': {(0,0): 1, (0,1): 1},
            'B': {(1, 2*j+1): 1},
            'C': {}
        })
        
        # M6 = (X21 - X11) * (Y11 + Y12)
        terms.append({
            'A': {(1,0): 1, (0,0): -1},
            'B': {(0, 2*j): 1, (0, 2*j+1): 1},
            'C': {}
        })
        
        # M7 = (X12 - X22) * (Y21 + Y22)
        terms.append({
            'A': {(0,1): 1, (1,1): -1},
            'B': {(1, 2*j): 1, (1, 2*j+1): 1},
            'C': {}
        })
        
        # Strassen for A2 * B2j
        # X = A2, Y = B2j
        # X11 = A[0,2], X12 = A[0,3], X21 = A[1,2], X22 = A[1,3]
        # Y11 = B[2,2j], Y12 = B[2,2j+1], Y21 = B[3,2j], Y22 = B[3,2j+1]
        
        # M8 = (X11 + X22) * (Y11 + Y22)
        terms.append({
            'A': {(0,2): 1, (1,3): 1},
            'B': {(2, 2*j): 1, (3, 2*j+1): 1},
            'C': {}
        })
        
        # M9 = (X21 + X22) * Y11
        terms.append({
            'A': {(1,2): 1, (1,3): 1},
            'B': {(2, 2*j): 1},
            'C': {}
        })
        
        # M10 = X11 * (Y12 - Y22)
        terms.append({
            'A': {(0,2): 1},
            'B': {(2, 2*j+1): 1, (3, 2*j+1): -1},
            'C': {}
        })
        
        # M11 = X22 * (Y21 - Y11)
        terms.append({
            'A': {(1,3): 1},
            'B': {(3, 2*j): 1, (2, 2*j): -1},
            'C': {}
        })
        
        # M12 = (X11 + X12) * Y22
        terms.append({
            'A': {(0,2): 1, (0,3): 1},
            'B': {(3, 2*j+1): 1},
            'C': {}
        })
        
        # M13 = (X21 - X11) * (Y11 + Y12)
        terms.append({
            'A': {(1,2): 1, (0,2): -1},
            'B': {(2, 2*j): 1, (2, 2*j+1): 1},
            'C': {}
        })
        
        # M14 = (X12 - X22) * (Y21 + Y22)
        terms.append({
            'A': {(0,3): 1, (1,3): -1},
            'B': {(3, 2*j): 1, (3, 2*j+1): 1},
            'C': {}
        })
    
    # Now compute the C contributions for each term.
    # For A1*B1j (terms 0-6 for each j):
    #   Cj11 = M1 + M4 - M5 + M7
    #   Cj12 = M3 + M5
    #   Cj21 = M2 + M4
    #   Cj22 = M1 - M2 + M3 + M6
    #
    # For A2*B2j (terms 7-13 for each j):
    #   Cj11 += M8 + M11 - M12 + M14
    #   Cj12 += M10 + M12
    #   Cj21 += M9 + M11
    #   Cj22 += M8 - M9 + M10 + M13
    
    for j in range(4):
        base = j * 14
        # M1 to M7 are at indices base+0 to base+6
        # M8 to M14 are at indices base+7 to base+13
        
        # Cj11: output index = (2j+0)*2 + 0 = 4j
        # Cj12: output index = (2j+1)*2 + 0 = 4j + 2
        # Cj21: output index = (2j+0)*2 + 1 = 4j + 1
        # Cj22: output index = (2j+1)*2 + 1 = 4j + 3
        
        # From A1*B1j:
        terms[base + 0]['C'] = {4*j: 1, 4*j+2: 0, 4*j+1: 0, 4*j+3: 1}  # M1: Cj11 += 1, Cj22 += 1
        terms[base + 1]['C'] = {4*j: 0, 4*j+2: 0, 4*j+1: 1, 4*j+3: -1}  # M2: Cj21 += 1, Cj22 -= 1
        terms[base + 2]['C'] = {4*j: 0, 4*j+2: 1, 4*j+1: 0, 4*j+3: 1}  # M3: Cj12 += 1, Cj22 += 1
        terms[base + 3]['C'] = {4*j: 1, 4*j+2: 0, 4*j+1: 1, 4*j+3: 0}  # M4: Cj11 += 1, Cj21 += 1
        terms[base + 4]['C'] = {4*j: -1, 4*j+2: 1, 4*j+1: 0, 4*j+3: 0}  # M5: Cj11 -= 1, Cj12 += 1
        terms[base + 5]['C'] = {4*j: 0, 4*j+2: 0, 4*j+1: 0, 4*j+3: 1}  # M6: Cj22 += 1
        terms[base + 6]['C'] = {4*j: 1, 4*j+2: 0, 4*j+1: 0, 4*j+3: 0}  # M7: Cj11 += 1
        
        # From A2*B2j:
        terms[base + 7]['C'] = {4*j: 1, 4*j+2: 0, 4*j+1: 0, 4*j+3: 1}  # M8: Cj11 += 1, Cj22 += 1
        terms[base + 8]['C'] = {4*j: 0, 4*j+2: 0, 4*j+1: 1, 4*j+3: -1}  # M9: Cj21 += 1, Cj22 -= 1
        terms[base + 9]['C'] = {4*j: 0, 4*j+2: 1, 4*j+1: 0, 4*j+3: 1}  # M10: Cj12 += 1, Cj22 += 1
        terms[base + 10]['C'] = {4*j: 1, 4*j+2: 0, 4*j+1: 1, 4*j+3: 0}  # M11: Cj11 += 1, Cj21 += 1
        terms[base + 11]['C'] = {4*j: -1, 4*j+2: 1, 4*j+1: 0, 4*j+3: 0}  # M12: Cj11 -= 1, Cj12 += 1
        terms[base + 12]['C'] = {4*j: 0, 4*j+2: 0, 4*j+1: 0, 4*j+3: 1}  # M13: Cj22 += 1
        terms[base + 13]['C'] = {4*j: 1, 4*j+2: 0, 4*j+1: 0, 4*j+3: 0}  # M14: Cj11 += 1
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, term in enumerate(terms):
        for (i, j), coeff in term['A'].items():
            U[i * m + j][c] = coeff
        for (j, k), coeff in term['B'].items():
            V[j * p + k][c] = coeff
        for (idx, coeff) in term['C'].items():
            if coeff != 0:
                W[idx][c] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
