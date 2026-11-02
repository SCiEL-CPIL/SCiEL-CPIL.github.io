"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from collections import defaultdict

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    # U: (n*m) x rank = 16 x rank
    # V: (m*p) x rank = 20 x rank  
    # W: (p*n) x rank = 20 x rank
    # Convention: W[k*n+i, r]
    
    start_time = time.time()
    deadline = 145  # 5s safety margin before 150s kill
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Schoolbook decomposition as baseline
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    best_rank = n * m * p  # 80
    
    # Strassen 2x2*2x2 decomposition (rank 7)
    def strassen_2x2():
        """Return U, V, W for 2x2*2x2 Strassen decomposition.
        U: 4x7, V: 4x7, W: 4x7
        Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # r=0: M1 = (A11+A22)*(B11+B22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # r=1: M2 = (A21+A22)*B11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        # r=2: M3 = A11*(B12-B22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1
        
        # r=3: M4 = A22*(B21-B11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # r=4: M5 = (A11+A12)*B22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # r=5: M6 = (A21-A11)*(B11+B12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # r=6: M7 = (A12-A22)*(B21+B22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1
        
        return U, V, W
    
    # Verify Strassen 2x2
    def test_strassen_2x2():
        U, V, W = strassen_2x2()
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0
                    for r in range(7):
                        s += U[i*2+j][r] * V[j*2+k][r] * W[k*2+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Build 4x4 * 4x5 using block decomposition
    # A is 4x4, split into 2x2 blocks: A11, A12, A21, A22
    # B is 4x5, split into blocks: B11(2x2), B12(2x3), B21(2x2), B22(2x3)
    # C is 4x5, split into blocks: C11(2x2), C12(2x3), C21(2x2), C22(2x3)
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # For 2x2*2x2 products, use Strassen (rank 7)
    # For 2x2*2x3 products, we need to figure out the rank.
    # Let's try to use a combination approach.
    
    # Actually, let's try a different approach: use the fact that we can
    # compute 4x4*4x4 (first 4 columns) using Strassen recursively,
    # and then handle the 5th column separately.
    
    # 4x4*4x4 via Strassen: 
    # Split into 2x2 blocks, use Strassen at top level (7 multiplications of 2x2 matrices)
    # Each 2x2*2x2 uses Strassen (7 scalar mults)
    # Total: 7*7 = 49 scalar mults for 4x4*4x4
    
    # 4x4*4x1 (last column): 4 dot products of length 4
    # Each dot product needs 4 mults, total 16 mults
    # But we can be smarter: the last column is just A * b where b is 4x1
    # This is 4 separate dot products, each of length 4
    # Rank for 4x4*4x1: this is just 4 independent dot products
    # Actually, the tensor for 4x4*4x1 has rank 16 (schoolbook)
    # Can we do better? The tensor T[i,j,k] where i is output row, j is input col of A, k is the single col of B
    # T[i,j,0] = 1 if j is in the right position... 
    # Actually, for 4x4 * 4x1, C[i] = sum_j A[i,j] * B[j,0]
    # This is 4 separate dot products. The tensor rank is 16 (schoolbook).
    # But we might be able to share some computations.
    
    # Let me try to construct the 4x4*4x4 part using Strassen and see if it works.
    
    def strassen_4x4():
        """Build 4x4*4x4 using Strassen recursively.
        Returns U, V, W for the 4x4*4x4 multiplication.
        U: 16xR, V: 16xR, W: 16xR
        """
        # Split A into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
        # Split B into 2x2 blocks: B11, B12, B21, B22 (each 2x2)
        # C = A*B, C is 4x4, split into 2x2 blocks: C11, C12, C21, C22
        
        # Strassen formulas:
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
        
        # Each M is a 2x2*2x2 product, which we compute using Strassen (rank 7)
        # So total rank is 7*7 = 49
        
        U = [[0]*49 for _ in range(16)]
        V = [[0]*49 for _ in range(16)]
        W = [[0]*49 for _ in range(16)]
        
        sU, sV, sW = strassen_2x2()
        
        # For each of the 7 Strassen multiplications, we need to:
        # 1. Form the left operand (2x2 matrix from A blocks)
        # 2. Form the right operand (2x2 matrix from B blocks)
        # 3. Multiply them using Strassen 2x2
        # 4. Add the result to the appropriate C blocks
        
        # Let's define the 7 multiplications:
        # Each multiplication r has:
        #   left operand: linear combination of A11, A12, A21, A22
        #   right operand: linear combination of B11, B12, B21, B22
        #   output: added to C11, C12, C21, C22 with certain coefficients
        
        # Multiplication 0: M1 = (A11+A22)*(B11+B22)
        # Left: A11 + A22, Right: B11 + B22
        # Output: C11 += 1*M1, C22 += 1*M1
        
        # Multiplication 1: M2 = (A21+A22)*B11
        # Left: A21 + A22, Right: B11
        # Output: C21 += 1*M2, C22 += -1*M2
        
        # Multiplication 2: M3 = A11*(B12-B22)
        # Left: A11, Right: B12 - B22
        # Output: C12 += 1*M3, C22 += 1*M3
        
        # Multiplication 3: M4 = A22*(B21-B11)
        # Left: A22, Right: B21 - B11
        # Output: C11 += 1*M4, C21 += 1*M4
        
        # Multiplication 4: M5 = (A11+A12)*B22
        # Left: A11 + A12, Right: B22
        # Output: C11 += -1*M5, C12 += 1*M5
        
        # Multiplication 5: M6 = (A21-A11)*(B11+B12)
        # Left: A21 - A11, Right: B11 + B12
        # Output: C22 += 1*M6
        
        # Multiplication 6: M7 = (A12-A22)*(B21+B22)
        # Left: A12 - A22, Right: B21 + B22
        # Output: C11 += 1*M7, C22 += 1*M7
        
        # For each multiplication, we need to embed the 2x2*2x2 Strassen decomposition
        # into the 4x4*4x4 problem.
        
        # The 2x2*2x2 Strassen decomposition has:
        # U_s: 4x7, where U_s[i*2+j, r] is the coefficient for A[i,j] in multiplication r
        # V_s: 4x7, where V_s[j*2+k, r] is the coefficient for B[j,k] in multiplication r
        # W_s: 4x7, where W_s[k*2+i, r] is the coefficient for C[i,k] in multiplication r
        
        # For the 4x4*4x4 problem:
        # A is indexed as A[i,j] where i,j in {0,1,2,3}
        # A11 = A[0:2,0:2], A12 = A[0:2,2:4], A21 = A[2:4,0:2], A22 = A[2:4,2:4]
        # B is indexed as B[j,k] where j,k in {0,1,2,3}
        # B11 = B[0:2,0:2], B12 = B[0:2,2:4], B21 = B[2:4,0:2], B22 = B[2:4,2:4]
        # C is indexed as C[i,k] where i,k in {0,1,2,3}
        # C11 = C[0:2,0:2], C12 = C[0:2,2:4], C21 = C[2:4,0:2], C22 = C[2:4,2:4]
        
        # For multiplication m (m in 0..6):
        # Left operand is a linear combination of A11, A12, A21, A22
        # Right operand is a linear combination of B11, B12, B21, B22
        # Output is added to C blocks with certain coefficients
        
        # Let's define the left and right operand coefficients for each multiplication:
        # left_coeffs[m][block] = coefficient for A_block in left operand
        # right_coeffs[m][block] = coefficient for B_block in right operand
        # out_coeffs[m][block] = coefficient for C_block in output
        
        left_coeffs = [
            [1, 0, 0, 1],  # M1: A11 + A22
            [0, 0, 1, 1],  # M2: A21 + A22
            [1, 0, 0, 0],  # M3: A11
            [0, 0, 0, 1],  # M4: A22
            [1, 1, 0, 0],  # M5: A11 + A12
            [-1, 0, 1, 0], # M6: A21 - A11
            [0, 1, 0, -1], # M7: A12 - A22
        ]
        
        right_coeffs = [
            [1, 0, 0, 1],  # M1: B11 + B22
            [1, 0, 0, 0],  # M2: B11
            [0, 1, 0, -1], # M3: B12 - B22
            [0, 0, 1, -1], # M4: B21 - B11
            [0, 0, 0, 1],  # M5: B22
            [1, 1, 0, 0],  # M6: B11 + B12
            [0, 0, 1, 1],  # M7: B21 + B22
        ]
        
        out_coeffs = [
            [1, 0, 0, 1],  # M1: C11 + C22
            [0, 0, 1, -1], # M2: C21 - C22
            [0, 1, 0, 1],  # M3: C12 + C22
            [1, 0, 1, 0],  # M4: C11 + C21
            [-1, 1, 0, 0], # M5: -C11 + C12
            [0, 0, 0, 1],  # M6: C22
            [1, 0, 0, 1],  # M7: C11 + C22
        ]
        
        # Block indices: 0=A11/B11/C11, 1=A12/B12/C12, 2=A21/B21/C21, 3=A22/B22/C22
        # Block 0: rows/cols 0-1, Block 1: rows 0-1, cols 2-3, Block 2: rows 2-3, cols 0-1, Block 3: rows 2-3, cols 2-3
        
        # For each multiplication m and each Strassen sub-multiplication r (0..6):
        # The U coefficient for A[i,j] is:
        #   sum over blocks b where A[i,j] is in block b: left_coeffs[m][b] * U_s[i_local*2+j_local, r]
        # Similarly for V and W.
        
        # Let's precompute the local indices for each global index:
        # For A[i,j]: block = 0 if i<2 and j<2, 1 if i<2 and j>=2, 2 if i>=2 and j<2, 3 if i>=2 and j>=2
        # local_i = i % 2, local_j = j % 2
        
        for m in range(7):
            for r in range(7):
                col = m * 7 + r
                
                # Fill U
                for i in range(4):
                    for j in range(4):
                        # Determine which block A[i,j] is in
                        if i < 2 and j < 2:
                            block = 0
                        elif i < 2 and j >= 2:
                            block = 1
                        elif i >= 2 and j < 2:
                            block = 2
                        else:
                            block = 3
                        local_i = i % 2
                        local_j = j % 2
                        U[i*4+j][col] = left_coeffs[m][block] * sU[local_i*2+local_j][r]
                
                # Fill V
                for j in range(4):
                    for k in range(4):
                        if j < 2 and k < 2:
                            block = 0
                        elif j < 2 and k >= 2:
                            block = 1
                        elif j >= 2 and k < 2:
                            block = 2
                        else:
                            block = 3
                        local_j = j % 2
                        local_k = k % 2
                        V[j*4+k][col] = right_coeffs[m][block] * sV[local_j*2+local_k][r]
                
                # Fill W
                for i in range(4):
                    for k in range(4):
                        if i < 2 and k < 2:
                            block = 0
                        elif i < 2 and k >= 2:
                            block = 1
                        elif i >= 2 and k < 2:
                            block = 2
                        else:
                            block = 3
                        local_i = i % 2
                        local_k = k % 2
                        W[k*4+i][col] = out_coeffs[m][block] * sW[local_k*2+local_i][r]
        
        return U, V, W
    
    # Verify 4x4*4x4 Strassen
    def test_strassen_4x4():
        U, V, W = strassen_4x4()
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = 0
                    for r in range(49):
                        s += U[i*4+j][r] * V[j*4+k][r] * W[k*4+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Now build 4x4*4x5:
    # First 4 columns: use 4x4*4x4 Strassen (rank 49)
    # Last column: 4x4*4x1, which is 4 dot products of length 4 (rank 16)
    # Total rank: 49 + 16 = 65
    
    def build_4x4_4x5():
        """Build 4x4*4x5 using Strassen for first 4 cols and schoolbook for last col."""
        U_44, V_44, W_44 = strassen_4x4()
        rank_44 = 49
        rank_41 = 16
        total_rank = rank_44 + rank_41
        
        U = [[0]*total_rank for _ in range(16)]
        V = [[0]*total_rank for _ in range(20)]
        W = [[0]*total_rank for _ in range(20)]
        
        # Copy the 4x4*4x4 part (first 4 columns of B)
        for i in range(16):
            for r in range(rank_44):
                U[i][r] = U_44[i][r]
        for j in range(16):
            for k in range(4):
                for r in range(rank_44):
                    V[j*5+k][r] = V_44[j*4+k][r]
        for i in range(4):
            for k in range(4):
                for r in range(rank_44):
                    W[k*4+i][r] = W_44[k*4+i][r]
        
        # Add the 4x4*4x1 part (last column of B)
        # C[i,4] = sum_j A[i,j] * B[j,4]
        # This is 4 separate dot products
        col_offset = rank_44
        for i in range(4):
            for j in range(4):
                col = col_offset + i*4 + j
                U[i*4+j][col] = 1
                V[j*5+4][col] = 1
                W[4*4+i][col] = 1  # k=4, so k*n+i = 4*4+i = 16+i
        
        return {"U": U, "V": V, "W": W}
    
    # Verify the combined 4x4*4x5 decomposition
    def test_4x4_4x5():
        result = build_4x4_4x5()
        U, V, W = result["U"], result["V"], result["W"]
        return verify(U, V, W)
    
    if test_4x4_4x5():
        result = build_4x4_4x5()
        if len(result["U"][0]) < best_rank:
            best = result
            best_rank = len(result["U"][0])
    
    # Try to optimize further: can we do better than 65?
    # The 4x4*4x1 part has rank 16 in schoolbook. Can we reduce it?
    # Actually, the tensor for 4x4*4x1 is just 4 independent dot products.
    # Each dot product is a rank-1 tensor, so the total rank is 16.
    # But we might be able to share some computations with the 4x4*4x4 part.
    
    # Let's try a different approach: use the full 4x4*4x5 tensor and try to find
    # a lower-rank decomposition using a more clever block structure.
    
    # For now, return the best we have.
    return best
    
# EVOLVE-BLOCK-END
