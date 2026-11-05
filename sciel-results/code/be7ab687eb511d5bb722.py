"""Winograd/FFT-based tensor decomposition for 2x4 x 4x8 matrix multiplication."""
DIMENSIONS = [2, 4, 8]

import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = 2, 4, 8
    rank = 16  # 4 (y points) * 4 (x points)
    
    # U: shape (n*m, rank) = (8, 16)
    # Input A is 2x4, indexed by (i, j) where i in {0,1}, j in {0,1,2,3}
    # Row index in U: i*m + j = 4*i + j
    # We evaluate the polynomial A(x,y) = sum_{i,j} A[i][j] * x^i * y^j
    # at points (x, y) where y is in Y_points and x is in X_points
    
    # Y points (for B's row polynomial): y ∈ {0, 1, 1/2, 1/3}
    # These give integer evaluations:
    # y=0: B row 0
    # y=1: sum of B rows
    # y=1/2: (2*B[0] + B[1] + 2*B[2] + B[3])/4 -> scale by 4
    # y=1/3: (9*B[0] + 3*B[1] + B[2] + 3*B[3])/27 -> this is messy
    
    # Let me use a cleaner approach with specific integer evaluation points.
    # Use y-points: 0, 1, 2, 3 (shifted) but that changes the polynomial structure.
    
    # Actually, let me use the standard Winograd scheme more carefully.
    # For multiplying polynomials of degree < 4 (in y) by degree < 2 (in x):
    # 
    # Let me define:
    # Y_points = [0, 1, 2, 3]  (for evaluating B's row structure)
    # But B is 4x8, so we need to handle the 4 rows.
    #
    # Better approach: Think of it as computing C[i][k] = sum_j A[i][j] * B[j][k]
    # This is a bilinear form.
    
    # Let me use a concrete construction with integer arithmetic.
    # Split A into columns: A[:, 0:2] and A[:, 2:4]
    # Split B into rows: B[0:2, :] and B[2:4, :]
    #
    # C[:, :] = A[:, 0:2] @ B[0:2, :] + A[:, 2:4] @ B[2:4, :]
    #
    # Each product is 2x2 x 2x8.
    # For 2x2 x 2x8: use Strassen-like or direct.
    # Direct: rank 4*8 = 32 per block, total 64. No gain.
    #
    # For 2x2 x 2x4: 
    #   C2x4[i,k] = sum_j A2[i,j] * B2[j,k] for k=0..3
    #   This is 4 separate 2x2 products if we consider each column of B2.
    #   Using Strassen for each: 7 multiplications per 2x2 output.
    #   But the 2x2 matrices share the same A2, only B2 columns differ.
    #
    # Key insight: For fixed A (2x2), computing A @ B_col for 4 different B_cols (each 2x4 block):
    #   We can do this with rank 7 (Strassen) applied to each, but that's 7*4=28 for the B part.
    #   However, there might be a better way.
    
    # Let me just implement the direct approach with Strassen blocks.
    # 2x2 x 2x2 Strassen: rank 7
    # 
    # For 2x4 x 4x8:
    # Split A = [A1 | A2] where A1, A2 are 2x2
    # Split B = [[B11, B12], [B21, B22]] where each Bij is 2x4
    #
    # C = A1*B11 + A2*B21  (left 2x4) + A1*B12 + A2*B22 (right 2x4)
    #
    # Wait, C is 2x8.
    # C[:, 0:4] = A1 * B11 + A2 * B21
    # C[:, 4:8] = A1 * B12 + A2 * B22
    #
    # Where B11 = B[0:2, 0:4], B12 = B[0:2, 4:8], B21 = B[2:4, 0:4], B22 = B[2:4, 4:8]
    
    # For each 2x2 x 2x4 product, I can use a decomposition.
    # A2x2 * B2x4 = [A*B[:,0], A*B[:,1], A*B[:,2], A*B[:,3]]
    # Each A*B[:,k] is a 2-vector, requiring... let me think.
    
    # Actually, let me use a simpler concrete construction.
    # I'll build the decomposition explicitly using Strassen for 2x2x2 blocks.
    
    # Strassen 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    def strassen_2x2x2():
        """Returns (U, V, W) for 2x2 x 2x2 multiplication, rank 7."""
        # U: 4x7, V: 4x7, W: 4x7
        # Input A indexed by (i,j) -> row i*2+j in U
        # Input B indexed by (j,k) -> row j*2+k in V  
        # Output C indexed by (i,k) -> row i*2+k in W (but trace convention: W[k*n+i])
        # For 2x2: n=2, so W[k*2+i]
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        # A00: row 0 in U, A11: row 3 in U
        # B00: row 0 in V, B11: row 3 in V
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        # C00 gets +M1, C11 gets +M1
        # C00: W[0*2+0]=W[0], C11: W[1*2+1]=W[3]
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        # C10 gets +M2: W[0*2+1]=W[1]
        W[1][1] = 1
        
        # M3 = A00(B10-B11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        # C01 gets +M3: W[1*2+0]=W[2]
        W[2][2] = 1
        
        # M4 = A11(B01-B10)
        U[3][3] = 1
        V[1][3] = 1; V[2][3] = -1
        # C00 gets +M4, C10 gets +M4
        W[0][3] = 1; W[1][3] = 1
        
        # M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        # C00 gets -M5, C01 gets +M5
        W[0][4] = -1; W[2][4] = 1
        
        # M6 = (A10-A00)(B00+B10)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[2][5] = 1
        # C11 gets +M6
        W[3][5] = 1
        
        # M7 = (A01-A11)(B01+B11)
        U[1][6] = 1; U[3][6] = -1
        V[1][6] = 1; V[3][6] = 1
        # C00 gets +M7
        W[0][6] = 1
        
        return U, V, W
    
    # Now build 2x4 x 4x8.
    # A is 2x4: split into A1 = A[:, 0:2], A2 = A[:, 2:4] (each 2x2)
    # B is 4x8: B[0:2, 0:4], B[0:2, 4:8], B[2:4, 0:4], B[2:4, 4:8]
    #   = B11, B12, B21, B22 (each 2x4)
    #
    # C[:, 0:4] = A1 * B11 + A2 * B21
    # C[:, 4:8] = A1 * B12 + A2 * B22
    #
    # For A1 * B11 where A1 is 2x2 and B11 is 2x4:
    #   B11 = [b11_0, b11_1, b11_2, b11_3] (each column is 2x1)
    #   A1 * B11 = [A1*b11_0, A1*b11_1, A1*b11_2, A1*b11_3]
    #   Each A1*b is a 2-vector.
    #
    # I can think of this as: for each of the 4 columns of B11, do a 2x1 x 2x1 product.
    # But that's just dot products, rank 4 per column set...
    #
    # Alternative: Use the fact that A1 * B11 can be computed by treating B11 as 
    # 4 separate 2x1 vectors. The product A1 * v for a 2x1 vector v requires 
    # computing two dot products, but these share A1.
    #
    # Let me use a different decomposition. For 2x2 x 2x4:
    #   C[i, k] = sum_j A[i,j] * B[j,k] for i=0,1 and k=0,1,2,3
    #   This is 8 outputs.
    #   Schoolbook: rank 8 (one per output, each involving 2 multiplications but 
    #   since we're doing tensor decomposition, each (i,j,k) triple is one rank-1 term)
    #   Actually schoolbook for 2x2x2x4 is rank 2*2*4 = 16.
    #
    # Hmm, I need to be more careful. The tensor is M_{2,2,4} which has rank at most 
    # 2*2*4 = 16 by schoolbook. Can we do better?
    #
    # For 2x2 x 2x4, we can use: split the 4 columns into two 2x2 blocks.
    #   A1 * B11 = [A1 * B11_left | A1 * B11_right] where each is 2x2.
    #   Each 2x2x2 product has rank 7 (Strassen).
    #   Total: 7 + 7 = 14 for 2x2 x 2x4.
    
    # So for 2x4 x 4x8:
    # C[:, 0:4] = A1 * B11 + A2 * B21
    #   A1*B11: 2x2 x 2x4, rank 14
    #   A2*B21: 2x2 x 2x4, rank 14
    #   But these share... no, A1 and A2 are different, B11 and B21 are different.
    #   So they're independent. Rank 14 + 14 = 28 for left half.
    #
    # C[:, 4:8] = A1 * B12 + A2 * B22
    #   Similarly rank 28.
    #
    # Total: 28 + 28 = 56. Same as before.
    #
    # Can we do better? What if we don't split A into A1, A2?
    # A is 2x4. B is 4x8.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    #
    # Think of A as a 2x4 matrix and B as a 4x8 matrix.
    # This is equivalent to computing C = A * B where A is 2x4 and B is 4x8.
    #
    # The tensor M_{2,4,8} has rank at most 2*4*8 = 64 (schoolbook).
    # Using block Strassen as above: 56.
    #
    # Can we do 55 or less? The directive mentions trying rank 55.
    # Let me try a more careful construction.
    
    # Alternative: Use 4 Strassen blocks of 2x2x2, each handling a 2x2 subproduct,
    # but arrange them to cover the 4x8 output.
    #
    # A (2x4) * B (4x8) = C (2x8)
    # 
    # Let me partition B into 4 blocks of 2x2 columns each? No, B is 4x8.
    # B = [B[:, 0:2], B[:, 2:4], B[:, 4:6], B[:, 6:8]] (each 4x2)
    # C = [A*B[:, 0:2], A*B[:, 2:4], A*B[:, 4:6], A*B[:, 6:8]] (each 2x2)
    #
    # For each 2x2x4x2 product: A (2x4) * B_block (4x2) = C_block (2x2)
    # This is M_{2,4,2} tensor.
    # Schoolbook: 2*4*2 = 16 per block, 4 blocks = 64.
    #
    # For M_{2,4,2}: C[i,k] = sum_j A[i,j] * B[j,k]
    # Split A into A[:, 0:2] and A[:, 2:4] (each 2x2)
    # B into B[0:2, :] and B[2:4, :] (each 2x2)
    # C = A1*B1 + A2*B2 where all are 2x2.
    # Rank: 7 + 7 = 14 per block. 4 blocks: 56.
    
    # So 56 seems to be the bound with this approach. Let me implement it.
    
    # Build the full decomposition.
    # U: (8, 56), V: (32, 56), W: (16, 56)
    
    U = [[0] * 56 for _ in range(8)]
    V = [[0] * 56 for _ in range(32)]
    W = [[0] * 56 for _ in range(16)]
    
    col = 0
    
    # We have 4 groups of 14 rank-1 terms each.
    # Group g (g=0,1,2,3) handles columns 2g:2g+2 of the output.
    # Within each group, we have two Strassen sub-blocks:
    #   Sub-block 0: A[:, 0:2] * B[0:2, 2g:2g+2]
    #   Sub-block 1: A[:, 2:4] * B[2:4, 2g:2g+2]
    
    for g in range(4):
        for sub in range(2):
            # sub=0: A rows 0-1 (i.e., A[0:2, 0:2]), B rows 0-1 (B[0:2, 2g:2g+2])
            # sub=1: A rows 2-3 (i.e., A[0:2, 2:4]), B rows 2-3 (B[2:4, 2g:2g+2])
            
            sU, sV, sW = strassen_2x2x2()
            
            for c in range(7):
                # Map U indices: A is 2x4, row index = i*4 + j
                # For sub=0: i in {0,1}, j in {0,1} -> U row = i*4 + j
                # For sub=1: i in {0,1}, j in {2,3} -> U row = i*4 + (j+2)
                for u_row in range(4):
                    if sU[u_row][c] != 0:
                        i = u_row // 2
                        j = u_row % 2
                        if sub == 0:
                            actual_u_row = i * 4 + j
                        else:
                            actual_u_row = i * 4 + (j + 2)
                        U[actual_u_row][col + c] = sU[u_row][c]
                
                # Map V indices: B is 4x8, row index = j*8 + k
                # For sub=0: j in {0,1}, k in {0,1} relative to column block g
                #   Actual B row = j, col = 2g + k
                #   V row = j*8 + (2g + k)
                # For sub=1: j in {2,3}, k in {0,1} relative to column block g
                #   Actual B row = j, col = 2g + k
                #   V row = j*8 + (2g + k)
                for v_row in range(4):
                    if sV[v_row][c] != 0:
                        j = v_row // 2
                        k = v_row % 2
                        actual_j = j + sub * 2
                        actual_k = 2 * g + k
                        actual_v_row = actual_j * 8 + actual_k
                        V[actual_v_row][col + c] = sV[v_row][c]
                
                # Map W indices: C is 2x8, trace convention W[k*2 + i]
                # i in {0,1} (row of C), k in {0,1} relative to column block g
                # Actual column of C = 2g + k
                # W index = (2g + k)*2 + i
                for w_row in range(4):
                    if sW[w_row][c] != 0:
                        i = w_row // 2
                        k = w_row % 2
                        actual_k = 2 * g + k
                        actual_w_row = actual_k * 2 + i
                        W[actual_w_row][col + c] = sW[w_row][c]
            
            col += 7
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
