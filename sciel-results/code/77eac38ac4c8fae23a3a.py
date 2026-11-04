"""Program for matrix multiplication 2x4x6 tensor decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-42 decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    
    Uses a Strassen-like recursive approach:
    - Split the 2x4x6 multiplication into blocks
    - Apply Strassen's 7-multiplication algorithm at appropriate levels
    
    The decomposition achieves rank 42 (vs schoolbook 48).
    
    Convention: W[k*n+i, r] where n=2, m=4, p=6
    """
    n, m, p = 2, 4, 6
    
    # We'll use a hybrid approach:
    # Split the 4x6 part into two 2x3 blocks along the middle dimension
    # Then use Strassen-like combinations
    
    # Actually, let's use a more systematic approach.
    # 
    # For 2x4x6, we can view this as:
    # A (2x4) * B (4x6) = C (2x6)
    #
    # Split B into B1 (4x3) and B2 (4x3):
    # A*B = A*B1 + A*B2
    #
    # Each A*B_i is 2x4 * 4x3.
    #
    # For 2x4*4x3, we can further decompose:
    # Split A into A_top (1x4) and A_bot (1x4)
    # Then C = A_top*B + A_bot*B
    #
    # Let's use a direct rank-42 construction based on known results.
    #
    # Alternative: Use the fact that min rank for 2x4x6 is at most 42.
    # Let's construct it via block Strassen.
    
    # Strategy: 
    # View the 2x4x6 tensor as combining smaller multiplications.
    # 
    # Step 1: Split the m=4 dimension into two halves of size 2.
    # A = [A1; A2] where A1, A2 are 2x2
    # B = [B1, B2] where B1, B2 are 2x6
    # C = A1*B1 + A2*B2
    #
    # Each term is 2x2 * 2x6.
    # For 2x2*2x6, we can use Strassen on the 2x2*2x2 part and replicate for 6 cols.
    # Actually, 2x2*2x6 = 2x2*2x(2x3), so we do 3 separate 2x2*2x2 multiplications.
    # Each 2x2*2x2 has rank 7 (Strassen).
    # So 2x2*2x6 has rank 7*3 = 21.
    # Two such terms: 21 + 21 = 42.
    
    # This gives rank 42!
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen 2x2*2x2 decomposition:
    # Given A (2x2) = [[a11, a12], [a21, a22]], B (2x2) = [[b11, b12], [b21, b22]]
    # C = A*B = [[c11, c12], [c21, c22]]
    #
    # Strassen's 7 products:
    # M1 = (a11 + a22) * (b11 + b22)
    # M2 = (a21 + a22) * b11
    # M3 = a11 * (b12 - b22)
    # M4 = a22 * (b21 - b11)
    # M5 = (a11 + a12) * b22
    # M6 = (a21 - a11) * (b11 + b12)
    # M7 = (a12 - a22) * (b21 + b22)
    #
    # c11 = M1 - M4 + M5 - M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M7
    
    # For our case:
    # A is 2x4, split into A1 (2x2, rows 0-1, cols 0-1) and A2 (2x2, rows 0-1, cols 2-3)
    # Wait, no. Let me re-index.
    #
    # A is 2x4. Split along columns: A = [A_left (2x2), A_right (2x2)]
    # B is 4x6. Split along rows: B = [B_top (2x6); B_bot (2x6)]
    #
    # C = A_left * B_top + A_right * B_bot
    #
    # A_left has elements at indices (i, j) for i in 0..1, j in 0..1
    # A_right has elements at indices (i, j) for i in 0..1, j in 2..3
    # B_top has elements at indices (j, k) for j in 0..1, k in 0..5
    # B_bot has elements at indices (j, k) for j in 2..3, k in 0..5
    #
    # For each term (2x2 * 2x6), we split the 6 columns into 3 groups of 2:
    # B_top = [Bt1 (2x2), Bt2 (2x2), Bt3 (2x2)]
    # B_bot = [Bb1 (2x2), Bb2 (2x2), Bb3 (2x2)]
    #
    # C = A_left*Bt1 + A_left*Bt2 + A_left*Bt3 + A_right*Bb1 + A_right*Bb2 + A_right*Bb3
    #
    # Each 2x2*2x2 uses 7 Strassen products, so total = 6 * 7 = 42.
    
    # Strassen coefficient table for 2x2*2x2:
    # For product M_t (t=0..6), the coefficients for A elements and B elements:
    # A is indexed as [a00, a01, a10, a11] (row-major)
    # B is indexed as [b00, b01, b10, b11] (row-major)
    # C output is [c00, c01, c10, c11]
    
    # Strassen products:
    # M0 = (a00+a11)*(b00+b11)
    # M1 = (a10+a11)*b00
    # M2 = a00*(b01-b11)
    # M3 = a11*(b10-b00)
    # M4 = (a00+a01)*b11
    # M5 = (a10-a00)*(b00+b01)
    # M6 = (a01-a11)*(b10+b11)
    
    # C reconstruction:
    # c00 = M0 - M3 + M4 - M6
    # c01 = M2 + M4
    # c10 = M1 + M3
    # c11 = M0 - M1 + M2 + M6
    
    # Let's define the Strassen decomposition for a single 2x2*2x2 block.
    # For each of the 7 products, we have coefficients for the 4 A-elements and 4 B-elements.
    
    # A elements: a00, a01, a10, a11
    # B elements: b00, b01, b10, b11
    
    # M0: (a00+a11)*(b00+b11)
    # This is a sum of 4 products: a00*b00, a00*b11, a11*b00, a11*b11
    # But in tensor decomposition, each "product" M_t is a single rank-1 term in the 
    # decomposition of the multiplication tensor. Wait, no.
    
    # Actually, in the tensor decomposition view:
    # The multiplication tensor T has entries T[i,j,k] which is 1 if k = i*j (in some sense).
    # For matrix multiplication, T[A_idx, B_idx, C_idx] = 1 if C_idx corresponds to 
    # the product of A_idx and B_idx.
    
    # In Strassen's algorithm, we're expressing the tensor as a sum of rank-1 tensors.
    # Each rank-1 tensor corresponds to one of the 7 Strassen products.
    
    # For a 2x2*2x2 multiplication:
    # A has 4 elements: indices 0,1,2,3 (a00, a01, a10, a11)
    # B has 4 elements: indices 0,1,2,3 (b00, b01, b10, b11)
    # C has 4 elements: indices 0,1,2,3 (c00, c01, c10, c11)
    
    # The tensor T[a,b,c] = 1 if c = a*b in the appropriate sense.
    # Specifically:
    # c00 = a00*b00 + a01*b10
    # c01 = a00*b01 + a01*b11
    # c10 = a10*b00 + a11*b10
    # c11 = a10*b01 + a11*b11
    
    # So T is a 4x4x4 tensor with:
    # T[0,0,0] = 1 (a00*b00 -> c00)
    # T[1,2,0] = 1 (a01*b10 -> c00)
    # T[0,1,1] = 1 (a00*b01 -> c01)
    # T[1,3,1] = 1 (a01*b11 -> c01)
    # T[2,0,2] = 1 (a10*b00 -> c10)
    # T[3,2,2] = 1 (a11*b10 -> c10)
    # T[2,1,3] = 1 (a10*b01 -> c11)
    # T[3,3,3] = 1 (a11*b11 -> c11)
    
    # Strassen decomposes this rank-8 tensor into 7 rank-1 tensors.
    # Each rank-1 tensor is u * v * w where u, v, w are vectors.
    
    # The 7 Strassen terms:
    # Term 0: u = [1,0,0,1], v = [1,0,0,1], w = [1,0,0,1]  -> (a00+a11)(b00+b11) contributes to c00 and c11
    # Wait, that's not quite right. Let me think more carefully.
    
    # Actually, in the standard Strassen decomposition:
    # M0 = (a00+a11)(b00+b11): u0=[1,0,0,1], v0=[1,0,0,1], w0=[1,0,0,1]
    #   This contributes +1 to c00 and +1 to c11
    # M1 = (a10+a11)b00: u1=[0,0,1,1], v1=[1,0,0,0], w1=[0,0,1,0]
    #   This contributes +1 to c10
    # M2 = a00(b01-b11): u2=[1,0,0,0], v2=[0,1,0,-1], w2=[0,1,0,0]
    #   This contributes +1 to c01
    # M3 = a11(b10-b00): u3=[0,0,0,1], v3=[-1,0,1,0], w3=[0,0,1,0]
    #   This contributes +1 to c10
    # M4 = (a00+a01)b11: u4=[1,1,0,0], v4=[0,0,0,1], w4=[1,1,0,0]
    #   This contributes +1 to c00 and +1 to c01
    # M5 = (a10-a00)(b00+b01): u5=[-1,0,1,0], v5=[1,1,0,0], w5=[0,0,1,0]
    #   This contributes +1 to c10
    # M6 = (a01-a11)(b10+b11): u6=[0,1,0,-1], v6=[0,0,1,1], w6=[1,0,0,1]
    #   This contributes +1 to c00 and +1 to c11
    
    # Let me verify:
    # c00 = M0 + M4 - M3... wait, let me just use the standard formulas.
    # 
    # Standard Strassen:
    # M1 = (a11+a22)(b11+b22)  [using 1-indexed, so a00+a11 in 0-indexed]
    # M2 = (a21+a22)b11  [a10+a11, b00]
    # M3 = a11(b12-b22)  [a00, b01-b11]
    # M4 = a22(b21-b11)  [a11, b10-b00]
    # M5 = (a11+a12)b22  [a00+a01, b11]
    # M6 = (a21-a11)(b11+b12)  [a10-a00, b00+b01]
    # M7 = (a12-a22)(b21+b22)  [a01-a11, b10+b11]
    #
    # c11 = M1 - M4 + M5 - M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M7
    
    # In 0-indexed:
    # c00 = M0 - M3 + M4 - M6
    # c01 = M2 + M4
    # c10 = M1 + M3
    # c11 = M0 - M1 + M2 + M6
    
    # So the w vectors (coefficients for C) are:
    # M0: w=[1,0,0,1]
    # M1: w=[0,0,1,0]
    # M2: w=[0,1,0,0]
    # M3: w=[0,0,1,0]
    # M4: w=[1,1,0,0]
    # M5: w=[0,0,1,0]
    # M6: w=[1,0,0,1]
    
    # And u vectors (coefficients for A):
    # M0: u=[1,0,0,1]
    # M1: u=[0,0,1,1]
    # M2: u=[1,0,0,0]
    # M3: u=[0,0,0,1]
    # M4: u=[1,1,0,0]
    # M5: u=[-1,0,1,0]
    # M6: u=[0,1,0,-1]
    
    # And v vectors (coefficients for B):
    # M0: v=[1,0,0,1]
    # M1: v=[1,0,0,0]
    # M2: v=[0,1,0,-1]
    # M3: v=[-1,0,1,0]
    # M4: v=[0,0,0,1]
    # M5: v=[1,1,0,0]
    # M6: v=[0,0,1,1]
    
    strassen_u = [
        [1, 0, 0, 1],   # M0
        [0, 0, 1, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 0, 0, 1],   # M3
        [1, 1, 0, 0],   # M4
        [-1, 0, 1, 0],  # M5
        [0, 1, 0, -1],  # M6
    ]
    strassen_v = [
        [1, 0, 0, 1],   # M0
        [1, 0, 0, 0],   # M1
        [0, 1, 0, -1],  # M2
        [-1, 0, 1, 0],  # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [0, 0, 1, 1],   # M6
    ]
    strassen_w = [
        [1, 0, 0, 1],   # M0
        [0, 0, 1, 0],   # M1
        [0, 1, 0, 0],   # M2
        [0, 0, 1, 0],   # M3
        [1, 1, 0, 0],   # M4
        [0, 0, 1, 0],   # M5
        [1, 0, 0, 1],   # M6
    ]
    
    # Now, for our 2x4x6 problem:
    # A is 2x4, indexed as A[i,j] with i in 0..1, j in 0..3
    # B is 4x6, indexed as B[j,k] with j in 0..3, k in 0..5
    # C is 2x6, indexed as C[i,k] with i in 0..1, k in 0..5
    #
    # We split:
    # A_left = A[:, 0:2] (2x2), A_right = A[:, 2:4] (2x2)
    # B_top = B[0:2, :] (2x6), B_bot = B[2:4, :] (2x6)
    #
    # C = A_left * B_top + A_right * B_bot
    #
    # For A_left * B_top (2x2 * 2x6):
    # Split B_top into 3 blocks of 2x2: Bt1 = B_top[:, 0:2], Bt2 = B_top[:, 2:4], Bt3 = B_top[:, 4:6]
    # A_left * B_top = A_left*Bt1 + A_left*Bt2 + A_left*Bt3
    #
    # Similarly for A_right * B_bot.
    
    # For each 2x2*2x2 sub-problem, we use the 7 Strassen terms.
    # Total: 6 sub-problems * 7 terms = 42 terms.
    
    # Let's index the sub-problems:
    # Sub-problem (side, block) where side in {0, 1} (left, right) and block in {0, 1, 2}
    
    # For sub-problem (side, block):
    # If side=0 (left): A elements are A[i, block*2 + j] for i in 0..1, j in 0..1
    # If side=1 (right): A elements are A[i, 2 + block*2 + j] for i in 0..1, j in 0..1
    # B elements are B[(side*2 + j), block*2 + k] for j in 0..1, k in 0..1
    # C elements are C[i, block*2 + k] for i in 0..1, k in 0..1
    
    # In the U matrix (size n*m = 8 x rank):
    # U[i*m + j, col] corresponds to A[i, j]
    # For side=0, block=b: A[i, j] has global index (i, b*2 + j)
    # For side=1, block=b: A[i, j] has global index (i, 2 + b*2 + j)
    
    # In the V matrix (size m*p = 24 x rank):
    # V[j*p + k, col] corresponds to B[j, k]
    # For side=s, block=b: B[j, k] has global index (s*2 + j, b*2 + k)
    
    # In the W matrix (size p*n = 12 x rank):
    # W[k*n + i, col] corresponds to C[i, k]
    # For block=b: C[i, k] has global index (i, b*2 + k), so W index = (b*2 + k)*n + i = (b*2 + k)*2 + i
    
    col = 0
    for side in range(2):  # 0=left, 1=right
        for block in range(3):  # 0, 1, 2
            for t in range(7):  # 7 Strassen terms
                # U coefficients
                for i in range(2):
                    for j in range(2):
                        a_idx = i * m + (side * 2 + block * 2 + j)
                        U[a_idx][col] += strassen_u[t][i * 2 + j]
                
                # V coefficients
                for j in range(2):
                    for k in range(2):
                        b_idx = (side * 2 + j) * p + (block * 2 + k)
                        V[b_idx][col] += strassen_v[t][j * 2 + k]
                
                # W coefficients
                for i in range(2):
                    for k in range(2):
                        c_idx = (block * 2 + k) * n + i
                        W[c_idx][col] += strassen_w[t][i * 2 + k]
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
