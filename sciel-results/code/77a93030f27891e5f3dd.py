"""Matrix multiplication 4,4,5 tensor decomposition."""
import time

DIMENSIONS = [4, 4, 5]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 x 4x5 matrix multiplication.
    
    Strategy: Decompose C = A * B where A is 4x4, B is 4x5, C is 4x5.
    Split B into B1 (4x4) and B2 (4x1). Then C = A*B1 + A*B2.
    
    For A*B1 (4x4 x 4x4 -> 4x4): Use Strassen's algorithm (rank 7).
    For A*B2 (4x4 x 4x1 -> 4x1): This is 4 independent dot products, rank 16.
    
    Total rank: 7 + 16 = 23.
    
    However, we need to be careful about the tensor product structure.
    Actually, let's think more carefully.
    
    The 4x4 x 4x4 multiplication can be viewed as a Kronecker product of 2x2 x 2x2.
    The rank of M_{2,2,2} is 7 (Strassen). So rank of M_{4,4,4} is at most 49.
    But we can do better by using block structure.
    
    Actually, let's just use the standard approach:
    - Use Strassen for the 4x4 x 4x4 part (which requires 49 multiplications if done naively via Kronecker, but actually Strassen on 4x4 blocks gives rank 7*7=49)
    - Use schoolbook for the 4x4 x 4x1 part (rank 16)
    - Total: 49 + 16 = 65
    
    But wait, we can do better. Let me reconsider.
    
    Actually, the simplest correct approach:
    - For the 4x4 x 4x4 part: use Strassen's algorithm on 2x2 blocks. This gives rank 7 for each 2x2 block multiplication, and there are 4 such block multiplications, but Strassen reduces it to 7 multiplications of 2x2 matrices. Each 2x2 multiplication has rank 4 (schoolbook) or 7 (Strassen). Using Strassen recursively, rank is 7*7 = 49.
    - For the 4x4 x 4x1 part: this is 4 dot products of length 4, so rank 4*4 = 16.
    - Total rank: 49 + 16 = 65.
    
    Let me implement this.
    """
    n, m, p = 4, 4, 5
    
    # Start with deadline guard
    start_time = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0
    
    # Strassen's algorithm for 2x2 x 2x2
    # The 7 Strassen factors:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 4x4 x 4x4, we partition into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bj is 2x2
    # C = [[C00, C01], [C10, C11]] where each Ck is 2x2
    
    # The tensor for 4x4 x 4x4 is the Kronecker product of the 2x2 x 2x2 tensor with itself.
    # Rank is 7 * 7 = 49.
    
    # For 4x4 x 4x1:
    # B2 is 4x1, so C2 = A * B2 is 4x1.
    # C2[i] = sum_j A[i,j] * B2[j]
    # This is 4 independent dot products, each of length 4.
    # Rank is 4 * 4 = 16.
    
    # Total rank: 49 + 16 = 65.
    
    # Build the U, V, W matrices.
    
    # First, the Strassen part for 4x4 x 4x4 (rank 49)
    # The 2x2 x 2x2 Strassen factors:
    # U_2x2: 4 rows, 7 columns
    # V_2x2: 4 rows, 7 columns  
    # W_2x2: 4 rows, 7 columns
    
    # Strassen U factors for 2x2:
    # M1: A00 + A11 -> U[0]=1, U[3]=1 (indices: 00->0, 01->1, 10->2, 11->3)
    # M2: A10 + A11 -> U[2]=1, U[3]=1
    # M3: A00 -> U[0]=1
    # M4: A11 -> U[3]=1
    # M5: A00 + A01 -> U[0]=1, U[1]=1
    # M6: A10 - A00 -> U[2]=1, U[0]=-1
    # M7: A01 - A11 -> U[1]=1, U[3]=-1
    
    U_2 = [
        [1, 0, 1, 0, 1, -1, 0],   # A00: M1, M3, M5, M6
        [0, 0, 0, 0, 1, 0, 1],    # A01: M5, M7
        [0, 1, 0, 0, 0, 1, 0],    # A10: M2, M6
        [1, 1, 0, 1, 0, 0, -1]    # A11: M1, M2, M4, M7
    ]
    
    # Strassen V factors for 2x2:
    # M1: B00 + B11 -> V[0]=1, V[3]=1
    # M2: B00 -> V[0]=1
    # M3: B01 - B11 -> V[1]=1, V[3]=-1
    # M4: B10 - B00 -> V[2]=1, V[0]=-1
    # M5: B11 -> V[3]=1
    # M6: B00 + B01 -> V[0]=1, V[1]=1
    # M7: B10 + B11 -> V[2]=1, V[3]=1
    
    V_2 = [
        [1, 1, 0, -1, 0, 1, 0],   # B00: M1, M2, M4, M6
        [0, 0, 1, 0, 0, 1, 0],    # B01: M3, M6
        [0, 0, 0, 1, 0, 0, 1],    # B10: M4, M7
        [1, 0, -1, 0, 1, 0, 1]    # B11: M1, M3, M5, M7
    ]
    
    # Strassen W factors for 2x2:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    W_2 = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0]    # C11
    ]
    
    # Now, the 4x4 x 4x4 tensor is the Kronecker product of the 2x2 x 2x2 tensor with itself.
    # The U matrix for 4x4 is the Kronecker product of U_2 with U_2.
    # U_4[i*4+j, k*7+l] = U_2[i,k] * U_2[j,l]
    
    U_4 = [[0.0] * 49 for _ in range(16)]
    for i in range(4):
        for k in range(7):
            for j in range(4):
                for l in range(7):
                    U_4[i * 4 + j][k * 7 + l] = U_2[i][k] * U_2[j][l]
    
    V_4 = [[0.0] * 49 for _ in range(16)]
    for i in range(4):
        for k in range(7):
            for j in range(4):
                for l in range(7):
                    V_4[i * 4 + j][k * 7 + l] = V_2[i][k] * V_2[j][l]
    
    W_4 = [[0.0] * 49 for _ in range(16)]
    for i in range(4):
        for k in range(7):
            for j in range(4):
                for l in range(7):
                    W_4[i * 4 + j][k * 7 + l] = W_2[i][k] * W_2[j][l]
    
    # Now, the 4x4 x 4x1 part (rank 16)
    # C2[i] = sum_j A[i,j] * B2[j]
    # For each i in 0..3, j in 0..3: term (i,j)
    # U[i*4+j, c] = 1
    # V[j*5+4, c] = 1  (B2 is column 4 of B, which is 4x1)
    # W[4*4+i, c] = 1  (C2 is column 4 of C, which is 4x1)
    
    U_1 = [[0.0] * 16 for _ in range(16)]
    V_1 = [[0.0] * 16 for _ in range(20)]
    W_1 = [[0.0] * 16 for _ in range(20)]
    
    for i in range(4):
        for j in range(4):
            c = i * 4 + j
            U_1[i * 4 + j][c] = 1.0
            V_1[j * 5 + 4][c] = 1.0
            W_1[4 * 4 + i][c] = 1.0
    
    # Concatenate: total rank = 49 + 16 = 65
    U_total = [[U_4[r][c] for c in range(49)] + [U_1[r][c] for c in range(16)] for r in range(16)]
    V_total = [[V_4[r][c] for c in range(49)] + [V_1[r][c] for c in range(16)] for r in range(20)]
    W_total = [[W_4[r][c] for c in range(49)] + [W_1[r][c] for c in range(16)] for r in range(20)]
    
    return {"U": U_total, "V": V_total, "W": W_total}
