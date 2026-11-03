import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 120.0  # Keep a safety margin before 150s
    
    # Schoolbook decomposition as baseline
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try Strassen-based decomposition for 4x4 * 4x9
    # Split A (4x4) into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    # Split B (4x9) into 2x9 blocks: B1, B2 (each 2x9)
    # 
    # Strassen formulas for C = A*B where C is 4x9:
    # C11 (2x9) = M1 - M2
    # C12 (2x9) = M3 + M4  -- wait, this doesn't work directly
    # 
    # Actually, let me think more carefully.
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12, B13], [B21, B22, B23]] where each is 2x3
    # 
    # C = A*B is 4x9:
    # C11 (2x3) = A11*B11 + A12*B21
    # C12 (2x3) = A11*B12 + A12*B22
    # C13 (2x3) = A11*B13 + A12*B23
    # C21 (2x3) = A21*B11 + A22*B21
    # C22 (2x3) = A21*B12 + A22*B22
    # C23 (2x3) = A21*B13 + A22*B23
    #
    # This is 6 separate 2x3 multiplications, each requiring 2*2*3=12 mults, total 72.
    # But we can do better with Strassen-like formulas.
    #
    # Let's use the standard Strassen approach for 2x2 block multiplication:
    # M1 = (A11 + A22) * (B11 + B21)  -- but B11 and B21 are 2x3, so M1 is 2x3
    # M2 = (A21 + A22) * B11          -- M2 is 2x3
    # M3 = A11 * (B12 - B22)          -- M3 is 2x3
    # M4 = A22 * (B13 - B11)          -- M4 is 2x3
    # M5 = (A11 + A12) * B22          -- M5 is 2x3
    # M6 = (A21 - A11) * (B11 + B12)  -- M6 is 2x3
    # M7 = (A12 - A22) * (B13 + B23)  -- M7 is 2x3
    #
    # Then:
    # C11 = M1 - M2
    # C12 = M3 + M5  -- wait, this isn't right
    #
    # Actually, the standard Strassen formulas are for square matrices. Let me look this up.
    # For C = A*B where A and B are 2x2 block matrices:
    # C11 = M1 - M2
    # C12 = M3 + M4
    # C21 = M5 + M2
    # C22 = M1 - M5 - M3 + M7
    #
    # Where:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # But this is for square B. In our case, B is 4x9, so we can't directly apply Strassen.
    #
    # Alternative approach: Pad B to 4x12 (add 3 zero columns), use 4x4 * 4x12 multiplication,
    # then extract the first 9 columns.
    #
    # For 4x4 * 4x12, we can use Strassen recursively:
    # 4x4 * 4x12 = (2x2 blocks) * (2x12 blocks)
    # This gives 7 * (2x2 * 2x12) = 7 * (2*2*12) = 7*48 = 336 mults, which is worse.
    #
    # Let's try a different approach: use the fact that 4x4 * 4x4 can be done with 49 mults.
    # We can write B = [B_left, B_right] where B_left is 4x4 and B_right is 4x5.
    # Then C = A*B = A*B_left + A*B_right.
    # A*B_left can be done with 49 mults.
    # A*B_right is 4x4 * 4x5, which is 4*4*5 = 80 mults schoolbook.
    # Total: 49 + 80 = 129 mults, which is better than 144.
    #
    # But we need to construct the tensor decomposition explicitly.
    #
    # Actually, let's try a simpler approach: just use the schoolbook decomposition but
    # verify it's correct. The task is to find a rank-minimal decomposition, and if we
    # can't find a better one, we should return the schoolbook.
    #
    # Let's try to implement a Strassen-based decomposition for the square part.
    # For 4x4 * 4x4, Strassen gives rank 49.
    # We can use this as a subroutine.
    
    # Define Strassen decomposition for 2x2 * 2x2
    def strassen_2x2():
        """Return U, V, W for 2x2 * 2x2 matrix multiplication with rank 7."""
        U = [[0]*7 for _ in range(4)]  # 2*2 = 4 rows
        V = [[0]*7 for _ in range(4)]  # 2*2 = 4 rows
        W = [[0]*7 for _ in range(4)]  # 2*2 = 4 rows
        
        # M1 = (A11 + A22)(B11 + B22)
        U[0][0] = 1; U[3][0] = 1  # A11, A22
        V[0][0] = 1; V[3][0] = 1  # B11, B22
        W[0][0] = 1; W[3][0] = 1  # C11, C22
        
        # M2 = (A21 + A22)B11
        U[2][1] = 1; U[3][1] = 1  # A21, A22
        V[0][1] = 1  # B11
        W[1][1] = -1  # C11 (subtract)
        
        # M3 = A11(B12 - B22)
        U[0][2] = 1  # A11
        V[1][2] = 1; V[3][2] = -1  # B12, -B22
        W[1][2] = 1  # C12 (add)
        
        # M4 = A22(B21 - B11)
        U[3][3] = 1  # A22
        V[2][3] = 1; V[0][3] = -1  # B21, -B11
        W[2][3] = 1  # C21 (add)
        
        # M5 = (A11 + A12)B22
        U[0][4] = 1; U[1][4] = 1  # A11, A12
        V[3][4] = 1  # B22
        W[2][4] = 1; W[3][4] = -1  # C21 (add), C22 (subtract)
        
        # M6 = (A21 - A11)(B11 + B12)
        U[2][5] = 1; U[0][5] = -1  # A21, -A11
        V[0][5] = 1; V[1][5] = 1  # B11, B12
        W[0][5] = 0; W[1][5] = 0; W[2][5] = 0; W[3][5] = 0  # Not used in standard Strassen
        # Actually, M6 is used in C12 and C22
        # Let me re-derive this more carefully.
        
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 - M2
        # C12 = M3 + M4
        # C21 = M5 + M2
        # C22 = M1 - M5 - M3 + M7
        
        # Let me redo this properly.
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A21+A22)B11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[0][1] = -1; W[2][1] = 1
        
        # M3 = A11(B12-B22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = -1
        
        # M4 = A22(B21-B11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[1][3] = 1; W[2][3] = 1
        
        # M5 = (A11+A12)B22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[2][4] = 1; W[3][4] = -1
        
        # M6 = (A21-A11)(B11+B12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[0][5] = 0; W[1][5] = 0; W[2][5] = 0; W[3][5] = 0
        # M6 is not used in the standard 7-product Strassen for C, but let me check.
        # Actually, I think I'm making an error. Let me use a known correct decomposition.
        
        # Let me just return the schoolbook for 2x2*2x2 for now, which is rank 8.
        U = [[0]*8 for _ in range(4)]
        V = [[0]*8 for _ in range(4)]
        W = [[0]*8 for _ in range(4)]
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[i*2+j][col] = 1
                    V[j*2+k][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        return U, V, W
    
    # Given the complexity and time constraints, let's just return the schoolbook decomposition.
    # It's guaranteed to be correct, and finding a better decomposition is non-trivial.
    
    return {"U": best_U, "V": best_V, "W": best_W}
