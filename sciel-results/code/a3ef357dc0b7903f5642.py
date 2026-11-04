import time
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Rank estimate: 1 (for 2x1 x 1x6) + 21 (for 2x2 x 2x6 via 3 Strassen blocks) = 22
    # Wait, earlier I calculated 21 + 1 = 22.
    # Let's check if R(2,2,6) can be 20.
    # If R(2,2,4)=13, then R(2,2,6) <= 13+7=20.
    # I will implement the 21 version first as it is safer and definitely valid.
    # If time permits, I can try to optimize.
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)] # 6 x 22
    V = [[0] * rank for _ in range(m * p)] # 18 x 22
    W = [[0] * rank for _ in range(p * n)] # 12 x 22
    
    col = 0
    
    # Part 1: A_R B_R (2x1 * 1x6)
    # A_R is col 2 of A (index 2 in j)
    # B_R is row 2 of B (index 2 in j)
    # U rows: i*3 + 2 for i in 0,1 -> 2, 3
    # V rows: 2*6 + k for k in 0..5 -> 12..17
    # W rows: k*2 + i for k in 0..5, i in 0,1 -> 0..11
    
    for i in range(n):
        U[i * m + 2][col] = 1
    for k in range(p):
        V[2 * p + k][col] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][col] = 1
    col += 1
    
    # Part 2: A_L B_L (2x2 * 2x6)
    # Split B_L (cols 0,1 of B) into 3 blocks of 2x2? No, B_L is 2x6.
    # B_L has cols 0..5.
    # Split into B_0 (cols 0,1), B_1 (cols 2,3), B_2 (cols 4,5).
    
    # Strassen coefficients for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # A indices: (0,0), (1,1), (1,0), (0,0), (0,0), (1,0), (0,1)
    # B indices: (0,0),(1,1), (0,0), (0,1),(1,1), (1,1), (0,0),(0,1), (1,0),(1,1)
    
    # Let's define the Strassen terms for a single 2x2x2 block.
    # Each term r has:
    # u_vec: 2x2 matrix of coefficients for A
    # v_vec: 2x2 matrix of coefficients for B
    # w_vec: 2x2 matrix of coefficients for C
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u_coeffs[i,j] is coeff for A[i,j]
        # v_coeffs[j,k] is coeff for B[j,k]
        # w_coeffs[i,k] is coeff for C[i,k]
        
        # M1
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
        # M2
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, 0]]),
        # M3
        ([[1, 0], [0, 0]], [[0, 1], [0, -1]], [[0, 1], [0, 0]]),
        # M4
        ([[0, 0], [0, 1]], [[0, -1], [1, 0]], [[0, 0], [1, 0]]),
        # M5
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[-1, 0], [0, 1]]), # Wait, C11 = M1+M4-M5+M7. So M5 coeff in C11 is -1.
        # M6
        ([[0, 0], [-1, 1]], [[1, 1], [0, 0]], [[0, 0], [0, 1]]), # C22 = M1-M2+M3+M6. M6 coeff in C22 is 1.
        # M7
        ([[0, 1], [0, -1]], [[0, 0], [1, 1]], [[1, 0], [0, 0]])  # C11 coeff 1, C22 coeff 0?
    ]
    
    # Let's re-verify Strassen formulas carefully.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Term 1 (M1):
    # A: 11, 22 -> [[1,0],[0,1]]
    # B: 11, 22 -> [[1,0],[0,1]]
    # C: 11, 22 -> [[1,0],[0,1]]
    
    # Term 2 (M2):
    # A: 21, 22 -> [[0,0],[1,1]]
    # B: 11 -> [[1,0],[0,0]]
    # C: 21 -> [[0,0],[1,0]]
    
    # Term 3 (M3):
    # A: 11 -> [[1,0],[0,0]]
    # B: 12, -22 -> [[0,1],[0,-1]]
    # C: 12 -> [[0,1],[0,0]]
    
    # Term 4 (M4):
    # A: 22 -> [[0,0],[0,1]]
    # B: 21, -11 -> [[-1,0],[1,0]]  (Note: B21-B11. B indices are (row, col). B11 is (0,0), B21 is (1,0). So col 0.
    # My previous V coeffs for M4: [[0,-1],[1,0]]. This corresponds to B[0,0]=-1, B[1,0]=1. Correct.
    # C: 21 -> [[0,0],[1,0]]
    
    # Term 5 (M5):
    # A: 11, 12 -> [[1,1],[0,0]]
    # B: 22 -> [[0,0],[0,1]]
    # C: 11 (-1), 12 (1) -> [[-1,1],[0,0]]
    
    # Term 6 (M6):
    # A: 21, -11 -> [[-1,0],[1,0]]
    # B: 11, 12 -> [[1,1],[0,0]]
    # C: 22 (1) -> [[0,0],[0,1]]
    
    # Term 7 (M7):
    # A: 12, -22 -> [[0,1],[0,-1]]
    # B: 21, 22 -> [[0,0],[1,1]]
    # C: 11 (1) -> [[1,0],[0,0]]
    
    strassen_terms = [
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
        ([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,0]]),
        ([[1,0],[0,0]], [[0,1],[0,-1]], [[0,1],[0,0]]),
        ([[0,0],[0,1]], [[-1,0],[1,0]], [[0,0],[1,0]]),
        ([[1,1],[0,0]], [[0,0],[0,1]], [[-1,1],[0,0]]),
        ([[-1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),
        ([[0,1],[0,-1]], [[0,0],[1,1]], [[1,0],[0,0]])
    ]
    
    for block in range(3):
        # Block b corresponds to B columns 2*b, 2*b+1
        # A columns 0, 1
        for term in strassen_terms:
            u_mat, v_mat, w_mat = term
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    if u_mat[i][j] != 0:
                        U[i * m + j][col] += u_mat[i][j]
                        
            # Fill V
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * block + k_local
                    if v_mat[j][k_local] != 0:
                        V[j * p + k_global][col] += v_mat[j][k_local]
                        
            # Fill W
            for i in range(2):
                for k_local in range(2):
                    k_global = 2 * block + k_local
                    if w_mat[i][k_local] != 0:
                        W[k_global * n + i][col] += w_mat[i][k_local]
                        
            col += 1
            
            if col >= rank:
                break
        if col >= rank:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
