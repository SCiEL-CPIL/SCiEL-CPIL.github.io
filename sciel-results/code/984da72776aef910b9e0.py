"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _build_strassen_2x2_2():
    # Strassen decomposition for 2x2x2 matrix multiplication
    # Returns U, V, W as lists of lists (2x2=4 rows for U and V, 2x2=4 rows for W)
    # Rank 7
    # Indices: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # For 2x2x2: n=2, m=2, p=2
    # U is 4x7, V is 4x7, W is 4x7
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen's 7 products for 2x2x2:
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M5
    # C21 = M1 + M2
    # C22 = M2 - M3 + M4 + M6
    # where:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Row indices for U (A part): 0=A11, 1=A12, 2=A21, 3=A22
    # Row indices for V (B part): 0=B11, 1=B21, 2=B12, 3=B22
    #   Wait, V[j*p+k, r] with m=2, p=2: j=0,1; k=0,1
    #   V[0*2+0] = V[0] = B11, V[0*2+1] = V[1] = B12
    #   V[1*2+0] = V[2] = B21, V[1*2+1] = V[3] = B22
    # Row indices for W (C part): W[k*n+i, r] with p=2, n=2: k=0,1; i=0,1
    #   W[0*2+0] = W[0] = C11, W[0*2+1] = W[1] = C12
    #   W[1*2+0] = W[2] = C21, W[1*2+1] = W[3] = C22
    
    # M1: (A11 + A22)(B11 + B22) -> contributes to C11 with +1
    # U: A11=1, A22=1; V: B11=1, B22=1; W: C11=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1
    
    # M2: (A21 + A22)B11 -> contributes to C21 with +1, C22 with +1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3: A11(B21 - B22) -> contributes to C12 with +1, C22 with -1
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = -1
    
    # M4: A22(B12 - B11) -> contributes to C11 with -1, C22 with +1
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[3][3] = 1
    
    # M5: (A11 + A12)B22 -> contributes to C11 with +1, C12 with +1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # M6: (A21 - A11)(B11 + B12) -> contributes to C22 with +1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A12 - A22)(B21 + B22) -> contributes to C11 with +1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_schoolbook_2x2_1():
    # Schoolbook decomposition for 2x2x1 matrix multiplication (multiply 2x2 by 2x1)
    # n=2, m=2, p=1
    # Rank = 2*2*1 = 4
    U = [[0]*4 for _ in range(4)]  # 2*2 = 4 rows
    V = [[0]*4 for _ in range(2)]  # 2*1 = 2 rows
    W = [[0]*4 for _ in range(2)]  # 1*2 = 2 rows
    
    # C[i,0] = sum_j A[i,j] B[j,0]
    for i in range(2):
        for j in range(2):
            col = i*2 + j
            U[i*2+j][col] = 1
            V[j*1+0][col] = 1
            W[0*2+i][col] = 1  # k=0 only, p=1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # Split the m=4 dimension into two blocks of size 2:
    # A is 2x4 = [A_left | A_right] where A_left is 2x2, A_right is 2x2
    # B is 4x5 = [B_top; B_bottom] where B_top is 2x5, B_bottom is 2x5
    # C = A_left @ B_top + A_right @ B_bottom (each term is 2x5)
    
    # For each block, we split p=5 into p=2 and p=1:
    # A_left @ B_top: B_top is 2x5, split into B_top_left (2x2) and B_top_right (2x1)
    #   = A_left @ B_top_left + A_left @ B_top_right
    # A_right @ B_bottom: B_bottom is 2x5, split into B_bottom_left (2x2) and B_bottom_right (2x1)
    #   = A_right @ B_bottom_left + A_right @ B_bottom_right
    
    # So total = 2*(2x2x2) + 2*(2x2x1)
    # Rank = 2*7 + 2*4 = 14 + 8 = 22
    
    strassen_U, strassen_V, strassen_W = _build_strassen_2x2_2()
    school_U, school_V, school_W = _build_schoolbook_2x2_1()
    
    # Build the full decomposition
    # U has n*m = 8 rows, V has m*p = 20 rows, W has p*n = 10 rows
    total_rank = 22
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col_offset = 0
    
    # Block 1: A_left (rows 0,1 of A) @ B_top_left (cols 0,1 of B_top, which are rows 0,1 of B)
    # A_left indices: i in {0,1}, j in {0,1} -> U[i*4 + j, ...]
    # B_top_left indices: j in {0,1}, k in {0,1} -> V[j*5 + k, ...]
    # C output: k in {0,1}, i in {0,1} -> W[k*2 + i, ...]
    for r in range(7):
        for i in range(2):
            for j in range(2):
                u_idx = i * 4 + j  # n=2, m=4: A[i,j] at row i*4+j
                U[u_idx][col_offset + r] = strassen_U[i*2+j][r]
        for j in range(2):
            for k in range(2):
                v_idx = j * 5 + k  # m=4, p=5: B[j,k] at row j*5+k
                V[v_idx][col_offset + r] = strassen_V[j*2+k][r]
        for k in range(2):
            for i in range(2):
                w_idx = k * 2 + i  # p=5, n=2: W[k*n+i]
                W[w_idx][col_offset + r] = strassen_W[k*2+i][r]
    col_offset += 7
    
    # Block 2: A_left @ B_top_right (cols 2 of B_top, which is row 2 of B)
    # B_top_right is 2x1: j in {0,1}, k=2 -> V[j*5+2, ...]
    # C output: k=2, i in {0,1} -> W[2*2+i, ...]
    for r in range(4):
        for i in range(2):
            for j in range(2):
                u_idx = i * 4 + j
                U[u_idx][col_offset + r] = school_U[i*2+j][r]
        for j in range(2):
            v_idx = j * 5 + 2  # k=2
            V[v_idx][col_offset + r] = school_V[j][r]
        for i in range(2):
            w_idx = 2 * 2 + i  # k=2
            W[w_idx][col_offset + r] = school_W[i][r]
    col_offset += 4
    
    # Block 3: A_right (rows 0,1 of A, cols 2,3) @ B_bottom_left (rows 2,3 of B, cols 0,1)
    # A_right: i in {0,1}, j in {2,3} -> U[i*4 + j, ...]
    # B_bottom_left: j in {2,3}, k in {0,1} -> V[j*5 + k, ...]
    # C output: k in {0,1}, i in {0,1} -> W[k*2 + i, ...]
    for r in range(7):
        for i in range(2):
            for j in range(2):
                u_idx = i * 4 + (j + 2)  # j+2 for A_right
                U[u_idx][col_offset + r] = strassen_U[i*2+j][r]
        for j in range(2):
            for k in range(2):
                v_idx = (j + 2) * 5 + k  # j+2 for B_bottom
                V[v_idx][col_offset + r] = strassen_V[j*2+k][r]
        for k in range(2):
            for i in range(2):
                w_idx = k * 2 + i
                W[w_idx][col_offset + r] = strassen_W[k*2+i][r]
    col_offset += 7
    
    # Block 4: A_right @ B_bottom_right (cols 2 of B_bottom, which is row 2 of B_bottom)
    # B_bottom_right is 2x1: j in {2,3}, k=2 -> V[j*5+2, ...]
    # C output: k=2, i in {0,1} -> W[2*2+i, ...]
    for r in range(4):
        for i in range(2):
            for j in range(2):
                u_idx = i * 4 + (j + 2)
                U[u_idx][col_offset + r] = school_U[i*2+j][r]
        for j in range(2):
            v_idx = (j + 2) * 5 + 2  # j+2, k=2
            V[v_idx][col_offset + r] = school_V[j][r]
        for i in range(2):
            w_idx = 2 * 2 + i
            W[w_idx][col_offset + r] = school_W[i][r]
    col_offset += 4
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
