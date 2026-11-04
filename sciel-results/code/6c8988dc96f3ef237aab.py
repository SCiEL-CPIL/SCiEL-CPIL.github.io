import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]
DEADLINE = 135.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p  # 84
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Strategy: Structured Kronecker product decomposition
    # Split middle dimension m=4 as 2x2
    # Use rank-7 decomposition for M_{2,2,2} (Strassen-like)
    # Combine with schoolbook for remaining parts
    
    # The rank-7 decomposition for 2x2 matrix multiplication (Strassen)
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    # r = 0..6
    strassen_U = np.zeros((4, 7))
    strassen_V = np.zeros((4, 7))
    strassen_W = np.zeros((4, 7))
    
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    #
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U matrix: A coefficients for each product
    # Index: i*2+j where i is row of A, j is col of A
    # A[0,0]=a11, A[0,1]=a12, A[1,0]=a21, A[1,1]=a22
    
    # M1: (a11+a22) -> U[0,0]=1, U[3,0]=1
    strassen_U[0, 0] = 1.0  # a11
    strassen_U[3, 0] = 1.0  # a22
    
    # M2: (a21+a22)*b11 -> U[2,1]=1, U[3,1]=1
    strassen_U[2, 1] = 1.0  # a21
    strassen_U[3, 1] = 1.0  # a22
    
    # M3: a11*(b12-b22) -> U[0,2]=1
    strassen_U[0, 2] = 1.0  # a11
    
    # M4: a22*(b21-b11) -> U[3,3]=1
    strassen_U[3, 3] = 1.0  # a22
    
    # M5: (a11+a12)*b22 -> U[0,4]=1, U[1,4]=1
    strassen_U[0, 4] = 1.0  # a11
    strassen_U[1, 4] = 1.0  # a12
    
    # M6: (a21-a11)*(b11+b12) -> U[2,5]=1, U[0,5]=-1
    strassen_U[2, 5] = 1.0  # a21
    strassen_U[0, 5] = -1.0  # -a11
    
    # M7: (a12-a22)*(b21+b22) -> U[1,6]=1, U[3,6]=-1
    strassen_U[1, 6] = 1.0  # a12
    strassen_U[3, 6] = -1.0  # -a22
    
    # V matrix: B coefficients for each product
    # Index: j*2+k where j is row of B, k is col of B
    # B[0,0]=b11, B[0,1]=b12, B[1,0]=b21, B[1,1]=b22
    
    # M1: *(b11+b22) -> V[0,0]=1, V[3,0]=1
    strassen_V[0, 0] = 1.0  # b11
    strassen_V[3, 0] = 1.0  # b22
    
    # M2: *b11 -> V[0,1]=1
    strassen_V[0, 1] = 1.0  # b11
    
    # M3: *(b12-b22) -> V[1,2]=1, V[3,2]=-1
    strassen_V[1, 2] = 1.0  # b12
    strassen_V[3, 2] = -1.0  # -b22
    
    # M4: *(b21-b11) -> V[2,3]=1, V[0,3]=-1
    strassen_V[2, 3] = 1.0  # b21
    strassen_V[0, 3] = -1.0  # -b11
    
    # M5: *b22 -> V[3,4]=1
    strassen_V[3, 4] = 1.0  # b22
    
    # M6: *(b11+b12) -> V[0,5]=1, V[1,5]=1
    strassen_V[0, 5] = 1.0  # b11
    strassen_V[1, 5] = 1.0  # b12
    
    # M7: *(b21+b22) -> V[2,6]=1, V[3,6]=1
    strassen_V[2, 6] = 1.0  # b21
    strassen_V[3, 6] = 1.0  # b22
    
    # W matrix: C coefficients for each product
    # Index: k*2+i where k is col of C, i is row of C
    # C[0,0]=c11, C[0,1]=c12, C[1,0]=c21, C[1,1]=c22
    
    # C11 = M1 + M4 - M5 + M7 -> W[0,0]=1, W[0,3]=1, W[0,4]=-1, W[0,6]=1
    strassen_W[0, 0] = 1.0   # M1 -> c11
    strassen_W[0, 3] = 1.0   # M4 -> c11
    strassen_W[0, 4] = -1.0  # -M5 -> c11
    strassen_W[0, 6] = 1.0   # M7 -> c11
    
    # C12 = M3 + M5 -> W[1,2]=1, W[1,4]=1
    strassen_W[1, 2] = 1.0   # M3 -> c12
    strassen_W[1, 4] = 1.0   # M5 -> c12
    
    # C21 = M2 + M4 -> W[2,1]=1, W[2,3]=1
    strassen_W[2, 1] = 1.0   # M2 -> c21
    strassen_W[2, 3] = 1.0   # M4 -> c21
    
    # C22 = M1 - M2 + M3 + M6 -> W[3,0]=1, W[3,1]=-1, W[3,2]=1, W[3,5]=1
    strassen_W[3, 0] = 1.0   # M1 -> c22
    strassen_W[3, 1] = -1.0  # -M2 -> c22
    strassen_W[3, 2] = 1.0   # M3 -> c22
    strassen_W[3, 5] = 1.0   # M6 -> c22
    
    # Now construct the full decomposition using Kronecker structure
    # Split m=4 into two blocks of size 2: m1=2, m2=2
    # Split n=3: we can split as n1=1, n2=2 or use other splits
    # Split p=7: we can split as p1=3, p2=4 or other splits
    
    # Approach: For each pair of (n-block, p-block), use appropriate decomposition
    # If both blocks have even dimensions, we can use Strassen
    # Otherwise, use schoolbook for that sub-block
    
    # Let's try: n=3 = 1 + 2, p=7 = 3 + 4
    # Block (n=1, p=3): 1x4 * 4x3 -> schoolbook, rank = 1*4*3 = 12
    # Block (n=1, p=4): 1x4 * 4x4 -> can use Strassen-like? No, n=1 is odd
    #   Actually for 1x4 * 4x4, it's just dot products, rank = 4
    # Block (n=2, p=3): 2x4 * 4x3 -> schoolbook, rank = 2*4*3 = 24
    # Block (n=2, p=4): 2x4 * 4x4 -> can use Strassen! rank = 7
    
    # Total rank: 12 + 4 + 24 + 7 = 47
    
    # Let's construct this decomposition
    n1, n2 = 1, 2  # Split n=3 as 1+2
    p1, p2 = 3, 4  # Split p=7 as 3+4
    
    total_rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    # Block 1: n=1, p=3 (rows 0 of A, cols 0-2 of B)
    # A_sub is 1x4, B_sub is 4x3, C_sub is 3x1
    # Schoolbook: rank = 1*4*3 = 12
    rank_b1 = 1 * 4 * 3
    U_b1 = np.zeros((4, rank_b1))
    V_b1 = np.zeros((4*3, rank_b1))
    W_b1 = np.zeros((3*1, rank_b1))
    col = 0
    for i in range(1):
        for j in range(4):
            for k in range(3):
                U_b1[j, col] = 1.0
                V_b1[j*3 + k, col] = 1.0
                W_b1[k*1 + i, col] = 1.0
                col += 1
    U_list.append(U_b1)
    V_list.append(V_b1)
    W_list.append(W_b1)
    total_rank += rank_b1
    
    # Block 2: n=1, p=4 (rows 0 of A, cols 3-6 of B)
    # A_sub is 1x4, B_sub is 4x4, C_sub is 4x1
    # This is just 4 dot products, rank = 4
    rank_b2 = 4
    U_b2 = np.zeros((4, rank_b2))
    V_b2 = np.zeros((4*4, rank_b2))
    W_b2 = np.zeros((4*1, rank_b2))
    for k in range(4):
        for j in range(4):
            U_b2[j, k] = 1.0
            V_b2[j*4 + k, k] = 1.0
            W_b2[k, k] = 1.0
    U_list.append(U_b2)
    V_list.append(V_b2)
    W_list.append(W_b2)
    total_rank += rank_b2
    
    # Block 3: n=2, p=3 (rows 1-2 of A, cols 0-2 of B)
    # A_sub is 2x4, B_sub is 4x3, C_sub is 3x2
    # Schoolbook: rank = 2*4*3 = 24
    rank_b3 = 2 * 4 * 3
    U_b3 = np.zeros((2*4, rank_b3))
    V_b3 = np.zeros((4*3, rank_b3))
    W_b3 = np.zeros((3*2, rank_b3))
    col = 0
    for i in range(2):
        for j in range(4):
            for k in range(3):
                U_b3[i*4 + j, col] = 1.0
                V_b3[j*3 + k, col] = 1.0
                W_b3[k*2 + i, col] = 1.0
                col += 1
    U_list.append(U_b3)
    V_list.append(V_b3)
    W_list.append(W_b3)
    total_rank += rank_b3
    
    # Block 4: n=2, p=4 (rows 1-2 of A, cols 3-6 of B)
    # A_sub is 2x4, B_sub is 4x4, C_sub is 4x2
    # Use Strassen: rank = 7
    # A_sub = [[a11, a12, a13, a14], [a21, a22, a23, a24]]
    # B_sub = [[b11, b12, b13, b14], [b21, b22, b23, b24], [b31, b32, b33, b34], [b41, b42, b43, b44]]
    # We need to split the 4x4 into 2x2 blocks of 2x2 matrices
    # A_sub = [[A11, A12], [A21, A22]] where each is 2x2
    # B_sub = [[B11, B12], [B21, B22]] where each is 2x2
    # C_sub = A_sub * B_sub = [[C11, C12], [C21, C22]] where each is 2x2
    #
    # For each 2x2 block product, we can use Strassen
    # But we have 4 block products: C11=A11*B11+A12*B21, etc.
    # This gets complex. Let me use a different approach.
    
    # Actually, for 2x4 * 4x4, we can view it as:
    # Split A into A_left (2x2) and A_right (2x2)
    # Split B into B_top (2x4) and B_bottom (2x4)
    # C = A_left * B_top + A_right * B_bottom
    # Each term is 2x2 * 2x4, which is two dot products per row
    # Rank for each term: 2*2*4 = 16, total = 32
    # That's worse than schoolbook (2*4*4=32). Same.
    
    # Let me try another split for Block 4:
    # Use the fact that we can apply Strassen to the 2x2 blocks
    # A_sub = [[a11, a12], [a21, a22]] (first 2 cols) + [[a13, a14], [a23, a24]] (last 2 cols)
    # B_sub = [[b11, b12, b13, b14], [b21, b22, b23, b24], [b31, b32, b33, b34], [b41, b42, b43, b44]]
    #
    # Let me just use schoolbook for Block 4 for now
    rank_b4 = 2 * 4 * 4
    U_b4 = np.zeros((2*4, rank_b4))
    V_b4 = np.zeros((4*4, rank_b4))
    W_b4 = np.zeros((4*2, rank_b4))
    col = 0
    for i in range(2):
        for j in range(4):
            for k in range(4):
                U_b4[i*4 + j, col] = 1.0
                V_b4[j*4 + k, col] = 1.0
                W_b4[k*2 + i, col] = 1.0
                col += 1
    U_list.append(U_b4)
    V_list.append(V_b4)
    W_list.append(W_b4)
    total_rank += rank_b4
    
    # Combine all blocks
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    col_offset = 0
    # Block 1: n=1 (row 0), p=3 (cols 0-2)
    U_full[0:4, col_offset:col_offset+rank_b1] = U_b1
    V_full[0:12, col_offset:col_offset+rank_b1] = V_b1
    W_full[0:3, col_offset:col_offset+rank_b1] = W_b1
    col_offset += rank_b1
    
    # Block 2: n=1 (row 0), p=4 (cols 3-6)
    U_full[0:4, col_offset:col_offset+rank_b2] = U_b2
    V_full[12:28, col_offset:col_offset+rank_b2] = V_b2
    W_full[3:7, col_offset:col_offset+rank_b2] = W_b2
    col_offset += rank_b2
    
    # Block 3: n=2 (rows 1-2), p=3 (cols 0-2)
    U_full[4:12, col_offset:col_offset+rank_b3] = U_b3
    V_full[0:12, col_offset:col_offset+rank_b3] = V_b3
    W_full[0:6, col_offset:col_offset+rank_b3] = W_b3
    col_offset += rank_b3
    
    # Block 4: n=2 (rows 1-2), p=4 (cols 3-6)
    U_full[4:12, col_offset:col_offset+rank_b4] = U_b4
    V_full[12:28, col_offset:col_offset+rank_b4] = V_b4
    W_full[3:11, col_offset:col_offset+rank_b4] = W_b4
    col_offset += rank_b4
    
    # Verify the decomposition
    # For each triple (i, j, k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(total_rank):
                    s += U_full[i*m+j, r] * V_full[j*p+k, r] * W_full[k*n+i, r]
                if abs(s - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid and total_rank < best_rank:
        best_rank = total_rank
        best_U = U_full.copy()
        best_V = V_full.copy()
        best_W = W_full.copy()
    
    # Try alternative splits
    # n=3 = 2+1, p=7 = 4+3
    n1a, n2a = 2, 1
    p1a, p2a = 4, 3
    
    total_rank_a = 0
    U_list_a = []
    V_list_a = []
    W_list_a = []
    
    # Block A1: n=2, p=4
    rank_a1 = 2 * 4 * 4
    U_a1 = np.zeros((2*4, rank_a1))
    V_a1 = np.zeros((4*4, rank_a1))
    W_a1 = np.zeros((4*2, rank_a1))
    col = 0
    for i in range(2):
        for j in range(4):
            for k in range(4):
                U_a1[i*4 + j, col] = 1.0
                V_a1[j*4 + k, col] = 1.0
                W_a1[k*2 + i, col] = 1.0
                col += 1
    U_list_a.append(U_a1)
    V_list_a.append(V_a1)
    W_list_a.append(W_a1)
    total_rank_a += rank_a1
    
    # Block A2: n=2, p=3
    rank_a2 = 2 * 4 * 3
    U_a2 = np.zeros((2*4, rank_a2))
    V_a2 = np.zeros((4*3, rank_a2))
    W_a2 = np.zeros((3*2, rank_a2))
    col = 0
    for i in range(2):
        for j in range(4):
            for k in range(3):
                U_a2[i*4 + j, col] = 1.0
                V_a2[j*3 + k, col] = 1.0
                W_a2[k*2 + i, col] = 1.0
                col += 1
    U_list_a.append(U_a2)
    V_list_a.append(V_a2)
    W_list_a.append(W_a2)
    total_rank_a += rank_a2
    
    # Block A3: n=1, p=4
    rank_a3 = 4
    U_a3 = np.zeros((4, rank_a3))
    V_a3 = np.zeros((4*4, rank_a3))
    W_a3 = np.zeros((4*1, rank_a3))
    for k in range(4):
        for j in range(4):
            U_a3[j, k] = 1.0
            V_a3[j*4 + k, k] = 1.0
            W_a3[k, k] = 1.0
    U_list_a.append(U_a3)
    V_list_a.append(V_a3)
    W_list_a.append(W_a3)
    total_rank_a += rank_a3
    
    # Block A4: n=1, p=3
    rank_a4 = 1 * 4 * 3
    U_a4 = np.zeros((4, rank_a4))
    V_a4 = np.zeros((4*3, rank_a4))
    W_a4 = np.zeros((3*1, rank_a4))
    col = 0
    for i in range(1):
        for j in range(4):
            for k in range(3):
                U_a4[j, col] = 1.0
                V_a4[j*3 + k, col] = 1.0
                W_a4[k*1 + i, col] = 1.0
                col += 1
    U_list_a.append(U_a4)
    V_list_a.append(V_a4)
    W_list_a.append(W_a4)
    total_rank_a += rank_a4
    
    U_full_a = np.zeros((n*m, total_rank_a))
    V_full_a = np.zeros((m*p, total_rank_a))
    W_full_a = np.zeros((p*n, total_rank_a))
    
    col_offset = 0
    # Block A1: n=2 (rows 0-1), p=4 (cols 0-3)
    U_full_a[0:8, col_offset:col_offset+rank_a1] = U_a1
    V_full_a[0:16, col_offset:col_offset+rank_a1] = V_a1
    W_full_a[0:8, col_offset:col_offset+rank_a1] = W_a1
    col_offset += rank_a1
    
    # Block A2: n=2 (rows 0-1), p=3 (cols 4-6)
    U_full_a[0:8, col_offset:col_offset+rank_a2] = U_a2
    V_full_a[16:28, col_offset:col_offset+rank_a2] = V_a2
    W_full_a[8:14, col_offset:col_offset+rank_a2] = W_a2
    col_offset += rank_a2
    
    # Block A3: n=1 (row 2), p=4 (cols 0-3)
    U_full_a[8:12, col_offset:col_offset+rank_a3] = U_a3
    V_full_a[0:16, col_offset:col_offset+rank_a3] = V_a3
    W_full_a[0:4, col_offset:col_offset+rank_a3] = W_a3
    col_offset += rank_a3
    
    # Block A4: n=1 (row 2), p=3 (cols 4-6)
    U_full_a[8:12, col_offset:col_offset+rank_a4] = U_a4
    V_full_a[16:28, col_offset:col_offset+rank_a4] = V_a4
    W_full_a[4:7, col_offset:col_offset+rank_a4] = W_a4
    col_offset += rank_a4
    
    valid_a = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(total_rank_a):
                    s += U_full_a[i*m+j, r] * V_full_a[j*p+k, r] * W_full_a[k*n+i, r]
                if abs(s - 1.0) > 1e-10:
                    valid_a = False
                    break
            if not valid_a:
                break
        if not valid_a:
            break
    
    if valid_a and total_rank_a < best_rank:
        best_rank = total_rank_a
        best_U = U_full_a.copy()
        best_V = V_full_a.copy()
        best_W = W_full_a.copy()
    
    # Check deadline
    if time.time() - start_time > DEADLINE - 5.0:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try using Strassen for 2x2x2 sub-blocks within the 2x4*4x4 case
    # For Block 4 (or A1): 2x4 * 4x4
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    # C = A*B = [[C11, C12], [C21, C22]]
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # Each 2x2*2x2 product can use Strassen (rank 7)
    # But we have 8 such products, so rank = 8*7 = 56, worse than schoolbook 32
    #
    # Alternative: Use the full Strassen on the 2x2 block structure
    # This requires careful construction. Let me try a different approach.
    
    # For now, return the best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
