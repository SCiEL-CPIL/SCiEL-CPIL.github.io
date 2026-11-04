import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0

    # Schoolbook decomposition
    rank_schoolbook = n * m * p

    def schoolbook():
        rank = rank_schoolbook
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

    best_result = schoolbook()
    best_rank = rank_schoolbook

    def check_decomposition(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        rank = len(U[0])
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    # F(2) algorithm for 2x2x2x2 matrix multiplication
    # A indices: 0=a00, 1=a01, 2=a10, 3=a11
    # B indices: 0=b00, 1=b01, 2=b10, 3=b11
    # C indices: 0=c00, 1=c01, 2=c10, 3=c11
    
    F2_U = [
        [1, 0, 1, 0, 1, -1, 0],   # a00
        [0, 0, 0, 0, 1, 0, 1],    # a01
        [0, 1, 0, 0, 0, 1, 0],    # a10
        [1, 1, 0, 1, 0, 0, -1],   # a11
    ]
    
    F2_V = [
        [1, 1, -1, 0, 0, 1, 0],   # b00
        [0, 0, 0, 1, 0, 0, 1],    # b01
        [0, 0, 1, 0, 0, 1, 0],    # b10
        [1, 0, 0, -1, 1, 0, 1],   # b11
    ]
    
    F2_W = [
        [1, 0, 1, -1, 0, 0, -1],  # c00
        [0, 0, 0, 1, 0, 0, 1],    # c01
        [0, 1, -1, 0, 0, 0, 0],   # c10
        [1, -1, 0, 0, 1, 1, 0],   # c11
    ]
    
    # Verify F(2) decomposition
    def verify_F2():
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    total = 0
                    for r in range(7):
                        total += F2_U[i][r] * F2_V[j][r] * F2_W[k][r]
                    # For 2x2x2x2, T[i,j,k] = 1 if i//2 == k//2 and i%2 == k%2 and i%2 == j%2... 
                    # Actually, for matrix multiplication: C[i,j] = sum_k A[i,k] * B[k,j]
                    # So T[A_idx, B_idx, C_idx] = 1 if A_idx row == C_idx row and B_idx col == C_idx col
                    # A_idx = i*2 + k, B_idx = k*2 + j, C_idx = i*2 + j
                    # So T[A_idx, B_idx, C_idx] = 1 if A_idx//2 == C_idx//2 and B_idx%2 == C_idx%2 and A_idx%2 == B_idx//2
                    a_row, a_col = i // 2, i % 2
                    b_row, b_col = j // 2, j % 2
                    c_row, c_col = k // 2, k % 2
                    expected = 1 if (a_row == c_row and a_col == b_row and b_col == c_col) else 0
                    if abs(total - expected) > 1e-9:
                        return False
        return True

    f2_valid = verify_F2()
    
    # Build 4x4x4x4 using tensor product of F(2) with itself
    # A is 4x4: A[i,j] where i,j in {0,1,2,3}
    # View A as 2x2 blocks of 2x2 matrices
    # A[i,j] where i = i1*2 + i0, j = j1*2 + j0
    # A_idx = i*4 + j = (i1*2+i0)*4 + (j1*2+j0) = i1*8 + i0*4 + j1*2 + j0
    
    # For the tensor product:
    # r = r1*7 + r0 where r1, r0 in {0,...,6}
    # A_idx = i1*8 + i0*4 + j1*2 + j0
    # B_idx = j1*8 + j0*4 + k1*2 + k0
    # C_idx = k1*8 + k0*4 + i1*2 + i0
    
    # U[A_idx, r] = F2_U[i1*2 + j1, r1] * F2_U[i0*2 + j0, r0]
    # V[B_idx, r] = F2_V[j1*2 + k1, r1] * F2_V[j0*2 + k0, r0]
    # W[C_idx, r] = F2_W[k1*2 + i1, r1] * F2_W[k0*2 + i0, r0]
    
    U_44 = [[0] * 49 for _ in range(16)]
    V_44 = [[0] * 49 for _ in range(16)]
    W_44 = [[0] * 49 for _ in range(16)]
    
    for i1 in range(2):
        for i0 in range(2):
            for j1 in range(2):
                for j0 in range(2):
                    A_idx = i1*8 + i0*4 + j1*2 + j0
                    for r1 in range(7):
                        for r0 in range(7):
                            r = r1*7 + r0
                            U_44[A_idx][r] = F2_U[i1*2 + j1][r1] * F2_U[i0*2 + j0][r0]
    
    for j1 in range(2):
        for j0 in range(2):
            for k1 in range(2):
                for k0 in range(2):
                    B_idx = j1*8 + j0*4 + k1*2 + k0
                    for r1 in range(7):
                        for r0 in range(7):
                            r = r1*7 + r0
                            V_44[B_idx][r] = F2_V[j1*2 + k1][r1] * F2_V[j0*2 + k0][r0]
    
    for k1 in range(2):
        for k0 in range(2):
            for i1 in range(2):
                for i0 in range(2):
                    C_idx = k1*8 + k0*4 + i1*2 + i0
                    for r1 in range(7):
                        for r0 in range(7):
                            r = r1*7 + r0
                            W_44[C_idx][r] = F2_W[k1*2 + i1][r1] * F2_W[k0*2 + i0][r0]
    
    # Verify the 4x4x4x4 decomposition
    def verify_4444(U, V, W):
        for i in range(16):
            for j in range(16):
                for k in range(16):
                    total = 0
                    for r in range(49):
                        total += U[i][r] * V[j][r] * W[k][r]
                    # A_idx = i1*8 + i0*4 + j1*2 + j0
                    # B_idx = j1*8 + j0*4 + k1*2 + k0
                    # C_idx = k1*8 + k0*4 + i1*2 + i0
                    i1, rem = divmod(i, 8)
                    i0, rem2 = divmod(rem, 4)
                    j1, j0 = divmod(rem2, 2)
                    
                    j1b, rem = divmod(j, 8)
                    j0b, rem2 = divmod(rem, 4)
                    k1, k0 = divmod(rem2, 2)
                    
                    k1c, rem = divmod(k, 8)
                    k0c, rem2 = divmod(rem, 4)
                    i1c, i0c = divmod(rem2, 2)
                    
                    # Check if this is a valid triple
                    # Need: i1 == i1c, i0 == i0c, j1 == j1b, j0 == j0b, k1 == k1c, k0 == k0c
                    # And the matrix multiplication structure: A[i,j] * B[j,k] -> C[i,k]
                    # So A_idx row part (i1,i0) should match C_idx row part (i1c,i0c)
                    # A_idx col part (j1,j0) should match B_idx row part (j1b,j0b)
                    # B_idx col part (k1,k0) should match C_idx col part (k1c,k0c)
                    expected = 1 if (i1 == i1c and i0 == i0c and j1 == j1b and j0 == j0b and k1 == k1c and k0 == k0c) else 0
                    if abs(total - expected) > 1e-9:
                        return False
        return True

    four44_valid = verify_4444(U_44, V_44, W_44)
    
    # Build full 4x4x4x5 decomposition
    # U: 16 rows (4*4), V: 20 rows (4*5), W: 20 rows (5*4)
    # First 49 terms: 4x4x4x4 part
    # Next 16 terms: 5th column (schoolbook)
    
    total_rank = 49 + 16
    U_full = [[0] * total_rank for _ in range(16)]
    V_full = [[0] * total_rank for _ in range(20)]
    W_full = [[0] * total_rank for _ in range(20)]
    
    # Copy 4x4x4x4 part
    for i in range(16):
        for r in range(49):
            U_full[i][r] = U_44[i][r]
    
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V_full[j*5+k][r] = V_44[j*4+k][r]
    
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W_full[k*4+i][r] = W_44[k*4+i][r]
    
    # 5th column: C[i,4] = sum_j A[i,j] * B[j,4]
    # Schoolbook: 16 terms
    for i in range(4):
        for j in range(4):
            r = 49 + i*4 + j
            U_full[i*4+j][r] = 1
            V_full[j*5+4][r] = 1
            W_full[4*4+i][r] = 1
    
    # Verify full decomposition
    full_valid = check_decomposition(U_full, V_full, W_full)
    
    if full_valid:
        best_result = {"U": U_full, "V": V_full, "W": W_full}
        best_rank = total_rank
    elif four44_valid and f2_valid:
        # 4x4x4x4 part works, but full doesn't? Shouldn't happen
        best_result = schoolbook()
        best_rank = rank_schoolbook
    else:
        # Fall back to schoolbook
        best_result = schoolbook()
        best_rank = rank_schoolbook
    
    return best_result
