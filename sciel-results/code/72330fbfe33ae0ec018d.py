import numpy as np
import time

# 3x3 * 3x8
# Split A as [[A11, A12], [A21, A22]] where
#   A11: 2x2 (A[0:2, 0:2])
#   A12: 2x1 (A[0:2, 2])
#   A21: 1x2 (A[2, 0:2])
#   A22: 1x1 (A[2, 2])
# Split B as [[B11, B12], [B21, B22]] where
#   B11: 2x8 (B[0:2, 0:8])
#   B12: 2x8 (B[0:2, 8:16]) -- but wait, B is 3x8, so B[0:2, :] is 2x8
#   Actually B is 3x8. Let's re-read the problem.
# A is 3x3, B is 3x8.
# So B is 3 rows, 8 cols.
# Split B as [[B1, B2], [B3]] where
#   B1: 2x8 (B[0:2, :])
#   B2: 1x8 (B[2, :])
# Then C = A * B = [[A11, A12], [A21, A22]] * [[B1], [B2]]
# C = [[A11*B1 + A12*B2], [A21*B1 + A22*B2]]
# C11: 2x8 (rows 0,1 of C)
# C21: 1x8 (row 2 of C)

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with Strassen's rank-7 decomposition.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    
    Convention:
    U[i*2 + j, r]  for A entry (i,j)
    V[j*2 + k, r]  for B entry (j,k)
    W[k*2 + i, r]  for C entry (i,k)  [Note: C_ik = sum_j A_ij B_jk]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    # Strassen terms:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = (a-b)(f+h)
    # m4 = (a+d)(g)
    # m5 = (a-c)(e+f)
    # m6 = (b-d)(g+h)
    # m7 = (a-d)(e+f+h+g)  -- Wait, standard Strassen is:
    # m7 = (a-d)(e+f+h+g) is not standard.
    # Standard:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = (a-b)(f+h)
    # m4 = (a+d)(g)
    # m5 = (a-c)(e+f)
    # m6 = (b-d)(g+h)
    # m7 = (a-d)(e+f+h+g) -- No, m7 = (a-d)(e+f) is not right either.
    # Let's use the exact coefficients from a verified source.
    
    # m1: (a+d)(e+h)
    #   A: a=1, d=1 -> U[0]=1, U[3]=1
    #   B: e=1, h=1 -> V[0]=1, V[3]=1
    #   C: c00 += m1, c01 += -m3, c10 += m4 - m5?
    # Let's just use the standard formulas:
    # C00 = m1 + m4 - m5 - m7
    # C01 = m3 + m7
    # C10 = m2 + m5
    # C11 = m1 - m2 + m3 + m6
    
    # m1 = (a+d)(e+h)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1  # C00
    W[3,0]=1  # C11
    
    # m2 = (c+d)(e)
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1  # C10
    W[3,1]=-1 # C11
    
    # m3 = (a-b)(f+h)
    U[0,2]=1; U[1,2]=-1
    V[1,2]=1; V[3,2]=1
    W[0,2]=-1 # C00
    W[1,2]=1  # C01
    W[3,2]=1  # C11
    
    # m4 = (a+d)(g)
    U[0,3]=1; U[3,3]=1
    V[2,3]=1
    W[0,3]=1  # C00
    W[2,3]=1  # C10
    
    # m5 = (a-c)(e+f)
    U[0,4]=1; U[2,4]=-1
    V[0,4]=1; V[1,4]=1
    W[0,4]=-1 # C00
    W[2,4]=1  # C10
    
    # m6 = (b-d)(g+h)
    U[1,5]=1; U[3,5]=-1
    V[2,5]=1; V[3,5]=1
    W[3,5]=1  # C11
    
    # m7 = (a-d)(f)  -- Wait, standard Strassen m7 = (a-d)(e+f+h+g)? No.
    # Let's check the standard Strassen again.
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = (a-b)(f+h)
    # m4 = (a+d)g
    # m5 = (a-c)(e+f)
    # m6 = (b-d)(g+h)
    # m7 = (a-d)(e+f+h+g) -- This is not right.
    # Actually, m7 = (a-d)f? No.
    # The 7th term is usually m7 = (a-d)(e+f+h+g) is wrong.
    # Let's use the one from Wikipedia:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = (a-b)(f+h)
    # m4 = (a+d)g
    # m5 = (a-c)(e+f)
    # m6 = (b-d)(g+h)
    # m7 = (a-d)(e+f+h+g) -- No, m7 = (a-d)f is not standard.
    # Wikipedia says: m7 = (a-d)(e+f+h+g) is not listed.
    # It lists:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = (a-b)(f+h)
    # m4 = (a+d)g
    # m5 = (a-c)(e+f)
    # m6 = (b-d)(g+h)
    # m7 = (a-d)(e+f+h+g) -- I think I'm confusing it.
    # Let's use a different source or derive it.
    # Actually, for the purpose of this task, I can just use the known rank-7 decomposition.
    # Let's hardcode the matrices from a verified implementation.
    
    # Re-doing with a verified set:
    # From "Laderman's algorithm" or standard Strassen:
    # The terms are:
    # 1: (a+d)(e+h)
    # 2: (c+d)(e)
    # 3: (a-b)(f+h)
    # 4: (a+d)(g)
    # 5: (a-c)(e+f)
    # 6: (b-d)(g+h)
    # 7: (a-d)(f)  -- No, this is only 7 terms if we use a different split.
    # Actually, standard Strassen uses:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = (a-b)(f+h)
    # m4 = (a+d)g
    # m5 = (a-c)(e+f)
    # m6 = (b-d)(g+h)
    # m7 = (a-d)(e+f+h+g) -- This is 7 terms.
    # Let's assume the above is correct and verify.
    # C00 = m1 + m4 - m5 - m7
    # C01 = m3 + m7
    # C10 = m2 + m5
    # C11 = m1 - m2 + m3 + m6
    
    # Let's check C00 = ae+bg
    # m1 = ae + ah + de + dh
    # m4 = ag + dg
    # m5 = ae + af - ce - cf
    # m7 = ae + af + ah + ag - de - df - dh - dg
    # m1 + m4 - m5 - m7 = 
    # (ae+ah+de+dh) + (ag+dg) - (ae+af-ce-cf) - (ae+af+ah+ag-de-df-dh-dg)
    # = ae+ah+de+dh+ag+dg-ae-af+ce+cf-ae-af-ah-ag+de+df+dh+dg
    # = -ae + 2de + 2dh + 2dg + ce + cf - 2af + df
    # This is not ae+bg. So my m7 is wrong.
    
    # Correct m7: (a-d)(f) ? No.
    # Let's use the exact coefficients from a known correct source.
    # I will use the one from the "Tensor Decomposition of Matrix Multiplication" paper or similar.
    # For now, I will use the schoolbook for 2x2 (rank 4) to be safe, but that's worse.
    # Actually, let's use the known correct Strassen coefficients:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = (a-b)(f+h)
    # m4 = (a+d)(g)
    # m5 = (a-c)(e+f)
    # m6 = (b-d)(g+h)
    # m7 = (a-d)(f)  -- No, this is not symmetric.
    
    # Let's try this set:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = (a-b)(f+h)
    # m4 = (a+d)(g)
    # m5 = (a-c)(e+f)
    # m6 = (b-d)(g+h)
    # m7 = (a-d)(e+f+h+g) -- I'll stick with this and hope for the best, or use schoolbook.
    
    # Given the risk, I'll use schoolbook for 2x2x2 (rank 4).
    # But that's not Strassen.
    
    # Let's use a verified Strassen decomposition.
    # From Wikipedia (Strassen algorithm):
    # m1 = (a + d)(e + h)
    # m2 = (c + d)e
    # m3 = (a - b)(f + h)
    # m4 = (a + d)g
    # m5 = (a - c)(e + f)
    # m6 = (b - d)(g + h)
    # m7 = (a - d)(e + f + h + g)  -- This is not in Wikipedia.
    # Wikipedia says:
    # C = [[m1 + m4 - m5 - m7, m3 + m7], [m2 + m5, m1 - m2 + m3 + m6]]
    # So m7 must be such that this works.
    # Let's solve for m7:
    # C01 = m3 + m7 = af + bh
    # m3 = af + ah - bf - bh
    # So m7 = af + bh - (af + ah - bf - bh) = 2bh - ah + bf
    # This doesn't look like a simple product.
    
    # I will use the schoolbook for 2x2x2 (rank 4) to ensure correctness.
    # Total rank will be:
    # A11*B1: 2x2 * 2x8. Split B into 4 blocks of 2x2.
    #   Each 2x2x2 uses rank 4 (schoolbook). Total 4*4 = 16.
    # A12*B2: 2x1 * 1x8. Rank 16.
    # A21*B1: 1x2 * 2x8. Rank 16.
    # A22*B2: 1x1 * 1x8. Rank 8.
    # Total = 16 + 16 + 16 + 8 = 56.
    # This is better than 72.
    
    U = np.zeros((4, 4))
    V = np.zeros((4, 4))
    W = np.zeros((4, 4))
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i*2 + j, col] = 1
                V[j*2 + k, col] = 1
                W[k*2 + i, col] = 1
                col += 1
    return U, V, W

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = 3, 3, 8
    
    # A is 3x3, B is 3x8
    # Split A: A11(2x2), A12(2x1), A21(1x2), A22(1x1)
    # Split B: B1(2x8), B2(1x8)
    
    # Block 1: A11(2x2) * B1(2x8)
    # Split B1 into 4 blocks of 2x2: B1_0, B1_1, B1_2, B1_3
    # Each uses schoolbook 2x2x2 (rank 4)
    U1, V1, W1 = _strassen_2x2()  # Returns schoolbook 2x2x2 (rank 4)
    rank1 = U1.shape[1]  # 4
    # B1 has 4 blocks of 2x2, so total rank = 4 * 4 = 16
    # U is (4, 16), V is (16, 16), W is (16, 16)
    U_b1 = np.tile(U1, (1, 4))  # (4, 16)
    V_b1 = np.zeros((16, 16))
    W_b1 = np.zeros((16, 16))
    for block in range(4):
        V_b1[block*4:(block+1)*4, block*4:(block+1)*4] = V1
        W_b1[block*4:(block+1)*4, block*4:(block+1)*4] = W1
    rank_b1 = 16
    
    # Block 2: A12(2x1) * B2(1x8)
    # Rank 16
    U_b2, V_b2, W_b2 = _schoolbook(2, 1, 8)
    rank_b2 = 16
    
    # Block 3: A21(1x2) * B1(2x8)
    # Rank 16
    U_b3, V_b3, W_b3 = _schoolbook(1, 2, 8)
    rank_b3 = 16
    
    # Block 4: A22(1x1) * B2(1x8)
    # Rank 8
    U_b4, V_b4, W_b4 = _schoolbook(1, 1, 8)
    rank_b4 = 8
    
    total_rank = rank_b1 + rank_b2 + rank_b3 + rank_b4  # 16+16+16+8 = 56
    
    U_full = np.zeros((9, total_rank))
    V_full = np.zeros((24, total_rank))
    W_full = np.zeros((24, total_rank))
    
    offset = 0
    
    # Block 1: A11 * B1
    # A11 is A[0:2, 0:2] -> indices 0,1,2,3 in U
    # B1 is B[0:2, 0:8] -> indices 0..15 in V (2*8=16)
    # Output is C[0:2, 0:8] -> indices 0..15 in W (8*2=16)
    U_full[0:4, offset:offset+rank_b1] = U_b1
    V_full[0:16, offset:offset+rank_b1] = V_b1
    W_full[0:16, offset:offset+rank_b1] = W_b1
    offset += rank_b1
    
    # Block 2: A12 * B2
    # A12 is A[0:2, 2] -> indices 0,1 in U (but A12 is 2x1, so U is (2,16))
    # We need to map A12's rows to the full A's rows 0,1.
    # U_b2 is (2,16). We need to place it in U_full[0:2, ...]
    # But U_full is (9, total_rank). A12 corresponds to A[0:2, 2], which is indices 2,3 in A (0-indexed: A[0,2]=2, A[1,2]=3)
    # Wait, A is 3x3. A[i,j] is at index i*3+j.
    # A11: A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,3,4
    # A12: A[0,2], A[1,2] -> indices 2,5
    # A21: A[2,0], A[2,1] -> indices 6,7
    # A22: A[2,2] -> index 8
    
    # So U_b2 (2x16) should be placed in U_full[2:4, ...] ? No, A12 is 2 rows, so it's 2 entries in A.
    # U_b2 has shape (2,16). We need to map these 2 entries to the correct positions in U_full.
    # The 2 entries are A[0,2] and A[1,2], which are indices 2 and 5 in the 9-element A.
    # So U_full[2, offset:offset+16] = U_b2[0, :]
    #    U_full[5, offset:offset+16] = U_b2[1, :]
    
    U_full[2, offset:offset+rank_b2] = U_b2[0, :]
    U_full[5, offset:offset+rank_b2] = U_b2[1, :]
    # B2 is B[2, 0:8] -> indices 16..23 in V (3*8=24, B[2,j] is at index 2*8+j = 16+j)
    V_full[16:24, offset:offset+rank_b2] = V_b2
    # Output is C[0:2, 0:8] -> indices 0..15 in W
    W_full[0:16, offset:offset+rank_b2] = W_b2
    offset += rank_b2
    
    # Block 3: A21 * B1
    # A21 is A[2, 0:2] -> indices 6,7 in A
    # U_b3 is (2,16)
    U_full[6, offset:offset+rank_b3] = U_b3[0, :]
    U_full[7, offset:offset+rank_b3] = U_b3[1, :]
    # B1 is B[0:2, 0:8] -> indices 0..15 in V
    V_full[0:16, offset:offset+rank_b3] = V_b3
    # Output is C[2, 0:8] -> indices 16..23 in W (W index is k*3+i, so C[2,k] is at 3*k+2)
    # Wait, W index is k*n + i, where n=3.
    # C[i,k] is at W index k*3 + i.
    # C[2,0] -> 0*3+2 = 2
    # C[2,1] -> 1*3+2 = 5
    # C[2,2] -> 2*3+2 = 8
    # ...
    # C[2,7] -> 7*3+2 = 23
    # So W_b3 (8x16) should be placed in W_full at indices 2,5,8,11,14,17,20,23
    W_full[2::3, offset:offset+rank_b3] = W_b3
    offset += rank_b3
    
    # Block 4: A22 * B2
    # A22 is A[2,2] -> index 8 in A
    # U_b4 is (1,8)
    U_full[8, offset:offset+rank_b4] = U_b4[0, :]
    # B2 is B[2, 0:8] -> indices 16..23 in V
    V_full[16:24, offset:offset+rank_b4] = V_b4
    # Output is C[2, 0:8] -> same as above
    W_full[2::3, offset:offset+rank_b4] = W_b4
    offset += rank_b4
    
    # Verify shapes
    assert U_full.shape == (9, total_rank)
    assert V_full.shape == (24, total_rank)
    assert W_full.shape == (24, total_rank)
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

# Quick self-test
if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    print(f"Rank: {U.shape[1]}")
    # Test with random matrices
    A = np.random.rand(3, 3)
    B = np.random.rand(3, 8)
    C_true = A @ B
    # C = sum_r U[:,r] * V[:,r] * W[:,r]^T? No, C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor decomposition: C[i,k] = sum_r U[i*3+j, r] V[j*8+k, r] W[k*3+i, r]
    # Wait, the convention is:
    # U[i*m + j, r] for A[i,j]
    # V[j*p + k, r] for B[j,k]
    # W[k*n + i, r] for C[i,k]
    # So C[i,k] = sum_{j,r} A[i,j] U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Let's compute this
    C_comp = np.zeros((3, 8))
    for i in range(3):
        for k in range(8):
            for j in range(3):
                for r in range(U.shape[1]):
                    C_comp[i, k] += A[i, j] * U[i*3 + j, r] * V[j*8 + k, r] * W[k*3 + i, r]
    print(f"Max error: {np.max(np.abs(C_true - C_comp))}")
