import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct_2x2_strassen():
    """Returns U, V, W for 2x2 matrix multiplication, rank 7."""
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i,j] corresponds to A[i,j], V[i,j] to B[i,j], W[i,j] to C[i,j]
    # C = A B
    # Strassen 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Map indices (i,j) to i*2+j
    # A11=0, A12=1, A21=2, A22=3
    # B11=0, B12=1, B21=2, B22=3
    # C11=0, C12=1, C21=2, C22=3
    
    # M1
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1 # C11, C22
    
    # M2
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[2, 1] += 1; W[3, 1] -= 1 # C21, C22
    
    # M3
    U[0, 2] += 1
    V[2, 2] += 1; V[3, 2] -= 1
    W[1, 2] += 1; W[3, 2] += 1 # C12, C22
    
    # M4
    U[3, 3] += 1
    V[1, 3] += 1; V[0, 3] -= 1
    W[0, 3] += 1; W[2, 3] += 1 # C11, C21
    
    # M5
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[0, 4] -= 1; W[1, 4] += 1 # C11, C12
    
    # M6
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[1, 5] += 1
    W[3, 5] += 1 # C22
    
    # M7
    U[1, 6] += 1; U[3, 6] -= 1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1 # C11
    
    return U, V, W

def construct_3x3_schoolbook():
    """Returns U, V, W for 3x3 matrix multiplication, rank 27."""
    rank = 27
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, col] = 1
                V[j*3+k, col] = 1
                W[k*3+i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Get base decompositions
    U2, V2, W2 = construct_2x2_strassen()
    U3, V3, W3 = construct_3x3_schoolbook()
    
    rank2 = 7
    rank3 = 27
    total_rank = rank2 * rank3
    
    # Construct 6x6 decomposition via tensor product
    # Index mapping: i = 3*i1 + i2, j = 3*j1 + j2, k = 3*k1 + k2
    # U6[i*6+j, r2*7 + r3] = U2[i2*2+j2, r2] * U3[i1*3+j1, r3]
    # Wait, the tensor product structure:
    # T_6[i,j,k] = sum_{r2,r3} U6[i,j,r2,r3] V6[j,k,r2,r3] W6[k,i,r2,r3]
    # U6[i,j, r2, r3] = U2[i2,j2, r2] * U3[i1,j1, r3]
    # where i = 3*i1 + i2, j = 3*j1 + j2.
    
    U6 = np.zeros((36, total_rank))
    V6 = np.zeros((36, total_rank))
    W6 = np.zeros((36, total_rank))
    
    for i in range(6):
        i1, i2 = divmod(i, 3) # i = 3*i1 + i2? No, 6 = 2*3. 
        # Let's use i = 2*i_big + i_small? 
        # My previous derivation: 6 = 3*2. 
        # i = 3*i1 + i2 is wrong because i2 goes to 2.
        # i = 2*i1 + i2 where i1 in {0,1,2}, i2 in {0,1}.
        # Let's stick to 3x2 factorization: 6 = 3 * 2.
        # i = 2*i1 + i2? No, 3*2 = 6.
        # Let's say i = 2*i1 + i2 is for 4.
        # For 6, let i = 3*i1 + i2 is not integer partition of 6 into 3 and 2 in that order?
        # 6 = 3 + 3? No, 3*2.
        # Let's use i = 2*i1 + i2 where i1 in {0,1,2} is wrong range.
        # i in 0..5.
        # i = 2 * (i // 2) + (i % 2)? No.
        # Let's use i = 3 * i1 + i2 where i1 in {0,1}, i2 in {0,1,2}? 
        # 3*1 + 2 = 5. Yes.
        # So i1 in {0,1} (2 values), i2 in {0,1,2} (3 values).
        # This corresponds to 2x2 and 3x3?
        # If i1 is from 2x2 part, i2 is from 3x3 part.
        # Then dimension is 2*3=6.
        
    # Let's re-evaluate indices.
    # Let i = 2*i_a + i_b where i_a in {0,1,2} is wrong.
    # i_a in {0,1} (2 choices), i_b in {0,1,2} (3 choices).
    # i = 2*i_a + i_b? Max 2*1+2=4. Not 5.
    # i = 3*i_a + i_b? Max 3*1+2=5. Yes.
    # So i_a is the "2x2" index (0,1), i_b is the "3x3" index (0,1,2).
    # Wait, 2x2 has indices 0,1. 3x3 has 0,1,2.
    # So i = 2*i_a + i_b is not correct mapping for product space.
    # The space is Z_2 x Z_3?
    # i = i_a * 3 + i_b? Max 1*3+2=5. Yes.
    # So i_a is the "first" factor (size 2), i_b is the "second" factor (size 3).
    # i = 3*i_a + i_b.
    
    for i in range(6):
        i_a, i_b = divmod(i, 3) # i_a in {0,1}, i_b in {0,1,2}
    for j in range(6):
        j_a, j_b = divmod(j, 3)
    for k in range(6):
        k_a, k_b = divmod(k, 3)
        
    # U6[i*6+j, r_a*7 + r_b] = U2[i_a*2+j_a, r_a] * U3[i_b*3+j_b, r_b]
    # V6[j*6+k, r_a*7 + r_b] = V2[j_a*2+k_a, r_a] * V3[j_b*3+k_b, r_b]
    # W6[k*6+i, r_a*7 + r_b] = W2[k_a*2+i_a, r_a] * W3[k_b*3+i_b, r_b]
    
    for i in range(6):
        i_a, i_b = divmod(i, 3)
        for j in range(6):
            j_a, j_b = divmod(j, 3)
            row_u = i * 6 + j
            for r_a in range(7):
                for r_b in range(27):
                    col = r_a * 27 + r_b
                    u2_val = U2[i_a * 2 + j_a, r_a]
                    u3_val = U3[i_b * 3 + j_b, r_b]
                    U6[row_u, col] = u2_val * u3_val
    for j in range(6):
        j_a, j_b = divmod(j, 3)
        for k in range(6):
            k_a, k_b = divmod(k, 3)
            row_v = j * 6 + k
            for r_a in range(7):
                for r_b in range(27):
                    col = r_a * 27 + r_b
                    v2_val = V2[j_a * 2 + k_a, r_a]
                    v3_val = V3[j_b * 3 + k_b, r_b]
                    V6[row_v, col] = v2_val * v3_val
    for k in range(6):
        k_a, k_b = divmod(k, 3)
        for i in range(6):
            i_a, i_b = divmod(i, 3)
            row_w = k * 6 + i
            for r_a in range(7):
                for r_b in range(27):
                    col = r_a * 27 + r_b
                    w2_val = W2[k_a * 2 + i_a, r_a]
                    w3_val = W3[k_b * 3 + i_b, r_b]
                    W6[row_w, col] = w2_val * w3_val
                    
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}

# EVOLVE-BLOCK-START
# (No changes needed if I put the logic inside construct, but I need to keep the interface)
# I will just define construct() as above.
# EVOLVE-BLOCK-END
