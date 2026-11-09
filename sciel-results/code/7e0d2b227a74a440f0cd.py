import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130.0  # Leave 20s margin for 150s limit

    # Build rank-112 decomposition using hierarchical Strassen 2x2x2
    # 4x4*4x8 = 4 copies of 4x4*4x2, each using 4 copies of 2x2*2x2 (rank 7)
    
    # Strassen 2x2x2 decomposition (rank 7)
    # U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
    U_s = np.zeros((4, 7), dtype=int)
    V_s = np.zeros((4, 7), dtype=int)
    W_s = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U_s[0,0]=1; U_s[3,0]=1
    V_s[0,0]=1; V_s[3,0]=1
    W_s[0,0]=1; W_s[3,0]=1
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 += M2
    U_s[2,1]=1; U_s[3,1]=1
    V_s[0,1]=1
    W_s[2,1]=1; W_s[3,1]=1
    
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    U_s[0,2]=1
    V_s[1,2]=1; V_s[3,2]=-1
    W_s[1,2]=1; W_s[3,2]=1
    
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    U_s[3,3]=1
    V_s[2,3]=1; V_s[0,3]=-1
    W_s[0,3]=1; W_s[2,3]=1
    
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    U_s[0,4]=1; U_s[1,4]=1
    V_s[3,4]=1
    W_s[0,4]=-1; W_s[1,4]=1
    
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    U_s[2,5]=1; U_s[0,5]=-1
    V_s[0,5]=1; V_s[1,5]=1
    W_s[3,5]=1
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    U_s[1,6]=1; U_s[3,6]=-1
    V_s[2,6]=1; V_s[3,6]=1
    W_s[0,6]=1
    
    # Verify 2x2x2
    valid_222 = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = sum(U_s[i*2+j,t] * V_s[j*2+k,t] * W_s[k*2+i,t] for t in range(7))
                if val != 1:
                    valid_222 = False
                    break
            if not valid_222: break
        if not valid_222: break
    
    if not valid_222:
        # Fallback to schoolbook
        rank = 128
        U = np.zeros((16, 128), dtype=int)
        V = np.zeros((32, 128), dtype=int)
        W = np.zeros((32, 128), dtype=int)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    U[i*4+j, col] = 1
                    V[j*8+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build 4x4*4x8 (rank 112)
    rank = 112
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((32, rank), dtype=int)
    W = np.zeros((32, rank), dtype=int)
    
    # For each of 4 column-pairs of B (each 4x2), use 4 Strassen 2x2x2 blocks
    # B column-pair b: columns 2b, 2b+1
    # A partitioned into 2x2 blocks: A[i_off, j_off] for i_off, j_off in {0,1}
    # For column-pair b, B is partitioned into 2x2 blocks: B[i_off, 0] and B[i_off, 1]
    #   where B[i_off, 0] is rows 2*i_off..2*i_off+1, cols 2b..2b+1
    #   and B[i_off, 1] is rows 2*i_off..2*i_off+1, cols 2b+2..2b+3 -- NO, that's wrong
    # 
    # Actually for 4x4 * 4x2:
    # A = [[A00, A01], [A10, A11]] (each 2x2)
    # B = [[B0], [B1]] (each 2x2)
    # C = [[A00*B0 + A01*B1], [A10*B0 + A11*B1]] (each 2x2)
    # 
    # For 4x4 * 4x8, split B into 4 blocks of 4x2:
    # B_b = B[:, 2b:2b+2] for b in 0..3
    # C_b = A * B_b for b in 0..3
    # Each C_b is 4x2.
    # For each b, use 4 Strassen 2x2x2:
    #   Strassen 1: A00 * B_b[0:2, 0:2] -> C_b[0:2, 0:2]
    #   Strassen 2: A01 * B_b[2:4, 0:2] -> C_b[0:2, 0:2]
    #   Strassen 3: A10 * B_b[0:2, 0:2] -> C_b[2:4, 0:2]
    #   Strassen 4: A11 * B_b[2:4, 0:2] -> C_b[2:4, 0:2]
    # 
    # Wait, that's not right. B_b is 4x2. Partition into 2x2 blocks:
    # B_b = [[B_b00], [B_b10]] where B_b00 is rows 0-1, cols 0-1; B_b10 is rows 2-3, cols 0-1
    # Then C_b = [[A00*B_b00 + A01*B_b10], [A10*B_b00 + A11*B_b10]]
    # Each block of C_b is 2x2.
    # 
    # So for each b:
    #   Strassen 1: A00 * B_b00 -> C_b[0:2, 0:2]
    #   Strassen 2: A01 * B_b10 -> C_b[0:2, 0:2]
    #   Strassen 3: A10 * B_b00 -> C_b[2:4, 0:2]
    #   Strassen 4: A11 * B_b10 -> C_b[2:4, 0:2]
    # 
    # Total: 4 * 4 * 7 = 112 rank.
    
    for b in range(4):
        for a_i_off in range(2):  # A row block offset (0 or 2)
            for a_j_off in range(2):  # A col block offset (0 or 2)
                for b_i_off in range(2):  # B row block offset (0 or 2)
                    # Determine which Strassen instance
                    # Strassen 1: a_i=0, a_j=0, b_i=0 -> C[0:2, :]
                    # Strassen 2: a_i=0, a_j=2, b_i=2 -> C[0:2, :]
                    # Strassen 3: a_i=2, a_j=0, b_i=0 -> C[2:4, :]
                    # Strassen 4: a_i=2, a_j=2, b_i=2 -> C[2:4, :]
                    
                    inst = (a_i_off // 2) * 2 + (a_j_off // 2)
                    # inst 0: A00*B00, inst 1: A01*B10, inst 2: A10*B00, inst 3: A11*B10
                    
                    # Check consistency: b_i_off should match
                    # For inst 0,2: b_i_off = 0
                    # For inst 1,3: b_i_off = 2
                    expected_b_i = (inst % 2) * 2
                    if b_i_off != expected_b_i:
                        continue
                    
                    base_t = b * 28 + inst * 7
                    
                    # U: A block at (a_i_off, a_j_off), 2x2
                    # U rows: (a_i_off)*4 + a_j_off, (a_i_off)*4 + a_j_off+1, (a_i_off+1)*4 + a_j_off, (a_i_off+1)*4 + a_j_off+1
                    for local_u in range(4):
                        li, lj = local_u // 2, local_u % 2
                        global_u = (a_i_off + li) * 4 + (a_j_off + lj)
                        U[global_u, base_t:base_t+7] = U_s[local_u]
                    
                    # V: B block at (b_i_off, 2b), 2x2
                    # V rows: (b_i_off)*8 + 2b, (b_i_off)*8 + 2b+1, (b_i_off+1)*8 + 2b, (b_i_off+1)*8 + 2b+1
                    for local_v in range(4):
                        li, lk = local_v // 2, local_v % 2
                        global_v = (b_i_off + li) * 8 + (2*b + lk)
                        V[global_v, base_t:base_t+7] = V_s[local_v]
                    
                    # W: C block at (a_i_off, 2b), 2x2
                    # C row block: a_i_off (0 or 2), col block: 2b
                    # W index: k*4 + i where k is col in C, i is row in C
                    # C[0:2, 2b:2b+2] -> k in {2b, 2b+1}, i in {0, 1}
                    # C[2:4, 2b:2b+2] -> k in {2b, 2b+1}, i in {2, 3}
                    c_i_off = a_i_off  # C row offset
                    for local_w in range(4):
                        lk, li = local_w // 2, local_w % 2
                        global_w = (2*b + lk) * 4 + (c_i_off + li)
                        W[global_w, base_t:base_t+7] = W_s[local_w]
    
    # Verify
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(8):
                val = np.dot(U[i*4+j], V[j*8+k] * W[k*4+i])
                if not np.isclose(val, 1, atol=1e-9):
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    rank = 128
    U = np.zeros((16, 128), dtype=int)
    V = np.zeros((32, 128), dtype=int)
    W = np.zeros((32, 128), dtype=int)
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(8):
                U[i*4+j, col] = 1
                V[j*8+k, col] = 1
                W[k*4+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
