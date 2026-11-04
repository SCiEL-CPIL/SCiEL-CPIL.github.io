import numpy as np

def get_strassen_2x2():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # Term 3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # Term 4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # Term 5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Term 6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def construct_4x4x4_strassen():
    U2, V2, W2 = get_strassen_2x2()
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    I = (i1*2+i2)*4 + (j1*2+j2)
                    u_idx1 = i1*2+j1
                    u_idx2 = i2*2+j2
                    for t1 in range(7):
                        for t2 in range(7):
                            R = t1*7+t2
                            U4[I][R] = U2[u_idx1][t1] * U2[u_idx2][t2]
                    
                    for k1 in range(2):
                        for k2 in range(2):
                            J = (j1*2+j2)*4 + (k1*2+k2)
                            v_idx1 = j1*2+k1
                            v_idx2 = j2*2+k2
                            for t1 in range(7):
                                for t2 in range(7):
                                    R = t1*7+t2
                                    V4[J][R] = V2[v_idx1][t1] * V2[v_idx2][t2]
                    
                    for k1 in range(2):
                        for k2 in range(2):
                            K = (k1*2+i1)*2 + (k2*2+i2) # Wait, W index is k*n+i. n=2.
                            # In 2x2x2, W index is k*2+i.
                            # Here K is the row index in W4.
                            # W4 row index is k*4+i.
                            # k = k1*2+k2, i = i1*2+i2.
                            K = (k1*2+k2)*4 + (i1*2+i2)
                            w_idx1 = k1*2+i1
                            w_idx2 = k2*2+i2
                            for t1 in range(7):
                                for t2 in range(7):
                                    R = t1*7+t2
                                    W4[K][R] = W2[w_idx1][t1] * W2[w_idx2][t2]
    return U4, V4, W4

def construct_2x2x3():
    # Rank 11: Strassen (7) + Schoolbook (4)
    U22, V22, W22 = get_strassen_2x2()
    
    # Schoolbook for 2x2x1
    # U shape (4, 4), V shape (2, 4), W shape (2, 4)
    U221 = [[0]*4 for _ in range(4)]
    V221 = [[0]*4 for _ in range(2)]
    W221 = [[0]*4 for _ in range(2)]
    for i in range(2):
        for j in range(2):
            t = i*2+j
            U221[i*2+j][t] = 1
            V221[j][t] = 1
            W221[0*2+i][t] = 1 # k=0, i=i. W index k*2+i.
            
    # Combine
    U223 = [[0]*11 for _ in range(4)]
    V223 = [[0]*11 for _ in range(6)] # 2x3
    W223 = [[0]*11 for _ in range(6)] # 3x2
    
    # Strassen part (cols 0-6)
    # V223 rows 0-1 correspond to B cols 0-1 (the 2x2 part)
    # V223 rows 2-5 correspond to B col 2? No.
    # In 2x2x3, B is 2x3.
    # Strassen handles 2x2x2. So B cols 0,1.
    # Schoolbook handles 2x2x1. So B col 2.
    
    for r in range(4):
        for t in range(7):
            U223[r][t] = U22[r][t]
    for r in range(2):
        for t in range(7):
            V223[r][t] = V22[r][t]
    # V223 rows 2-5 are for the 3rd column of B?
    # No, V223 has 6 rows. Rows 0-1 for B col 0, 1?
    # V index is j*p+k. p=3.
    # Row j*3+k.
    # j=0, k=0 -> 0. j=0, k=1 -> 1. j=0, k=2 -> 2.
    # j=1, k=0 -> 3. j=1, k=1 -> 4. j=1, k=2 -> 5.
    
    # Strassen V22 has 4 rows. Index j*2+k.
    # j=0, k=0 -> 0. j=0, k=1 -> 1.
    # j=1, k=0 -> 2. j=1, k=1 -> 3.
    
    # So V223[0] = V22[0]
    # V223[1] = V22[1]
    # V223[3] = V22[2]
    # V223[4] = V22[3]
    # V223[2] and V223[5] are 0 for Strassen part.
    
    for j in range(2):
        for k in range(2):
            v22_row = j*2+k
            v223_row = j*3+k
            for t in range(7):
                V223[v223_row][t] = V22[v22_row][t]
                
    # W223 rows 0-5. Index k*2+i.
    # k=0, i=0 -> 0. k=0, i=1 -> 1.
    # k=1, i=0 -> 2. k=1, i=1 -> 3.
    # k=2, i=0 -> 4. k=2, i=1 -> 5.
    
    for k in range(2):
        for i in range(2):
            w22_row = k*2+i
            w223_row = k*2+i
            for t in range(7):
                W223[w223_row][t] = W22[w22_row][t]
                
    # Schoolbook part (cols 7-10)
    # U221 rows 0-3.
    # V221 rows 0-1. Corresponds to B col 2.
    # V223 row for B col 2 is j*3+2.
    # j=0 -> 2. j=1 -> 5.
    for r in range(4):
        for t in range(4):
            U223[r][7+t] = U221[r][t]
            
    for j in range(2):
        v221_row = j
        v223_row = j*3+2
        for t in range(4):
            V223[v223_row][7+t] = V221[v221_row][t]
            
    # W221 rows 0-1. Corresponds to k=0.
    # W223 row for k=0 is 0*2+i = i.
    for i in range(2):
        w221_row = i # k=0, i=i
        w223_row = i
        for t in range(4):
            W223[w223_row][7+t] = W221[w221_row][t]
            
    return U223, V223, W223

def construct_4x4x3():
    U223, V223, W223 = construct_2x2x3()
    
    U443 = [[0]*44 for _ in range(16)]
    V443 = [[0]*44 for _ in range(12)] # 4x3
    W443 = [[0]*44 for _ in range(12)] # 3x4
    
    # 4 subproblems
    # 1. A11 B11' -> C11'
    # A11: i in 0,1; j in 0,1.
    # B11': j in 0,1; k in 0,1,2 (local).
    # C11': i in 0,1; k in 0,1,2 (local).
    
    # 2. A12 B21' -> C11'
    # A12: i in 0,1; j in 2,3.
    # B21': j in 2,3; k in 0,1,2.
    # C11': i in 0,1; k in 0,1,2.
    
    # 3. A21 B11' -> C21'
    # A21: i in 2,3; j in 0,1.
    # B11': j in 0,1; k in 0,1,2.
    # C21': i in 2,3; k in 0,1,2.
    
    # 4. A22 B21' -> C21'
    # A22: i in 2,3; j in 2,3.
    # B21': j in 2,3; k in 0,1,2.
    # C21': i in 2,3; k in 0,1,2.
    
    configs = [
        (0, 0, 0, 0, 0), # i_offset, j_offset_A, j_offset_B, i_offset_C, k_offset_C
        (0, 2, 2, 0, 0),
        (2, 0, 0, 2, 0),
        (2, 2, 2, 2, 0)
    ]
    
    for idx, (io, jo_a, jo_b, io_c, ko_c) in enumerate(configs):
        rank_offset = idx * 11
        for r in range(4):
            i_local = r // 2
            j_local = r % 2
            i_global = io + i_local
            j_global = jo_a + j_local
            u_row = i_global * 4 + j_global
            for t in range(11):
                U443[u_row][rank_offset + t] = U223[r][t]
                
        for r in range(6):
            j_local = r // 3
            k_local = r % 3
            j_global = jo_b + j_local
            k_global = k_local
            v_row = j_global * 3 + k_global
            for t in range(11):
                V443[v_row][rank_offset + t] = V223[r][t]
                
        for r in range(6):
            k_local = r // 2
            i_local = r % 2
            i_global = io_c + i_local
            k_global = ko_c + k_local
            w_row = k_global * 4 + i_global
            for t in range(11):
                W443[w_row][rank_offset + t] = W223[r][t]
                
    return U443, V443, W443

def construct():
    U44, V44, W44 = construct_4x4x4_strassen()
    U443, V443, W443 = construct_4x4x3()
    
    rank1 = 49
    rank2 = 44
    total_rank = rank1 + rank2
    
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(28)]
    W = [[0]*total_rank for _ in range(28)]
    
    # Fill U
    for i in range(16):
        for t in range(rank1):
            U[i][t] = U44[i][t]
        for t in range(rank2):
            U[i][rank1+t] = U443[i][t]
            
    # Fill V
    # V44 row j*4+k -> V row j*7+k (k=0..3)
    for j in range(4):
        for k in range(4):
            v44_row = j*4+k
            v_row = j*7+k
            for t in range(rank1):
                V[v_row][t] = V44[v44_row][t]
                
    # V443 row j*3+k' -> V row j*7+(4+k') (k'=0..2)
    for j in range(4):
        for k in range(3):
            v443_row = j*3+k
            v_row = j*7+(4+k)
            for t in range(rank2):
                V[v_row][rank1+t] = V443[v443_row][t]
                
    # Fill W
    # W44 row k*4+i -> W row k*4+i (k=0..3)
    for k in range(4):
        for i in range(4):
            w44_row = k*4+i
            w_row = k*4+i
            for t in range(rank1):
                W[w_row][t] = W44[w44_row][t]
                
    # W443 row k'*4+i -> W row (4+k')*4+i (k'=0..2)
    for k in range(3):
        for i in range(4):
            w443_row = k*4+i
            w_row = (4+k)*4+i
            for t in range(rank2):
                W[w_row][rank1+t] = W443[w443_row][t]
                
    return {"U": U, "V": V, "W": W}
