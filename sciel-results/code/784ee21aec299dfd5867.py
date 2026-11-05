import time
from fractions import Fraction as F

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return (U, V, W) for 2x2 * 2x2 multiplication, rank 7.
    U is (4, 7), V is (4, 7), W is (4, 7).
    Indices: U row = i*2 + j, V row = j*2 + k, W row = k*2 + i.
    """
    n = 2
    U = [[0]*7 for _ in range(n*n)]
    V = [[0]*7 for _ in range(n*n)]
    W = [[0]*7 for _ in range(n*n)]
    
    # M1 = (A11+A22)(B11+B22)
    U[0][0] += 1
    U[3][0] += 1
    V[0][0] += 1
    V[3][0] += 1
    W[0][0] += 1
    W[3][0] += 1
    
    # M2 = (A21+A22)B11
    U[2][1] += 1
    U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1
    W[3][1] -= 1
    
    # M3 = A11(B21-B22)
    U[0][2] += 1
    V[2][2] += 1
    V[3][2] -= 1
    W[1][2] += 1
    W[3][2] += 1
    
    # M4 = A22(B12-B11)
    U[3][3] += 1
    V[1][3] += 1
    V[0][3] -= 1
    W[0][3] += 1
    W[2][3] += 1
    
    # M5 = (A11+A12)B22
    U[0][4] += 1
    U[1][4] += 1
    V[3][4] += 1
    W[0][4] -= 1
    W[1][4] += 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2][5] += 1
    U[0][5] -= 1
    V[0][5] += 1
    V[1][5] += 1
    W[3][5] += 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] += 1
    U[3][6] -= 1
    V[2][6] += 1
    V[3][6] += 1
    W[0][6] += 1
    
    return U, V, W

def _schoolbook_2x2_1():
    """
    Return (U, V, W) for 2x2 * 2x1 multiplication, rank 4.
    U is (4, 4), V is (2, 4), W is (2, 4).
    Indices: U row = i*2 + j, V row = j*1 + 0 (k=0), W row = 0*2 + i (k=0).
    """
    n = 2
    m = 2
    p = 1
    rank = n * m * p # 4
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for i in range(n):
        for j in range(m):
            col = i*m*p + j*p + 0
            U[i*m + j][col] = 1
            V[j*p + 0][col] = 1
            W[0*n + i][col] = 1
    return U, V, W

def _combine_2x2_3():
    """
    Return (U, V, W) for 2x2 * 2x3 multiplication, rank 11.
    U is (4, 11), V is (6, 11), W is (6, 11).
    Indices: U row = i*2 + j, V row = j*3 + k, W row = k*2 + i.
    """
    U_s, V_s, W_s = _strassen_2x2()
    U_b, V_b, W_b = _schoolbook_2x2_1()
    
    rank = 11
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(6)]
    
    # Copy Strassen part (cols 0-6)
    for i in range(4):
        for r in range(7):
            U[i][r] = U_s[i][r]
    
    # V_s indices: j*2 + k, k in {0,1}
    # V_total indices: j*3 + k, k in {0,1,2}
    for j in range(2):
        for k in range(2):
            for r in range(7):
                V[j*3 + k][r] = V_s[j*2 + k][r]
                
    # W_s indices: k*2 + i, k in {0,1}
    # W_total indices: k*2 + i, k in {0,1,2}
    for k in range(2):
        for i in range(2):
            for r in range(7):
                W[k*2 + i][r] = W_s[k*2 + i][r]
                
    # Copy Schoolbook part (cols 7-10)
    for i in range(4):
        for r in range(4):
            U[i][7 + r] = U_b[i][r]
            
    # V_b indices: j*1 + 0
    # V_total indices: j*3 + 2
    for j in range(2):
        for r in range(4):
            V[j*3 + 2][7 + r] = V_b[j][r]
            
    # W_b indices: 0*2 + i
    # W_total indices: 2*2 + i
    for i in range(2):
        for r in range(4):
            W[2*2 + i][7 + r] = W_b[i][r]
            
    return U, V, W

def _kronecker_4x4_3():
    """
    Return (U, V, W) for 4x4 * 4x3 multiplication, rank 44.
    U is (16, 44), V is (12, 44), W is (12, 44).
    Indices: U row = i*4 + j, V row = j*3 + k, W row = k*4 + i.
    """
    U1, V1, W1 = _combine_2x2_3() # Rank 11
    U2, V2, W2 = _schoolbook_2x2_1() # Rank 4
    
    rank1 = 11
    rank2 = 4
    total_rank = rank1 * rank2 # 44
    
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(12)]
    W = [[0]*total_rank for _ in range(12)]
    
    # U1 size 4x11, U2 size 4x4
    # U index: i1*2+i2, j1*2+j2
    # U_total index: (i1*2+i2)*4 + (j1*2+j2) = i1*8 + i2*4 + j1*2 + j2
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    u_idx = (i1*2 + i2)*4 + (j1*2 + j2)
                    u1_idx = i1*2 + j1
                    u2_idx = i2*2 + j2
                    for r1 in range(rank1):
                        for r2 in range(rank2):
                            val = U1[u1_idx][r1] * U2[u2_idx][r2]
                            if val != 0:
                                U[u_idx][r1*rank2 + r2] += val
                                
    # V1 size 6x11 (j1*3+k1), V2 size 2x4 (j2*1+k2)
    # V_total index: j*3+k = (j1*2+j2)*3 + k1*1+k2 = j1*6 + j2*3 + k1 + k2
    # k2 is always 0.
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(3):
                v_idx = (j1*2 + j2)*3 + k1
                v1_idx = j1*3 + k1
                v2_idx = j2
                for r1 in range(rank1):
                    for r2 in range(rank2):
                        val = V1[v1_idx][r1] * V2[v2_idx][r2]
                        if val != 0:
                            V[v_idx][r1*rank2 + r2] += val
                            
    # W1 size 6x11 (k1*2+i1), W2 size 2x4 (k2*2+i2)
    # W_total index: k*4+i = (k1*1+k2)*4 + (i1*2+i2) = k1*4 + k2*4 + i1*2 + i2
    # k2 is always 0.
    for k1 in range(3):
        for i1 in range(2):
            for i2 in range(2):
                w_idx = k1*4 + (i1*2 + i2)
                w1_idx = k1*2 + i1
                w2_idx = i2
                for r1 in range(rank1):
                    for r2 in range(rank2):
                        val = W1[w1_idx][r1] * W2[w2_idx][r2]
                        if val != 0:
                            W[w_idx][r1*rank2 + r2] += val
                            
    return U, V, W

def _strassen_4x4_4():
    """
    Return (U, V, W) for 4x4 * 4x4 Strassen, rank 49.
    """
    n = 4
    rank = 49
    U = [[0]*rank for _ in range(n*n)]
    V = [[0]*rank for _ in range(n*n)]
    W = [[0]*rank for _ in range(n*n)]
    
    U2, V2, W2 = _strassen_2x2()
    
    A_coeffs_list = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,1,1), (1,1,-1)],
    ]
    
    B_coeffs_list = [
        [(0,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (0,0,-1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (1,1,1)],
    ]
    
    C_coeffs_list = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (1,1,1)],
        [(0,0,1), (1,0,1)],
        [(0,0,-1), (0,1,1)],
        [(1,1,1)],
        [(0,0,1)],
    ]
    
    for s in range(7):
        for t in range(7):
            col = s*7 + t
            for (ar, ac, a_coeff) in A_coeffs_list[s]:
                for i2 in range(2):
                    for j2 in range(2):
                        u_idx = (ar*2 + i2)*4 + (ac*2 + j2)
                        U[u_idx][col] += a_coeff * U2[i2*2 + j2][t]
            
            for (br, bc, b_coeff) in B_coeffs_list[s]:
                for j2 in range(2):
                    for k2 in range(2):
                        v_idx = (br*2 + j2)*4 + (bc*2 + k2)
                        V[v_idx][col] += b_coeff * V2[j2*2 + k2][t]
            
            for (cr, cc, c_coeff) in C_coeffs_list[s]:
                for i2 in range(2):
                    for k2 in range(2):
                        w_idx = (cc*2 + k2)*4 + (cr*2 + i2)
                        W[w_idx][col] += c_coeff * W2[k2*2 + i2][t]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    U_s, V_s, W_s = _strassen_4x4_4()
    U_k, V_k, W_k = _kronecker_4x4_3()
    
    rank_s = 49
    rank_k = 44
    total_rank = rank_s + rank_k
    
    U_full = [[0]*total_rank for _ in range(n*m)]
    V_full = [[0]*total_rank for _ in range(m*p)]
    W_full = [[0]*total_rank for _ in range(p*n)]
    
    for i in range(n*m):
        for r in range(rank_s):
            U_full[i][r] = U_s[i][r]
    
    for j in range(m):
        for k1 in range(4):
            for r in range(rank_s):
                V_full[j*p + k1][r] = V_s[j*4 + k1][r]
    
    for k1 in range(4):
        for i in range(n):
            for r in range(rank_s):
                W_full[k1*n + i][r] = W_s[k1*4 + i][r]
                
    for i in range(n*m):
        for r in range(rank_k):
            U_full[i][rank_s + r] = U_k[i][r]
            
    for j in range(m):
        for k2 in range(3):
            for r in range(rank_k):
                V_full[j*p + (4 + k2)][rank_s + r] = V_k[j*3 + k2][r]
                
    for k2 in range(3):
        for i in range(n):
            for r in range(rank_k):
                W_full[(4 + k2)*n + i][rank_s + r] = W_k[k2*4 + i][r]
                
    return {"U": U_full, "V": V_full, "W": W_full}

# EVOLVE-BLOCK-END
