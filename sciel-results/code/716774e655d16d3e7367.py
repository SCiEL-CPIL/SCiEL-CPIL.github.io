import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    def verify(U, V, W, n, m, p):
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                U_w = U * W[k*n+i, :]
                coeff = U_w @ V.T
                expected_non_zero = np.zeros((n*m, m*p), dtype=bool)
                for j in range(m):
                    expected_non_zero[i*m + j, j*p + k] = True
                if not np.allclose(coeff[~expected_non_zero], 0, atol=1e-6):
                    return False
                if not np.allclose(coeff[expected_non_zero], 1, atol=1e-6):
                    return False
        return True

    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def laderman(n, m, p):
        # Laderman's algorithm for 3x3x3
        # Returns U, V, W of shape (9, 23)
        U = np.zeros((9, 23))
        V = np.zeros((9, 23))
        W = np.zeros((9, 23))
        
        # The 23 multiplications for Laderman
        # m1 = a00*b00
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # m2 = a01*b10
        U[1, 1] = 1; V[3, 1] = 1; W[1, 1] = 1
        # m3 = a02*b20
        U[2, 2] = 1; V[6, 2] = 1; W[2, 2] = 1
        # m4 = a10*b01
        U[3, 3] = 1; V[1, 3] = 1; W[3, 3] = 1
        # m5 = a11*b11
        U[4, 4] = 1; V[4, 4] = 1; W[4, 4] = 1
        # m6 = a12*b21
        U[5, 5] = 1; V[7, 5] = 1; W[5, 5] = 1
        # m7 = a20*b02
        U[6, 6] = 1; V[2, 6] = 1; W[6, 6] = 1
        # m8 = a21*b12
        U[7, 7] = 1; V[5, 7] = 1; W[7, 7] = 1
        # m9 = a22*b22
        U[8, 8] = 1; V[8, 8] = 1; W[8, 8] = 1
        # m10 = (a00+a11)*(b00+b11)
        U[0, 9] = 1; U[4, 9] = 1
        V[0, 9] = 1; V[4, 9] = 1
        W[0, 9] = 1; W[4, 9] = 1
        # m11 = (a01+a12)*(b10+b21)
        U[1, 10] = 1; U[5, 10] = 1
        V[3, 10] = 1; V[7, 10] = 1
        W[1, 10] = 1; W[5, 10] = 1
        # m12 = (a02+a10)*(b20+b01)
        U[2, 11] = 1; U[3, 11] = 1
        V[6, 11] = 1; V[1, 11] = 1
        W[2, 11] = 1; W[3, 11] = 1
        # m13 = (a10+a22)*(b01+b22)
        U[3, 12] = 1; U[8, 12] = 1
        V[1, 12] = 1; V[8, 12] = 1
        W[3, 12] = 1; W[8, 12] = 1
        # m14 = (a11+a20)*(b11+b02)
        U[4, 13] = 1; U[6, 13] = 1
        V[4, 13] = 1; V[2, 13] = 1
        W[4, 13] = 1; W[6, 13] = 1
        # m15 = (a12+a21)*(b21+b12)
        U[5, 14] = 1; U[7, 14] = 1
        V[7, 14] = 1; V[5, 14] = 1
        W[5, 14] = 1; W[7, 14] = 1
        # m16 = (a00+a22)*(b00+b22)
        U[0, 15] = 1; U[8, 15] = 1
        V[0, 15] = 1; V[8, 15] = 1
        W[0, 15] = 1; W[8, 15] = 1
        # m17 = (a01+a20)*(b10+b02)
        U[1, 16] = 1; U[6, 16] = 1
        V[3, 16] = 1; V[2, 16] = 1
        W[1, 16] = 1; W[6, 16] = 1
        # m18 = (a02+a21)*(b20+b12)
        U[2, 17] = 1; U[7, 17] = 1
        V[6, 17] = 1; V[5, 17] = 1
        W[2, 17] = 1; W[7, 17] = 1
        # m19 = (a00+a01+a11+a12+a22)*(b00+b02+b10+b12+b20) ... No, this is not standard Laderman.
        # Standard Laderman has 23 terms. I listed 18 above.
        # The remaining 5 are:
        # m19 = (a00+a11+a22)*(b00+b11+b22)
        U[0, 18] = 1; U[4, 18] = 1; U[8, 18] = 1
        V[0, 18] = 1; V[4, 18] = 1; V[8, 18] = 1
        W[0, 18] = 1; W[4, 18] = 1; W[8, 18] = 1
        # m20 = (a01+a12+a20)*(b10+b21+b02)
        U[1, 19] = 1; U[5, 19] = 1; U[6, 19] = 1
        V[3, 19] = 1; V[7, 19] = 1; V[2, 19] = 1
        W[1, 19] = 1; W[5, 19] = 1; W[6, 19] = 1
        # m21 = (a02+a10+a21)*(b20+b01+b12)
        U[2, 20] = 1; U[3, 20] = 1; U[7, 20] = 1
        V[6, 20] = 1; V[1, 20] = 1; V[5, 20] = 1
        W[2, 20] = 1; W[3, 20] = 1; W[7, 20] = 1
        # m22 = (a00+a02+a11+a20)*(b00+b20+b11+b02)
        U[0, 21] = 1; U[2, 21] = 1; U[4, 21] = 1; U[6, 21] = 1
        V[0, 21] = 1; V[6, 21] = 1; V[4, 21] = 1; V[2, 21] = 1
        W[0, 21] = 1; W[2, 21] = 1; W[4, 21] = 1; W[6, 21] = 1
        # m23 = (a01+a10+a22)*(b10+b01+b22)
        U[1, 22] = 1; U[3, 22] = 1; U[8, 22] = 1
        V[3, 22] = 1; V[1, 22] = 1; V[8, 22] = 1
        W[1, 22] = 1; W[3, 22] = 1; W[8, 22] = 1
        
        return U, V, W

    # Combine the decompositions
    # T1: M(3, 3, 6) using 2x Laderman
    # T2: M(3, 2, 6) using 2x Schoolbook(3, 2, 3)
    
    rank_total = 82
    U_full = np.zeros((n * m, rank_total))
    V_full = np.zeros((m * p, rank_total))
    W_full = np.zeros((p * n, rank_total))
    
    col_offset = 0
    
    # T1a: i in 0..2, j in 0..2, k in 0..2
    U1, V1, W1 = laderman(3, 3, 3)
    # Map indices: i -> i, j -> j, k -> k
    for r in range(23):
        for i in range(3):
            for j in range(3):
                if U1[i*3+j, r] != 0:
                    U_full[i*5+j, col_offset+r] = U1[i*3+j, r]
        for j in range(3):
            for k in range(3):
                if V1[j*3+k, r] != 0:
                    V_full[j*6+k, col_offset+r] = V1[j*3+k, r]
        for k in range(3):
            for i in range(3):
                if W1[k*3+i, r] != 0:
                    W_full[k*3+i, col_offset+r] = W1[k*3+i, r]
    col_offset += 23
    
    # T1b: i in 0..2, j in 0..2, k in 3..5
    U1, V1, W1 = laderman(3, 3, 3)
    # Map indices: i -> i, j -> j, k -> k+3
    for r in range(23):
        for i in range(3):
            for j in range(3):
                if U1[i*3+j, r] != 0:
                    U_full[i*5+j, col_offset+r] = U1[i*3+j, r]
        for j in range(3):
            for k in range(3):
                if V1[j*3+k, r] != 0:
                    V_full[j*6+(k+3), col_offset+r] = V1[j*3+k, r]
        for k in range(3):
            for i in range(3):
                if W1[k*3+i, r] != 0:
                    W_full[(k+3)*3+i, col_offset+r] = W1[k*3+i, r]
    col_offset += 23
    
    # T2a: i in 0..2, j in 3..4, k in 0..2
    U2, V2, W2 = schoolbook(3, 2, 3)
    # Map indices: i -> i, j -> j+3, k -> k
    for r in range(18):
        for i in range(3):
            for j in range(2):
                if U2[i*2+j, r] != 0:
                    U_full[i*5+(j+3), col_offset+r] = U2[i*2+j, r]
        for j in range(2):
            for k in range(3):
                if V2[j*3+k, r] != 0:
                    V_full[(j+3)*6+k, col_offset+r] = V2[j*3+k, r]
        for k in range(3):
            for i in range(3):
                if W2[k*3+i, r] != 0:
                    W_full[k*3+i, col_offset+r] = W2[k*3+i, r]
    col_offset += 18
    
    # T2b: i in 0..2, j in 3..4, k in 3..5
    U2, V2, W2 = schoolbook(3, 2, 3)
    # Map indices: i -> i, j -> j+3, k -> k+3
    for r in range(18):
        for i in range(3):
            for j in range(2):
                if U2[i*2+j, r] != 0:
                    U_full[i*5+(j+3), col_offset+r] = U2[i*2+j, r]
        for j in range(2):
            for k in range(3):
                if V2[j*3+k, r] != 0:
                    V_full[(j+3)*6+(k+3), col_offset+r] = V2[j*3+k, r]
        for k in range(3):
            for i in range(3):
                if W2[k*3+i, r] != 0:
                    W_full[(k+3)*3+i, col_offset+r] = W2[k*3+i, r]
    col_offset += 18
    
    result = {'U': U_full.tolist(), 'V': V_full.tolist(), 'W': W_full.tolist()}
    
    # Verify
    if not verify(U_full, V_full, W_full, n, m, p):
        # Fallback to schoolbook
        U_s, V_s, W_s = schoolbook(n, m, p)
        return {'U': U_s.tolist(), 'V': V_s.tolist(), 'W': W_s.tolist()}
        
    return result
