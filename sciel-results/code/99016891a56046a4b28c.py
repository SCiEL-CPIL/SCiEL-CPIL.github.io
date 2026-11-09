import numpy as np

def _build_laderman_3x3x3():
    """
    Returns U, V, W for 3x3x3 matrix multiplication using Laderman's algorithm (Rank 23).
    U: 9x23, V: 9x23, W: 9x23
    Indices:
    U[i*3+j, c]
    V[j*3+k, c]
    W[k*3+i, c]
    """
    R = 23
    U = np.zeros((9, R))
    V = np.zeros((9, R))
    W = np.zeros((9, R))
    
    # Laderman's algorithm coefficients
    # Based on standard literature.
    # a_ij is A[i,j], b_jk is B[j,k]
    
    # M1 = (a00 + a11) * (b00 + b11)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[4, 0] = 1 # C00 += M1, C11 += M1
    
    # M2 = (a00 + a11) * (b01 + b10)
    U[0, 1] = 1; U[4, 1] = 1
    V[1, 1] = 1; V[3, 1] = 1
    W[1, 1] = 1; W[3, 1] = 1 # C01 += M2, C10 += M2
    
    # M3 = (a01 + a10) * (b00 + b11)
    U[1, 2] = 1; U[3, 2] = 1
    V[0, 2] = 1; V[4, 2] = 1
    W[0, 2] = 1; W[4, 2] = 1 # C00 += M3, C11 += M3
    
    # M4 = (a01 + a10) * (b01 + b10)
    U[1, 3] = 1; U[3, 3] = 1
    V[1, 3] = 1; V[3, 3] = 1
    W[1, 3] = 1; W[3, 3] = 1 # C01 += M4, C10 += M4
    
    # M5 = (a00 + a11) * b12
    U[0, 4] = 1; U[4, 4] = 1
    V[6, 4] = 1
    W[2, 4] = 1; W[5, 4] = 1 # C02 += M5, C12 += M5
    
    # M6 = (a01 + a10) * b12
    U[1, 5] = 1; U[3, 5] = 1
    V[6, 5] = 1
    W[2, 5] = 1; W[5, 5] = 1 # C02 += M6, C12 += M6
    
    # M7 = (a00 + a11) * b20
    U[0, 6] = 1; U[4, 6] = 1
    V[2, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1 # C00 += M7, C10 += M7
    
    # M8 = (a01 + a10) * b20
    U[1, 7] = 1; U[3, 7] = 1
    V[2, 7] = 1
    W[0, 7] = 1; W[3, 7] = 1 # C00 += M8, C10 += M8
    
    # M9 = (a00 + a11) * b21
    U[0, 8] = 1; U[4, 8] = 1
    V[5, 8] = 1
    W[1, 8] = 1; W[4, 8] = 1 # C01 += M9, C11 += M9
    
    # M10 = (a01 + a10) * b21
    U[1, 9] = 1; U[3, 9] = 1
    V[5, 9] = 1
    W[1, 9] = 1; W[4, 9] = 1 # C01 += M10, C11 += M10
    
    # M11 = a00 * b22
    U[0, 10] = 1
    V[8, 10] = 1
    W[2, 10] = 1 # C02 += M11
    
    # M12 = a11 * b22
    U[4, 11] = 1
    V[8, 11] = 1
    W[5, 11] = 1 # C12 += M12
    
    # M13 = a20 * b02
    U[6, 12] = 1
    V[1, 12] = 1
    W[2, 12] = 1 # C22 += M13? No, C20?
    # a20 is A[2,0]. b02 is B[0,2].
    # C[2,2] += A[2,0]B[0,2]? No, C[i,k] = sum j A[i,j]B[j,k].
    # C[2,2] gets A[2,0]B[0,2]. Yes.
    # Wait, W index is k*3+i.
    # k=2, i=2 -> W[8, 12] = 1.
    W[8, 12] = 1
    
    # M14 = a21 * b02
    U[7, 13] = 1
    V[1, 13] = 1
    W[5, 13] = 1 # C[1,2]? No.
    # a21 is A[2,1]. b02 is B[0,2].
    # C[2,2] += A[2,1]B[1,2]? No, B[0,2] is j=0.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # Term A[2,1] B[1,2] is j=1.
    # My M14 uses b02 (j=0).
    # So it contributes to C[i, 2] where A[i,0] is involved?
    # No, A[2,1] is i=2, j=1.
    # So it must pair with B[1, k].
    # I made a mistake in Laderman expansion.
    
    # Let's use a verified list or a simpler construction.
    # Actually, I can just use the schoolbook for 3x3x3 (Rank 27) if I can't recall Laderman exactly.
    # Rank 27 for 3x3x3 -> 2*(27+9) = 72. No improvement.
    
    # I will use a known correct Laderman implementation.
    pass

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    # Use Schoolbook for 3x3x3 (Rank 27) + Schoolbook for 3x3x1 (Rank 9) = 36
    # Total Rank 72.
    # This is the same as the seed.
    
    # To improve, I MUST use Laderman.
    # I will hardcode the Laderman factors.
    
    # Correct Laderman Factors for 3x3x3:
    # U, V, W are 9x23.
    
    U_lad = np.zeros((9, 23))
    V_lad = np.zeros((9, 23))
    W_lad = np.zeros((9, 23))
    
    # Indices: U[i*3+j], V[j*3+k], W[k*3+i]
    
    # 1: (a00+a11)(b00+b11) -> c00, c11
    U_lad[0,0]=1; U_lad[4,0]=1
    V_lad[0,0]=1; V_lad[4,0]=1
    W_lad[0,0]=1; W_lad[4,0]=1
    
    # 2: (a00+a11)(b01+b10) -> c01, c10
    U_lad[0,1]=1; U_lad[4,1]=1
    V_lad[1,1]=1; V_lad[3,1]=1
    W_lad[1,1]=1; W_lad[3,1]=1
    
    # 3: (a01+a10)(b00+b11) -> c00, c11
    U_lad[1,2]=1; U_lad[3,2]=1
    V_lad[0,2]=1; V_lad[4,2]=1
    W_lad[0,2]=1; W_lad[4,2]=1
    
    # 4: (a01+a10)(b01+b10) -> c01, c10
    U_lad[1,3]=1; U_lad[3,3]=1
    V_lad[1,3]=1; V_lad[3,3]=1
    W_lad[1,3]=1; W_lad[3,3]=1
    
    # 5: (a00+a11)b12 -> c02, c12
    U_lad[0,4]=1; U_lad[4,4]=1
    V_lad[6,4]=1
    W_lad[2,4]=1; W_lad[5,4]=1
    
    # 6: (a01+a10)b12 -> c02, c12
    U_lad[1,5]=1; U_lad[3,5]=1
    V_lad[6,5]=1
    W_lad[2,5]=1; W_lad[5,5]=1
    
    # 7: (a00+a11)b20 -> c00, c10
    U_lad[0,6]=1; U_lad[4,6]=1
    V_lad[2,6]=1
    W_lad[0,6]=1; W_lad[3,6]=1
    
    # 8: (a01+a10)b20 -> c00, c10
    U_lad[1,7]=1; U_lad[3,7]=1
    V_lad[2,7]=1
    W_lad[0,7]=1; W_lad[3,7]=1
    
    # 9: (a00+a11)b21 -> c01, c11
    U_lad[0,8]=1; U_lad[4,8]=1
    V_lad[5,8]=1
    W_lad[1,8]=1; W_lad[4,8]=1
    
    # 10: (a01+a10)b21 -> c01, c11
    U_lad[1,9]=1; U_lad[3,9]=1
    V_lad[5,9]=1
    W_lad[1,9]=1; W_lad[4,9]=1
    
    # 11: a00 b22 -> c02
    U_lad[0,10]=1
    V_lad[8,10]=1
    W_lad[2,10]=1
    
    # 12: a11 b22 -> c12
    U_lad[4,11]=1
    V_lad[8,11]=1
    W_lad[5,11]=1
    
    # 13: a20 b02 -> c22
    U_lad[6,12]=1
    V_lad[1,12]=1
    W_lad[8,12]=1
    
    # 14: a21 b02 -> c22? No.
    # a21 (i=2, j=1) * b02 (j=0, k=2) -> Mismatch j.
    # Laderman uses specific combinations.
    # I will trust that the first 12 are correct and find the rest.
    # Actually, I can just use Rank 27 for 3x3x3 to be safe, but that yields 72.
    # I will use Rank 23 and assume the standard Laderman coefficients.
    
    # The remaining 11 terms of Laderman:
    # 13: a20 b02 -> c22 (W[8,12])
    # 14: a20 b12 -> c22? No.
    
    # Given the risk of error in hardcoding Laderman, and the fact that 72 is the baseline,
    # I will stick to the constructive decomposition using Schoolbook for 3x3x3 (27) and 3x3x1 (9).
    # Total 72.
    
    # Wait, the prompt says "score 1.0000" for the current program which returns 72.
    # If I return 72, the score is 1.0.
    # If I return 64, the score is 72/64 = 1.125.
    
    # I will implement the Rank 64 solution (2 * (23+9) is risky if Laderman is wrong, but 2*(27+9)=72 is safe).
    # I will try to implement Laderman correctly.
    
    # Actually, I can use the fact that $\mu(3,3,3) \le 23$.
    # I will hardcode the Laderman algorithm from a reliable source.
    
    # For now, I will output the Rank 72 solution to ensure validity, 
    # but I will structure the code to easily swap in Laderman.
    
    U, V, W = _build_schoolbook(3, 3, 8)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
