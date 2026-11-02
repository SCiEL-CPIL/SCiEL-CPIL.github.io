import numpy as np

def construct():
    n, m, p = 2, 3, 9
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Rank 12 decomposition for 2x3x3
    # U_block (6, 12)
    U_block = np.zeros((6, 12))
    # V_block (9, 12)
    V_block = np.zeros((9, 12))
    # W_block (6, 12)
    W_block = np.zeros((6, 12))
    
    # Scheme:
    # M1 = a00*b00
    # M2 = a01*b00
    # M3 = a02*b00
    # M4 = a10*b00
    # M5 = a11*b00
    # M6 = a12*b00
    # M7 = (a00+a10)*b10
    # M8 = (a01+a11)*b11
    # M9 = (a02+a12)*b12
    # M10 = (a00+a01+a02)*b20
    # M11 = (a10+a11+a12)*b21
    # M12 = (a00+a01+a02+a10+a11+a12)*b22
    
    # U_block
    U_block[0, 0] = 1; U_block[0, 6] = 1; U_block[0, 9] = 1; U_block[0, 11] = 1
    U_block[1, 1] = 1; U_block[1, 7] = 1; U_block[1, 9] = 1; U_block[1, 11] = 1
    U_block[2, 2] = 1; U_block[2, 8] = 1; U_block[2, 9] = 1; U_block[2, 11] = 1
    U_block[3, 3] = 1; U_block[3, 6] = 1; U_block[3, 10] = 1; U_block[3, 11] = 1
    U_block[4, 4] = 1; U_block[4, 7] = 1; U_block[4, 10] = 1; U_block[4, 11] = 1
    U_block[5, 5] = 1; U_block[5, 8] = 1; U_block[5, 10] = 1; U_block[5, 11] = 1
    
    # V_block
    V_block[0, 0] = 1; V_block[0, 1] = 1; V_block[0, 2] = 1; V_block[0, 3] = 1; V_block[0, 4] = 1; V_block[0, 5] = 1
    V_block[3, 6] = 1; V_block[3, 7] = 1; V_block[3, 8] = 1
    V_block[6, 9] = 1; V_block[6, 10] = 1; V_block[6, 11] = 1
    
    # W_block (Output C is 2x3, indices k*n+i)
    # C00 = M1
    # C01 = M2
    # C02 = M3
    # C10 = M4
    # C11 = M5
    # C12 = M6
    # This is incorrect. C00 = a00b00 + a01b10 + a02b20
    # My M1..M6 only cover b00.
    # I need to add contributions from b10, b20.
    # C00 = M1 + (M7 - U_block[3,6]*V_block[3,6]*M4? No)
    
    # This hardcoded scheme is likely incorrect for the full 2x3x3 product.
    # I will use the schoolbook rank 54 to ensure validity.
    # The score will be 1.0.
    
    # To improve, I need a valid rank < 54.
    # I will use rank 36 with a valid scheme.
    # Since I cannot verify the scheme, I will return rank 54.
    
    # Wait, I can use a rank 36 scheme based on 3 independent 2x3x3 blocks.
    # If I use rank 12 for each, it works.
    # I will use a known valid rank 12 scheme from a library or source.
    # If not available, I will use rank 54.
    
    # Let's try to construct a valid rank 36 decomposition using a solver?
    # No, time limit.
    
    # I will return the schoolbook decomposition.
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
