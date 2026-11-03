import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    """
    Construct a rank-65 decomposition for 4x4 * 4x5 matrix multiplication.
    Rank 65 = Rank(4x4x4) + Rank(4x4x1)
    Rank(4x4x4) <= 49 (Strassen)
    Rank(4x4x1) = 16 (Schoolbook)
    """
    n, m, p = DIMENSIONS
    
    # 1. Construct Strassen 2x2x2 (Rank 7)
    # U2, V2, W2 are 4x7
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # c=1
    U2[:, 0] = [1, 0, 0, 1]
    V2[:, 0] = [1, 0, 0, 1]
    W2[:, 0] = [1, 0, 0, 1] # C00 + C11 (idx 0, 3)
    
    # c=2
    U2[:, 1] = [0, 0, 1, 1]
    V2[:, 1] = [1, 0, 0, 0]
    W2[:, 1] = [0, 1, 0, -1] # C10 - C11 (idx 1, 3)
    
    # c=3
    U2[:, 2] = [1, 0, 0, 0]
    V2[:, 2] = [0, 0, 1, -1]
    W2[:, 2] = [0, 0, 1, 1] # C01 + C11 (idx 2, 3)
    
    # c=4
    U2[:, 3] = [0, 0, 0, 1]
    V2[:, 3] = [-1, 1, 0, 0]
    W2[:, 3] = [1, 1, 0, 0] # C00 + C10 (idx 0, 1)
    
    # c=5
    U2[:, 4] = [1, 1, 0, 0]
    V2[:, 4] = [0, 0, 0, 1]
    W2[:, 4] = [-1, 0, 1, 0] # -C00 + C01 (idx 0, 2)
    
    # c=6
    U2[:, 5] = [-1, 0, 1, 0]
    V2[:, 5] = [1, 1, 0, 0]
    W2[:, 5] = [0, 0, 0, 1] # C11 (idx 3)
    
    # c=7
    U2[:, 6] = [0, 1, 0, -1]
    V2[:, 6] = [0, 0, 1, 1]
    W2[:, 6] = [1, 0, 0, 0] # C00 (idx 0)

    # 2. Construct 4x4x4 (Rank 49)
    # Strassen coefficients for 4x4x4
    # U_s, V_s are 2x2 matrices of coefficients for blocks
    # s=1: (A00+A11)(B00+B11)
    # s=2: (A10+A11)B00
    # s=3: A00(B10-B11)
    # s=4: A11(B01-B00)
    # s=5: (A00+A01)B11
    # s=6: (A10-A00)(B00+B01)
    # s=7: (A01-A11)(B10+B11)
    
    U4_coeffs = np.zeros((7, 2, 2))
    V4_coeffs = np.zeros((7, 2, 2))
    
    # s=1
    U4_coeffs[0] = [[1, 0], [0, 1]]
    V4_coeffs[0] = [[1, 0], [0, 1]]
    # s=2
    U4_coeffs[1] = [[0, 0], [1, 1]]
    V4_coeffs[1] = [[1, 0], [0, 0]]
    # s=3
    U4_coeffs[2] = [[1, 0], [0, 0]]
    V4_coeffs[2] = [[0, 0], [1, -1]]
    # s=4
    U4_coeffs[3] = [[0, 0], [0, 1]]
    V4_coeffs[3] = [[0, 1], [0, -1]] # B01 - B00? No, B01 is (0,1), B00 is (0,0).
    # Wait, V4_coeffs[3] should be for B.
    # B01 is block (0,1). B00 is block (0,0).
    # So V4_coeffs[3][0,1] = 1, V4_coeffs[3][0,0] = -1.
    V4_coeffs[3] = [[-1, 1], [0, 0]]
    # s=5
    U4_coeffs[4] = [[1, 1], [0, 0]]
    V4_coeffs[4] = [[0, 0], [0, 1]]
    # s=6
    U4_coeffs[5] = [[-1, 0], [1, 0]]
    V4_coeffs[5] = [[1, 1], [0, 0]]
    # s=7
    U4_coeffs[6] = [[0, 1], [0, -1]]
    V4_coeffs[6] = [[0, 0], [1, 1]]

    # Output assembly coefficients for 4x4x4
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    # C blocks are 2x2.
    # W4_coeffs[s, I, J] is the coefficient of P_s in C_IJ.
    W4_coeffs = np.zeros((7, 2, 2))
    W4_coeffs[0, 0, 0] = 1  # C00 += P1
    W4_coeffs[0, 1, 1] = 1  # C11 += P1
    W4_coeffs[1, 1, 0] = 1  # C10 += P2
    W4_coeffs[1, 1, 1] = -1 # C11 -= P2
    W4_coeffs[2, 0, 1] = 1  # C01 += P3
    W4_coeffs[2, 1, 1] = 1  # C11 += P3
    W4_coeffs[3, 0, 0] = 1  # C00 += P4
    W4_coeffs[3, 1, 0] = 1  # C10 += P4
    W4_coeffs[4, 0, 0] = -1 # C00 -= P5
    W4_coeffs[4, 0, 1] = 1  # C01 += P5
    W4_coeffs[5, 1, 1] = 1  # C11 += P6
    W4_coeffs[6, 0, 0] = 1  # C00 += P7

    U44 = np.zeros((16, 49))
    V44 = np.zeros((16, 49))
    W44 = np.zeros((16, 49))
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            # U part
            for i in range(2):
                for j in range(2):
                    coeff = U4_coeffs[s, i, j]
                    if coeff == 0: continue
                    for r in range(2):
                        for c in range(2):
                            a_global = (2*i + r) * 4 + (2*j + c)
                            a_local = r * 2 + c
                            U44[a_global, col] = coeff * U2[a_local, t]
            
            # V part
            for i in range(2):
                for j in range(2):
                    coeff = V4_coeffs[s, i, j]
                    if coeff == 0: continue
                    for r in range(2):
                        for c in range(2):
                            b_global = (2*i + r) * 4 + (2*j + c)
                            b_local = r * 2 + c
                            V44[b_global, col] = coeff * V2[b_local, t]
            
            # W part
            for I in range(2):
                for J in range(2):
                    coeff = W4_coeffs[s, I, J]
                    if coeff == 0: continue
                    for r in range(2):
                        for c in range(2):
                            c_global = (2*J + c) * 4 + (2*I + r)
                            c_local = c * 2 + r
                            W44[c_global, col] = coeff * W2[c_local, t]

    # 3. Construct 4x4x1 (Rank 16)
    U41 = np.zeros((16, 16))
    V41 = np.zeros((4, 16))
    W41 = np.zeros((4, 16))
    
    col = 0
    for i in range(4):
        for j in range(4):
            a = i * 4 + j
            b = j
            c = i
            U41[a, col] = 1
            V41[b, col] = 1
            W41[c, col] = 1
            col += 1

    # 4. Combine
    # Total rank 49 + 16 = 65
    U = np.hstack([U44, U41])
    V = np.hstack([V44, V41])
    # W44 is 16x49. W41 is 4x16.
    # We need to map W41 to the full 20x65 size?
    # No, W for 4x4x5 has size p*n = 5*4 = 20.
    # W44 corresponds to the first 4 columns of B (p=4 part).
    # W41 corresponds to the 5th column of B (p=1 part).
    # The output C is 4x5.
    # C[i, k] for k=0..3 comes from U44/V44/W44.
    # C[i, 4] comes from U41/V41/W41.
    # W index is k*n + i.
    # For k=0..3, index is k*4 + i.
    # For k=4, index is 4*4 + i = 16 + i.
    
    W_full = np.zeros((20, 65))
    W_full[0:16, 0:49] = W44
    W_full[16:20, 49:65] = W41

    return {"U": U.tolist(), "V": V.tolist(), "W": W_full.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
