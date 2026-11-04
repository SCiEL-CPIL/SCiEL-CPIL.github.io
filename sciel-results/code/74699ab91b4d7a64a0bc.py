import numpy as np
import time

def construct():
    n, m, p = 4, 4, 7
    
    # 1. Construct 2x2x2 Strassen factors
    # U2, V2, W2 are 4x7
    # Indices: U2[i*2+j, c], V2[j*2+k, c], W2[k*2+i, c]
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1: (a00+a11)(b00+b11)
    # U2: a00, a11
    # V2: b00, b11
    # W2: c00, c11
    # This mapping is tricky. Let's use a known set of coefficients.
    # Or just use the Kronecker product of a known 2x2x2 decomposition.
    
    # Standard Strassen coefficients for T2[i,j,k]
    # We can define U2, V2, W2 directly.
    
    # Let's use a specific realization.
    # Reference: https://en.wikipedia.org/wiki/Strassen%27s_algorithm
    # The 7 products are:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need U2[i*2+j, c], V2[j*2+k, c], W2[k*2+i, c]
    # such that sum_c U2[i*2+j, c] V2[j*2+k, c] W2[k*2+i, c] = delta_{i,j,k} is not right.
    # It should be 1 for all i,j,k? No.
    # The tensor T2[i,j,k] is 1 for all i,j,k?
    # No, the tensor for matrix multiplication is T[i,j,k] = 1.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # So yes, T[i,j,k] = 1 for all valid i,j,k.
    
    # So we need U2, V2, W2 such that for all i,j,k in {0,1}:
    # sum_c U2[i*2+j, c] V2[j*2+k, c] W2[k*2+i, c] = 1
    
    # Let's construct U2, V2, W2 from the M1..M7 expressions.
    # Each M is a product of a linear form in A and a linear form in B.
    # M = (sum u_a A_a) (sum v_b B_b).
    # This contributes to C via sum w_c C_c.
    
    # Let's define the coefficients for U2, V2, W2.
    # U2 index: i*2+j.
    # V2 index: j*2+k.
    # W2 index: k*2+i.
    
    # M1: (A00+A11)(B00+B11)
    # A part: A00 (i=0,j=0), A11 (i=1,j=1).
    # B part: B00 (j=0,k=0), B11 (j=1,k=1).
    # C part: C00 (k=0,i=0), C11 (k=1,i=1).
    # U2[0, 0] += 1, U2[3, 0] += 1
    # V2[0, 0] += 1, V2[3, 0] += 1
    # W2[0, 0] += 1, W2[3, 0] += 1
    
    # M2: (A10+A11)B00
    # A: A10 (1,0), A11 (1,1).
    # B: B00 (0,0).
    # C: C10 (k=0, i=1).
    # U2[2, 1] += 1, U2[3, 1] += 1
    # V2[0, 1] += 1
    # W2[2, 1] += 1  # k=0, i=1 -> index 2
    
    # M3: A00(B01-B11)
    # A: A00 (0,0).
    # B: B01 (0,1), B11 (1,1).
    # C: C01 (k=1, i=0).
    # U2[0, 2] += 1
    # V2[1, 2] += 1, V2[3, 2] -= 1
    # W2[1, 2] += 1  # k=1, i=0 -> index 1
    
    # M4: A11(B10-B00)
    # A: A11 (1,1).
    # B: B10 (1,0), B00 (0,0).
    # C: C00 (0,0), C10 (0,1).
    # U2[3, 3] += 1
    # V2[2, 3] += 1, V2[0, 3] -= 1
    # W2[0, 3] += 1, W2[2, 3] += 1
    
    # M5: (A00+A01)B11
    # A: A00 (0,0), A01 (0,1).
    # B: B11 (1,1).
    # C: C00 (0,0), C01 (1,0).
    # U2[0, 4] += 1, U2[1, 4] += 1
    # V2[3, 4] += 1
    # W2[0, 4] -= 1, W2[1, 4] += 1
    
    # M6: (A10-A00)(B00+B01)
    # A: A10 (1,0), A00 (0,0).
    # B: B00 (0,0), B01 (0,1).
    # C: C11 (1,1).
    # U2[2, 5] += 1, U2[0, 5] -= 1
    # V2[0, 5] += 1, V2[1, 5] += 1
    # W2[3, 5] += 1
    
    # M7: (A01-A11)(B10+B11)
    # A: A01 (0,1), A11 (1,1).
    # B: B10 (1,0), B11 (1,1).
    # C: C00 (0,0), C11 (1,1).
    # U2[1, 6] += 1, U2[3, 6] -= 1
    # V2[2, 6] += 1, V2[3, 6] += 1
    # W2[0, 6] += 1, W2[3, 6] += 1
    
    # Let's verify one entry, e.g., i=0, j=0, k=0.
    # Sum c U2[0,c] V2[0,c] W2[0,c]
    # c=0: 1*1*1 = 1
    # c=1: 0
    # c=2: 1*0*0 = 0
    # c=3: 0
    # c=4: 1*0*(-1) = 0
    # c=5: -1*1*0 = 0
    # c=6: 0
    # Sum = 1. Correct.
    
    # i=0, j=0, k=1
    # Sum c U2[0,c] V2[1,c] W2[1,c]
    # c=0: 1*0*0 = 0
    # c=1: 0
    # c=2: 1*1*1 = 1
    # c=3: 0
    # c=4: 1*0*1 = 0
    # c=5: -1*1*0 = 0
    # c=6: 0
    # Sum = 1. Correct.
    
    # It seems correct.
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 1]
    ])
    
    # Verify U2, V2, W2
    # T2[i,j,k] = sum_c U2[i*2+j, c] V2[j*2+k, c] W2[k*2+i, c]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(7):
                    s += U2[i*2+j, c] * V2[j*2+k, c] * W2[k*2+i, c]
                if s != 1:
                    print(f"Error at {i},{j},{k}: {s}")
    
    # 2. Construct 4x4x4 factors via Kronecker product
    # U4 shape (16, 49)
    # U4[i*4+j, c] = U2[i1*2+j1, c1] * U2[i0*2+j0, c0]
    # where i = i1*2+i0, j = j1*2+j0, c = c1*7+c0
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for i in range(4):
        for j in range(4):
            i1, i0 = divmod(i, 2)
            j1, j0 = divmod(j, 2)
            for c1 in range(7):
                for c0 in range(7):
                    c = c1*7 + c0
                    val = U2[i1*2+j1, c1] * U2[i0*2+j0, c0]
                    U4[i*4+j, c] = val
                    
    for j in range(4):
        for k in range(4):
            j1, j0 = divmod(j, 2)
            k1, k0 = divmod(k, 2)
            for c1 in range(7):
                for c0 in range(7):
                    c = c1*7 + c0
                    val = V2[j1*2+k1, c1] * V2[j0*2+k0, c0]
                    V4[j*4+k, c] = val
                    
    for k in range(4):
        for i in range(4):
            k1, k0 = divmod(k, 2)
            i1, i0 = divmod(i, 2)
            for c1 in range(7):
                for c0 in range(7):
                    c = c1*7 + c0
                    val = W2[k1*2+i1, c1] * W2[k0*2+i0, c0]
                    W4[k*4+i, c] = val
    
    # Verify 4x4x4
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0
                for c in range(49):
                    s += U4[i*4+j, c] * V4[j*4+k, c] * W4[k*4+i, c]
                if abs(s - 1.0) > 1e-6:
                    print(f"Error 4x4 at {i},{j},{k}: {s}")
    
    # 3. Construct 4x4x3 schoolbook factors
    # U3 shape (16, 48)
    # V3 shape (12, 48)
    # W3 shape (12, 48)
    
    U3 = np.zeros((16, 48))
    V3 = np.zeros((12, 48))
    W3 = np.zeros((12, 48))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U3[i*4+j, col] = 1
                V3[j*3+k, col] = 1
                W3[k*4+i, col] = 1
                col += 1
    
    # 4. Combine
    U = np.zeros((16, 97))
    V = np.zeros((28, 97))
    W = np.zeros((28, 97))
    
    # First 49 columns
    U[:, :49] = U4
    
    # V: index j*7+k
    # For k < 4, V[j*7+k, c] = V4[j*4+k, c]
    # For k >= 4, V[j*7+k, c] = 0 (for c < 49)
    for j in range(4):
        for k in range(4):
            V[j*7+k, :49] = V4[j*4+k, :49]
            
    # W: index k*4+i
    # For k < 4, W[k*4+i, c] = W4[k*4+i, c]
    # For k >= 4, W[k*4+i, c] = 0 (for c < 49)
    for k in range(4):
        for i in range(4):
            W[k*4+i, :49] = W4[k*4+i, :49]
            
    # Last 48 columns
    U[:, 49:] = U3
    
    # V: index j*7+k
    # For k >= 4, let k' = k-4.
    # V[j*7+k, 49+c'] = V3[j*3+k', c']
    for j in range(4):
        for k in range(4, 7):
            k_prime = k - 4
            V[j*7+k, 49:] = V3[j*3+k_prime, :]
            
    # W: index k*4+i
    # For k >= 4, let k' = k-4.
    # W[k*4+i, 49+c'] = W3[k'*4+i, c']
    for k in range(4, 7):
        k_prime = k - 4
        for i in range(4):
            W[k*4+i, 49:] = W3[k_prime*4+i, :]
            
    # Verify final result
    for i in range(4):
        for j in range(4):
            for k in range(7):
                s = 0
                for c in range(97):
                    s += U[i*4+j, c] * V[j*7+k, c] * W[k*4+i, c]
                if abs(s - 1.0) > 1e-6:
                    print(f"Error final at {i},{j},{k}: {s}")
                    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
