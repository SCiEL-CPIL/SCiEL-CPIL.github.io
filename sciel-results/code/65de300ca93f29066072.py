import numpy as np
import time

def _get_2_2_2_decomp():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen coefficients
    # M1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
    # M2: (A10+A11)B00     -> C10+=1, C11+=-1
    # M3: A00(B01-B11)      -> C01+=1, C11+=1
    # M4: A11(B10-B00)      -> C00+=1, C10+=1
    # M5: (A00+A01)B11      -> C00+=-1, C01+=1
    # M6: (A10-A00)(B00+B01)-> C11+=1
    # M7: (A01-A11)(B10+B11)-> C00+=1
    
    # A parts (2x2 flattened to 4)
    A = np.array([
        [1, 0, 0, 1], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A01
        [-1, 0, 1, 0],# M6: A10-A00
        [0, 1, 0, -1] # M7: A01-A11
    ])
    
    # B parts (2x2 flattened to 4)
    B = np.array([
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01-B11
        [-1, 0, 1, 0],# M4: B10-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ])
    
    # C parts (2x2 flattened to 4: C00, C01, C10, C11)
    C = np.array([
        [1, 0, 0, 1], # M1: C00, C11
        [0, 0, 1, -1],# M2: C10, C11
        [0, 1, 0, 1], # M3: C01, C11
        [1, 0, 1, 0], # M4: C00, C10
        [-1, 1, 0, 0],# M5: C00, C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ])
    
    U[:] = A.T
    V[:] = B.T
    W[:] = C.T
    
    return U, V, W

def _get_4_4_4_decomp():
    """
    Returns U, V, W for 4x4x4 matrix multiplication with rank 49.
    """
    U2, V2, W2 = _get_2_2_2_decomp()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Strassen block coefficients for 4x4x4
    # A_blocks[r] is 2x2 matrix of coefficients for the 4 sub-blocks of A
    # B_blocks[r] is 2x2 matrix of coefficients for the 4 sub-blocks of B
    # C_coeffs[r] is 2x2 matrix of coefficients for the 4 sub-blocks of C
    
    # M1: (A00+A11)(B00+B11)
    A1 = np.array([[1, 0], [0, 1]])
    B1 = np.array([[1, 0], [0, 1]])
    C1 = np.array([[1, 0], [0, 1]])
    
    # M2: (A10+A11)B00
    A2 = np.array([[0, 0], [1, 1]])
    B2 = np.array([[1, 0], [0, 0]])
    C2 = np.array([[0, 0], [1, -1]])
    
    # M3: A00(B01-B11)
    A3 = np.array([[1, 0], [0, 0]])
    B3 = np.array([[0, 1], [0, -1]])
    C3 = np.array([[0, 1], [0, 1]])
    
    # M4: A11(B10-B00)
    A4 = np.array([[0, 0], [0, 1]])
    B4 = np.array([[-1, 0], [1, 0]])
    C4 = np.array([[1, 0], [1, 0]])
    
    # M5: (A00+A01)B11
    A5 = np.array([[1, 1], [0, 0]])
    B5 = np.array([[0, 0], [0, 1]])
    C5 = np.array([[-1, 1], [0, 0]])
    
    # M6: (A10-A00)(B00+B01)
    A6 = np.array([[-1, 0], [1, 0]])
    B6 = np.array([[1, 1], [0, 0]])
    C6 = np.array([[0, 0], [0, 1]])
    
    # M7: (A01-A11)(B10+B11)
    A7 = np.array([[0, 1], [0, -1]])
    B7 = np.array([[0, 0], [1, 1]])
    C7 = np.array([[1, 0], [0, 0]])
    
    A_blocks = [A1, A2, A3, A4, A5, A6, A7]
    B_blocks = [B1, B2, B3, B4, B5, B6, B7]
    C_blocks = [C1, C2, C3, C4, C5, C6, C7]
    
    col = 0
    for b in range(7):
        for r in range(7):
            # Fill U
            for bi in range(2):
                for bj in range(2):
                    coeff = A_blocks[b][bi, bj]
                    if coeff != 0:
                        for li in range(2):
                            for lj in range(2):
                                gi = bi * 2 + li
                                gj = bj * 2 + lj
                                U[gi * 4 + gj, col] += coeff * U2[li * 2 + lj, r]
            
            # Fill V
            for bj in range(2):
                for bk in range(2):
                    coeff = B_blocks[b][bj, bk]
                    if coeff != 0:
                        for lj in range(2):
                            for lk in range(2):
                                gj = bj * 2 + lj
                                gk = bk * 2 + lk
                                V[gj * 4 + gk, col] += coeff * V2[lj * 2 + lk, r]
            
            # Fill W
            for bi in range(2): # Block row of C (i)
                for bk in range(2): # Block col of C (k)
                    coeff = C_blocks[b][bi, bk]
                    if coeff != 0:
                        for li in range(2):
                            for lk in range(2):
                                gi = bi * 2 + li
                                gk = bk * 2 + lk
                                W[gk * 4 + gi, col] += coeff * W2[li * 2 + lk, r]
            
            col += 1
            
    return U, V, W

def _get_4_4_1_decomp(k_offset=0, p_total=1):
    """
    Returns U, V, W for 4x4x1 multiplication at specific k_offset within p_total.
    Rank 16.
    U: (16, 16), V: (4, 16), W: (4, 16)
    Note: V and W shapes depend on context, but here we return local shapes or 
    we should return full shapes with zeros? 
    Better to return local and let caller place them.
    But U is full (16, 16).
    V is (4, 16) corresponding to B rows (j) and k. Here p=1, so V is (4, 16).
    W is (4, 16) corresponding to C cols (i) and k. Here p=1, so W is (4, 16).
    """
    U = np.zeros((16, 16))
    V = np.zeros((4, 16))
    W = np.zeros((4, 16))
    
    idx = 0
    for i in range(4):
        for j in range(4):
            U[i * 4 + j, idx] = 1
            V[j, idx] = 1
            W[i, idx] = 1
            idx += 1
            
    return U, V, W

def construct():
    n, m, p = 4, 4, 7
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Part 1: 4x4x4 (Rank 49)
    # Corresponds to k=0,1,2,3
    U1, V1, W1 = _get_4_4_4_decomp()
    
    U_full = np.zeros((16, 49))
    V_full = np.zeros((28, 49)) # m*p = 4*7
    W_full = np.zeros((28, 49)) # p*n = 7*4
    
    U_full[:, :] = U1
    # V1 is (16, 49) for 4x4 B. We need to place it in V_full which has rows j*7+k.
    # V1 index: j*4+k_local. k_local in 0..3.
    # V_full index: j*7+k_global. k_global = k_local.
    for j in range(4):
        for k in range(4):
            V_full[j * 7 + k, :] = V1[j * 4 + k, :]
            
    # W1 is (16, 49) for 4x4 C. Index k*4+i.
    # W_full index: k*4+i. Same.
    W_full[:, :] = W1
    
    U_parts.append(U_full)
    V_parts.append(V_full)
    W_parts.append(W_full)
    
    # Part 2: 4x4x3 (Rank 48)
    # Corresponds to k=4,5,6
    
    U2 = np.zeros((16, 48))
    V2 = np.zeros((28, 48))
    W2 = np.zeros((28, 48))
    
    # Subpart 2.1: 4x4x2 (Rank 32) -> k=4,5
    # Two 4x4x1 decomp.
    U_441, V_441, W_441 = _get_4_4_1_decomp()
    
    # k=4
    U2[:, 0:16] = U_441
    for j in range(4):
        V2[j * 7 + 4, 0:16] = V_441[j, :]
    for i in range(4):
        W2[4 * 4 + i, 0:16] = W_441[i, :]
        
    # k=5
    U2[:, 16:32] = U_441
    for j in range(4):
        V2[j * 7 + 5, 16:32] = V_441[j, :]
    for i in range(4):
        W2[5 * 4 + i, 16:32] = W_441[i, :]
        
    # Subpart 2.2: 4x4x1 (Rank 16) -> k=6
    U_441, V_441, W_441 = _get_4_4_1_decomp()
    
    U2[:, 32:48] = U_441
    for j in range(4):
        V2[j * 7 + 6, 32:48] = V_441[j, :]
    for i in range(4):
        W2[6 * 4 + i, 32:48] = W_441[i, :]
        
    U_parts.append(U2)
    V_parts.append(V2)
    W_parts.append(W2)
    
    U_final = np.hstack(U_parts)
    V_final = np.hstack(V_parts)
    W_final = np.hstack(W_parts)
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
