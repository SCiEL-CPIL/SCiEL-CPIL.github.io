import time
import numpy as np

def construct():
    start_time = time.time()
    
    # Dimensions
    n, m, p = 2, 3, 8
    
    # Target Rank 44
    # Construction:
    # 1. Strassen 2x2x2 (Rank 7)
    # 2. 2x2x4 (Rank 14) by doubling Strassen
    # 3. 2x1x4 (Rank 8) by schoolbook
    # 4. 2x3x4 (Rank 22) by combining 2x2x4 and 2x1x4
    # 5. 2x3x8 (Rank 44) by doubling 2x3x4
    
    # Step 1: Strassen 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    # Indices:
    # A: a(0), b(1), c(2), d(3)
    # B: e(0), f(1), g(2), h(3)
    # C: c11(0), c12(1), c21(2), c22(3)
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h)
    U_s[:, 0] = [1, 0, 0, 1]
    V_s[:, 0] = [1, 1, 0, 0]
    W_s[:, 0] = [1, 0, 0, 1]
    
    # M2 = (c+d)e
    U_s[:, 1] = [0, 0, 1, 1]
    V_s[:, 1] = [1, 0, 0, 0]
    W_s[:, 1] = [0, 0, 1, -1] # C21 += M2, C22 -= M2
    
    # M3 = a(f-h)
    U_s[:, 2] = [1, 0, 0, 0]
    V_s[:, 2] = [0, 1, 0, -1]
    W_s[:, 2] = [0, 1, 0, 1] # C12 += M3, C22 += M3
    
    # M4 = d(g-e)
    U_s[:, 3] = [0, 0, 0, 1]
    V_s[:, 3] = [-1, 0, 1, 0]
    W_s[:, 3] = [1, 0, 1, 0] # C11 += M4, C21 += M4
    
    # M5 = (a+b)h
    U_s[:, 4] = [1, 1, 0, 0]
    V_s[:, 4] = [0, 0, 0, 1]
    W_s[:, 4] = [-1, 1, 0, 0] # C11 -= M5, C12 += M5
    
    # M6 = (c-a)(e+f)
    U_s[:, 5] = [-1, 0, 1, 0]
    V_s[:, 5] = [1, 1, 0, 0]
    W_s[:, 5] = [0, 0, 0, 1] # C22 += M6
    
    # M7 = (b-d)(g+h)
    U_s[:, 6] = [0, 1, 0, -1]
    V_s[:, 6] = [0, 0, 1, 1]
    W_s[:, 7] = [1, 0, 0, 0] if False else [1, 0, 0, 0] # C11 += M7
    # Correction: W_s[:, 6]
    W_s[:, 6] = [1, 0, 0, 0]
    
    # Step 2: 2x2x4 (Rank 14)
    # A: 2x2 (idx 0-3)
    # B: 2x4 (idx 0-7)
    # C: 2x4 (idx 0-7)
    
    U_224 = np.zeros((4, 14))
    V_224 = np.zeros((8, 14))
    W_224 = np.zeros((8, 14))
    
    # Left half (cols 0-1 of B, cols 0-1 of C)
    U_224[:, 0:7] = U_s
    # V_s is 4x7 (j in 0-1, k in 0-1). B is 2x4 (j in 0-1, k in 0-3).
    # For k in 0-1, index in B is j*4 + k.
    for j in range(2):
        for k in range(2):
            V_224[j*4 + k, 0:7] = V_s[j*2 + k, 0:7]
    # W_s is 4x7 (k in 0-1, i in 0-1). C is 2x4 (k in 0-3, i in 0-1).
    # For k in 0-1, index in C is k*2 + i.
    for k in range(2):
        for i in range(2):
            W_224[k*2 + i, 0:7] = W_s[k*2 + i, 0:7]
            
    # Right half (cols 2-3 of B, cols 2-3 of C)
    U_224[:, 7:14] = U_s
    # For k in 2-3, index in B is j*4 + k.
    for j in range(2):
        for k in range(2, 4):
            V_224[j*4 + k, 7:14] = V_s[j*2 + (k-2), 0:7]
    # For k in 2-3, index in C is k*2 + i.
    for k in range(2, 4):
        for i in range(2):
            W_224[k*2 + i, 7:14] = W_s[(k-2)*2 + i, 0:7]

    # Step 3: 2x1x4 (Rank 8)
    # A: 2x1 (idx 0-1)
    # B: 1x4 (idx 0-3)
    # C: 2x4 (idx 0-7)
    
    U_214 = np.zeros((2, 8))
    V_214 = np.zeros((4, 8))
    W_214 = np.zeros((8, 8))
    
    for t in range(8):
        i = t // 4
        k = t % 4
        U_214[i, t] = 1
        V_214[k, t] = 1
        W_214[k*2 + i, t] = 1

    # Step 4: 2x3x4 (Rank 22)
    # A: 2x3 (idx 0-5)
    # B: 3x4 (idx 0-11)
    # C: 2x4 (idx 0-7)
    
    U_234 = np.zeros((6, 22))
    V_234 = np.zeros((12, 22))
    W_234 = np.zeros((8, 22))
    
    # Part 1: 2x2x4 (A[:, 0:2], B[0:2, :])
    # Map A indices: i in 0-1, j in 0-1 -> i*2+j. In 2x3: i*3+j.
    for i in range(2):
        for j in range(2):
            U_234[i*3 + j, 0:14] = U_224[i*2 + j, 0:14]
    
    # Map B indices: j in 0-1, k in 0-3 -> j*4+k. In 3x4: j*4+k.
    # So V_224 rows 0-7 map directly to V_234 rows 0-7.
    V_234[0:8, 0:14] = V_224
    
    # Map C indices: same.
    W_234[:, 0:14] = W_224
    
    # Part 2: 2x1x4 (A[:, 2], B[2, :])
    # Map A indices: i in 0-1 -> i. In 2x3: i*3+2.
    for i in range(2):
        U_234[i*3 + 2, 14:22] = U_214[i, 0:8]
        
    # Map B indices: k in 0-3 -> k. In 3x4: 2*4+k = 8+k.
    for k in range(4):
        V_234[8 + k, 14:22] = V_214[k, 0:8]
        
    # Map C indices: same.
    W_234[:, 14:22] = W_214

    # Step 5: 2x3x8 (Rank 44)
    # A: 2x3 (idx 0-5)
    # B: 3x8 (idx 0-23)
    # C: 2x8 (idx 0-15)
    
    U = np.zeros((6, 44))
    V = np.zeros((24, 44))
    W = np.zeros((16, 44))
    
    # Left half (B[0:4], C[0:7])
    U[:, 0:22] = U_234
    # B indices: j in 0-2, k in 0-3 -> j*4+k. In 3x8: j*8+k.
    for j in range(3):
        for k in range(4):
            V[j*8 + k, 0:22] = V_234[j*4 + k, 0:22]
    # C indices: k in 0-3, i in 0-1 -> k*2+i. In 2x8: k*4+i? 
    # Wait, C is 2x8. Output index is usually row*cols + col.
    # Row i (0-1), Col k (0-7). Index i*8 + k?
    # The problem statement says: "output factor follows the official trace convention W[k*n+i,r]"
    # This implies the index is k*n + i, where k is the column index of C (0..p-1) and i is the row index (0..n-1).
    # So for 2x8, index is k*2 + i.
    
    # Let's re-verify the W indexing convention.
    # "W[k*n+i,r]"
    # For 2x3x8, n=2, p=8.
    # k is 0..7, i is 0..1.
    # Index = k*2 + i.
    # This matches my W_224 and W_234 construction where I used k*2+i.
    
    for k in range(4):
        for i in range(2):
            W[k*2 + i, 0:22] = W_234[k*2 + i, 0:22]
            
    # Right half (B[4:8], C[4:7])
    U[:, 22:44] = U_234
    for j in range(3):
        for k in range(4, 8):
            V[j*8 + k, 22:44] = V_234[j*4 + (k-4), 0:22]
    for k in range(4, 8):
        for i in range(2):
            W[k*2 + i, 22:44] = W_234[(k-4)*2 + i, 0:22]

    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    construct()
