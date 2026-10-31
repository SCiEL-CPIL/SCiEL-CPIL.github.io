import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 8.0

    n, m, p = 4, 4, 4

    # Build the 2x2 Strassen factor matrices (7 multiplications)
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # M0 = (A00 + A11) * (B00 + B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # M1 = (A10 + A11) * B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1

    # M2 = A00 * (B01 - B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1

    # M3 = A11 * (B10 - B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1

    # M4 = (A00 + A01) * B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1

    # M5 = (A10 - A00) * (B00 + B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    # M6 = (A01 - A11) * (B10 + B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Compute Kronecker products
    # K_U[r_kron, c_kron] = U2[r1, s] * U2[r2, t]
    # where r_kron = r1*4 + r2, c_kron = s*7 + t
    K_U = np.kron(U2, U2)
    K_V = np.kron(V2, V2)
    K_W = np.kron(W2, W2)
    
    # Permutation for U and V
    # Output index r_out = i*4 + j
    # Kron index r_kron = ((i//2)*2 + (j//2))*4 + ((i%2)*2 + (j%2))
    perm_UV = np.zeros(16, dtype=int)
    for i in range(4):
        for j in range(4):
            r_out = i * 4 + j
            r1 = (i // 2) * 2 + (j // 2)
            r2 = (i % 2) * 2 + (j % 2)
            r_kron = r1 * 4 + r2
            perm_UV[r_out] = r_kron
            
    U_arr = K_U[perm_UV]
    V_arr = K_V[perm_UV]
    
    # Permutation for W
    # Output index r_out = j*4 + i (Trace convention W[k*n+i, r] with k=j, n=4, i=i)
    # Factor indices are derived from i and j in the same way as U/V
    # r1 = (i // 2) * 2 + (j // 2)
    # r2 = (i % 2) * 2 + (j % 2)
    perm_W = np.zeros(16, dtype=int)
    for i in range(4):
        for j in range(4):
            r_out = j * 4 + i
            r1 = (i // 2) * 2 + (j // 2)
            r2 = (i % 2) * 2 + (j % 2)
            r_kron = r1 * 4 + r2
            perm_W[r_out] = r_kron
            
    W_arr = K_W[perm_W]

    if time.time() > deadline:
        # Fallback to empty if somehow slow (should not happen)
        return {"U": [], "V": [], "W": []}

    return {
        "U": U_arr.tolist(),
        "V": V_arr.tolist(),
        "W": W_arr.tolist()
    }
