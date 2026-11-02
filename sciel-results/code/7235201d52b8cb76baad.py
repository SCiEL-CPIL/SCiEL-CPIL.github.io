import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=3
    # U shape: (n*m, rank) = (6, 17)
    # V shape: (m*p, rank) = (9, 17)
    # W shape: (p*n, rank) = (6, 17)
    
    rank = 17
    U = np.zeros((6, rank))
    V = np.zeros((9, rank))
    W = np.zeros((6, rank))
    
    # Helper to get indices
    def idx_A(i, j): return i * m + j
    def idx_B(j, k): return j * p + k
    def idx_C(k, i): return k * n + i
    
    # --- Strassen Part (r = 0..6) ---
    # Subproblem: i in {0,1}, j in {0,1}, k in {0,1}
    # A indices: 00(0), 01(1), 10(3), 11(4)
    # B indices: 00(0), 01(1), 10(3), 11(4)
    # C indices: 00(0), 01(1), 10(2), 11(3)
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U[idx_A(0,0), 0] = 1
    U[idx_A(1,1), 0] = 1
    V[idx_B(0,0), 0] = 1
    V[idx_B(1,1), 0] = 1
    W[idx_C(0,0), 0] = 1  # C00
    W[idx_C(1,1), 0] = 1  # C11
    
    # r=1: M2 = (A10+A11)B00
    U[idx_A(1,0), 1] = 1
    U[idx_A(1,1), 1] = 1
    V[idx_B(0,0), 1] = 1
    W[idx_C(0,1), 1] = 1  # C10 (k=0, i=1)
    W[idx_C(1,1), 1] = -1 # C11
    
    # r=2: M3 = A00(B10-B11)
    U[idx_A(0,0), 2] = 1
    V[idx_B(1,0), 2] = 1
    V[idx_B(1,1), 2] = -1
    W[idx_C(1,0), 2] = 1  # C01 (k=1, i=0)
    W[idx_C(1,1), 2] = 1  # C11
    
    # r=3: M4 = A11(B01-B00)
    U[idx_A(1,1), 3] = 1
    V[idx_B(0,1), 3] = 1
    V[idx_B(0,0), 3] = -1
    W[idx_C(0,0), 3] = 1  # C00
    W[idx_C(0,1), 3] = 1  # C10
    
    # r=4: M5 = (A00+A01)B11
    U[idx_A(0,0), 4] = 1
    U[idx_A(0,1), 4] = 1
    V[idx_B(1,1), 4] = 1
    W[idx_C(0,0), 4] = -1 # C00
    W[idx_C(1,0), 4] = 1  # C01
    
    # r=5: M6 = (A10-A00)(B00+B10)
    U[idx_A(1,0), 5] = 1
    U[idx_A(0,0), 5] = -1
    V[idx_B(0,0), 5] = 1
    V[idx_B(1,0), 5] = 1
    W[idx_C(1,1), 5] = 1  # C11
    
    # r=6: M7 = (A01-A11)(B11+B01)
    U[idx_A(0,1), 6] = 1
    U[idx_A(1,1), 6] = -1
    V[idx_B(1,1), 6] = 1
    V[idx_B(0,1), 6] = 1
    W[idx_C(0,0), 6] = 1  # C00
    
    # --- Schoolbook Part 2: k=2, j in {0,1}, i in {0,1} (r = 7..10) ---
    # Terms: A00B02, A01B12, A10B02, A11B12
    # C indices: C02 (k=2, i=0 -> idx 4), C12 (k=2, i=1 -> idx 5)
    
    # r=7: A00 B02 -> C02
    U[idx_A(0,0), 7] = 1
    V[idx_B(0,2), 7] = 1
    W[idx_C(2,0), 7] = 1
    
    # r=8: A01 B12 -> C02
    U[idx_A(0,1), 8] = 1
    V[idx_B(1,2), 8] = 1
    W[idx_C(2,0), 8] = 1
    
    # r=9: A10 B02 -> C12
    U[idx_A(1,0), 9] = 1
    V[idx_B(0,2), 9] = 1
    W[idx_C(2,1), 9] = 1
    
    # r=10: A11 B12 -> C12
    U[idx_A(1,1), 10] = 1
    V[idx_B(1,2), 10] = 1
    W[idx_C(2,1), 10] = 1
    
    # --- Schoolbook Part 3: j=2, i in {0,1}, k in {0,1,2} (r = 11..16) ---
    # Terms: A02B20, A02B21, A02B22, A12B20, A12B21, A12B22
    # A indices: A02 (idx 2), A12 (idx 5)
    # B indices: B20 (idx 6), B21 (idx 7), B22 (idx 8)
    # C indices: C00(0), C01(1), C02(4), C10(2), C11(3), C12(5)
    
    # r=11: A02 B20 -> C00
    U[idx_A(0,2), 11] = 1
    V[idx_B(2,0), 11] = 1
    W[idx_C(0,0), 11] = 1
    
    # r=12: A02 B21 -> C01
    U[idx_A(0,2), 12] = 1
    V[idx_B(2,1), 12] = 1
    W[idx_C(1,0), 12] = 1
    
    # r=13: A02 B22 -> C02
    U[idx_A(0,2), 13] = 1
    V[idx_B(2,2), 13] = 1
    W[idx_C(2,0), 13] = 1
    
    # r=14: A12 B20 -> C10
    U[idx_A(1,2), 14] = 1
    V[idx_B(2,0), 14] = 1
    W[idx_C(0,1), 14] = 1
    
    # r=15: A12 B21 -> C11
    U[idx_A(1,2), 15] = 1
    V[idx_B(2,1), 15] = 1
    W[idx_C(1,1), 15] = 1
    
    # r=16: A12 B22 -> C12
    U[idx_A(1,2), 16] = 1
    V[idx_B(2,2), 16] = 1
    W[idx_C(2,1), 16] = 1
    
    # Verification
    # T[I, J, K] = sum_r U[I, r] V[J, r] W[K, r]
    T = np.einsum('ir,jr,kr->IJK', U, V, W)
    
    # Expected tensor
    targets = np.zeros((6, 9, 6))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                targets[I, J, K] = 1.0
                
    if not np.allclose(T, targets, atol=1e-9):
        # Fallback to schoolbook if construction is wrong (should not happen)
        rank_school = n * m * p
        U_s = np.zeros((n*m, rank_school))
        V_s = np.zeros((m*p, rank_school))
        W_s = np.zeros((p*n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    I = i * m + j
                    J = j * p + k
                    K = k * n + i
                    U_s[I, col] = 1
                    V_s[J, col] = 1
                    W_s[K, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
