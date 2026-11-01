import numpy as np
import time

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p  # 12
    U_sb = [[0] * rank_school for _ in range(n * m)]
    V_sb = [[0] * rank_school for _ in range(m * p)]
    W_sb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_school

    # Verify helper
    def verify(U, V, W):
        rank = len(U[0]) if U else 0
        T_prime = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            u_r = np.array(U[:, r] if hasattr(U, '__getitem__') else U, dtype=float)
            v_r = np.array(V[:, r] if hasattr(V, '__getitem__') else V, dtype=float)
            w_r = np.array(W[:, r] if hasattr(W, '__getitem__') else W, dtype=float)
            T_prime += np.einsum('i,j,k->ijk', u_r, v_r, w_r)
        
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_target[a, b, c] = 1.0
        return np.allclose(T_prime, T_target, atol=1e-9)
    
    # --- Rank 11 decomposition using Strassen for 2x2 block + schoolbook for 2x1 ---
    # A is 2x2, B is 2x3.
    # Split B into B1 (2x2) and B2 (2x1).
    # C1 = A * B1 (2x2), C2 = A * B2 (2x1).
    # Use Strassen for C1 (rank 7) and schoolbook for C2 (rank 2, since A is 2x2, B2 is 2x1).
    # Total rank = 7 + 2 = 9? Wait, let's check the dimensions.
    # Strassen uses 7 multiplications for 2x2*2x2.
    # Schoolbook for 2x2*2x1 uses 4 multiplications? No, 2x2*2x1 is sum of 2 products per row, so 4 products total.
    # So rank 7 + 4 = 11.
    
    # Let's build the rank 11 decomposition.
    # Indices:
    # A: 2x2, indices (i, j) -> i in {0,1}, j in {0,1}. U index: i*m + j = i*2 + j.
    # B: 2x3, indices (j, k) -> j in {0,1}, k in {0,1,2}. V index: j*p + k = j*3 + k.
    # C: 2x3, indices (k, i) -> k in {0,1,2}, i in {0,1}. W index: k*n + i = k*2 + i.
    
    U11 = np.zeros((4, 11))
    V11 = np.zeros((6, 11))
    W11 = np.zeros((6, 11))
    
    # Part 1: Strassen for A * B1 where B1 = B[:, 0:2]
    # B1 has columns 0 and 1. So B1[j, 0] = B[j, 0], B1[j, 1] = B[j, 1].
    # V indices for B1: 
    #   B[0,0] -> V[0], B[0,1] -> V[1]
    #   B[1,0] -> V[3], B[1,1] -> V[4]
    # Strassen terms for 2x2 * 2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U for A parts (4x):
    # A00: U[0], A01: U[1], A10: U[2], A11: U[3]
    
    # Term 1 (M1): (A00 + A11) * (B00 + B11)
    U11[:, 0] = [1, 0, 0, 1]
    V11[:, 0] = [1, 0, 0, 1, 0, 0]  # B[0,0] + B[1,1] -> V[0] + V[4]
    # C00: W[0], C11: W[3]
    W11[:, 0] = [1, 0, 0, 1, 0, 0]  # W[0] + W[3]
    
    # Term 2 (M2): (A10 + A11) * B00
    U11[:, 1] = [0, 0, 1, 1]
    V11[:, 1] = [1, 0, 0, 0, 0, 0]  # B[0,0]
    # C10: W[2], C11: W[3] (with -)
    W11[:, 1] = [0, 0, 1, -1, 0, 0]  # W[2] - W[3]
    
    # Term 3 (M3): A00 * (B10 - B11)
    U11[:, 2] = [1, 0, 0, 0]
    V11[:, 2] = [0, 0, 0, 1, -1, 0]  # B[1,0] - B[1,1] -> V[3] - V[4]
    # C01: W[1], C11: W[3]
    W11[:, 2] = [0, 1, 0, 1, 0, 0]  # W[1] + W[3]
    
    # Term 4 (M4): A11 * (B01 - B00)
    U11[:, 3] = [0, 0, 0, 1]
    V11[:, 3] = [-1, 1, 0, 0, 0, 0]  # B[0,1] - B[0,0] -> -V[0] + V[1]
    # C00: W[0], C10: W[2]
    W11[:, 3] = [1, 0, 1, 0, 0, 0]  # W[0] + W[2]
    
    # Term 5 (M5): (A00 + A01) * B11
    U11[:, 4] = [1, 1, 0, 0]
    V11[:, 4] = [0, 0, 0, 0, 1, 0]  # B[1,1] -> V[4]
    # C00: W[0] (with -), C01: W[1]
    W11[:, 4] = [-1, 1, 0, 0, 0, 0]  # -W[0] + W[1]
    
    # Term 6 (M6): (A10 - A00) * (B00 + B01)
    U11[:, 5] = [-1, 0, 1, 0]
    V11[:, 5] = [1, 1, 0, 0, 0, 0]  # B[0,0] + B[0,1] -> V[0] + V[1]
    # C11: W[3]
    W11[:, 5] = [0, 0, 0, 1, 0, 0]
    
    # Term 7 (M7): (A01 - A11) * (B10 + B11)
    U11[:, 6] = [0, 1, 0, -1]
    V11[:, 6] = [0, 0, 0, 1, 1, 0]  # B[1,0] + B[1,1] -> V[3] + V[4]
    # C00: W[0], C11: W[3] (with -)
    W11[:, 6] = [1, 0, 0, -1, 0, 0]
    
    # Part 2: Schoolbook for A * B2 where B2 = B[:, 2:3] (2x1)
    # B2 has column 2. So B2[j, 0] = B[j, 2].
    # V indices for B2:
    #   B[0,2] -> V[2], B[1,2] -> V[5]
    # C2 = A * B2 (2x1).
    # C2[0] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C2[1] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    # This requires 4 multiplications.
    
    # Term 8: A00 * B02
    U11[:, 7] = [1, 0, 0, 0]
    V11[:, 7] = [0, 0, 1, 0, 0, 0]  # B[0,2] -> V[2]
    # C02: W[4], C12: W[5]
    W11[:, 7] = [0, 0, 0, 0, 1, 0]  # W[4]
    
    # Term 9: A01 * B12
    U11[:, 8] = [0, 1, 0, 0]
    V11[:, 8] = [0, 0, 0, 0, 0, 1]  # B[1,2] -> V[5]
    # C02: W[4]
    W11[:, 8] = [0, 0, 0, 0, 1, 0]  # W[4]
    
    # Term 10: A10 * B02
    U11[:, 9] = [0, 0, 1, 0]
    V11[:, 9] = [0, 0, 1, 0, 0, 0]  # B[0,2] -> V[2]
    # C12: W[5]
    W11[:, 9] = [0, 0, 0, 0, 0, 1]  # W[5]
    
    # Term 11: A11 * B12
    U11[:, 10] = [0, 0, 0, 1]
    V11[:, 10] = [0, 0, 0, 0, 0, 1]  # B[1,2] -> V[5]
    # C12: W[5]
    W11[:, 10] = [0, 0, 0, 0, 0, 1]  # W[5]
    
    # Verify rank 11
    if verify(U11, V11, W11):
        best_U = U11.tolist()
        best_V = V11.tolist()
        best_W = W11.tolist()
        best_rank = 11
    
    # Try to find rank 10 or lower? 
    # Lower bound is 6. Known best is 10? Or 11?
    # Let's stick with 11 for now as it's a solid improvement over 12.
    
    return {"U": best_U, "V": best_V, "W": best_W}
