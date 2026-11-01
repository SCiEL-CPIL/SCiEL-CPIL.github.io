"""Strassen-based rank-49 decomposition for 2x2 * 2x14 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-49 tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    Strategy:
    - Treat the 2x14 matrix B as 7 independent 2x2 blocks.
    - Apply Strassen's algorithm (rank 7) to each 2x2 block.
    - Total rank = 7 * 7 = 49.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    nm = n * m  # 4
    mp = m * p  # 28
    pn = p * n  # 28
    
    num_blocks = p // 2  # 7
    rank = 7 * num_blocks  # 49
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Strassen terms for 2x2 * 2x2
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # U vectors (coeffs for A)
    # M1 = (A00+A11)(B00+B11)
    U_m1 = [1, 0, 0, 1]
    # M2 = (A10+A11)B00
    U_m2 = [0, 0, 1, 1]
    # M3 = A00(B01-B11)
    U_m3 = [1, 0, 0, 0]
    # M4 = A11(B10-B00)
    U_m4 = [0, 0, 0, 1]
    # M5 = (A00+A01)B11
    U_m5 = [1, 1, 0, 0]
    # M6 = (A10-A00)(B00+B01)
    U_m6 = [-1, 0, 1, 0]
    # M7 = (A01-A11)(B10+B11)
    U_m7 = [0, 1, 0, -1]
    
    U_strassen = [U_m1, U_m2, U_m3, U_m4, U_m5, U_m6, U_m7]
    
    # V vectors (coeffs for B)
    # M1: B00+B11 -> [1, 0, 0, 1]
    V_m1 = [1, 0, 0, 1]
    # M2: B00 -> [1, 0, 0, 0]
    V_m2 = [1, 0, 0, 0]
    # M3: B01-B11 -> [0, 1, 0, -1]
    V_m3 = [0, 1, 0, -1]
    # M4: B10-B00 -> [-1, 0, 1, 0]
    V_m4 = [-1, 0, 1, 0]
    # M5: B11 -> [0, 0, 0, 1]
    V_m5 = [0, 0, 0, 1]
    # M6: B00+B01 -> [1, 1, 0, 0]
    V_m6 = [1, 1, 0, 0]
    # M7: B10+B11 -> [0, 0, 1, 1]
    V_m7 = [0, 0, 1, 1]
    
    V_strassen = [V_m1, V_m2, V_m3, V_m4, V_m5, V_m6, V_m7]
    
    # W vectors (coeffs for C)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W for M1: C00 + C11 -> [1, 0, 0, 1]
    W_m1 = [1, 0, 0, 1]
    # W for M2: C10 - C11 -> [0, 0, 1, -1]
    W_m2 = [0, 0, 1, -1]
    # W for M3: C01 + C11 -> [0, 1, 0, 1]
    W_m3 = [0, 1, 0, 1]
    # W for M4: C00 + C10 -> [1, 0, 1, 0]
    W_m4 = [1, 0, 1, 0]
    # W for M5: -C00 + C01 -> [-1, 1, 0, 0]
    W_m5 = [-1, 1, 0, 0]
    # W for M6: C11 -> [0, 0, 0, 1]
    W_m6 = [0, 0, 0, 1]
    # W for M7: C00 -> [1, 0, 0, 0]
    W_m7 = [1, 0, 0, 0]
    
    W_strassen = [W_m1, W_m2, W_m3, W_m4, W_m5, W_m6, W_m7]

    for l in range(num_blocks):
        for q in range(7):
            col = l * 7 + q
            
            # Fill U
            U[:, col] = U_strassen[q]
            
            # Fill V
            v = np.zeros(mp)
            v_coeffs = V_strassen[q]
            # B indices in block l:
            # B00 -> j=0, k=2l
            # B01 -> j=0, k=2l+1
            # B10 -> j=1, k=2l
            # B11 -> j=1, k=2l+1
            v[0 * p + 2*l] = v_coeffs[0]
            v[0 * p + 2*l + 1] = v_coeffs[1]
            v[1 * p + 2*l] = v_coeffs[2]
            v[1 * p + 2*l + 1] = v_coeffs[3]
            V[:, col] = v
            
            # Fill W
            w = np.zeros(pn)
            w_coeffs = W_strassen[q]
            # C indices in block l:
            # C00 -> i=0, k=2l -> idx 2l*n + 0
            # C01 -> i=0, k=2l+1 -> idx (2l+1)*n + 0
            # C10 -> i=1, k=2l -> idx 2l*n + 1
            # C11 -> i=1, k=2l+1 -> idx (2l+1)*n + 1
            w[2*l * n + 0] = w_coeffs[0]
            w[(2*l+1) * n + 0] = w_coeffs[1]
            w[2*l * n + 1] = w_coeffs[2]
            w[(2*l+1) * n + 1] = w_coeffs[3]
            W[:, col] = w
            
    # Verify
    def verify(Uf, Vf, Wf):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += Uf[i*m+j, r] * Vf[j*p+k, r] * Wf[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    U_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U_list]
    V_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V_list]
    W_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
