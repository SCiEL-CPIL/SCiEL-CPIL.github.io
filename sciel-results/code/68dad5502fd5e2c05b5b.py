import numpy as np

def construct():
    n, m, p = 2, 2, 5
    rank = 18
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # --- Block 1: k in {0, 1} (Strassen) ---
    # Strassen 2x2x2
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # U_s (4x7)
    U_s = np.zeros((4, 7))
    # a00, a01, a10, a11 indices: 0, 1, 2, 3
    U_s[0, 0] = 1; U_s[3, 0] = 1 # a00+a11
    U_s[2, 1] = 1; U_s[3, 1] = 1 # a10+a11
    U_s[0, 2] = 1 # a00
    U_s[3, 3] = 1 # a11
    U_s[0, 4] = 1; U_s[1, 4] = 1 # a00+a01
    U_s[2, 5] = 1; U_s[0, 5] = -1 # a10-a00
    U_s[1, 6] = 1; U_s[3, 6] = -1 # a01-a11
    
    # V_s (4x7) for 2x2 B
    V_s = np.zeros((4, 7))
    # b00, b01, b10, b11 indices: 0, 1, 2, 3
    V_s[0, 0] = 1; V_s[3, 0] = 1 # b00+b11
    V_s[0, 1] = 1 # b00
    V_s[1, 2] = 1; V_s[3, 2] = -1 # b01-b11
    V_s[2, 3] = 1; V_s[0, 3] = -1 # b10-b00
    V_s[3, 4] = 1 # b11
    V_s[0, 5] = 1; V_s[1, 5] = 1 # b00+b01
    V_s[2, 6] = 1; V_s[3, 6] = 1 # b10+b11
    
    # W_s (4x7) for 2x2 C (trace convention W[k*n+i])
    # C00, C01, C10, C11
    # W indices: k*2+i
    # M1 -> C00+=, C01-=, C10-=, C11+=
    W_s = np.zeros((4, 7))
    W_s[0, 0] = 1; W_s[1, 0] = -1; W_s[2, 0] = -1; W_s[3, 0] = 1
    # M2 -> C01+=, C10+=, C11+=
    W_s[1, 1] = 1; W_s[2, 1] = 1; W_s[3, 1] = 1
    # M3 -> C00+=
    W_s[0, 2] = 1
    # M4 -> C11+=
    W_s[3, 3] = 1
    # M5 -> C00+=, C10+=
    W_s[0, 4] = 1; W_s[2, 4] = 1
    # M6 -> C00+=, C01+=
    W_s[0, 5] = 1; W_s[1, 5] = 1
    # M7 -> C01+=, C10+=
    W_s[1, 6] = 1; W_s[2, 6] = 1
    
    # Assign Block 1 to global
    # U[:, 0:7] = U_s
    U[:, 0:7] = U_s
    # V rows for k in {0,1} are j*5+0 and j*5+1
    # V_s row j*2+k.
    # j=0, k=0 -> V_s[0] -> V[0]
    # j=0, k=1 -> V_s[1] -> V[1]
    # j=1, k=0 -> V_s[2] -> V[5]
    # j=1, k=1 -> V_s[3] -> V[6]
    V[0, 0:7] = V_s[0]
    V[1, 0:7] = V_s[1]
    V[5, 0:7] = V_s[2]
    V[6, 0:7] = V_s[3]
    
    # W rows for k in {0,1} are k*2+i
    # W_s row k*2+i (since n=2)
    # k=0, i=0 -> W_s[0] -> W[0]
    # k=0, i=1 -> W_s[1] -> W[1]
    # k=1, i=0 -> W_s[2] -> W[2]
    # k=1, i=1 -> W_s[3] -> W[3]
    W[0, 0:7] = W_s[0]
    W[1, 0:7] = W_s[1]
    W[2, 0:7] = W_s[2]
    W[3, 0:7] = W_s[3]
    
    # --- Block 2: k in {2, 3} (Strassen) ---
    U[:, 7:14] = U_s
    # V rows for k in {2,3}
    # j=0, k=2 -> V[2]
    # j=0, k=3 -> V[3]
    # j=1, k=2 -> V[7]
    # j=1, k=3 -> V[8]
    V[2, 7:14] = V_s[0]
    V[3, 7:14] = V_s[1]
    V[7, 7:14] = V_s[2]
    V[8, 7:14] = V_s[3]
    
    # W rows for k in {2,3}
    # k=2, i=0 -> W[4]
    # k=2, i=1 -> W[5]
    # k=3, i=0 -> W[6]
    # k=3, i=1 -> W[7]
    W[4, 7:14] = W_s[0]
    W[5, 7:14] = W_s[1]
    W[6, 7:14] = W_s[2]
    W[7, 7:14] = W_s[3]
    
    # --- Block 3: k in {4} (Schoolbook) ---
    # Rank 4
    # C04 = a00 b04 + a01 b14
    # C14 = a10 b04 + a11 b14
    # Terms:
    # 1. a00 b04
    # 2. a01 b14
    # 3. a10 b04
    # 4. a11 b14
    
    U_b = np.zeros((4, 4))
    U_b[0, 0] = 1 # a00
    U_b[1, 1] = 1 # a01
    U_b[2, 2] = 1 # a10
    U_b[3, 3] = 1 # a11
    
    V_b = np.zeros((2, 4)) # j*1+k, k=0
    V_b[0, 0] = 1 # b04 (j=0)
    V_b[1, 1] = 1 # b14 (j=1)
    V_b[0, 2] = 1 # b04 (j=0)
    V_b[1, 3] = 1 # b14 (j=1)
    
    W_b = np.zeros((2, 4)) # k*2+i, k=0
    W_b[0, 0] = 1 # i=0 (C04)
    W_b[1, 1] = 1 # i=1 (C14)
    W_b[0, 2] = 1 # i=0 (C04)
    W_b[1, 3] = 1 # i=1 (C14)
    
    U[:, 14:18] = U_b
    # V rows for k=4
    # j=0, k=4 -> V[4]
    # j=1, k=4 -> V[9]
    V[4, 14:18] = V_b[0]
    V[9, 14:18] = V_b[1]
    
    # W rows for k=4
    # k=4, i=0 -> W[8]
    # k=4, i=1 -> W[9]
    W[8, 14:18] = W_b[0]
    W[9, 14:18] = W_b[1]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-START
def construct():
    # Placeholder to satisfy structure if needed, but we replace the whole block
    pass
# EVOLVE-BLOCK-END
