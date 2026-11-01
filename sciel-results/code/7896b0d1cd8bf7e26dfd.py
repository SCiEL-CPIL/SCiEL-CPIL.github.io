import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 6
    
    # Rank 42 solution based on Strassen blocks
    rank = 42
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # We split m=4 into two blocks of size 2: j in {0,1} and j in {2,3}
    # We split p=6 into three blocks of size 2: k in {0,1}, {2,3}, {4,5}
    # For each (j_block, k_block) pair, we use Strassen for 2x2 @ 2x2.
    
    # Strassen 2x2 @ 2x2
    # Indices: A(i,j), B(j,k) -> C(i,k)
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # Standard Strassen for 2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 - M5 + M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # strassen_U[prod_idx] maps local 2x2 index (0..3) to coefficient
    # local index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_U = [
        {0: 1, 3: 1},       # M1: A00 + A11
        {2: 1, 3: 1},       # M2: A10 + A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00 + A01
        {2: 1, 0: -1},      # M6: A10 - A00
        {1: 1, 3: -1},      # M7: A01 - A11
    ]
    strassen_V = [
        {0: 1, 3: 1},       # M1: B00 + B11
        {0: 1},             # M2: B00
        {1: 1, 3: -1},      # M3: B01 - B11
        {2: 1, 0: -1},      # M4: B10 - B00
        {3: 1},             # M5: B11
        {0: 1, 1: 1},       # M6: B00 + B01
        {2: 1, 3: 1},       # M7: B10 + B11
    ]
    strassen_W = [
        {0: 1, 3: -1, 2: 1, 1: 1},   # C00 = M1 - M5 + M4 + M7
        {1: 1, 3: 1},                 # C01 = M3 + M5
        {2: 1, 0: 1},                 # C10 = M2 + M4
        {0: 1, 2: -1, 1: 1, 3: 1},   # C11 = M1 - M2 + M3 + M6
    ]
    
    # Wait, the W matrix should have one entry per (i,k) per product.
    # Let's re-verify the standard Strassen output reconstruction.
    # C00 = M1 - M5 + M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for each product idx (0..6), we add coeff * U_prod * V_prod to W.
    # W[i,k] += coeff
    
    # Let's define W coefficients for each product idx and each output (i,k)
    # w_coeff[prod_idx][local_ik] = coeff
    # local_ik: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    w_coeff = [
        {0: 1, 1: 0, 2: 0, 3: 0},  # M1 contributes to C00
        {0: 0, 1: 0, 2: 0, 3: -1}, # M2 contributes to C11
        {0: 0, 1: 1, 2: 0, 3: 1},  # M3 contributes to C01, C11
        {0: 1, 1: 0, 2: 1, 3: 0},  # M4 contributes to C00, C10
        {0: -1, 1: 1, 2: 0, 3: 0}, # M5 contributes to C00, C01
        {0: 0, 1: 0, 2: 0, 3: 1},  # M6 contributes to C11
        {0: 1, 1: 0, 2: 0, 3: 0},  # M7 contributes to C00
    ]
    
    # Wait, let's double check C00.
    # C00 = M1 - M5 + M4 + M7
    # M1: +1
    # M2: 0
    # M3: 0
    # M4: +1
    # M5: -1
    # M6: 0
    # M7: +1
    # So w_coeff for C00 (local 0):
    # [1, 0, 0, 1, -1, 0, 1]
    
    # Let's build the W matrix directly from the formulas.
    # W row index = k*n + i
    # For local i, k: global i = local_i, global k = local_k
    # But we are in a block, so global i = local_i, global k = k_block*2 + local_k
    
    col = 0
    for j_block in range(2): # j=0,1 and j=2,3
        for k_block in range(3): # k=0,1 and k=2,3 and k=4,5
            for prod_idx in range(7):
                r = col + prod_idx
                
                # U part: A[i, j]
                # i in {0,1}, j in {j_block*2, j_block*2+1}
                for local_idx, sign in strassen_U[prod_idx].items():
                    # local_idx 0..3 for the 2x2 matrix A
                    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                    gi = local_idx // 2
                    gj = local_idx % 2
                    U[gi * m + (j_block * 2 + gj), r] = sign
                
                # V part: B[j, k]
                # j in {j_block*2, j_block*2+1}, k in {k_block*2, k_block*2+1}
                for local_idx, sign in strassen_V[prod_idx].items():
                    # local_idx 0..3 for the 2x2 matrix B
                    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                    gj = local_idx // 2
                    gk = local_idx % 2
                    V[(j_block * 2 + gj) * p + (k_block * 2 + gk), r] = sign
                
                # W part: C[i, k]
                # i in {0,1}, k in {k_block*2, k_block*2+1}
                for local_idx, sign in w_coeff[prod_idx].items():
                    if sign != 0:
                        gi = local_idx // 2
                        gk = local_idx % 2
                        W[(k_block * 2 + gk) * n + gi, r] = sign
            
            col += 7
            
    if time.time() - start_time > 110:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
