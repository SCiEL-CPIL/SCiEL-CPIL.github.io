import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 6
    rank = 42
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen 2x2 factors
    # U (A): indices 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V (B): indices 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W (C): indices 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # P1: (a+d)(e+h)
    # P2: (c+d)e
    # P3: a(f-h)
    # P4: d(g-e)
    # P5: (a+b)h
    # P6: (c-a)(e+f)
    # P7: (b-d)(g+h)
    
    strassen_U = [
        {0: 1, 3: 1},       # P1
        {2: 1, 3: 1},       # P2
        {0: 1},             # P3
        {3: 1},             # P4
        {0: 1, 1: 1},       # P5
        {0: -1, 2: 1},      # P6
        {1: 1, 3: -1},      # P7
    ]
    strassen_V = [
        {0: 1, 3: 1},       # P1
        {0: 1},             # P2
        {1: 1, 3: -1},      # P3
        {2: 1, 0: -1},      # P4
        {3: 1},             # P5
        {0: 1, 1: 1},       # P6
        {2: 1, 3: 1},       # P7
    ]
    strassen_W = [
        {0: 1, 3: 1},       # P1
        {2: 1, 3: -1},      # P2
        {1: 1, 3: 1},       # P3
        {0: 1, 2: 1},       # P4
        {0: -1, 1: 1},      # P5
        {3: 1},             # P6
        {0: 1},             # P7
    ]
    
    col = 0
    # Blocks of A: 2 (left, right)
    # Blocks of B: 3 (top-mid-bottom? No, B is 4x6. Split rows into 2+2. Split cols into 2+2+2)
    # A is 2x4. Split cols into 2+2.
    # B is 4x6. Split rows into 2+2. Split cols into 2+2+2.
    
    # Term: A_i B_j
    # i in {0,1} corresponds to A cols 2i, 2i+1
    # j in {0,1} corresponds to B rows 2j, 2j+1
    # We only have terms where the A block and B block "match" in the summation index?
    # C = sum_k A_{:,k} B_{k,:}
    # A_{:,0} is cols 0,1. B_{0,:} is row 0.
    # A_{:,1} is cols 2,3. B_{1,:} is row 1.
    # A_{:,2} is cols 4,5? No, A is 2x4. Cols 0,1,2,3.
    # B is 4x6. Rows 0,1,2,3.
    # Sum over k=0 to 3.
    # Term k=0: A[:, 0:1] (2x1) * B[0, :] (1x6)
    # Term k=1: A[:, 1:2] (2x1) * B[1, :] (1x6)
    # Term k=2: A[:, 2:3] (2x1) * B[2, :] (1x6)
    # Term k=3: A[:, 3:4] (2x1) * B[3, :] (1x6)
    
    # This grouping is not 2x2 blocks directly.
    # Let's stick to the 2x2 block multiplication.
    # A = [A_L, A_R] (2x2 each)
    # B = [B_T; B_B] (2x6 each)
    # C = A_L B_T + A_R B_B
    # A_L B_T: (2x2) * (2x6). Split B_T into 3 (2x2) blocks.
    # A_R B_B: (2x2) * (2x6). Split B_B into 3 (2x2) blocks.
    
    # Total 6 Strassen blocks.
    
    for i_block in range(2): # A_L (0), A_R (1)
        for j_block in range(2): # B_T (0), B_B (1)
            # Only i_block == j_block contributes?
            # A_L multiplies B_T (rows 0,1 of B).
            # A_R multiplies B_B (rows 2,3 of B).
            # Yes, the sum is A_L B_T + A_R B_B.
            # So we iterate i_block in 0..1, and j_block must equal i_block?
            # No, A_L is cols 0,1 of A. It multiplies rows 0,1 of B.
            # A_R is cols 2,3 of A. It multiplies rows 2,3 of B.
            # So yes, i_block and j_block are linked.
            
            if i_block != j_block:
                continue
                
            # Now we have a 2x2 A part and a 2x6 B part.
            # Split B part into 3 2x2 blocks.
            for k_block in range(3):
                for prod_idx in range(7):
                    r = col + prod_idx
                    
                    # U (A)
                    # A is 2x2 local.
                    # Global A indices:
                    # i (row) in 0..1
                    # j (col) in i_block*2 .. i_block*2+1
                    for local_idx, sign in strassen_U[prod_idx].items():
                        li = local_idx // 2
                        lj = local_idx % 2
                        gi = li
                        gj = i_block * 2 + lj
                        U[gi * m + gj, r] = sign
                    
                    # V (B)
                    # B is 2x2 local.
                    # Global B indices:
                    # j (row) in j_block*2 .. j_block*2+1
                    # k (col) in k_block*2 .. k_block*2+1
                    for local_idx, sign in strassen_V[prod_idx].items():
                        lj = local_idx // 2
                        lk = local_idx % 2
                        gj = j_block * 2 + lj
                        gk = k_block * 2 + lk
                        V[gj * p + gk, r] = sign
                    
                    # W (C)
                    # C is 2x2 local.
                    # Global C indices:
                    # i (row) in 0..1
                    # k (col) in k_block*2 .. k_block*2+1
                    # W index: k*n + i
                    for local_idx, sign in strassen_W[prod_idx].items():
                        li = local_idx // 2
                        lk = local_idx % 2
                        gi = li
                        gk = k_block * 2 + lk
                        W[gk * n + gi, r] = sign
                    
                col += 7
    
    if time.time() - start_time > 110:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
