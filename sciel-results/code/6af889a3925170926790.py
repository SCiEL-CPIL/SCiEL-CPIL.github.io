import numpy as np

def construct():
    n, m, p = 2, 3, 4
    rank = 22
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Helper to add a Strassen block
    # i_off: row offset in A
    # j_off: col offset in A / row offset in B
    # k_off: col offset in B / row offset in W (output index k)
    # Note: W index is k*n + i
    
    def add_strassen(i_off, j_off, k_off, col_start):
        # Indices:
        # A: i_off..i_off+1, j_off..j_off+1
        # B: j_off..j_off+1, k_off..k_off+1
        # C: i_off..i_off+1, k_off..k_off+1
        
        # Strassen terms
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # U coefficients (A)
        # V coefficients (B)
        # W coefficients (C)
        
        terms = [
            # (U, V, W)
            # M1
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1, (0,1):-1, (1,0):-1}), # Wait, W maps to C_{k,i}
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # Let's define W for each term based on the final sum.
            # It's easier to define the contribution to C for each M.
        ]
        
        # Let's just hardcode the 7 terms' contributions to U, V, W.
        # U is (2x2) -> (4,)
        # V is (2x2) -> (4,)
        # W is (2x2) -> (4,) (indices k, i)
        
        # Term 1
        u = [1, 0, 1, 0] # A00, A01, A10, A11 -> (0,0)=1, (1,1)=1
        v = [1, 0, 0, 1] # B00, B01, B10, B11 -> (0,0)=1, (1,1)=1
        w = [1, 1, -1, -1] # C00, C01, C10, C11? No, W index is k*n+i.
        # n=2.
        # C00 (k=0, i=0) -> idx 0
        # C01 (k=0, i=1) -> idx 1
        # C10 (k=1, i=0) -> idx 2
        # C11 (k=1, i=1) -> idx 3
        # C00 = M1 ... so w[0] += 1
        # C11 = M1 ... so w[3] += 1
        # C01 = -M5 ... 
        # C10 = -M5 ...
        # Wait, M1 contributes to C00 (+1) and C11 (+1).
        w = [1, 0, 0, 1]
        
        # Term 2: M2 = (A10+A11)B00
        u = [0, 0, 1, 1]
        v = [1, 0, 0, 0]
        # C10 = M2 (+1), C11 = -M2 (-1)
        w = [0, 0, 1, -1]
        
        # Term 3: M3 = A00(B01-B11)
        u = [1, 0, 0, 0]
        v = [0, 1, 0, -1]
        # C01 = M3 (+1), C11 = M3 (+1)
        w = [0, 1, 0, 1]
        
        # Term 4: M4 = A11(B10-B00)
        u = [0, 0, 0, 1]
        v = [0, 0, 1, -1] # B10 - B00
        # C00 = M4 (+1), C10 = M4 (+1)
        w = [1, 0, 1, 0]
        
        # Term 5: M5 = (A00+A01)B11
        u = [1, 1, 0, 0]
        v = [0, 0, 0, 1]
        # C00 = -M5 (-1), C01 = M5 (+1)
        w = [-1, 1, 0, 0]
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        u = [0, 0, 1, -1]
        v = [1, 1, 0, 0]
        # C11 = M6 (+1)
        w = [0, 0, 0, 1]
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        u = [0, 1, 0, -1]
        v = [0, 0, 1, 1]
        # C00 = M7 (+1)
        w = [1, 0, 0, 0]
        
        all_terms = [
            (u, v, w),
            (u, v, w), # placeholder
        ]
        # I will list them properly in the code.
        
        terms_list = [
            ([1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
        ]
        
        for t_idx, (u_t, v_t, w_t) in enumerate(terms_list):
            c = col_start + t_idx
            # Fill U
            for idx, val in enumerate(u_t):
                if val != 0:
                    i_local, j_local = idx // 2, idx % 2
                    i_global = i_off + i_local
                    j_global = j_off + j_local
                    U[i_global * m + j_global][c] = val
            # Fill V
            for idx, val in enumerate(v_t):
                if val != 0:
                    j_local, k_local = idx // 2, idx % 2
                    j_global = j_off + j_local
                    k_global = k_off + k_local
                    V[j_global * p + k_global][c] = val
            # Fill W
            for idx, val in enumerate(w_t):
                if val != 0:
                    k_local, i_local = idx // 2, idx % 2
                    k_global = k_off + k_local
                    i_global = i_off + i_local
                    W[k_global * n + i_global][c] = val

    # Part 1: A[0:2, 0:2] x B[0:2, 0:2]
    add_strassen(0, 0, 0, 0)
    
    # Part 2: A[0:2, 0:2] x B[0:2, 2:4]
    add_strassen(0, 0, 2, 7)
    
    # Part 3: A[0:2, 2:3] x B[2:3, 0:4]
    # C_{i,k} = A_{i,2} B_{2,k}
    # 8 terms.
    col = 14
    for i in range(2):
        for k in range(4):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
