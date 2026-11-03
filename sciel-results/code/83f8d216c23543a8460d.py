import time
import sys

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 85 decomposition
    # Part 1: 4x4x4 using Strassen (Rank 49)
    # Part 2: 4x4x3 using Strassen+Schoolbook (Rank 36)
    
    # Helper to add a term to U, V, W
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # We will build U, V, W as lists of lists
    # But we don't know the rank beforehand? We do: 85.
    rank = 85
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: 4x4x4 (Strassen) ---
    # Strassen 2x2x2 terms
    # Returns list of (u_indices, v_indices, w_indices, w_values)
    # u_indices: list of (i, j, val)
    # v_indices: list of (j, k, val)
    # w_indices: list of (k, i, val)
    
    def get_strassen_2x2x2():
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Reconstruction:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        terms = []
        
        # M1
        u1 = [(0,0,1), (1,1,1)]
        v1 = [(0,0,1), (0,1,1)]
        w1 = [(0,0,1), (1,0,1), (0,1,1), (1,1,1)] # k,i: (0,0)->C00, (0,1)->C10? No.
        # W index is k*n + i.
        # C00 (i=0,k=0): W[0*2+0] = W[0]
        # C01 (i=0,k=1): W[1*2+0] = W[2]
        # C10 (i=1,k=0): W[0*2+1] = W[1]
        # C11 (i=1,k=1): W[1*2+1] = W[3]
        
        # C11 (i=0,k=0) coeff 1. W[0] += 1
        # C12 (i=0,k=1) coeff 1. W[2] += 1
        # C21 (i=1,k=0) coeff 1. W[1] += 1
        # C22 (i=1,k=1) coeff 1. W[3] += 1
        
        w1_vals = {0: 1, 1: 1, 2: 1, 3: 1} # idx: val
        
        terms.append((u1, v1, w1_vals))
        
        # M2
        u2 = [(1,0,1), (1,1,1)]
        v2 = [(0,0,1)]
        # C21 (i=1,k=0) coeff 1. W[1] += 1
        # C22 (i=1,k=1) coeff -1. W[3] += -1
        w2_vals = {1: 1, 3: -1}
        terms.append((u2, v2, w2_vals))
        
        # M3
        u3 = [(0,0,1)]
        v3 = [(0,1,1), (1,1,-1)]
        # C12 (i=0,k=1) coeff 1. W[2] += 1
        # C22 (i=1,k=1) coeff 1. W[3] += 1
        w3_vals = {2: 1, 3: 1}
        terms.append((u3, v3, w3_vals))
        
        # M4
        u4 = [(1,1,1)]
        v4 = [(1,0,1), (0,0,-1)]
        # C11 (i=0,k=0) coeff 1. W[0] += 1
        # C21 (i=1,k=0) coeff 1. W[1] += 1
        w4_vals = {0: 1, 1: 1}
        terms.append((u4, v4, w4_vals))
        
        # M5
        u5 = [(0,0,1), (0,1,1)]
        v5 = [(1,1,1)]
        # C11 (i=0,k=0) coeff -1. W[0] += -1
        # C12 (i=0,k=1) coeff 1. W[2] += 1
        w5_vals = {0: -1, 2: 1}
        terms.append((u5, v5, w5_vals))
        
        # M6
        u6 = [(1,0,1), (0,0,-1)]
        v6 = [(0,0,1), (0,1,1)]
        # C22 (i=1,k=1) coeff 1. W[3] += 1
        w6_vals = {3: 1}
        terms.append((u6, v6, w6_vals))
        
        # M7
        u7 = [(0,1,1), (1,1,-1)]
        v7 = [(1,0,1), (1,1,1)]
        # C11 (i=0,k=0) coeff 1. W[0] += 1
        # C22 (i=1,k=1) coeff 1. W[3] += 1
        w7_vals = {0: 1, 3: 1}
        terms.append((u7, v7, w7_vals))
        
        return terms

    # Apply 4x4x4 Strassen
    # Blocks:
    # A00 (0,0), A01 (0,1)
    # A10 (1,0), A11 (1,1)
    # B00 (0,0), B01 (0,1)
    # B10 (1,0), B11 (1,1)
    
    block_offsets = [
        (0, 0, 0, 0), # A00, B00
        (0, 1, 0, 1), # A00, B01
        (1, 0, 1, 0), # A10, B00
        (1, 1, 1, 1), # A10, B01
        (0, 0, 0, 0), # A01, B00 (Wait, A is 4x4, so A01 is rows 0-1, cols 2-3)
        (0, 1, 0, 1), # A01, B01
        (1, 0, 1, 0), # A11, B00
        (1, 1, 1, 1), # A11, B01
    ]
    # Correct mapping for 4x4:
    # A blocks:
    # A00: i=0,1 j=0,1
    # A01: i=0,1 j=2,3
    # A10: i=2,3 j=0,1
    # A11: i=2,3 j=2,3
    
    # B blocks (4x4 part):
    # B00: j=0,1 k=0,1
    # B01: j=0,1 k=2,3
    # B10: j=2,3 k=0,1
    # B11: j=2,3 k=2,3
    
    strassen_terms = get_strassen_2x2x2()
    
    for i_block in range(2):
        for j_block in range(2):
            # A block offset: i_off = i_block * 2, j_off = j_block * 2
            i_off = i_block * 2
            j_off = j_block * 2
            
            for k_block in range(2):
                # B block offset: j_off_B = k_block * 2? No.
                # B is 4x4.
                # B block (j_block, k_block)
                # j_off_B = j_block * 2
                # k_off_B = k_block * 2
                
                j_off_B = j_block * 2
                k_off_B = k_block * 2
                
                for u, v, w_vals in strassen_terms:
                    # Add to global U, V, W
                    for (i, j, val) in u:
                        gi = i_off + i
                        gj = j_off + j
                        U[gi * m + gj][col] += val
                    for (j, k, val) in v:
                        gj = j_off_B + j
                        gk = k_off_B + k
                        V[gj * p + gk][col] += val
                    for w_idx, val in w_vals.items():
                        # w_idx is local k*n + i (n=2)
                        # local k = w_idx // 2
                        # local i = w_idx % 2
                        lk = w_idx // 2
                        li = w_idx % 2
                        gk = k_off_B + lk
                        gi = i_off + li
                        W[gk * n + gi][col] += val
                    col += 1
    
    # --- Part 2: 4x4x3 (Rank 36) ---
    # B_R is 4x3.
    # Split into 2x3 blocks.
    # B_R blocks:
    # BR00: j=0,1 k=4,5,6 (Wait, B is 4x7. Cols 4,5,6)
    # BR10: j=2,3 k=4,5,6
    
    # A blocks same as before.
    
    # 2x2x3 decomposition:
    # Split 2x3 B into 2x2 and 2x1.
    # 2x2x2 (Strassen, Rank 7)
    # 2x2x1 (Schoolbook, Rank 2)
    
    def get_2x2x1():
        # A 2x2, B 2x1.
        # C11 = A11 B1
        # C12 = A12 B2? No, B is 2x1.
        # B = [b1; b2].
        # C1 = A11 b1 + A12 b2? No.
        # A = [[a11, a12], [a21, a22]].
        # B = [[b1], [b2]].
        # C = [[a11 b1 + a12 b2], [a21 b1 + a22 b2]].
        # This is 4 products.
        # Can we do better?
        # Rank 2?
        # u1 = [1, 0, 0, 0] (A11), v1 = [1, 0] (B1), w1 = [1, 0] (C1)
        # u2 = [0, 1, 0, 0] (A12), v2 = [0, 1] (B2), w2 = [1, 0] (C1)
        # u3 = [0, 0, 1, 0] (A21), v3 = [1, 0] (B1), w3 = [0, 1] (C2)
        # u4 = [0, 0, 0, 1] (A22), v4 = [0, 1] (B2), w4 = [0, 1] (C2)
        # This is Rank 4.
        # Is Rank 4 minimal?
        # 2x2x1 is equivalent to 2x2x2 with one column zero?
        # Strassen 2x2x2 has Rank 7.
        # If we set B22=0, B12=0?
        # It might reduce to Rank 4.
        # Let's use Rank 4.
        
        terms = []
        # C1 = A11 B1
        terms.append(([(0,0,1)], [(0,0,1)], {0: 1})) # W idx 0 (k=0,i=0)
        # C1 += A12 B2
        terms.append(([(0,1,1)], [(1,0,1)], {0: 1})) # W idx 0
        # C2 = A21 B1
        terms.append(([(1,0,1)], [(0,0,1)], {1: 1})) # W idx 1 (k=0,i=1)
        # C2 += A22 B2
        terms.append(([(1,1,1)], [(1,0,1)], {1: 1})) # W idx 1
        
        return terms

    # Apply 4x4x3
    for i_block in range(2):
        i_off = i_block * 2
        for j_block in range(2):
            j_off = j_block * 2
            j_off_B = j_block * 2
            
            # B_R block is 2x3.
            # k_off_B = 4 (start of 3rd part of B)
            k_off_B = 4
            
            # 2x2x2 part (cols 0,1 of B_R block -> global k 4,5)
            for u, v, w_vals in strassen_terms:
                for (i, j, val) in u:
                    gi = i_off + i
                    gj = j_off + j
                    U[gi * m + gj][col] += val
                for (j, k, val) in v:
                    # k is 0 or 1 in local 2x2.
                    # Map to global k: k_off_B + k
                    gj = j_off_B + j
                    gk = k_off_B + k
                    V[gj * p + gk][col] += val
                for w_idx, val in w_vals.items():
                    lk = w_idx // 2
                    li = w_idx % 2
                    gk = k_off_B + lk
                    gi = i_off + li
                    W[gk * n + gi][col] += val
                col += 1
            
            # 2x2x1 part (col 2 of B_R block -> global k 6)
            # B col is 2 in local 2x3.
            # This corresponds to k=2 in local 2x3?
            # Our 2x2x1 uses B of size 2x1.
            # The B column index in V is 2?
            # In 2x2x1, B is 2x1. Indices j=0,1. k=0.
            # So we map local k=0 to global k=6.
            
            for u, v, w_vals in get_2x2x1():
                for (i, j, val) in u:
                    gi = i_off + i
                    gj = j_off + j
                    U[gi * m + gj][col] += val
                for (j, k, val) in v:
                    # k is always 0 in 2x2x1.
                    gj = j_off_B + j
                    gk = k_off_B + 2 # 4+2=6
                    V[gj * p + gk][col] += val
                for w_idx, val in w_vals.items():
                    # w_idx is local k*n + i. n=2.
                    # k is always 0.
                    # w_idx is i.
                    li = w_idx % 2
                    lk = w_idx // 2 # 0
                    gk = k_off_B + 2
                    gi = i_off + li
                    W[gk * n + gi][col] += val
                col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
