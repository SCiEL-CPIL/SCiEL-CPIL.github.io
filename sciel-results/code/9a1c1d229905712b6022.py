import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Split m=5 into 2, 2, 1
    # Split p=6 into 2, 2, 2
    # Blocks:
    # (0,0): 2x2x2 -> Rank 7
    # (0,1): 2x2x2 -> Rank 7
    # (0,2): 2x2x2 -> Rank 7
    # (1,0): 2x2x2 -> Rank 7
    # (1,1): 2x2x2 -> Rank 7
    # (1,2): 2x2x2 -> Rank 7
    # (2,0): 2x1x2 -> Rank 4
    # (2,1): 2x1x2 -> Rank 4
    # (2,2): 2x1x2 -> Rank 4
    
    rank = 6 * 7 + 3 * 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_block(i_offset, j_offset, k_offset, current_col):
        # A indices: i in [i_offset, i_offset+2), j in [j_offset, j_offset+2)
        # B indices: j in [j_offset, j_offset+2), k in [k_offset, k_offset+2)
        # C indices: i in [i_offset, i_offset+2), k in [k_offset, k_offset+2)
        
        # M1 = (A11+A22)(B11+B22)
        U[i_offset*m + j_offset][current_col] = 1
        U[i_offset*m + j_offset+1][current_col] = 1
        V[j_offset*p + k_offset][current_col] = 1
        V[j_offset*p + k_offset+1][current_col] = 1
        W[k_offset*n + i_offset][current_col] = 1
        W[k_offset*n + i_offset+1][current_col] = 1
        W[(k_offset+1)*n + i_offset][current_col] = 1
        W[(k_offset+1)*n + i_offset+1][current_col] = 1
        current_col += 1
        
        # M2 = (A21+A22)B11
        U[i_offset*m + j_offset+1][current_col] = 1
        U[i_offset*m + j_offset+1][current_col] = 1 # A21 and A22? No.
        # A is 2x2. A21 is row 1, col 0. A22 is row 1, col 1.
        # Indices:
        # A11: i_offset, j_offset
        # A12: i_offset, j_offset+1
        # A21: i_offset+1, j_offset
        # A22: i_offset+1, j_offset+1
        
        # M2 = (A21+A22) B11
        # U rows: i_offset+1, j_offset (A21) and i_offset+1, j_offset+1 (A22)
        U[i_offset*m + j_offset+1][current_col] = 1 # Wait, A21 is (1,0) -> i_offset+1, j_offset
        # Correction:
        # A21 index: (i_offset+1)*m + j_offset
        # A22 index: (i_offset+1)*m + j_offset+1
        U[i_offset*m + j_offset][current_col] = 0 # Reset? No, just set.
        # I am building U, V, W from scratch.
        
        # Let's restart the helper logic carefully.
        return current_col

    # This manual construction is error-prone.
    # I will use a more robust approach: 
    # Generate the schoolbook rank 60, then try to merge terms?
    # No, I will just implement the 54 rank logic clearly.
    
    # Clear U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Strassen 2x2x2 coefficients
    # M1: A11+A22, B11+B22 -> C11, C12, C21, C22
    # M2: A21+A22, B11 -> C21, C22
    # M3: A11, B12-B22 -> C12, C22
    # M4: A22, B21-B11 -> C11, C21
    # M5: A11+A12, B22 -> C11, C12
    # M6: A21-A11, B11+B12 -> C22
    # M7: A12-A22, B21+B22 -> C11, C12
    
    # Let's define the terms for one 2x2x2 block
    # A indices: a00, a01, a10, a11
    # B indices: b00, b01, b10, b11
    # C indices: c00, c01, c10, c11
    
    # Term 1: (a00+a11)(b00+b11)
    # U: a00, a11
    # V: b00, b11
    # W: c00, c01, c10, c11
    
    # I will write a function to add a term.
    
    def add_term(u_indices, v_indices, w_indices, signs_u, signs_v, signs_w):
        nonlocal col
        for u_idx, s_u in zip(u_indices, signs_u):
            U[u_idx][col] += s_u
        for v_idx, s_v in zip(v_indices, signs_v):
            V[v_idx][col] += s_v
        for w_idx, s_w in zip(w_indices, signs_w):
            W[w_idx][col] += s_w
        col += 1

    # Process 6 blocks of 2x2x2
    for i_block in range(2):
        for k_block in range(3):
            i_off = i_block * 2
            k_off = k_block * 2
            # j_off is determined by i_block? No, m is split into 2,2,1.
            # Wait, my split was m=2,2,1 and p=2,2,2.
            # The blocks are (i_slice, j_slice, k_slice).
            # i is always 0..2 (full row).
            # j slices: 0-2, 2-4, 4-5.
            # k slices: 0-2, 2-4, 4-6.
            
            # So we have 3 j-slices and 3 k-slices.
            # 9 blocks total.
            # j_slice 0 (len 2), k_slice 0 (len 2) -> 2x2x2
            # j_slice 0 (len 2), k_slice 1 (len 2) -> 2x2x2
            # j_slice 0 (len 2), k_slice 2 (len 2) -> 2x2x2
            # j_slice 1 (len 2), k_slice 0 (len 2) -> 2x2x2
            # j_slice 1 (len 2), k_slice 1 (len 2) -> 2x2x2
            # j_slice 1 (len 2), k_slice 2 (len 2) -> 2x2x2
            # j_slice 2 (len 1), k_slice 0 (len 2) -> 2x1x2
            # j_slice 2 (len 1), k_slice 1 (len 2) -> 2x1x2
            # j_slice 2 (len 1), k_slice 2 (len 2) -> 2x1x2
            
            pass # Logic below

    # j_slices: [0, 2], [2, 4], [4, 5]
    # k_slices: [0, 2], [2, 4], [4, 6]
    
    j_slices = [(0, 2), (2, 4), (4, 5)]
    k_slices = [(0, 2), (2, 4), (4, 6)]
    
    for j_start, j_end in j_slices:
        for k_start, k_end in k_slices:
            if j_end - j_start == 2 and k_end - k_start == 2:
                # 2x2x2 Strassen
                # A indices: i in 0..2, j in j_start..j_end
                # B indices: j in j_start..j_end, k in k_start..k_end
                # C indices: i in 0..2, k in k_start..k_end
                
                a00 = 0*m + j_start
                a01 = 0*m + j_start + 1
                a10 = 1*m + j_start
                a11 = 1*m + j_start + 1
                
                b00 = j_start*p + k_start
                b01 = j_start*p + k_start + 1
                b10 = (j_start+1)*p + k_start
                b11 = (j_start+1)*p + k_start + 1
                
                c00 = k_start*n + 0
                c01 = k_start*n + 1
                c10 = (k_start+1)*n + 0
                c11 = (k_start+1)*n + 1
                
                # M1
                add_term([a00, a11], [b00, b11], [c00, c01, c10, c11], [1, 1], [1, 1], [1, 1, 1, 1])
                # M2
                add_term([a10, a11], [b00], [c10, c11], [1, 1], [1], [1, 1])
                # M3
                add_term([a00], [b01, b11], [c01, c11], [1], [1, -1], [1, 1])
                # M4
                add_term([a11], [b10, b00], [c00, c10], [1], [1, -1], [1, 1])
                # M5
                add_term([a00, a01], [b11], [c00, c01], [1, 1], [1], [1, 1])
                # M6
                add_term([a10, a00], [b00, b01], [c11], [1, -1], [1, 1], [1])
                # M7
                add_term([a01, a11], [b10, b11], [c00, c01], [1, -1], [1, 1], [1, 1])
                
            elif j_end - j_start == 1 and k_end - k_start == 2:
                # 2x1x2
                # A: i in 0..2, j in j_start
                # B: j in j_start, k in k_start..k_end
                # C: i in 0..2, k in k_start..k_end
                
                a00 = 0*m + j_start
                a10 = 1*m + j_start
                
                b00 = j_start*p + k_start
                b01 = j_start*p + k_start + 1
                
                c00 = k_start*n + 0
                c01 = k_start*n + 1
                c10 = (k_start+1)*n + 0
                c11 = (k_start+1)*n + 1
                
                # 4 terms
                add_term([a00], [b00], [c00], [1], [1], [1])
                add_term([a00], [b01], [c01], [1], [1], [1])
                add_term([a10], [b00], [c10], [1], [1], [1])
                add_term([a10], [b01], [c11], [1], [1], [1])
                
    return {"U": U, "V": V, "W": W}
