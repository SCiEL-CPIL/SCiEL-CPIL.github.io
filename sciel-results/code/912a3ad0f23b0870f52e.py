import numpy as np

DIMENSIONS = [2, 2, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=5
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (10, rank)
    # W: (p*n, rank) = (10, rank)
    
    rank = 18
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx, col] = u_val
        V[v_idx, col] = v_val
        W[w_idx, col] = w_val
        col += 1
        
    # Strassen block 1: cols 0, 1 of B (indices 0, 1 in p)
    # B indices: j*p + k. j in 0,1. k in 0,1.
    # B_00 is index 0*5+0=0. B_01 is 0*5+1=1.
    # B_10 is 1*5+0=5. B_11 is 1*5+1=6.
    
    # A indices: i*m + j.
    # A_00=0, A_01=1, A_10=2, A_11=3.
    
    # C indices: k*n + i.
    # C_00=0, C_01=1 (i=1, k=0? No. W[k*n+i]. k=0, i=0 -> 0. k=0, i=1 -> 1.
    # k=1, i=0 -> 2. k=1, i=1 -> 3.
    # Wait. C is 2x5.
    # C_{i,k}. i in 0,1. k in 0..4.
    # W index is k*n + i.
    
    # Strassen for A (2x2) * B_1 (2x2, cols 0,1)
    # m1 = (a00+a01)(b00+b10)
    # m2 = (a10+a11)(b00+b10)
    # m3 = a01(b10-b00)
    # m4 = a10(b11-b01)
    # m5 = (a00+a11)(b00+b11)
    # m6 = (a01+a11)(b10-b11)
    # m7 = (a00-a10)(b01+b11)
    
    # Outputs C_00, C_01, C_10, C_11 (for k=0,1)
    # C_00 = m5 + m4 - m6 - m7
    # C_01 = m1 - m3 - m4 - m7  (Wait, C_01 corresponds to k=1? No.
    # Standard Strassen:
    # C11 = m5 + m4 - m6 - m7
    # C12 = m1 - m3 - m4 - m7
    # C21 = m2 + m3 - m5 + m6
    # C22 = m1 + m2 - m3 + m4
    
    # Mapping to our indices:
    # C_{i,k}.
    # i=0, k=0 -> C_00. W index 0*2+0=0.
    # i=0, k=1 -> C_01. W index 1*2+0=2.
    # i=1, k=0 -> C_10. W index 0*2+1=1.
    # i=1, k=1 -> C_11. W index 1*2+1=3.
    
    # m1: U: a00+a01. V: b00+b10. W: ?
    # m1 contributes to C_01 and C_11.
    # C_01 = ... + m1 ...
    # C_11 = ... + m1 ...
    
    # This is getting complex to hardcode manually.
    # I will generate the Strassen terms programmatically.
    
    def add_strassen_block(b_col_offset):
        nonlocal col
        # b_col_offset is the starting column index in B (0, 2, 4)
        # B indices:
        # b00 = 0*5 + b_col_offset
        # b01 = 0*5 + b_col_offset + 1
        # b10 = 1*5 + b_col_offset
        # b11 = 1*5 + b_col_offset + 1
        
        # A indices: 0,1,2,3
        # W indices for this block:
        # k=0, i=0 -> 0
        # k=0, i=1 -> 1
        # k=1, i=0 -> 2
        # k=1, i=1 -> 3
        # But we have offset.
        # k_abs = b_col_offset + k_rel
        # W index = k_abs * n + i
        
        # m1 = (a00+a01) * (b00+b10)
        # U: a00+a01. V: b00+b10.
        # Contributes to C_01 (k=1, i=0) and C_11 (k=1, i=1) with coeff 1.
        # Wait, Strassen formulas:
        # C12 (i=0, k=1) has +m1.
        # C22 (i=1, k=1) has +m1.
        # So W[1*2+0] += 1, W[1*2+1] += 1.
        
        # I will just add 7 terms for each block.
        
        # Term 1: m1
        U[0, col] = 1; U[1, col] = 1
        V[b_col_offset, col] = 1; V[5+b_col_offset, col] = 1
        W[(b_col_offset+1)*2+0, col] = 1
        W[(b_col_offset+1)*2+1, col] = 1
        col += 1
        
        # Term 2: m2 = (a10+a11)(b00+b10)
        U[2, col] = 1; U[3, col] = 1
        V[b_col_offset, col] = 1; V[5+b_col_offset, col] = 1
        W[(b_col_offset+0)*2+1, col] = 1 # C21 (i=1, k=0)
        W[(b_col_offset+1)*2+1, col] = 1 # C22 (i=1, k=1)
        col += 1
        
        # Term 3: m3 = a01(b10-b00)
        U[1, col] = 1
        V[b_col_offset, col] = -1; V[5+b_col_offset, col] = 1
        W[(b_col_offset+1)*2+0, col] = -1 # C12
        W[(b_col_offset+1)*2+1, col] = -1 # C22
        col += 1
        
        # Term 4: m4 = a10(b11-b01)
        U[2, col] = 1
        V[b_col_offset+1, col] = -1; V[5+b_col_offset+1, col] = 1
        W[(b_col_offset+0)*2+0, col] = 1 # C11
        W[(b_col_offset+1)*2+0, col] = -1 # C12
        W[(b_col_offset+1)*2+1, col] = 1 # C22
        col += 1
        
        # Term 5: m5 = (a00+a11)(b00+b11)
        U[0, col] = 1; U[3, col] = 1
        V[b_col_offset, col] = 1; V[5+b_col_offset+1, col] = 1
        W[(b_col_offset+0)*2+0, col] = 1 # C11
        W[(b_col_offset+0)*2+1, col] = -1 # C21
        col += 1
        
        # Term 6: m6 = (a01+a11)(b10-b11)
        U[1, col] = 1; U[3, col] = 1
        V[5+b_col_offset, col] = 1; V[5+b_col_offset+1, col] = -1
        W[(b_col_offset+0)*2+0, col] = -1 # C11
        W[(b_col_offset+0)*2+1, col] = 1 # C21
        col += 1
        
        # Term 7: m7 = (a00-a10)(b01+b11)
        U[0, col] = 1; U[2, col] = -1
        V[b_col_offset+1, col] = 1; V[5+b_col_offset+1, col] = 1
        W[(b_col_offset+0)*2+0, col] = -1 # C11
        W[(b_col_offset+1)*2+0, col] = -1 # C12
        col += 1

    # Add two Strassen blocks for cols 0-1 and 2-3
    add_strassen_block(0)
    add_strassen_block(2)
    
    # Last block: col 4 (2x1)
    # Schoolbook 4 terms
    # y0 = a00*b0 + a01*b1
    # y1 = a10*b0 + a11*b1
    # B indices: 0*5+4=4, 1*5+4=9.
    # W indices: k=4.
    # C_04: W[4*2+0]=8.
    # C_14: W[4*2+1]=9.
    
    # t1: a00*b0 -> C_04
    U[0, col] = 1
    V[4, col] = 1
    W[8, col] = 1
    col += 1
    
    # t2: a01*b1 -> C_04
    U[1, col] = 1
    V[9, col] = 1
    W[8, col] = 1
    col += 1
    
    # t3: a10*b0 -> C_14
    U[2, col] = 1
    V[4, col] = 1
    W[9, col] = 1
    col += 1
    
    # t4: a11*b1 -> C_14
    U[3, col] = 1
    V[9, col] = 1
    W[9, col] = 1
    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
