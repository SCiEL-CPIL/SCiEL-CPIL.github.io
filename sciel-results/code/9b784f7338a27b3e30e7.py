"""Rewritten program for matrix multiplication 2,2,15 using structured algebraic decomposition."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x15 matrix.
    
    Strategy: Decompose the 2x15 multiplication into blocks:
    - A 2x4 block using rank-14 decomposition (two independent Strassen 2x2x2x2 blocks)
    - A 2x11 block using rank-39 decomposition (five Strassen 2x2x2x2 blocks + schoolbook 2x2x2x1)
    
    Total rank = 14 + 39 = 53
    
    The output factor W follows the official trace convention W[k*n+i, r] where:
    - k is the column index of the 2x15 matrix (0 to 14)
    - i is the row index of the 2x2 result (0 to 1)
    - r is the rank index
    """
    n, m, p = 2, 2, 15
    rank = 53
    
    U = [[0] * rank for _ in range(n * m)]  # shape (4, 53)
    V = [[0] * rank for _ in range(m * p)]  # shape (30, 53)
    W = [[0] * rank for _ in range(p * n)]  # shape (30, 53)
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, c, u_val=1, v_val=1, w_val=1):
        U[u_idx][c] += u_val
        V[v_idx][c] += v_val
        W[w_idx][c] += w_val
    
    # Strassen 2x2 x 2x2 block
    # A = [[a, b], [c, d]] indexed as U[0]=a, U[1]=b, U[2]=c, U[3]=d
    # B = [[e, f], [g, h]] where e=B[j*2+k0], f=B[j*2+k0+1], g=B[(j+1)*2+k0], h=B[(j+1)*2+k0+1]
    # C = A*B: C[0,0]=ae+bi, C[0,1]=af+bh, C[1,0]=ce+dg, C[1,1]=cf+dh
    # W indices: W[k*n+i] so W[0*2+0]=W[0] for C[0,0], W[1*2+0]=W[2] for C[1,0],
    #            W[0*2+1]=W[1] for C[0,1], W[1*2+1]=W[3] for C[1,1]
    # But wait: W[k*n+i] where k is column of B, i is row of C
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # W index for output C[i,j] is j*n + i = j*2 + i
    
    def strassen_block(base_col, b_col_offset):
        """
        Add a Strassen 2x2 x 2x2 block.
        base_col: starting column in U,V,W
        b_col_offset: which columns of B (the 2x15 matrix) this block uses (0-3 for first, etc.)
        """
        global col
        start = col
        col += 7
        
        # U indices: 0=a, 1=b, 2=c, 3=d (for A = [[a,b],[c,d]])
        # V indices: b_col_offset*2 + 0 = e (B[0, b_col_offset])
        #             b_col_offset*2 + 1 = f (B[0, b_col_offset+1])
        #             b_col_offset*2 + 2 = g (B[1, b_col_offset])
        #             b_col_offset*2 + 3 = h (B[1, b_col_offset+1])
        
        e_idx = b_col_offset * 2 + 0
        f_idx = b_col_offset * 2 + 1
        g_idx = b_col_offset * 2 + 2
        h_idx = b_col_offset * 2 + 3
        
        # M1 = (a+d)(e+i) -> but i here is B[1, b_col_offset] = g
        # Actually Strassen: M1 = (a+d)(e+i) where i is B[1,0] in standard
        # Let me re-derive for B = [[e, f], [g, h]]:
        # M1 = (a+d)(e+g)
        # M2 = (c+d)(e+g)  
        # M3 = a(f-g)
        # M4 = d(h-f)
        # M5 = (a+b)(h-g)
        # M6 = (c+b)(g-h)
        # M7 = (a-c)(e+f)
        
        # Then:
        # C[0,0] = M1 - M2 - M4 + M7
        # C[0,1] = M4 + M5
        # C[1,0] = M2 + M6
        # C[1,1] = M1 - M3 - M5 + M6
        
        # W indices: C[0,0] -> W[0*2+0] = W[0], C[0,1] -> W[1*2+0] = W[2]
        #            C[1,0] -> W[0*2+1] = W[1], C[1,1] -> W[1*2+1] = W[3]
        
        # M1 = (a+d)(e+g)
        add_term(0, e_idx, 0, start + 0, 1, 1, 1)   # a*e
        add_term(0, g_idx, 0, start + 0, 1, 1, 1)   # a*g
        add_term(3, e_idx, 0, start + 0, 1, 1, 1)   # d*e
        add_term(3, g_idx, 0, start + 0, 1, 1, 1)   # d*g
        
        # M2 = (c+d)(e+g)
        add_term(2, e_idx, 0, start + 1, 1, 1, 1)   # c*e
        add_term(2, g_idx, 0, start + 1, 1, 1, 1)   # c*g
        add_term(3, e_idx, 0, start + 1, 1, 1, 1)   # d*e
        add_term(3, g_idx, 0, start + 1, 1, 1, 1)   # d*g
        
        # M3 = a(f-g)
        add_term(0, f_idx, 0, start + 2, 1, 1, 1)   # a*f
        add_term(0, g_idx, 0, start + 2, 1, -1, 1)  # a*(-g)
        
        # M4 = d(h-f)
        add_term(3, h_idx, 0, start + 3, 1, 1, 1)   # d*h
        add_term(3, f_idx, 0, start + 3, 1, -1, 1)  # d*(-f)
        
        # M5 = (a+b)(h-g)
        add_term(0, h_idx, 0, start + 4, 1, 1, 1)   # a*h
        add_term(0, g_idx, 0, start + 4, 1, -1, 1)  # a*(-g)
        add_term(1, h_idx, 0, start + 4, 1, 1, 1)   # b*h
        add_term(1, g_idx, 0, start + 4, 1, -1, 1)  # b*(-g)
        
        # M6 = (c+b)(g-h)
        add_term(2, g_idx, 0, start + 5, 1, 1, 1)   # c*g
        add_term(2, h_idx, 0, start + 5, 1, -1, 1)  # c*(-h)
        add_term(1, g_idx, 0, start + 5, 1, 1, 1)   # b*g
        add_term(1, h_idx, 0, start + 5, 1, -1, 1)  # b*(-h)
        
        # M7 = (a-c)(e+f)
        add_term(0, e_idx, 0, start + 6, 1, 1, 1)   # a*e
        add_term(0, f_idx, 0, start + 6, 1, 1, 1)   # a*f
        add_term(2, e_idx, 0, start + 6, -1, 1, 1)  # (-c)*e
        add_term(2, f_idx, 0, start + 6, -1, 1, 1)  # (-c)*f
        
        # Now combine to get C entries:
        # C[0,0] = M1 - M2 - M4 + M7 -> W[0]
        # C[0,1] = M4 + M5 -> W[2]
        # C[1,0] = M2 + M6 -> W[1]
        # C[1,1] = M1 - M3 - M5 + M6 -> W[3]
        
        # We need to add the W coefficients. The W factor combines the M terms.
        # Actually, in tensor decomposition, each rank-1 term has U, V, W entries.
        # The way I've set it up, each add_term creates a rank-1 term.
        # But Strassen combines multiple products. Let me reconsider.
        
        # Actually, the standard approach: each multiplication M_i is a separate rank-1 term
        # where U has the first factor, V has the second factor, and W distributes to outputs.
        
        # Let me redo this properly. Each M_i is one rank-1 term.
        # M1 = (a+d)(e+g): U=[1,0,0,1] for (a+d), V=[1,0,1,0] for (e+g), W distributes
        # But W needs to be distributed to C[0,0] and C[1,1].
        
        # Hmm, this is getting complex. Let me use a different approach.
        
        # Actually, I think my add_term approach is wrong. Let me restart the Strassen implementation.
        
        return start
    
    # Let me restart with a cleaner approach
    col = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(b_col_offset):
        """Add 7 rank-1 terms for a Strassen 2x2 x 2x2 block using B columns b_col_offset, b_col_offset+1"""
        global col
        start = col
        col += 7
        
        e_idx = b_col_offset * 2 + 0  # B[0, b_col_offset]
        f_idx = b_col_offset * 2 + 1  # B[0, b_col_offset+1]
        g_idx = b_col_offset * 2 + 2  # B[1, b_col_offset]
        h_idx = b_col_offset * 2 + 3  # B[1, b_col_offset+1]
        
        # M1 = (a+d)(e+g)
        # U: a=1, d=1; V: e=1, g=1; W: C[0,0]+=1, C[1,1]+=1
        U[0][start+0] += 1
        U[3][start+0] += 1
        V[e_idx][start+0] += 1
        V[g_idx][start+0] += 1
        W[0][start+0] += 1  # C[0,0]: k=0, i=0
        W[3][start+0] += 1  # C[1,1]: k=1, i=1
        
        # M2 = (c+d)(e+g)
        U[2][start+1] += 1
        U[3][start+1] += 1
        V[e_idx][start+1] += 1
        V[g_idx][start+1] += 1
        W[1][start+1] += 1  # C[1,0]: k=0, i=1
        W[3][start+1] += 1  # C[1,1]: k=1, i=1
        
        # M3 = a(f-g)
        U[0][start+2] += 1
        V[f_idx][start+2] += 1
        V[g_idx][start+2] -= 1
        W[3][start+2] -= 1  # C[1,1]: k=1, i=1, coefficient -1
        
        # M4 = d(h-f)
        U[3][start+3] += 1
        V[h_idx][start+3] += 1
        V[f_idx][start+3] -= 1
        W[0][start+3] -= 1  # C[0,0]: k=0, i=0, coefficient -1
        W[2][start+3] += 1  # C[0,1]: k=1, i=0, coefficient +1
        
        # M5 = (a+b)(h-g)
        U[0][start+4] += 1
        U[1][start+4] += 1
        V[h_idx][start+4] += 1
        V[g_idx][start+4] -= 1
        W[2][start+4] += 1  # C[0,1]: k=1, i=0, coefficient +1
        W[3][start+4] -= 1  # C[1,1]: k=1, i=1, coefficient -1
        
        # M6 = (c+b)(g-h)
        U[2][start+5] += 1
        U[1][start+5] += 1
        V[g_idx][start+5] += 1
        V[h_idx][start+5] -= 1
        W[1][start+5] += 1  # C[1,0]: k=0, i=1, coefficient +1
        W[3][start+5] += 1  # C[1,1]: k=1, i=1, coefficient +1
        
        # M7 = (a-c)(e+f)
        U[0][start+6] += 1
        U[2][start+6] -= 1
        V[e_idx][start+6] += 1
        V[f_idx][start+6] += 1
        W[0][start+6] += 1  # C[0,0]: k=0, i=0, coefficient +1
        W[3][start+6] += 1  # C[1,1]: k=1, i=1, coefficient +1
        
        return start
    
    # Block 1: 2x2 x 2x4 using two Strassen blocks
    # First Strassen: B columns 0,1
    add_strassen_block(0)
    # Second Strassen: B columns 2,3
    add_strassen_block(2)
    
    # Block 2: 2x2 x 2x11 using five Strassen blocks (columns 4-13) + schoolbook for columns 14
    # Five Strassen blocks for columns 4,5,6,7,8,9,10,11,12,13
    add_strassen_block(4)
    add_strassen_block(6)
    add_strassen_block(8)
    add_strassen_block(10)
    add_strassen_block(12)
    
    # Schoolbook for 2x2 x 2x1 (column 14)
    # C[0,14] = a*B[0,14] + b*B[1,14]
    # C[1,14] = c*B[0,14] + d*B[1,14]
    # 4 terms
    b_idx = 14 * 2 + 0  # B[0, 14]
    c_idx = 14 * 2 + 1  # B[1, 14]
    
    # Term 1: a * B[0,14] -> C[0,14]
    U[0][col] = 1
    V[b_idx][col] = 1
    W[14*2+0][col] = 1  # C[0,14]: k=14, i=0
    col += 1
    
    # Term 2: b * B[1,14] -> C[0,14]
    U[1][col] = 1
    V[c_idx][col] = 1
    W[14*2+0][col] = 1  # C[0,14]: k=14, i=0
    col += 1
    
    # Term 3: c * B[0,14] -> C[1,14]
    U[2][col] = 1
    V[b_idx][col] = 1
    W[14*2+1][col] = 1  # C[1,14]: k=14, i=1
    col += 1
    
    # Term 4: d * B[1,14] -> C[1,14]
    U[3][col] = 1
    V[c_idx][col] = 1
    W[14*2+1][col] = 1  # C[1,14]: k=14, i=1
    col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
