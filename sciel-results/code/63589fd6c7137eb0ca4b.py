"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split m=5 into 2+2+1
    # Split n=4 into 2+2
    # Split p=6 into 2+2+2
    
    # We will construct U, V, W for the 108 rank decomposition.
    # Rank 108 = 42 (for 4x2x6) + 42 (for 4x2x6) + 24 (for 4x1x6)
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        U[u_idx].append(val)
        V[v_idx].append(val)
        W[w_idx].append(val)
        
    U = [[] for _ in range(n*m)]
    V = [[] for _ in range(m*p)]
    W = [[] for _ in range(p*n)]
    
    # Strassen 2x2x2 decomposition
    # C = AB
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B12)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map the Strassen terms to the global U, V, W.
    # The Strassen algorithm computes 7 products.
    # Each product is a scalar multiplication in the tensor decomposition.
    
    # For a 2x2x2 block at position (i, j, k) in the 4x5x6 grid:
    # i in {0, 1} (row block of A)
    # j in {0, 1, 2} (col block of A / row block of B)
    # k in {0, 1, 2} (col block of B)
    
    # Block (i, j, k) is A[i*2:(i+1)*2, j*2:(j+1)*2] * B[j*2:(j+1)*2, k*2:(k+1)*2]
    # This is a 2x2x2 multiplication.
    
    # We iterate over all such blocks.
    # For j=0, 1: 2x2x2 blocks (Strassen)
    # For j=2: 2x1x2 blocks (Schoolbook)
    
    for i in range(2): # 2 row blocks of A
        for j in range(3): # 3 col blocks of A
            for k in range(3): # 3 col blocks of B
                # Determine block size
                # A block: rows i*2 to i*2+1, cols j*2 to j*2+1 (if j<2) or j*2 to j*2 (if j==2)
                # B block: rows j*2 to j*2+1 (if j<2) or j*2 to j*2 (if j==2), cols k*2 to k*2+1
                
                # Actually, the split of m=5 is 2+2+1.
                # So j=0: cols 0-1, j=1: cols 2-3, j=2: col 4.
                
                a_row_start = i * 2
                a_col_start = 0
                if j == 0: a_col_start = 0
                elif j == 1: a_col_start = 2
                else: a_col_start = 4
                
                b_row_start = a_col_start
                b_col_start = k * 2
                
                # A block size: 2 x (2 if j<2 else 1)
                # B block size: (2 if j<2 else 1) x 2
                
                if j < 2:
                    # 2x2x2 Strassen
                    # We add 7 terms for this block.
                    # The terms are linear combinations of A and B entries.
                    # We need to map the Strassen M1..M7 to the global U, V, W.
                    
                    # Strassen M1: (A11+A22)(B11+B22)
                    # U indices: A11, A22
                    # V indices: B11, B22
                    # W index: C11, C22
                    
                    # This is complex to map directly.
                    # Let's use the explicit tensor decomposition for Strassen.
                    
                    # Strassen 2x2x2 tensor has rank 7.
                    # We can hardcode the 7 terms for each 2x2x2 block.
                    
                    # Let's define the local indices.
                    # A local: (0,0), (0,1), (1,0), (1,1)
                    # B local: (0,0), (0,1), (1,0), (1,1)
                    # C local: (0,0), (0,1), (1,0), (1,1)
                    
                    # Global indices:
                    # A global: (a_row_start + r, a_col_start + c)
                    # B global: (b_row_start + r, b_col_start + c)
                    # C global: (b_col_start + c, a_row_start + r) -> W[k*n+i, r]
                    # Wait, W index is k*n+i?
                    # The task says W[k*n+i, r].
                    # k is the column of B, i is the row of A.
                    # So W index = (b_col_start + c) * n + (a_row_start + r).
                    
                    # Let's add the 7 Strassen terms.
                    # M1: (A11+A22)(B11+B22)
                    # U: A11, A22
                    # V: B11, B22
                    # W: C11, C22
                    
                    # We need to create a term for each product in the expansion.
                    # M1 = A11B11 + A11B22 + A22B11 + A22B22
                    # This is 4 terms.
                    # But Strassen uses 7 products, not 8.
                    # The products are:
                    # P1 = (A11+A22)(B11+B22)
                    # P2 = (A21+A22)B11
                    # P3 = A11(B21-B22)
                    # P4 = A22(B11-B12)
                    # P5 = (A11+A12)B22
                    # P6 = (A21-A11)(B11+B12)
                    # P7 = (A12-A22)(B21+B22)
                    
                    # Each P is a product of a linear combination of A and a linear combination of B.
                    # In the tensor decomposition, each P corresponds to a rank-1 term.
                    # U vector has the coefficients of A in the linear combination.
                    # V vector has the coefficients of B in the linear combination.
                    # W vector has the coefficients of C in the result.
                    
                    # So for P1:
                    # U: A11=1, A22=1
                    # V: B11=1, B22=1
                    # W: C11=1, C22=1 (since P1 contributes to C11 and C22)
                    
                    # Let's implement this.
                    
                    # Local to global mapping
                    def get_a_idx(r, c):
                        return (a_row_start + r) * m + (a_col_start + c)
                    def get_b_idx(r, c):
                        return (b_row_start + r) * p + (b_col_start + c)
                    def get_w_idx(r, c):
                        # W[k*n+i, r] -> index = k*n + i
                        # k is global col of B, i is global row of A
                        # k = b_col_start + c
                        # i = a_row_start + r
                        return (b_col_start + c) * n + (a_row_start + r)
                    
                    # P1: (A11+A22)(B11+B22)
                    # U: A11(0,0), A22(1,1)
                    # V: B11(0,0), B22(1,1)
                    # W: C11(0,0), C22(1,1)
                    u1 = {get_a_idx(0,0): 1, get_a_idx(1,1): 1}
                    v1 = {get_b_idx(0,0): 1, get_b_idx(1,1): 1}
                    w1 = {get_w_idx(0,0): 1, get_w_idx(1,1): 1}
                    
                    # P2: (A21+A22)B11
                    # U: A21(1,0), A22(1,1)
                    # V: B11(0,0)
                    # W: C21(1,0)
                    u2 = {get_a_idx(1,0): 1, get_a_idx(1,1): 1}
                    v2 = {get_b_idx(0,0): 1}
                    w2 = {get_w_idx(1,0): 1}
                    
                    # P3: A11(B21-B22)
                    # U: A11(0,0)
                    # V: B21(1,0), B22(1,1) with -1
                    # W: C12(0,1)
                    u3 = {get_a_idx(0,0): 1}
                    v3 = {get_b_idx(1,0): 1, get_b_idx(1,1): -1}
                    w3 = {get_w_idx(0,1): 1}
                    
                    # P4: A22(B11-B12)
                    # U: A22(1,1)
                    # V: B11(0,0), B12(0,1) with -1
                    # W: C21(1,0)
                    u4 = {get_a_idx(1,1): 1}
                    v4 = {get_b_idx(0,0): 1, get_b_idx(0,1): -1}
                    w4 = {get_w_idx(1,0): 1}
                    
                    # P5: (A11+A12)B22
                    # U: A11(0,0), A12(0,1)
                    # V: B22(1,1)
                    # W: C12(0,1)
                    u5 = {get_a_idx(0,0): 1, get_a_idx(0,1): 1}
                    v5 = {get_b_idx(1,1): 1}
                    w5 = {get_w_idx(0,1): 1}
                    
                    # P6: (A21-A11)(B11+B12)
                    # U: A21(1,0), A11(0,0) with -1
                    # V: B11(0,0), B12(0,1)
                    # W: C22(1,1)
                    u6 = {get_a_idx(1,0): 1, get_a_idx(0,0): -1}
                    v6 = {get_b_idx(0,0): 1, get_b_idx(0,1): 1}
                    w6 = {get_w_idx(1,1): 1}
                    
                    # P7: (A12-A22)(B21+B22)
                    # U: A12(0,1), A22(1,1) with -1
                    # V: B21(1,0), B22(1,1)
                    # W: C22(1,1)
                    u7 = {get_a_idx(0,1): 1, get_a_idx(1,1): -1}
                    v7 = {get_b_idx(1,0): 1, get_b_idx(1,1): 1}
                    w7 = {get_w_idx(1,1): 1}
                    
                    # Add these 7 terms to U, V, W
                    for u, v, w in [(u1, v1, w1), (u2, v2, w2), (u3, v3, w3), (u4, v4, w4), (u5, v5, w5), (u6, v6, w6), (u7, v7, w7)]:
                        col = len(U[0])
                        for idx in range(n*m):
                            U[idx].append(u.get(idx, 0))
                        for idx in range(m*p):
                            V[idx].append(v.get(idx, 0))
                        for idx in range(p*n):
                            W[idx].append(w.get(idx, 0))
                else:
                    # j == 2: 2x1x2 block
                    # A: 2x1, B: 1x2
                    # Schoolbook: 4 terms
                    # C11 = A11 B11
                    # C12 = A11 B12
                    # C21 = A21 B11
                    # C22 = A21 B12
                    
                    # A local: (0,0), (1,0)
                    # B local: (0,0), (0,1)
                    # C local: (0,0), (0,1), (1,0), (1,1)
                    
                    # P1: A11 B11
                    u1 = {get_a_idx(0,0): 1}
                    v1 = {get_b_idx(0,0): 1}
                    w1 = {get_w_idx(0,0): 1}
                    
                    # P2: A11 B12
                    u2 = {get_a_idx(0,0): 1}
                    v2 = {get_b_idx(0,1): 1}
                    w2 = {get_w_idx(0,1): 1}
                    
                    # P3: A21 B11
                    u3 = {get_a_idx(1,0): 1}
                    v3 = {get_b_idx(0,0): 1}
                    w3 = {get_w_idx(1,0): 1}
                    
                    # P4: A21 B12
                    u4 = {get_a_idx(1,0): 1}
                    v4 = {get_b_idx(0,1): 1}
                    w4 = {get_w_idx(1,1): 1}
                    
                    for u, v, w in [(u1, v1, w1), (u2, v2, w2), (u3, v3, w3), (u4, v4, w4)]:
                        col = len(U[0])
                        for idx in range(n*m):
                            U[idx].append(u.get(idx, 0))
                        for idx in range(m*p):
                            V[idx].append(v.get(idx, 0))
                        for idx in range(p*n):
                            W[idx].append(w.get(idx, 0))
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
