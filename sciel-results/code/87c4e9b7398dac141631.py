"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 3x8 matrix B into four 3x2 blocks.
    # For each block, we use a rank-11 decomposition for 2x3x2 multiplication.
    # This gives a total rank of 4 * 11 = 44.
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 11 terms for 2x3x2 multiplication
    # A is 2x3, B is 3x2
    # C = AB
    # Terms 1-7: Strassen for the 2x2 submatrix (cols 0,1 of A and rows 0,1 of B)
    # Terms 8-11: Cross terms for col 2 of A and row 2 of B
    
    # Strassen terms for 2x2
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # Cross terms
    # M8 = a13*b31
    # M9 = a13*b32
    # M10 = a23*b31
    # M11 = a23*b32
    
    # We need to map these to the global indices.
    # Global A index: i*m + j
    # Global B index: j*p + k
    # Global C index: k*n + i
    
    # For each of the 4 blocks of B (k blocks of size 2)
    for block in range(4):
        base_col = block * 2
        # U factors for this block
        # We use the same A indices for all blocks, so U is the same for all 4 blocks?
        # No, U is part of the tensor decomposition.
        # The tensor is a sum of rank-1 terms.
        # Each term is u ⊗ v ⊗ w.
        # For block b, the v factor only has non-zero entries in the columns corresponding to block b.
        # The u factor is the same for all blocks (it depends on A).
        # The w factor only has non-zero entries in the rows corresponding to block b.
        
        # Let's define the 11 (u, v, w) vectors for the 2x3x2 case.
        # Then for block b, we shift the v and w indices.
        
        # Indices in 2x3x2:
        # A: 0,1 (row 0), 2,3 (row 1)? No, A is 2x3.
        # Flattened A: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
        # Flattened B: 0:(0,0), 1:(1,0), 2:(2,0), 3:(0,1), 4:(1,1), 5:(2,1)
        # Flattened C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # Strassen terms (u, v, w)
        # M1: u=[1,0,0, 1,1,0], v=[1,1,0, 1,1,0], w=[1,0, 1,1] -> C11 += M1, C12 += M1, C21 += M1, C22 += M1?
        # No, Strassen gives:
        # C11 = M1 - M4 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M6
        
        # Let's list the (u, v, w) for each multiplication.
        # u is in R^6, v is in R^6, w is in R^4.
        
        terms = []
        
        # M1 = (a11+a22)(b11+b22)
        u1 = [1, 0, 0, 1, 1, 0]
        v1 = [1, 1, 0, 1, 1, 0]
        # Contribution to C:
        # C11 += M1, C12 += M1, C21 += M1, C22 += M1? No.
        # C11 = M1 - M4 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M6
        
        # So w1 for M1 is [1, 0, 0, 1]
        terms.append((u1, v1, [1, 0, 0, 1]))
        
        # M2 = (a21+a22)b11
        u2 = [0, 0, 0, 1, 1, 0]
        v2 = [1, 0, 0, 1, 0, 0]
        # C21 += M2
        terms.append((u2, v2, [0, 0, 1, 0]))
        
        # M3 = a11(b12-b22)
        u3 = [1, 0, 0, 0, 0, 0]
        v3 = [0, 0, 0, 1, -1, 0]
        # C12 += M3, C22 -= M3
        terms.append((u3, v3, [0, 1, 0, -1]))
        
        # M4 = a22(b21-b11)
        u4 = [0, 0, 0, 0, 1, 0]
        v4 = [0, 1, 0, 0, 0, 0] # b21 is index 1, b11 is index 0. So v4 = [0, 1, 0, 0, 0, 0] - [1, 0, 0, 0, 0, 0] = [-1, 1, 0, 0, 0, 0]
        v4 = [-1, 1, 0, 0, 0, 0]
        # C11 -= M4, C21 += M4
        terms.append((u4, v4, [-1, 0, 1, 0]))
        
        # M5 = (a11+a12)b22
        u5 = [1, 1, 0, 0, 0, 0]
        v5 = [0, 0, 0, 0, 1, 0]
        # C12 += M5
        terms.append((u5, v5, [0, 1, 0, 0]))
        
        # M6 = (a21-a11)(b11+b12)
        u6 = [-1, 0, 0, 1, 0, 0]
        v6 = [1, 0, 0, 1, 0, 0]
        # C22 += M6
        terms.append((u6, v6, [0, 0, 0, 1]))
        
        # M7 = (a12-a22)(b21+b22)
        u7 = [0, 1, 0, 0, -1, 0]
        v7 = [0, 1, 0, 0, 1, 0]
        # C11 += M7
        terms.append((u7, v7, [1, 0, 0, 0]))
        
        # M8 = a13*b31
        u8 = [0, 0, 1, 0, 0, 0]
        v8 = [0, 0, 1, 0, 0, 0]
        # C11 += M8
        terms.append((u8, v8, [1, 0, 0, 0]))
        
        # M9 = a13*b32
        u9 = [0, 0, 1, 0, 0, 0]
        v9 = [0, 0, 0, 0, 0, 1]
        # C12 += M9
        terms.append((u9, v9, [0, 1, 0, 0]))
        
        # M10 = a23*b31
        u10 = [0, 0, 0, 0, 0, 1]
        v10 = [0, 0, 1, 0, 0, 0]
        # C21 += M10
        terms.append((u10, v10, [0, 0, 1, 0]))
        
        # M11 = a23*b32
        u11 = [0, 0, 0, 0, 0, 1]
        v11 = [0, 0, 0, 0, 0, 1]
        # C22 += M11
        terms.append((u11, v11, [0, 0, 0, 1]))
        
        for t_idx, (u, v, w) in enumerate(terms):
            col = block * 11 + t_idx
            # Fill U
            for i in range(6):
                U[i][col] = u[i]
            # Fill V
            # B indices for this block:
            # Row 0: base_col, base_col+1
            # Row 1: base_col+2, base_col+3
            # Row 2: base_col+4, base_col+5
            # v indices: 0:(0,0), 1:(1,0), 2:(2,0), 3:(0,1), 4:(1,1), 5:(2,1)
            # Global V index: j*p + k
            # j=0, k=base_col -> 0*3 + base_col = base_col
            # j=0, k=base_col+1 -> base_col+1
            # j=1, k=base_col -> 1*3 + base_col = 3 + base_col
            # j=1, k=base_col+1 -> 3 + base_col + 1
            # j=2, k=base_col -> 2*3 + base_col = 6 + base_col
            # j=2, k=base_col+1 -> 6 + base_col + 1
            
            v_indices = [base_col, base_col+1, 3+base_col, 3+base_col+1, 6+base_col, 6+base_col+1]
            for i in range(6):
                V[v_indices[i]][col] = v[i]
            
            # Fill W
            # C indices for this block:
            # C is 2x8.
            # Block block has columns base_col, base_col+1.
            # C indices: k*n + i
            # k=0, i=0 -> 0*2 + 0 = 0
            # k=0, i=1 -> 0*2 + 1 = 1
            # k=1, i=0 -> 1*2 + 0 = 2
            # k=1, i=1 -> 1*2 + 1 = 3
            # Wait, C is 2x8.
            # C[k, i] where k is column of C (0..7), i is row of C (0..1).
            # W[k*n + i, col]
            # For block `block`, the columns of C are `base_col` and `base_col+1`.
            # So k takes values `base_col` and `base_col+1`.
            # i takes values 0 and 1.
            # W indices:
            # (k=base_col, i=0) -> base_col*2 + 0
            # (k=base_col, i=1) -> base_col*2 + 1
            # (k=base_col+1, i=0) -> (base_col+1)*2 + 0
            # (k=base_col+1, i=1) -> (base_col+1)*2 + 1
            
            w_indices = [base_col*2, base_col*2+1, (base_col+1)*2, (base_col+1)*2+1]
            for i in range(4):
                W[w_indices[i]][col] = w[i]
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
