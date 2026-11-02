"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    # Use Strassen's algorithm on 2x2 blocks of the right matrix B.
    # B is 2x10, so we split it into 5 blocks of size 2x2.
    # Each block multiplication uses rank 7 instead of rank 4.
    # Total rank = 5 * 7 = 35.
    n, m, p = DIMENSIONS
    num_blocks = p // m  # 10 // 2 = 5
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products for 2x2 matrix multiplication:
    # A = [[a, b], [c, d]], B = [[x, y], [z, w]]
    # M1 = (a+d)(x+w)
    # M2 = (c+d)x
    # M3 = a(y-w)
    # M4 = d(z-x)
    # M5 = (a+b)w
    # M6 = (c+a)(y-w)
    # M7 = (b+d)(x+z)
    # C = [[a+d, b], [c, d]] * [[x, y], [z, w]]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the U, V, W contributions for one Strassen block
    # U has shape (n*m, 7) = (4, 7)
    # V has shape (m*p_block, 7) = (2*2, 7) = (4, 7) for a single block
    # W has shape (p_block*n, 7) = (2*2, 7) = (4, 7) for a single block
    
    # Index mapping:
    # U row: i*m + j where i in [0,1], j in [0,1]
    # So U[0] = U[0,0], U[1] = U[0,1], U[2] = U[1,0], U[3] = U[1,1]
    # i=0,j=0 -> idx 0; i=0,j=1 -> idx 1; i=1,j=0 -> idx 2; i=1,j=1 -> idx 3
    
    # V row: j*p + k where j in [0,1], k in [0,1] for a 2x2 block
    # V[0] = V[0,0], V[1] = V[0,1], V[2] = V[1,0], V[3] = V[1,1]
    
    # W row: k*n + i where k in [0,1], i in [0,1]
    # W[0] = W[0,0], W[1] = W[0,1], W[2] = W[1,0], W[3] = W[1,1]
    
    # Strassen terms (7 columns):
    # col 0: M1 = (a+d)(x+w)
    # col 1: M2 = (c+d)x
    # col 2: M3 = a(y-w)
    # col 3: M4 = d(z-x)
    # col 4: M5 = (a+b)w
    # col 5: M6 = (c+a)(y-w)
    # col 6: M7 = (b+d)(x+z)
    
    for block in range(num_blocks):
        base_col = block * 7
        for col_offset in range(7):
            col = base_col + col_offset
            
            # Set U values for this Strassen term
            if col_offset == 0:  # M1: (a+d)(x+w)
                U[0][col] = 1  # a
                U[3][col] = 1  # d
            elif col_offset == 1:  # M2: (c+d)x
                U[2][col] = 1  # c
                U[3][col] = 1  # d
            elif col_offset == 2:  # M3: a(y-w)
                U[0][col] = 1  # a
            elif col_offset == 3:  # M4: d(z-x)
                U[3][col] = 1  # d
            elif col_offset == 4:  # M5: (a+b)w
                U[0][col] = 1  # a
                U[1][col] = 1  # b
            elif col_offset == 5:  # M6: (c+a)(y-w)
                U[2][col] = 1  # c
                U[0][col] = 1  # a
            elif col_offset == 6:  # M7: (b+d)(x+z)
                U[1][col] = 1  # b
                U[3][col] = 1  # d
            
            # Set V values for this Strassen term
            # V indices: 0=(j=0,k=0)=x, 1=(j=0,k=1)=y, 2=(j=1,k=0)=z, 3=(j=1,k=1)=w
            if col_offset == 0:  # M1: (x+w)
                V[0][col] = 1  # x
                V[3][col] = 1  # w
            elif col_offset == 1:  # M2: x
                V[0][col] = 1  # x
            elif col_offset == 2:  # M3: (y-w)
                V[1][col] = 1  # y
                V[3][col] = -1  # w
            elif col_offset == 3:  # M4: (z-x)
                V[2][col] = 1  # z
                V[0][col] = -1  # x
            elif col_offset == 4:  # M5: w
                V[3][col] = 1  # w
            elif col_offset == 5:  # M6: (y-w)
                V[1][col] = 1  # y
                V[3][col] = -1  # w
            elif col_offset == 6:  # M7: (x+z)
                V[0][col] = 1  # x
                V[2][col] = 1  # z
            
            # Set W values for this Strassen term
            # W indices: 0=(k=0,i=0)=C00, 1=(k=0,i=1)=C01, 2=(k=1,i=0)=C10, 3=(k=1,i=1)=C11
            if col_offset == 0:  # M1 contributes to C00 (+1) and C11 (+1)
                W[0][col] = 1   # C00
                W[3][col] = 1   # C11
            elif col_offset == 1:  # M2 contributes to C10 (+1) and C11 (-1)
                W[2][col] = 1   # C10
                W[3][col] = -1  # C11
            elif col_offset == 2:  # M3 contributes to C01 (+1) and C11 (+1)
                W[1][col] = 1   # C01
                W[3][col] = 1   # C11
            elif col_offset == 3:  # M4 contributes to C00 (+1) and C10 (+1)
                W[0][col] = 1   # C00
                W[2][col] = 1   # C10
            elif col_offset == 4:  # M5 contributes to C00 (-1) and C01 (+1)
                W[0][col] = -1  # C00
                W[1][col] = 1   # C01
            elif col_offset == 5:  # M6 contributes to C11 (+1)
                W[3][col] = 1   # C11
            elif col_offset == 6:  # M7 contributes to C00 (+1)
                W[0][col] = 1   # C00
            
            # Map V and W to the correct block positions in the full matrices
            # For block `block`, the B columns are at positions block*m through block*m + m - 1
            # The V matrix has shape (m*p, rank), so for block `block`, the local V rows 0-3
            # correspond to global rows:
            # local row 0 (j=0,k=0) -> global row 0*p + block*m + 0 = block*m
            # local row 1 (j=0,k=1) -> global row 0*p + block*m + 1 = block*m + 1
            # local row 2 (j=1,k=0) -> global row 1*p + block*m + 0 = p + block*m
            # local row 3 (j=1,k=1) -> global row 1*p + block*m + 1 = p + block*m + 1
            
            # The W matrix has shape (p*n, rank), so for block `block`, the local W rows 0-3
            # correspond to global rows:
            # local row 0 (k=0,i=0) -> global row (block*m + 0)*n + 0 = (block*m)*n
            # local row 1 (k=0,i=1) -> global row (block*m + 0)*n + 1 = (block*m)*n + 1
            # local row 2 (k=1,i=0) -> global row (block*m + 1)*n + 0 = (block*m + 1)*n
            # local row 3 (k=1,i=1) -> global row (block*m + 1)*n + 1 = (block*m + 1)*n + 1
            
            # Actually, let me re-derive this. The W index is k*n + i.
            # For block `block`, the output C columns are block*m through block*m + m - 1.
            # k ranges from 0 to m-1 within the block, so global k = block*m + k_local.
            # i ranges from 0 to n-1.
            # So W row = (block*m + k_local) * n + i
            
            # Let me redo the V and W indexing properly.
            pass
    
    # I need to fix the V and W indexing. Let me rewrite the inner loop properly.
    
    # Clear and rebuild
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for block in range(num_blocks):
        base_col = block * 7
        for col_offset in range(7):
            col = base_col + col_offset
            
            # Set U values for this Strassen term
            if col_offset == 0:  # M1: (a+d)(x+w)
                U[0][col] = 1  # a (i=0,j=0)
                U[3][col] = 1  # d (i=1,j=1)
            elif col_offset == 1:  # M2: (c+d)x
                U[2][col] = 1  # c (i=1,j=0)
                U[3][col] = 1  # d (i=1,j=1)
            elif col_offset == 2:  # M3: a(y-w)
                U[0][col] = 1  # a (i=0,j=0)
            elif col_offset == 3:  # M4: d(z-x)
                U[3][col] = 1  # d (i=1,j=1)
            elif col_offset == 4:  # M5: (a+b)w
                U[0][col] = 1  # a (i=0,j=0)
                U[1][col] = 1  # b (i=0,j=1)
            elif col_offset == 5:  # M6: (c+a)(y-w)
                U[2][col] = 1  # c (i=1,j=0)
                U[0][col] = 1  # a (i=0,j=0)
            elif col_offset == 6:  # M7: (b+d)(x+z)
                U[1][col] = 1  # b (i=0,j=1)
                U[3][col] = 1  # d (i=1,j=1)
            
            # Set V values for this Strassen term
            # V row index = j*p + k, where j in [0,1], k in [0,1] for the local block
            # For block `block`, global k = block*m + k_local
            # So V row = j*p + (block*m + k_local)
            
            if col_offset == 0:  # M1: (x+w)
                V[0*p + block*m + 0][col] = 1   # x: j=0, k_local=0
                V[1*p + block*m + 1][col] = 1   # w: j=1, k_local=1
            elif col_offset == 1:  # M2: x
                V[0*p + block*m + 0][col] = 1   # x: j=0, k_local=0
            elif col_offset == 2:  # M3: (y-w)
                V[0*p + block*m + 1][col] = 1   # y: j=0, k_local=1
                V[1*p + block*m + 1][col] = -1  # w: j=1, k_local=1
            elif col_offset == 3:  # M4: (z-x)
                V[1*p + block*m + 0][col] = 1   # z: j=1, k_local=0
                V[0*p + block*m + 0][col] = -1  # x: j=0, k_local=0
            elif col_offset == 4:  # M5: w
                V[1*p + block*m + 1][col] = 1   # w: j=1, k_local=1
            elif col_offset == 5:  # M6: (y-w)
                V[0*p + block*m + 1][col] = 1   # y: j=0, k_local=1
                V[1*p + block*m + 1][col] = -1  # w: j=1, k_local=1
            elif col_offset == 6:  # M7: (x+z)
                V[0*p + block*m + 0][col] = 1   # x: j=0, k_local=0
                V[1*p + block*m + 0][col] = 1   # z: j=1, k_local=0
            
            # Set W values for this Strassen term
            # W row index = k*n + i, where k in [0,1], i in [0,1] for the local block
            # For block `block`, global k = block*m + k_local
            # So W row = (block*m + k_local)*n + i
            
            if col_offset == 0:  # M1 contributes to C00 (+1) and C11 (+1)
                W[(block*m + 0)*n + 0][col] = 1   # C00: k_local=0, i=0
                W[(block*m + 1)*n + 1][col] = 1   # C11: k_local=1, i=1
            elif col_offset == 1:  # M2 contributes to C10 (+1) and C11 (-1)
                W[(block*m + 1)*n + 0][col] = 1   # C10: k_local=1, i=0
                W[(block*m + 1)*n + 1][col] = -1  # C11: k_local=1, i=1
            elif col_offset == 2:  # M3 contributes to C01 (+1) and C11 (+1)
                W[(block*m + 0)*n + 1][col] = 1   # C01: k_local=0, i=1
                W[(block*m + 1)*n + 1][col] = 1   # C11: k_local=1, i=1
            elif col_offset == 3:  # M4 contributes to C00 (+1) and C10 (+1)
                W[(block*m + 0)*n + 0][col] = 1   # C00: k_local=0, i=0
                W[(block*m + 1)*n + 0][col] = 1   # C10: k_local=1, i=0
            elif col_offset == 4:  # M5 contributes to C00 (-1) and C01 (+1)
                W[(block*m + 0)*n + 0][col] = -1  # C00: k_local=0, i=0
                W[(block*m + 0)*n + 1][col] = 1   # C01: k_local=0, i=1
            elif col_offset == 5:  # M6 contributes to C11 (+1)
                W[(block*m + 1)*n + 1][col] = 1   # C11: k_local=1, i=1
            elif col_offset == 6:  # M7 contributes to C00 (+1)
                W[(block*m + 0)*n + 0][col] = 1   # C00: k_local=0, i=0
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
