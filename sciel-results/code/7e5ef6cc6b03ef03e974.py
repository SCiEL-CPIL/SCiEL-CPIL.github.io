import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    rank = 44
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 11 multiplications for a 3x2x2 block
    # Indices for U (i, j_local), V (j_local, k_local), W (k_local, i)
    # j_local, k_local in {0, 1}
    
    # m0: (X00 + X11)(Y00 + Y11)
    # U: (0,0), (1,1)
    # V: (0,0), (1,1)
    # W: (0,0) coeff 1, (1,1) coeff 1 -> C00, C11
    # C00 is i=0, k=0. C11 is i=1, k=1.
    
    # m1: (X10 + X11)Y00
    # U: (1,0), (1,1)
    # V: (0,0)
    # W: (0,1) coeff 1, (1,1) coeff -1 -> C10, C11
    
    # m2: X00(Y10 - Y11)
    # U: (0,0)
    # V: (1,0), (1,1)
    # W: (1,0) coeff 1 -> C01
    
    # m3: X11(Y01 - Y00)
    # U: (1,1)
    # V: (0,1), (0,0)
    # W: (0,0) coeff 1, (0,1) coeff 1 -> C00, C10
    
    # m4: (X00 + X10)Y11
    # U: (0,0), (1,0)
    # V: (1,1)
    # W: (0,0) coeff -1, (1,0) coeff 1 -> C00, C01
    
    # m5: (X10 - X00)(Y00 + Y01)
    # U: (1,0), (0,0)
    # V: (0,0), (0,1)
    # W: (1,1) coeff 1 -> C11
    
    # m6: (X01 - X11)(Y10 + Y11)
    # U: (0,1), (1,1)
    # V: (1,0), (1,1)
    # W: (0,0) coeff 1 -> C00
    
    # m7: X20 Y00
    # U: (2,0)
    # V: (0,0)
    # W: (0,2) coeff 1 -> C20
    
    # m8: X21 Y10
    # U: (2,1)
    # V: (1,0)
    # W: (0,2) coeff 1 -> C20
    
    # m9: X20 Y01
    # U: (2,0)
    # V: (0,1)
    # W: (1,2) coeff 1 -> C21
    
    # m10: X21 Y11
    # U: (2,1)
    # V: (1,1)
    # W: (1,2) coeff 1 -> C21

    # List of (u_indices, v_indices, w_indices) for the 11 mults
    # u_indices: list of (i, j) and coeff
    # v_indices: list of (j, k) and coeff
    # w_indices: list of (k, i) and coeff
    
    # Note: U index is i*m + j. V index is j*p + k. W index is k*n + i.
    # Here m=4, p=4, n=3.
    # But for the block, we use local j, k.
    # We will shift these based on the block.
    
    block_defs = []
    
    # m0
    block_defs.append((
        [((0, 0), 1), ((1, 1), 1)],
        [((0, 0), 1), ((1, 1), 1)],
        [((0, 0), 1), ((1, 1), 1)]
    ))
    # m1
    block_defs.append((
        [((1, 0), 1), ((1, 1), 1)],
        [((0, 0), 1)],
        [((0, 1), 1), ((1, 1), -1)]
    ))
    # m2
    block_defs.append((
        [((0, 0), 1)],
        [((1, 0), 1), ((1, 1), -1)],
        [((1, 0), 1)]
    ))
    # m3
    block_defs.append((
        [((1, 1), 1)],
        [((0, 1), 1), ((0, 0), -1)],
        [((0, 0), 1), ((0, 1), 1)]
    ))
    # m4
    block_defs.append((
        [((0, 0), 1), ((1, 0), 1)],
        [((1, 1), 1)],
        [((0, 0), -1), ((1, 0), 1)]
    ))
    # m5
    block_defs.append((
        [((1, 0), 1), ((0, 0), -1)],
        [((0, 0), 1), ((0, 1), 1)],
        [((1, 1), 1)]
    ))
    # m6
    block_defs.append((
        [((0, 1), 1), ((1, 1), -1)],
        [((1, 0), 1), ((1, 1), 1)],
        [((0, 0), 1)]
    ))
    # m7
    block_defs.append((
        [((2, 0), 1)],
        [((0, 0), 1)],
        [((0, 2), 1)]
    ))
    # m8
    block_defs.append((
        [((2, 1), 1)],
        [((1, 0), 1)],
        [((0, 2), 1)]
    ))
    # m9
    block_defs.append((
        [((2, 0), 1)],
        [((0, 1), 1)],
        [((1, 2), 1)]
    ))
    # m10
    block_defs.append((
        [((2, 1), 1)],
        [((1, 1), 1)],
        [((1, 2), 1)]
    ))
    
    # 4 Blocks
    # Block 0: j_shift=0, k_shift=0, w_k_shift=0
    # Block 1: j_shift=2, k_shift=0, w_k_shift=0
    # Block 2: j_shift=0, k_shift=2, w_k_shift=2
    # Block 3: j_shift=2, k_shift=2, w_k_shift=2
    
    for b in range(4):
        base_col = b * 11
        j_shift = (b % 2) * 2
        k_shift = (b // 2) * 2
        w_k_shift = k_shift # W depends on global k
        
        for m_idx in range(11):
            col = base_col + m_idx
            u_defs, v_defs, w_defs = block_defs[m_idx]
            
            for (i, j_local), coeff in u_defs:
                j_global = j_local + j_shift
                u_idx = i * m + j_global
                U[u_idx][col] += coeff
                
            for (j_local, k_local), coeff in v_defs:
                j_global = j_local + j_shift
                k_global = k_local + k_shift
                v_idx = j_global * p + k_global
                V[v_idx][col] += coeff
                
            for (k_local, i), coeff in w_defs:
                k_global = k_local + w_k_shift
                w_idx = k_global * n + i
                W[w_idx][col] += coeff
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
