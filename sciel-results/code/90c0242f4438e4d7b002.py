"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Recursive Strassen decomposition for 4x4x4 matrix multiplication.
    Rank = 7 * 7 = 49.
    """
    n, m, p = DIMENSIONS  # 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Inner Strassen 2x2x2 coefficients
    # x indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # y indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # z indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    inner_u = [
        [(0, 1), (3, 1)],    # t=0: x0+x3
        [(2, 1), (3, 1)],    # t=1: x2+x3
        [(0, 1)],            # t=2: x0
        [(3, 1)],            # t=3: x3
        [(0, 1), (1, 1)],    # t=4: x0+x1
        [(2, 1), (1, -1)],   # t=5: x2-x1
        [(3, 1), (2, -1)],   # t=6: x3-x2
    ]
    
    inner_v = [
        [(0, 1), (3, 1)],    # t=0: y0+y3
        [(0, 1)],            # t=1: y0
        [(2, 1), (3, 1)],    # t=2: y2+y3
        [(2, 1)],            # t=3: y2
        [(3, 1)],            # t=4: y3
        [(0, 1), (1, 1)],    # t=5: y0+y1
        [(2, 1), (3, 1)],    # t=6: y2+y3
    ]
    
    # Z[0,0] = P1 + P4 - P5 + P7
    # Z[0,1] = P3 + P5
    # Z[1,0] = P2 + P4
    # Z[1,1] = P1 - P2 + P3 + P6
    inner_w = [
        [(0, 1), (3, 1)],    # t=0 (P1): z00+=1, z11+=1
        [(2, 1), (3, -1)],   # t=1 (P2): z10+=1, z11+=-1
        [(1, 1), (3, 1)],    # t=2 (P3): z01+=1, z11+=1
        [(0, 1), (2, 1)],    # t=3 (P4): z00+=1, z10+=1
        [(0, -1), (1, 1)],   # t=4 (P5): z00+=-1, z01+=1
        [(3, 1)],            # t=5 (P6): z11+=1
        [(0, 1), (3, -1)],   # t=6 (P7): z00+=1, z11+=-1
    ]
    
    # Outer Strassen products for 4x4
    # x_blocks: (block_row, block_col, sign) for A
    # y_blocks: (block_row, block_col, sign) for B
    # res: list of (block_row, block_col, sign) for C contribution
    
    outer_products = [
        {'x': [(0,0,1), (1,1,1)], 'y': [(0,0,1), (1,1,1)], 'res': [(0,0,1)]},
        {'x': [(1,0,1), (1,1,1)], 'y': [(0,0,1)], 'res': [(1,0,1)]},
        {'x': [(0,0,1)], 'y': [(1,0,1), (1,1,1)], 'res': [(0,1,1)]},
        {'x': [(1,1,1)], 'y': [(1,0,1)], 'res': [(0,0,1), (1,0,1)]},
        {'x': [(0,0,1), (0,1,1)], 'y': [(1,1,1)], 'res': [(0,0,-1), (0,1,1)]},
        {'x': [(1,0,1), (0,1,-1)], 'y': [(0,0,1), (0,1,1)], 'res': [(1,1,1)]},
        {'x': [(1,1,1), (1,0,-1)], 'y': [(1,0,1), (1,1,1)], 'res': [(0,0,1), (1,1,-1)]},
    ]
    
    def a_idx(bi, bj, di, dj):
        return (bi*2+di)*4 + (bj*2+dj)
    
    def b_idx(bi, bj, di, dj):
        return (bi*2+di)*4 + (bj*2+dj)
    
    def w_idx(bi, bj, di, dj):
        # C[bi*2+di, bj*2+dj] -> W[(bj*2+dj)*4 + (bi*2+di)]
        return (bj*2+dj)*4 + (bi*2+di)
    
    for s in range(7):
        prod = outer_products[s]
        for t in range(7):
            col = s * 7 + t
            
            # Build U contributions
            for di in range(2):
                for dj in range(2):
                    local_idx = di * 2 + dj
                    for (pos, coeff) in inner_u[t]:
                        if pos == local_idx:
                            for (bi, bj, sign) in prod['x']:
                                a_ind = a_idx(bi, bj, di, dj)
                                U[a_ind][col] += sign * coeff
            
            # Build V contributions
            for di in range(2):
                for dj in range(2):
                    local_idx = di * 2 + dj
                    for (pos, coeff) in inner_v[t]:
                        if pos == local_idx:
                            for (bi, bj, sign) in prod['y']:
                                b_ind = b_idx(bi, bj, di, dj)
                                V[b_ind][col] += sign * coeff
            
            # Build W contributions
            for (res_bi, res_bj, res_sign) in prod['res']:
                for di in range(2):
                    for dj in range(2):
                        local_idx = di * 2 + dj
                        for (pos, coeff) in inner_w[t]:
                            if pos == local_idx:
                                w_ind = w_idx(res_bi, res_bj, di, dj)
                                W[w_ind][col] += res_sign * coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
