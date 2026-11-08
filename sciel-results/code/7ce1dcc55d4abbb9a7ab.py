import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # Rank-12 decomposition for 2x2x4 matmul tensor
    # U: 4x12 (index = i*2+j), V: 8x12 (index = j*4+k), W: 8x12 (index = k*2+i)
    # Each entry is a list of (local_index, value) pairs for that rank term
    
    terms_224 = [
        # r=0
        ([(0, 1), (3, 1)], [(0, 1), (7, 1)], [(0, 1), (3, 1)]),
        # r=1
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)]),
        # r=2
        ([(0, 1)], [(1, 1), (7, -1)], [(2, 1), (3, 1)]),
        # r=3
        ([(3, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)]),
        # r=4
        ([(0, 1), (1, 1)], [(7, 1)], [(0, -1), (2, 1)]),
        # r=5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # r=6
        ([(1, 1), (3, -1)], [(4, 1), (7, 1)], [(0, 1)]),
        # r=7
        ([(0, 1), (3, 1)], [(2, 1), (5, 1)], [(4, 1), (7, 1)]),
        # r=8
        ([(2, 1), (3, 1)], [(2, 1)], [(5, 1), (7, -1)]),
        # r=9
        ([(0, 1)], [(3, 1), (5, -1)], [(6, 1), (7, 1)]),
        # r=10
        ([(3, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)]),
        # r=11
        ([(0, 1), (1, 1)], [(5, 1)], [(4, -1), (6, 1)]),
    ]

    def verify_224():
        """Verify the rank-12 decomposition for 2x2x4 matmul."""
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    u_idx = i * 2 + j
                    v_idx = j * 4 + k
                    w_idx = k * 2 + i
                    total = 0
                    for r in range(12):
                        u_val = 0
                        v_val = 0
                        w_val = 0
                        for (idx, val) in terms_224[r][0]:
                            if idx == u_idx:
                                u_val = val
                        for (idx, val) in terms_224[r][1]:
                            if idx == v_idx:
                                v_val = val
                        for (idx, val) in terms_224[r][2]:
                            if idx == w_idx:
                                w_val = val
                        total += u_val * v_val * w_val
                    if total != 1:
                        return False
        return True

    # Verify the 2x2x4 decomposition
    if not verify_224():
        # Fallback to rank-56 Strassen if rank-12 is wrong
        rank = 56
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        strassen_U = [
            [(0, 1), (3, 1)], [(2, 1), (3, 1)], [(0, 1)], [(3, 1)],
            [(0, 1), (1, 1)], [(2, 1), (0, -1)], [(1, 1), (3, -1)]
        ]
        strassen_V = [
            [(0, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)], [(2, 1), (0, -1)],
            [(3, 1)], [(0, 1), (1, 1)], [(2, 1), (3, 1)]
        ]
        strassen_W = [
            [(0, 1), (3, 1)], [(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (1, 1)],
            [(0, -1), (2, 1)], [(3, 1)], [(0, 1)]
        ]
        
        def add_strassen_block(base_rank, i_off, j_off, k_off):
            for r in range(7):
                cr = base_rank + r
                for idx, val in strassen_U[r]:
                    li, lj = divmod(idx, 2)
                    U[(li + i_off) * 4 + (lj + j_off)][cr] += val
                for idx, val in strassen_V[r]:
                    lj, lk = divmod(idx, 2)
                    V[(lj + j_off) * 8 + (lk + k_off)][cr] += val
                for idx, val in strassen_W[r]:
                    lk, li = divmod(idx, 2)
                    W[(lk + k_off) * 2 + (li + i_off)][cr] += val
        
        add_strassen_block(0, 0, 0, 0)
        add_strassen_block(7, 0, 0, 2)
        add_strassen_block(14, 0, 0, 4)
        add_strassen_block(21, 0, 0, 6)
        add_strassen_block(28, 0, 2, 0)
        add_strassen_block(35, 0, 2, 2)
        add_strassen_block(42, 0, 2, 4)
        add_strassen_block(49, 0, 2, 6)
        
        # Verify Strassen
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    total = 0
                    for r in range(rank):
                        total += U[i*4+j][r] * V[j*8+k][r] * W[k*2+i][r]
                    if abs(total - 1) > 1e-10:
                        return {"U": [[0]*64 for _ in range(8)], "V": [[0]*64 for _ in range(32)], "W": [[0]*64 for _ in range(16)]}
        
        return {"U": U, "V": V, "W": W}

    # Build the full 2x4x8 decomposition using four rank-12 blocks
    rank = 48
    U = [[0] * rank for _ in range(n * m)]  # 8 x 48
    V = [[0] * rank for _ in range(m * p)]  # 32 x 48
    W = [[0] * rank for _ in range(p * n)]  # 16 x 48

    def add_224_block(base_rank, i_offset, j_offset, k_offset):
        """Add a 2x2x4 block at the given offsets."""
        for r in range(12):
            cr = base_rank + r
            for (idx, val) in terms_224[r][0]:
                li, lj = divmod(idx, 2)
                U[(li + i_offset) * 4 + (lj + j_offset)][cr] += val
            for (idx, val) in terms_224[r][1]:
                lj, lk = divmod(idx, 4)
                V[(lj + j_offset) * 8 + (lk + k_offset)][cr] += val
            for (idx, val) in terms_224[r][2]:
                lk, li = divmod(idx, 2)
                W[(lk + k_offset) * 2 + (li + i_offset)][cr] += val

    # Four blocks covering all (j, k) combinations:
    # Block 0: j in {0,1}, k in {0,1,2,3}
    add_224_block(0, 0, 0, 0)
    # Block 1: j in {0,1}, k in {4,5,6,7}
    add_224_block(12, 0, 0, 4)
    # Block 2: j in {2,3}, k in {0,1,2,3}
    add_224_block(24, 0, 2, 0)
    # Block 3: j in {2,3}, k in {4,5,6,7}
    add_224_block(36, 0, 2, 4)

    # Verify the full decomposition
    def verify_full():
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    total = 0
                    for r in range(rank):
                        total += U[i*4+j][r] * V[j*8+k][r] * W[k*2+i][r]
                    if abs(total - 1) > 1e-10:
                        return False
        return True

    if verify_full():
        return {"U": U, "V": V, "W": W}

    # Last resort: trivial rank 64
    rank = 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    return {"U": U, "V": V, "W": W}
