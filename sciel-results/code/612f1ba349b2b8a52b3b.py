    def build_4x4x4_strassen():
        U2, V2, W2 = get_strassen_2x2()
        
        # 7 steps
        # Each step has:
        # A_coeffs: dict mapping (block_i, block_j) to coeff
        # B_coeffs: dict mapping (block_j, block_k) to coeff
        # C_mappings: list of (block_i, block_k, sign)
        
        steps = [
            # M1
            {
                'A': {(0,0): 1, (1,1): 1},
                'B': {(0,0): 1, (1,1): 1},
                'C': [(0,0, 1), (1,1, 1)]
            },
            # M2
            {
                'A': {(1,0): 1, (1,1): 1},
                'B': {(0,0): 1},
                'C': [(1,0, 1), (1,1, -1)]
            },
            # M3
            {
                'A': {(0,0): 1},
                'B': {(0,1): 1, (1,1): -1},
                'C': [(0,1, 1), (1,1, 1)]
            },
            # M4
            {
                'A': {(1,1): 1},
                'B': {(1,0): 1, (0,0): -1},
                'C': [(0,0, -1), (1,0, 1)]
            },
            # M5
            {
                'A': {(0,0): 1, (0,1): 1},
                'B': {(1,1): 1},
                'C': [(0,0, 1), (0,1, 1)]
            },
            # M6
            {
                'A': {(1,0): 1, (0,0): -1},
                'B': {(0,0): 1, (0,1): 1},
                'C': [(1,1, 1)]
            },
            # M7
            {
                'A': {(0,1): 1, (1,1): -1},
                'B': {(1,0): 1, (1,1): 1},
                'C': [(0,0, -1)]
            }
        ]
        
        rank = 49
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        
        for s_idx, step in enumerate(steps):
            r_offset = s_idx * 7
            
            # Fill U
            for (bi, bj), coeff in step['A'].items():
                # A block (bi, bj) is rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
                # U index: i*4+j
                # U2 index: li*2+lj
                for li in range(2):
                    for lj in range(2):
                        gi = 2*bi + li
                        gj = 2*bj + lj
                        U[gi*4+gj, r_offset:r_offset+7] += coeff * U2[li*2+lj, :]
                        
            # Fill V
            for (bj, bk), coeff in step['B'].items():
                # B block (bj, bk) is rows 2*bj..2*bj+1, cols 2*bk..2*bk+1
                # V index: j*4+k
                # V2 index: lj*2+lk
                for lj in range(2):
                    for lk in range(2):
                        gj = 2*bj + lj
                        gk = 2*bk + lk
                        V[gj*4+gk, r_offset:r_offset+7] += coeff * V2[lj*2+lk, :]
                        
            # Fill W
            for (bi, bk), sign in step['C']:
                # C block (bi, bk) is rows 2*bi..2*bi+1, cols 2*bk..2*bk+1
                # W index: k*4+i
                # W2 index: lk*2+li
                for li in range(2):
                    for lk in range(2):
                        gi = 2*bi + li
                        gk = 2*bk + lk
                        W[gk*4+gi, r_offset:r_offset+7] += sign * W2[lk*2+li, :]
                        
        return U, V, W
    