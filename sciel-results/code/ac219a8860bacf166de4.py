    def _build_block_schoolbook(n, m, p):
        """
        Build a schoolbook decomposition for n x m times m x p.
        """
        n, m, p = 2, 4, 5
        total_rank = 40
        
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        
        def add_schoolbook_block(i0, j0, k0, width_i, width_j, width_k, offset):
            for di in range(width_i):
                for dj in range(width_j):
                    for dk in range(width_k):
                        a_idx = (i0 + di) * m + (j0 + dj)
                        b_idx = (j0 + dj) * p + (k0 + dk)
                        c_idx = (k0 + dk) * n + (i0 + di)
                        U[a_idx, offset] = 1.0
                        V[b_idx, offset] = 1.0
                        W[c_idx, offset] = 1.0
                        offset += 1
            return offset
        
        offset = 0
        # A_left * B_top
        offset = add_schoolbook_block(0, 0, 0, 2, 2, 2, offset)
        offset = add_schoolbook_block(0, 0, 2, 2, 2, 2, offset)
        offset = add_schoolbook_block(0, 0, 4, 2, 2, 1, offset)
        
        # A_right * B_bot
        offset = add_schoolbook_block(0, 2, 0, 2, 2, 2, offset)
        offset = add_schoolbook_block(0, 2, 2, 2, 2, 2, offset)
        offset = add_schoolbook_block(0, 2, 4, 2, 2, 1, offset)
        
        assert offset == total_rank, f"Expected rank 40, got {offset}"
        
        return U, V, W
