# ... inside add_strassen_block
            # Update W
            # w_c is [C00, C01, C10, C11] -> index i*2 + k_local
            for k_local in range(2):
                for i in range(n):
                    # w_c index: i * 2 + k_local
                    w_idx = (k_offset + k_local) * n + i
                    W[w_idx][col] += w_c[i * 2 + k_local]
