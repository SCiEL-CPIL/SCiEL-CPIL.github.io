    def strassen_4x4x1_tail():
        """
        Generates terms for 4x4 * 4x1 (col 6) using Strassen with zeroed columns.
        Local cols 1, 2, 3 are zero. Only col 0 is non-zero.
        """
        outer_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        inner_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]

        def get_block_indices_b(block_idx, local_idx):
            br = block_idx // 2
            bc = block_idx % 2
            lr = local_idx // 2
            lc = local_idx % 2
            
            row = br * 2 + lr
            local_col = bc * 2 + lc
            # Only local_col 0 is non-zero for this tail
            if local_col != 0:
                return None
            global_col = 6 + local_col
            return row * p + global_col

        for outer_a, outer_b, outer_c in outer_terms:
            for inner_x, inner_y, inner_z in inner_terms:
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                
                # Accumulate U
                for x_local, x_coeff in inner_x:
                    for a_block, a_coeff in outer_a:
                        br = a_block // 2
                        bc = a_block % 2
                        lr = x_local // 2
                        lc = x_local % 2
                        row = br * 2 + lr
                        col = bc * 2 + lc
                        idx = row * m + col
                        u[idx] += a_coeff * x_coeff
                        
                # Accumulate V
                v_non_zero = False
                for y_local, y_coeff in inner_y:
                    for b_block, b_coeff in outer_b:
                        idx = get_block_indices_b(b_block, y_local)
                        if idx is not None:
                            v[idx] += b_coeff * y_coeff
                            v_non_zero = True
                            
                if not v_non_zero:
                    continue

                # Accumulate W
                for z_local, z_coeff in inner_z:
                    for c_block, c_coeff in outer_c:
                        br = c_block // 2
                        bc = c_block % 2
                        lr = z_local // 2
                        lc = z_local % 2
                        row = br * 2 + lr
                        col = bc * 2 + lc
                        global_col = 6 + col
                        idx = global_col * n + row
                        w[idx] += c_coeff * z_coeff
                        
                add_term(u, v, w)
