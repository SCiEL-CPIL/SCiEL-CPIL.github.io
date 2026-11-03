        def construct():
            n, m, p = 2, 4, 8
            rank = 56
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            col = 0
            for b in range(4):
                k0 = b * 2
                k1 = b * 2 + 1
                
                # Process T1: A_L * B_b_top
                # A_L indices in U: 0,1,4,5
                # B_b_top indices in V: 0*8+k0, 0*8+k1, 1*8+k0, 1*8+k1
                add_strassen_terms(U, V, W, col, 
                                   u_indices=[0,1,4,5], 
                                   v_indices=[0*8+k0, 0*8+k1, 1*8+k0, 1*8+k1],
                                   w_base=k0)
                col += 7
                
                # Process T2: A_R * B_b_bot
                # A_R indices in U: 2,3,6,7
                # B_b_bot indices in V: 2*8+k0, 2*8+k1, 3*8+k0, 3*8+k1
                add_strassen_terms(U, V, W, col,
                                   u_indices=[2,3,6,7],
                                   v_indices=[2*8+k0, 2*8+k1, 3*8+k0, 3*8+k1],
                                   w_base=k0)
                col += 7
            
            return {"U": U, "V": V, "W": W}
        