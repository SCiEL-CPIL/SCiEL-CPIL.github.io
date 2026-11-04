<<<<<<< SEARCH
        # Apply naive blocks for j=4
        for k_base in [0, 2, 4]:
            add_naive_2112_block(0, 4, k_base)
        
=======
        # Apply rank-1 block for j=4
        for i in range(n):
            U[i * m + 4, col] = 1
        for k in range(p):
            V[4 * p + k, col] = 1
            for i in range(n):
                W[k * n + i, col] = 1
        col += 1
        
>>>>>>> REPLACE
