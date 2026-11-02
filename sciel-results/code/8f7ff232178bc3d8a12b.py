    def add_schoolblock(col_start, k_start, num_cols):
        for i in range(num_cols):
            k = k_start + i
            base = col_start + i
            
            # Term 1: A00 * B0,k
            U[0, base] += 1
            V[0 * 7 + k, base] += 1
            W[2 * k + 0, base] += 1
            
            # Term 2: A01 * B1,k
            U[1, base + 1] += 1
            V[1 * 7 + k, base + 1] += 1
            W[2 * k + 0, base + 1] += 1
            
            # Term 3: A10 * B0,k
            U[2, base + 2] += 1
            V[0 * 7 + k, base + 2] += 1
            W[2 * k + 1, base + 2] += 1
            
            # Term 4: A11 * B1,k
            U[3, base + 3] += 1
            V[1 * 7 + k, base + 3] += 1
            W[2 * k + 1, base + 3] += 1
    