<<<<<<< SEARCH
        # For j=4, add rank-1 terms for each (i, k)
        offset = 77
        for i in range(4):
            for k in range(6):
                U[i * m + 4][offset] = 1
                V[4 * p + k][offset] = 1
                W[k * n + i][offset] = 1
                offset += 1
=======
        # For j=4, use a single rank-1 term (outer product structure)
        offset = 77
        for i in range(4):
            U[i * m + 4][offset] = 1
        for k in range(6):
            V[4 * p + k][offset] = 1
        for i in range(4):
            for k in range(6):
                W[k * n + i][offset] = 1
        offset += 1
>>>>>>> REPLACE
