<<<<<<< SEARCH
            for lr, lc, coeff in term['U']:
                for i in range(3):
                    for j in range(3):
                        u_row = (2 * lr + i) * m + (2 * lc + j)
                        U[u_row, col] += coeff * U3[i * 3 + j, r_3]
            
            for lr, lc, coeff in term['V']:
                for j in range(3):
                    for k in range(3):
                        v_row = (2 * lr + j) * p + (2 * lc + k)
                        V[v_row, col] += coeff * V3[j * 3 + k, r_3]
            
            for lr, lc, coeff in term['W']:
                for i in range(3):
                    for k in range(3):
                        w_row = (2 * lc + k) * n + (2 * lr + i)
                        W[w_row, col] += coeff * W3[k * 3 + i, r_3]
=======
            for lr, lc, coeff in term['U']:
                for i in range(3):
                    for j in range(3):
                        u_row = (3 * lr + i) * m + (3 * lc + j)
                        U[u_row, col] += coeff * U3[i * 3 + j, r_3]
            
            for lr, lc, coeff in term['V']:
                for j in range(3):
                    for k in range(3):
                        v_row = (3 * lr + j) * p + (3 * lc + k)
                        V[v_row, col] += coeff * V3[j * 3 + k, r_3]
            
            for lr, lc, coeff in term['W']:
                for i in range(3):
                    for k in range(3):
                        w_row = (3 * lc + k) * n + (3 * lr + i)
                        W[w_row, col] += coeff * W3[k * 3 + i, r_3]
>>>>>>> REPLACE
