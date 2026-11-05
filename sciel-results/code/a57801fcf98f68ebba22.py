<<<<<<< SEARCH
for (pi, pj), val in a_coeffs.items():
                            r = I * block_size + pi
                            c = J * block_size + pj
                            U[r * m + c, col_idx] += val

                        for (pi, pj), val in b_coeffs.items():
                            r = J * block_size + pi
                            c = K * block_size + pj
                            V[r * p + c, col_idx] += val

                        for (pi, pj), val in c_coeffs.items():
                            i = I * block_size + pi
                            k = K * block_size + pj
                            W[k * n + i, col_idx] += val
=======
    for (pi, pj), val in a_coeffs.items():
        r = I * block_size + pi
        c = J * block_size + pj
        U[r * m + c, col_idx] += val

    for (pi, pj), val in b_coeffs.items():
        r = J * block_size + pi
        c = K * block_size + pj
        V[r * p + c, col_idx] += val

    for (pi, pj), val in c_coeffs.items():
        i = I * block_size + pi
        k = K * block_size + pj
        W[k * n + i, col_idx] += val
>>>>>>> REPLACE
