for i in range(2):
    for j in range(2):
        for (r, c, coeff) in a_blocks:
            global_i = r + i
            global_j = c + j
            u[global_i * m + global_j] += coeff * t['A'][i][j]
