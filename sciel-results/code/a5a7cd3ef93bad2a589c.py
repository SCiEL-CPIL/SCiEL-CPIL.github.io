for i in range(2):
    for j in range(2):
        for (r, c, coeff) in a_blocks:
            # ...
            u[global_i * m + global_j] += coeff * t['A'][i][j]
