            for (ze, zf, cz) in st['z']:
                for i in range(3):
                    for j in range(3):
                        # Find coeff for P_l in C[i,j]
                        coeff_l = 0
                        for (idx, c) in recon[i][j]:
                            if idx == l:
                                coeff_l = c
                                break
                        if coeff_l != 0:
                            # C[i,j] is at rows 2i..2i+1, cols 2j..2j+1
                            # Entry [ze, zf] is at global row 2i+ze, col 2j+zf
                            # W index: col * n + row
                            w_row = (2*j + zf) * n + (2*i + ze)
                            W[w_row][col] += coeff_l * cz
