<<<<<<< SEARCH
            if score_s == 0:
                colors_out = []
                for i in range(n_new):
                    for j in range(i + 1, n_new):
                    colors_out.append(color_s[i][j])
                return {'n': n_new, 'colors': colors_out}
=======
            if score_s == 0:
                colors_out = []
                for i in range(n_new):
                    for j in range(i + 1, n_new):
                        colors_out.append(color_s[i][j])
                return {'n': n_new, 'colors': colors_out}
>>>>>>> REPLACE
