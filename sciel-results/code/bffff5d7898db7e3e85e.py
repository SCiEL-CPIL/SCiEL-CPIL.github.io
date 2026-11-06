        for i in range(n_moduli):
            m = moduli[i]
            for r in range(m):
                if is_valid(i, r):
                    classes.append([m, r])
                    assigned[i] = r
                    break
