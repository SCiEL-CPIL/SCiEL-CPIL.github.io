        for p in range(2, target + 1):
            if P[p] == 0:
                for j in range(p, target + 1, p):
                    P[j] = p
        