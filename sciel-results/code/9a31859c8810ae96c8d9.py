valid = True
for i in range(2):
    for j in range(2):
        for k in range(2):
            sum_val = 0
            for u, v, w in strassen_terms:
                if (i,j) in u and (j,k) in v and (i,k) in w:
                    sum_val += u[(i,j)] * v[(j,k)] * w[(i,k)]
            if abs(sum_val - 1) > 1e-9:
                valid = False
                break
