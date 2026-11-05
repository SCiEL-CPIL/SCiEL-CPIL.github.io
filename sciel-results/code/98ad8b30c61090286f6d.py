def _schoolbook_3x3_terms():
    terms = []
    for i in range(3):
        for k in range(3):
            for j in range(3):
                u = [0]*9; v = [0]*9; w = [0]*9
                u[i*3+k] = 1
                v[k*3+j] = 1
                w[j*3+i] = 1
                terms.append((u, v, w))
    return terms
