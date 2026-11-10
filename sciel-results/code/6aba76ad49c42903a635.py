    d3_vecs = []
    for i in range(3):
        for j in range(i + 1, 3):
            k = 3 - i - j
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(3, dtype=np.int64)
                    v[i] = 2 * s1
                    v[j] = 2 * s2
                    d3_vecs.append(v)
    