def construct():
    vectors = []
    for i, j in combinations(range(D), 2):
        for si, sj in product((1, -1), (1, -1)):
            v = [0] * D
            v[i] = si
            v[j] = sj
            vectors.append(v)
    return np.array(vectors, dtype=np.int64)
