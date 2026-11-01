    idx_map = np.zeros((v, v, v), dtype=np.int32)
    rank = 0
    for a in range(v):
        for b in range(a + 1, v):
            for c in range(b + 1, v):
                idx_map[a, b, c] = rank
                rank += 1
    n_triples = rank
