    # D3 lattice minimal vectors (scaled by 2) in the last 3 dimensions
    # These are vectors with two non-zero coordinates of ±2 in the last 3 dims
    for i in range(8, 11):
        for j in range(i + 1, 11):
            for si in [-2, 2]:
                for sj in [-2, 2]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points.append(v)
   