# Precompute indices for i < j < k
triples = []
for i in range(N):
    for j in range(i+1, N):
        for k in range(j+1, N):
            triples.append((i, j, k))
triples = np.array(triples).T # Shape (3, 165)
idx_i = triples[0]
idx_j = triples[1]
idx_k = triples[2]
