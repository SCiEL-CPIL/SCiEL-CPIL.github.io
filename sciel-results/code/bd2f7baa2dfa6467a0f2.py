edge_idx = [[-1]*n for _ in range(n)]
for idx, (i, j) in enumerate(edges):
    edge_idx[i][j] = idx
    edge_idx[j][i] = idx
