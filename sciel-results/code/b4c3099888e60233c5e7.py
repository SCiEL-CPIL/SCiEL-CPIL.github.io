    idx_new = 0
    for i in range(m):
        if i == a or i == b:
            continue
        for j in range(m):
            if j == a or j == b:
                continue
            if i == j:
                continue
            new_dist_matrix[idx_new, idx_new] = 0
            idx_j = 0
            for jj in range(m):
                if jj == a or jj == b:
                    continue
                if jj == j:
                    new_dist_matrix[idx_new, idx_j] = current_dist[i, j]
                    break
                idx_j += 1
        idx_new += 1
    