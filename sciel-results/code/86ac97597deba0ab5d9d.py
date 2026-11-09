def calculate_score(edge_set, degs):
    max_sum = 0
    es = edge_set
    ds = degs
    for i, j, k in triangles:
        e1 = (i, j) if i < j else (j, i)
        e2 = (j, k) if j < k else (k, j)
        e3 = (i, k) if i < k else (k, i)
        if e1 in es and e2 in es and e3 in es:
            s = ds[i] + ds[j] + ds[k]
            if s > max_sum:
                max_sum = s
    return max_sum
