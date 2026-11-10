    for i in range(32):
        u = node_to_class[i]
        for k in range(5):
            j = i ^ (1 << k)
            v = node_to_class[j]
            if u < v:
                edges_set.add((u, v))
