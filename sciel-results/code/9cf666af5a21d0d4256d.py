    for i, (idx, mask, tri) in enumerate(unique_triples):
        for v in tri:
            vertex_to_triple_indices[v].append(i)
