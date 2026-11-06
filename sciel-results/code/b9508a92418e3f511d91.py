    result = []
    for e_idx in range(total_edges):
        if best_set is not None and best_set[e_idx]:
            u, v = all_edges[e_idx]
            result.append([u, v])
    
    return result
