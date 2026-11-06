for u, v in ap3s_by_elem[x]:
    if in_A[u] and in_A[v]:
        pass # Complete
    elif in_A[u] and not in_A[v]:
        missing_3ap[v] += 1
    elif not in_A[u] and in_A[v]:
        missing_3ap[u] += 1
