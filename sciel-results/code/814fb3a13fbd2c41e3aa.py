    cycle_others = [[] for _ in range(M)]
    for c_idx, c in enumerate(cycles):
        for i in range(4):
            e = c[i]
            others = [c[j] for j in range(4) if j != i]
            cycle_others[e].append(tuple(others))
