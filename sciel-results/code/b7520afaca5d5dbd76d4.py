def bitonic_sort(n):
    comps = []
    for k in range(2, n + 1, 2):
        for i in range(0, n, k):
            for j in range(k // 2, 0, -1):
                for p in range(0, k, 2):
                    if p + j < k:
                        a = i + p
                        b = i + p + j
                        if a < b:
                            comps.append((a, b))
    return list(set(comps))
