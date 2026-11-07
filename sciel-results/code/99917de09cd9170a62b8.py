def construct():
    n = 16
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            result.append([u, v])
    return result
