def construct():
    n = 10
    L1 = [[0]*n for _ in range(n)]
    for r in range(n):
        i = r // 2
        j = r % 2
        for c in range(n):
            k = c // 2
            l = c % 2
            L1[r][c] = 2 * ((i + k) % 5) + (j + l) % 2

    L2 = [[0]*n for _ in range(n)]
    for r in range(n):
        i = r // 2
        j = r % 2
        for c in range(n):
            k = c // 2
            l = c % 2
            L2[r][c] = 2 * ((i + 2 * k + l) % 5) + (j + k) % 2

    # Verify L2 is a Latin square and orthogonal to L1
    for r in range(n):
        if sorted(L2[r]) != list(range(n)):
            return [L1]
    for c in range(n):
        col = [L2[r][c] for r in range(n)]
        if sorted(col) != list(range(n)):
            return [L1]
            
    pairs = set()
    for r in range(n):
        for c in range(n):
            p = (L1[r][c], L2[r][c])
            if p in pairs:
                return [L1]
            pairs.add(p)
            
    return [L1, L2]
