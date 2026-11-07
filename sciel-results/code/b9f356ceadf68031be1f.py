def construct():
    n = 10
    S1 = [[0]*n for _ in range(n)]
    S2 = [[0]*n for _ in range(n)]
    
    for r in range(n):
        r1 = r % 5
        r2 = r // 5
        for c in range(n):
            c1 = c % 5
            c2 = c // 5
            
            # S1
            u1 = (r1 + c1) % 5
            u2 = (r2 + c2) % 2
            S1[r][c] = 5 * u1 + u2
            
            # S2
            v1 = (r1 + 2 * c1) % 5
            v2 = (r2 + 2 * c2) % 2
            S2[r][c] = 5 * v1 + v2
            
    return [S1, S2]
