def _score(points):
    n = len(points)
    if n != 16:
        return 0
    if len(set(map(tuple, points))) != 16:
        return 0
    for p in points:
        if abs(p[0]) > 1000 or abs(p[1]) > 1000:
            return 0

    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = A * x1 + B * y1
            
            g = _gcd(_gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                continue
            A //= g
            B //= g
            C //= g
            
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            
            key = (A, B, C)
            
            if key not in lines:
                lines[key] = set()
            
            lines[key].add(i)
            lines[key].add(j)
            
            for k in range(n):
                if k == i or k == j:
                    continue
                xk, yk = points[k]
                if A * xk + B * yk == C:
                    lines[key].add(k)
                    if len(lines[key]) >= 5:
                        return 0
                    
    score = 0
    for key, pts_in_line in lines.items():
        if len(pts_in_line) == 4:
            score += 1
    return score
