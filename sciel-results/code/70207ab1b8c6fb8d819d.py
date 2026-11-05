def count_ordinary(points):
    lines = {}
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Normalize line equation ax + by + c = 0
            a = y1 - y2
            b = x2 - x1
            c = x1*y2 - x2*y1
            # Normalize
            g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
            if g != 0:
                a //= g
                b //= g
                c //= g
            # Ensure consistent sign
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
            key = (a, b, c)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    
    ordinary = 0
    for key, pts in lines.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary
