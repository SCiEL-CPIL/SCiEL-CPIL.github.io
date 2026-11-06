def _score_and_check(points):
    """Returns (score, is_valid) where score is number of 4-point lines."""
    n = len(points)
    line_points = defaultdict(set)
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            c = dy * x1 - dx * y1
            line_key = (dx, dy, c)
            line_points[line_key].add(i)
            line_points[line_key].add(j)
            
    score = 0
    for line_key, indices in line_points.items():
        if len(indices) == 4:
            score += 1
        elif len(indices) >= 5:
            return 0, False
            
    return score, True
