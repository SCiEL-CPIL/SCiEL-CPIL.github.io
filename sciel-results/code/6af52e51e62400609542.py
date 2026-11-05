def count_ordinary(points):
    n = len(points)
    ordinary = 0
    lines_seen = set()
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
            a = dy
            b = -dx
            c = a*x1 + b*y1
            g2 = gcd(abs(a), gcd(abs(b), abs(c)))
            if g2 > 0:
                a //= g2
                b //= g2
                c //= g2
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
            key = (a, b, c)
            if key in lines_seen:
                continue
            count = 0
            for k in range(n):
                xk, yk = points[k]
                if a*xk + b*yk == c:
                    count += 1
            lines_seen.add(key)
            if count == 2:
                ordinary += 1
    return ordinary

def construct():
    # Approach: Deterministic search over a set of structured candidates
    # This avoids the instability and speed issues of the previous local search.
    
    def make_grid(rows, cols, start_x=0, start_y=0):
        pts = []
        for r in range(rows):
            for c in range(cols):
                pts.append([start_x + c, start_y + r])
        return pts

    candidates = []
    
    # 1. Standard Grids
    candidates.append(make_grid(3, 4))
    candidates.append(make_grid(4, 3))
    candidates.append(make_grid(2, 6))
    candidates.append(make_grid(6, 2))
    
    # 2. Perturbed Grids (Shift corners or edges by 1)
    base = make_grid(3, 4)
    # Move (0,0) to (1,1) ? No, keep distinct.
    # Try small shifts
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if dx == 0 and dy == 0: continue
            p = [pt[:] for pt in base]
            # Shift first point
            p[0] = [p[0][0] + dx, p[0][1] + dy]
            if len(set(tuple(x) for x in p)) == 12:
                candidates.append(p)
                
            # Shift last point
            p = [pt[:] for pt in base]
            p[11] = [p[11][0] + dx, p[11][1] + dy]
            if len(set(tuple(x) for x in p)) == 12:
                candidates.append(p)
                
            # Shift middle of top row
            p = [pt[:] for pt in base]
            p[1] = [p[1][0] + dx, p[1][1] + dy]
            if len(set(tuple(x) for x in p)) == 12:
                candidates.append(p)
                
            # Shift middle of bottom row
            p = [pt[:] for pt in base]
            p[10] = [p[10][0] + dx, p[10][1] + dy]
            if len(set(tuple(x) for x in p)) == 12:
                candidates.append(p)

    best_points = None
    best_score = float('inf')
    
    for c in candidates:
        # Check bounds
        valid = True
        for x, y in c:
            if abs(x) > 10**9 or abs(y) > 10**9:
                valid = False
                break
        if not valid:
            continue
        if len(set(tuple(x) for x in c)) != 12:
            continue
            
        score = count_ordinary(c)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in c]
            
        if best_score <= 6:
            break
            
    return best_points
