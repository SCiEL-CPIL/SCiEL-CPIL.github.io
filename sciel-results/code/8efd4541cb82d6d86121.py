<<<<<<< SEARCH
def _line_key(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    g = math.gcd(abs(dy), abs(dx))
    if g == 0:
        return None
    a = dy // g
    b = -dx // g
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
    c = a * x1 + b * y1
    return (a, b, c)
=======
def _line_key(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    g = math.gcd(abs(dy), abs(dx))
    if g == 0:
        return None
    a = dy // g
    b = -dx // g
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
    c = a * x1 + b * y1
    return (a, b, c)

def _is_all_collinear(points):
    if len(points) <= 2:
        return True
    p0 = points[0]
    p1 = points[1]
    for p in points[2:]:
        key = _line_key(p0[0], p0[1], p1[0], p1[1])
        if key is None:
            return False
        a, b, c = key
        if a * p[0] + b * p[1] != c:
            return False
    return True
>>>>>>> REPLACE

