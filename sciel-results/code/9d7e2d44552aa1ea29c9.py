import time

N = 64
DEADLINE = 320.0  # safety margin before 330s kill

def _forbidden_for_pair(a, b):
    """Return the set of grid points P in [0,N)^2 that, together with a and b,
    form an isosceles triangle (i.e. any two of PA, PB, AB are equal)."""
    x1, y1 = a
    x2, y2 = b
    dx = x2 - x1
    dy = y2 - y1
    d2 = dx * dx + dy * dy
    forbidden = set()

    # --- Apex case: |P - A| == |P - B|  (P on perpendicular bisector) ---
    # 2*dx*x + 2*dy*y = (x2^2 - x1^2) + (y2^2 - y1^2)
    C = (x2 * x2 - x1 * x1) + (y2 * y2 - y1 * y1)
    Acoef = 2 * dx
    Bcoef = 2 * dy
    if Acoef == 0 and Bcoef == 0:
        return forbidden  # a == b, should not happen
    if Acoef == 0:
        # Bcoef*y = C
        if C % Bcoef == 0:
            y = C // Bcoef
            if 0 <= y < N:
                for x in range(N):
                    forbidden.add((x, y))
    elif Bcoef == 0:
        # Acoef*x = C
        if C % Acoef == 0:
            x = C // Acoef
            if 0 <= x < N:
                for y in range(N):
                    forbidden.add((x, y))
    else:
        # Acoef*x + Bcoef*y = C ; enumerate x, solve for y
        for x in range(N):
            rem = C - Acoef * x
            if rem % Bcoef == 0:
                y = rem // Bcoef
                if 0 <= y < N:
                    forbidden.add((x, y))

    # --- Base case: |P - A| == d  (circle radius sqrt(d2) around A) ---
    # (x - x1)^2 + (y - y1)^2 == d2
    # and |P - B| == d similarly.
    # Enumerate x, solve for y.
    for x in range(N):
        dxp = x - x1
        t = d2 - dxp * dxp
        if t >= 0:
            y1sq = y1
            # y^2 - 2*y1*y + (y1^2 - t) == 0  -> y = y1 +/- sqrt(t)
            # check integer y
            # y must satisfy (y - y1)^2 == t
            # so y = y1 + s where s^2 == t, s integer, s >= 0, and also y1 - s
            # find integer s with s*s == t
            s = int(round(t ** 0.5))
            for cand in (s, -s):
                if cand * cand == t:
                    y = y1 + cand
                    if 0 <= y < N:
                        forbidden.add((x, y))
    # circle around B
    for x in range(N):
        dxp = x - x2
        t = d2 - dxp * dxp
        if t >= 0:
            s = int(round(t ** 0.5))
            for cand in (s, -s):
                if cand * cand == t:
                    y = y2 + cand
                    if 0 <= y < N:
                        forbidden.add((x, y))

    return forbidden


def construct():
    start = time.time()
    best = []

    # Order candidate points in a deterministic but spread-out way.
    # Use a space-filling-ish ordering: iterate over a shuffled grid but
    # keep deterministic seed for reproducibility.
    import random
    rng = random.Random(12345)
    candidates = [(x, y) for x in range(N) for y in range(N)]
    rng.shuffle(candidates)

    points = []
    forbidden = set()  # points that would create an isosceles triangle

    for p in candidates:
        if time.time() > DEADLINE:
            break
        if p in forbidden:
            continue
        # p is safe. Add it. Now forbid all points that would form an
        # isosceles triangle with p and each existing point.
        new_forbidden = set()
        for q in points:
            new_forbidden |= _forbidden_for_pair(p, q)
        forbidden |= new_forbidden
        points.append(p)

    # If we ran out of time, return what we have; else return all.
    if points:
        best = points
    else:
        # Fallback: at least return a valid small set.
        best = [(0, 0), (1, 0)]

    return best
