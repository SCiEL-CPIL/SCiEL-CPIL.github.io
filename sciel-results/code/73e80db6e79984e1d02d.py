<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    base_points, D = _build_known_5points()
    if not _validate(base_points, D):
        raise RuntimeError("Base configuration failed validation")
    
    best = list(base_points)
    
    result = _try_pythagorean_expansion(best, D, deadline)
    if result is not None and len(result) > len(best):
        best = result
    
    expanded = _search_expand(best, D, deadline, max_extra=3)
    if len(expanded) > len(best):
        best = expanded
    
    if time.time() < deadline - 10:
        rng = random.Random(999)
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
            pts = []
            for _ in range(6):
                x = rng.randint(-200, 200)
                y = rng.randint(-200, 200)
                pts.append((Fraction(x), Fraction(y)))
            if _validate(pts, D) and len(pts) > len(best):
                best = pts
                break
    
    result_points = []
    for x, y in best:
        result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': D, 'points': result_points}
=======
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Try to build a 6-point configuration first
    base_points, D = _build_known_6points()
    if not _validate(base_points, D):
        # Fall back to 5 points
        base_points, D = _build_known_5points()
        if not _validate(base_points, D):
            raise RuntimeError("Base configuration failed validation")
    
    best = list(base_points)
    
    # Try to expand further
    expanded = _search_expand(best, D, deadline, max_extra=5)
    if len(expanded) > len(best):
        best = expanded
    
    # If we have time, try random restarts with different starting points
    if time.time() < deadline - 20:
        rng = random.Random(999)
        for attempt in range(50):
            if time.time() > deadline - 20:
                break
            pts = []
            for _ in range(6):
                x = rng.randint(-300, 300)
                y = rng.randint(-300, 300)
                pts.append((Fraction(x), Fraction(y)))
            if _validate(pts, D) and len(pts) > len(best):
                best = pts
                break
    
    result_points = []
    for x, y in best:
        result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': D, 'points': result_points}
>>>>>>> REPLACE
