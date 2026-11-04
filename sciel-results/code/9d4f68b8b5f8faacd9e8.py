def _minimize_over_rotation(pts, n_coarse=60, n_fine=150):
    # coarse scan to find best basin
    best_theta = 0.0
    best_side = float('inf')
    for i in range(n_coarse):
        th = (i / n_coarse) * (math.pi/3)
        s = _enclosing_hex_side(pts, th)
        if s < best_side:
            best_side = s
            best_theta = th
    # local golden section around best_theta
    span = (math.pi/3) / n_coarse
    lo = best_theta - span
    hi = best_theta + span
    # wrap into [0, pi/3] not needed for local search
    gr = (math.sqrt(5) - 1) / 2
    a = hi - gr*(hi-lo)
    b = lo + gr*(hi-lo)
    fa = _enclosing_hex_side(pts, a)
    fb = _enclosing_hex_side(pts, b)
    for _ in range(n_fine):
        if fa < fb:
            hi = b; b = a; fb = fa
            a = hi - gr*(hi-lo); fa = _enclosing_hex_side(pts, a)
        else:
            lo = a; a = b; fa = fb
            b = lo + gr*(hi-lo); fb = _enclosing_hex_side(pts, b)
    theta = (lo+hi)/2.0
    return theta, _enclosing_hex_side(pts, theta)
