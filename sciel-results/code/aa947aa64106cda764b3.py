def score_points(points):
    """
    Returns (score, is_valid)
    score: number of lines with exactly 4 points
    is_valid: False if any line has >= 5 points
    """
    line_counts = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            if key is None:
                continue
            line_counts[key] = line_counts.get(key, 0) + 1
            
    score = 0
    is_valid = True
    for count in line_counts.values():
        # count is number of pairs.
        # k points -> k*(k-1)/2 pairs.
        # We need to find k such that k*(k-1)/2 == count.
        # Since max points is 16, max pairs is 120.
        # k=3 -> 3
        # k=4 -> 6
        # k=5 -> 10
        # k=6 -> 15
        # etc.
        
        # Check if count corresponds to an integer k
        # k = (1 + sqrt(1 + 8*count)) / 2
        import math
        disc = 1 + 8 * count
        sqrt_disc = math.isqrt(disc)
        if sqrt_disc * sqrt_disc != disc:
            # This shouldn't happen if counts are derived from actual points
            # But if it does, it's an error in counting? 
            # Actually, every line with k points contributes exactly C(k,2) to the count.
            # So count MUST be a triangular number.
            continue
            
        k = (1 + sqrt_disc) // 2
        
        if k >= 5:
            is_valid = False
        elif k == 4:
            score += 1
            
    return score, is_valid
