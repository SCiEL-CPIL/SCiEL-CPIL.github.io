def verify_code(words):
    """
    Verify all pairwise distances.
    Optimized pure Python version for small N.
    """
    n = len(words)
    if n <= 1:
        return True
    
    min_dist = MIN_DIST
    for i in range(n):
        w_i = words[i]
        for j in range(i + 1, n):
            if (w_i ^ words[j]).bit_count() < min_dist:
                return False
    return True
