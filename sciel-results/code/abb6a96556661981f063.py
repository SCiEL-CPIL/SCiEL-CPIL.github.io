def check_overlap(inner, side=1.0, tol=1e-9):
    """Check if any two inner hexagons overlap."""
    min_dist_sq = (np.sqrt(3) * side + tol) ** 2
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, _ = inner[i]
            x2, y2, _ = inner[j]
            dx = x1 - x2
            dy = y1 - y2
            if dx * dx + dy * dy < min_dist_sq:
                return False
    return True
