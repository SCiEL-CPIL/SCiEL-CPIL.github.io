def _compute_score(points):
    """Compute the score: A_min / A_hull."""
    a_min, _ = _compute_min_triangle_area(points)
    a_hull = _compute_hull_area(points)
    if a_hull <= 0 or not np.isfinite(a_min):
        return 0.0
    return a_min / a_hull
