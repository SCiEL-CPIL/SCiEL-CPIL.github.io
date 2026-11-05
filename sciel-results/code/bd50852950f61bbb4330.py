    # Safety check: ensure we return a valid configuration
    if best_points is None or not _is_valid(best_points):
        # Fallback to a known valid configuration (4x4 grid)
        best_points = [[x, y] for x in range(4) for y in range(4)]
        
    return best_points
