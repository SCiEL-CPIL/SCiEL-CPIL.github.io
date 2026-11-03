def _convex_hull_area(points):
    """
    Compute the area of the convex hull of a set of 2D points.
    """
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    # Find a point on the hull (e.g., the one with min y, then min x)
    min_idx = np.lexsort((points[:, 0], points[:, 1]))[0]
    pivot = points[min_idx]
    
    # Sort other points by angle relative to pivot
    other_pts = np.delete(points, min_idx, axis=0)
    angles = np.arctan2(other_pts[:, 1] - pivot[1], other_pts[:, 0] - pivot[0])
    sorted_indices = np.argsort(angles)
    sorted_pts = other_pts[sorted_indices]
    
    # Andrew's monotone chain or just shoelace on the sorted points?
    # Sorting by angle around a pivot gives the convex hull order IF the pivot is on the hull.
    # The min-y point is always on the hull.
    # However, if there are collinear points, this might include interior points if not careful.
    # But for area calculation, including interior points in the "polygon" defined by sorted angles
    # will result in a self-intersecting or non-convex polygon if not strictly convex.
    # Actually, sorting by angle around a hull point yields the convex hull vertices in order,
    # provided we remove points that are "inside" the triangle formed by neighbors.
    
    # Simpler: Just use the standard gift wrapping or monotone chain.
    # Given N=14, even O(N^2) is fine.
    
    # Let's use the property that the area is half the sum of cross products of hull vertices.
    # We need the hull vertices in CCW order.
    
    # Sort all points by x, then y
    sort_idx = np.lexsort((points[:, 1], points[:, 0]))
    sorted_points = points[sort_idx]
    
    # Lower hull
    lower = []
    for p in sorted_points:
        while len(lower) >= 2 and np.cross(lower[-1] - lower[-2], p - lower[-2]) <= 0:
            lower.pop()
        lower.append(p)
        
    # Upper hull
    upper = []
    for p in sorted_points[::-1]:
        while len(upper) >= 2 and np.cross(upper[-1] - upper[-2], p - upper[-2]) <= 0:
            upper.pop()
        upper.append(p)
        
    # Concatenate, removing last point of each (duplicates)
    hull = np.vstack([lower[:-1], upper[:-1]])
    
    if len(hull) < 3:
        return 0.0
        
    # Shoelace formula
    x = hull[:, 0]
    y = hull[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area
