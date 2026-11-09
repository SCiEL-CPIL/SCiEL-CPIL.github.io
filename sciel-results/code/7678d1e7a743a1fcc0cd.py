def _triangle_areas_fast(pts):
    n = len(pts)
    # Get all combinations of 3 indices
    # Using np.indices or similar might be memory heavy for n=13?
    # 13 choose 3 = 286. Very small.
    # But we can vectorize the computation for all 286 triangles at once.
    i, j, k = np.indices((n, n, n)).reshape(3, -1).T
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p1 = pts[i]
    p2 = pts[j]
    p3 = pts[k]
    
    # Area = 0.5 * |det(p1-p3, p2-p3)|
    # Or 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return areas
