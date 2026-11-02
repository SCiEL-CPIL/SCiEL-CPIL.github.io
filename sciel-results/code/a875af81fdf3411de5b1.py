def get_all_areas_vectorized(points):
    n = len(points)
    if n < 3:
        return np.array([])
    
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    det = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
          (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
          
    areas = 0.5 * np.abs(det)
    
    return areas
