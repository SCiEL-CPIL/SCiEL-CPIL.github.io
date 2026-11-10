def is_valid(points):
    if points.shape != (33, 2):
        return False
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return False
    if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
        return False
    # Check distances
    # 33 points, 33*32/2 = 528 pairs
    for i in range(33):
        for j in range(i+1, 33):
            d2 = np.sum((points[i] - points[j])**2)
            if d2 < 1e-12:
                return False
    # Check collinearity
    for i in range(33):
        for j in range(i+1, 33):
            for k in range(j+1, 33):
                cross = points[i][0]*(points[j][1]-points[k][1]) + points[j][0]*(points[k][1]-points[i][1]) + points[k][0]*(points[i][1]-points[j][1])
                if abs(cross) < 1e-9:
                    return False
    return True
