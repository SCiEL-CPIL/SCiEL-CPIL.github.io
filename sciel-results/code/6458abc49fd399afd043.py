    def compute_min_enclosing_hexagon(points):
        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
        except Exception:
            hull_points = points
            
        centroid = np.mean(hull_points, axis=0)
        centered_points = hull_points - centroid
        
        # Sample theta
        thetas = np.linspace(0, np.pi/3, 50)
        min_L = float('inf')
        best_theta = 0
        for theta in thetas:
            normals = np.array([[np.cos(theta + k * np.pi/3), np.sin(theta + k * np.pi/3)] for k in range(6)])
            projections = np.abs(centered_points @ normals.T)
            max_proj = np.max(projections)
            L = max_proj / (np.sqrt(3) / 2)
            if L < min_L:
                min_L = L
                best_theta = theta
        return min_L, best_theta
