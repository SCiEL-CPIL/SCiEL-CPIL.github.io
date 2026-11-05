def construct():
    """
    Construct 30 distinct integer points in [-30,30]^2 that maximize the
    largest multiplicity of a single positive doubled area.
    
    Strategy: Use a dense rectangular grid to maximize the number of
    triangles with small doubled area. The 5x6 grid (x in [0,4], y in [0,5])
    produces many triangles with doubled area = 2 (unit area triangles).
    
    To improve, we can try different grid shapes or add some perturbation
    to create more triangles with a common small doubled area.
    
    The doubled area of triangle with vertices (x1,y1), (x2,y2), (x3,y3) is:
    |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    
    For a rectangular grid, many triangles will have small doubled areas.
    Let's compute and verify the best configuration.
    """
    # Start with the 5x6 grid
    points = []
    for y in range(6):
        for x in range(5):
            points.append((x, y))
    
    # Try to improve by trying nearby configurations
    # We'll do a local search: try shifting individual points slightly
    # and see if it improves the max multiplicity
    
    def compute_max_multipity(pts):
        """Compute the largest multiplicity of a single positive doubled area."""
        from collections import Counter
        n = len(pts)
        areas = Counter()
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i+1, n):
                x2, y2 = pts[j]
                for k in range(j+1, n):
                    x3, y3 = pts[k]
                    # Doubled area = |det|
                    det = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
                    adet = abs(det)
                    if adet > 0:
                        areas[adet] += 1
        if not areas:
            return 0
        return max(areas.values())
    
    base_score = compute_max_multipity(points)
    
    # Local search: try small perturbations
    import random
    random.seed(42)
    
    best_points = list(points)
    best_score = base_score
    improved = True
    iterations = 0
    
    while improved and iterations < 200:
        improved = False
        iterations += 1
        for idx in range(len(best_points)):
            x, y = best_points[idx]
            # Try moving this point by 1 in each direction
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                        continue
                    # Check if (nx, ny) is already in the set
                    new_pts = list(best_points)
                    new_pts[idx] = (nx, ny)
                    if len(set(new_pts)) != len(new_pts):
                        continue
                    score = compute_max_multipity(new_pts)
                    if score > best_score:
                        best_points = new_pts
                        best_score = score
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return best_points
