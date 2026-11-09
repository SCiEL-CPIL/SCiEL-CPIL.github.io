<<<<<<< SEARCH
def _generate_grid(rng):
    """
    Generate points in a grid pattern.
    """
    # 6x6 grid is 36 points. We need 33.
    # Let's use a 6x6 grid and remove 3 random points.
    xs = np.linspace(0.1, 0.9, 6)
    ys = np.linspace(0.1, 0.9, 6)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Remove 3 random points
    idx_to_remove = rng.choice(36, size=3, replace=False)
    mask = np.ones(36, dtype=bool)
    mask[idx_to_remove] = False
    pts = pts[mask]
    
    # Add noise
    pts += rng.normal(0, 0.01, (33, 2))
    pts = np.clip(pts, 0.01, 0.99)
    
    attempts = 0
    while not _is_valid_fast(pts) and attempts < 50:
        pts += rng.normal(0, 0.005, (33, 2))
        pts = np.clip(pts, 0.01, 0.99)
        attempts += 1
    return pts
=======
def _generate_corner_cluster(rng):
    """
    Generate points clustered near the four corners and edges.
    Points near corners/edges are less likely to form convex 7-gons.
    """
    pts = []
    # 4 corner clusters of ~7 points each
    corners = [(0.08, 0.08), (0.92, 0.08), (0.08, 0.92), (0.92, 0.92)]
    for cx, cy in corners:
        for _ in range(7):
            x = cx + rng.normal(0, 0.04)
            y = cy + rng.normal(0, 0.04)
            pts.append((x, y))
    # 4 edge-midpoint clusters of ~1 point each (16 so far, need 17 more)
    edges = [(0.5, 0.06), (0.5, 0.94), (0.06, 0.5), (0.94, 0.5)]
    for ex, ey in edges:
        for _ in range(4):
            x = ex + rng.normal(0, 0.03)
            y = ey + rng.normal(0, 0.03)
            pts.append((x, y))
    # Remaining 1 point near center
    pts.append((0.5 + rng.normal(0, 0.02), 0.5 + rng.normal(0, 0.02)))
    
    pts = np.array(pts)
    pts = np.clip(pts, 0.01, 0.99)
    
    attempts = 0
    while not _is_valid_fast(pts) and attempts < 100:
        pts += rng.normal(0, 0.003, (33, 2))
        pts = np.clip(pts, 0.01, 0.99)
        attempts += 1
    
    return pts
>>>>>>> REPLACE

