<<<<<<< SEARCH
def _generate_initial_candidates():
    """Generate diverse initial point sets."""
    candidates = []
    
    # 1. Best Rectangular Grids
    grid_dims = [(5, 6), (6, 5), (4, 8), (8, 4), (3, 10), (10, 3), (2, 15), (15, 2)]
    for w, h in grid_dims:
        pts = [(i, j) for i in range(w) for j in range(h)]
        if len(pts) == 30:
            cx = (w - 1) // 2
            cy = (h - 1) // 2
            centered = [(x - cx, y - cy) for x, y in pts]
            candidates.append(centered)

    # 2. Diamond Shape (Manhattan distance)
    diamond_pts = []
    for x in range(-10, 11):
        for y in range(-10, 11):
            diamond_pts.append((x, y, abs(x) + abs(y)))
    diamond_pts.sort(key=lambda p: p[2])
    seen = set()
    cand_diamond = []
    for x, y, d in diamond_pts:
        if (x, y) not in seen:
            seen.add((x, y))
            cand_diamond.append((x, y))
            if len(cand_diamond) == 30:
                break
    candidates.append(cand_diamond)

    # 3. Circle Shape (Euclidean distance)
    circle_pts = []
    for x in range(-10, 11):
        for y in range(-10, 11):
            circle_pts.append((x, y, x*x + y*y))
    circle_pts.sort(key=lambda p: p[2])
    seen = set()
    cand_circle = []
    for x, y, d in circle_pts:
        if (x, y) not in seen:
            seen.add((x, y))
            cand_circle.append((x, y))
            if len(cand_circle) == 30:
                break
    candidates.append(cand_circle)
    
    # 4. Lattice patterns with different bases
    # Hexagonal-like lattice
    hex_pts = []
    for row in range(6):
        for col in range(5):
            x = col * 2 + (row % 2)
            y = row
            hex_pts.append((x, y))
    if len(hex_pts) == 30:
        cx = 4
        cy = 2
        candidates.append([(x - cx, y - cy) for x, y in hex_pts])
    
    # Triangular lattice
    tri_pts = []
    for row in range(6):
        for col in range(5):
            x = col
            y = row
            tri_pts.append((x, y))
    if len(tri_pts) == 30:
        cx = 2
        cy = 2
        candidates.append([(x - cx, y - cy) for x, y in tri_pts])
    
    # 5. Random Clusters
    for _ in range(10):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-5, 5)
            y = random.randint(-5, 5)
            pts.add((x, y))
        candidates.append(list(pts))
    
    # 6. More structured random: pick from small grid with some randomness
    for _ in range(10):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-4, 4)
            y = random.randint(-4, 4)
            pts.add((x, y))
        candidates.append(list(pts))

    return candidates
=======
def _generate_initial_candidates():
    """Generate diverse initial point sets."""
    candidates = []
    
    # 1. Best Rectangular Grids
    grid_dims = [(5, 6), (6, 5), (4, 8), (8, 4), (3, 10), (10, 3), (2, 15), (15, 2)]
    for w, h in grid_dims:
        pts = [(i, j) for i in range(w) for j in range(h)]
        if len(pts) == 30:
            cx = (w - 1) // 2
            cy = (h - 1) // 2
            centered = [(x - cx, y - cy) for x, y in pts]
            candidates.append(centered)

    # 2. Diamond Shape (Manhattan distance)
    diamond_pts = []
    for x in range(-10, 11):
        for y in range(-10, 11):
            diamond_pts.append((x, y, abs(x) + abs(y)))
    diamond_pts.sort(key=lambda p: p[2])
    seen = set()
    cand_diamond = []
    for x, y, d in diamond_pts:
        if (x, y) not in seen:
            seen.add((x, y))
            cand_diamond.append((x, y))
            if len(cand_diamond) == 30:
                break
    candidates.append(cand_diamond)

    # 3. Circle Shape (Euclidean distance)
    circle_pts = []
    for x in range(-10, 11):
        for y in range(-10, 11):
            circle_pts.append((x, y, x*x + y*y))
    circle_pts.sort(key=lambda p: p[2])
    seen = set()
    cand_circle = []
    for x, y, d in circle_pts:
        if (x, y) not in seen:
            seen.add((x, y))
            cand_circle.append((x, y))
            if len(cand_circle) == 30:
                break
    candidates.append(cand_circle)
    
    # 4. Lattice patterns with different bases
    # Hexagonal-like lattice
    hex_pts = []
    for row in range(6):
        for col in range(5):
            x = col * 2 + (row % 2)
            y = row
            hex_pts.append((x, y))
    if len(hex_pts) == 30:
        cx = 4
        cy = 2
        candidates.append([(x - cx, y - cy) for x, y in hex_pts])
    
    # Triangular lattice
    tri_pts = []
    for row in range(6):
        for col in range(5):
            x = col
            y = row
            tri_pts.append((x, y))
    if len(tri_pts) == 30:
        cx = 2
        cy = 2
        candidates.append([(x - cx, y - cy) for x, y in tri_pts])
    
    # 5. Diagonal line patterns
    for offset in range(-5, 6, 2):
        diag_pts = []
        for i in range(30):
            x = i - 15
            y = i + offset
            if -30 <= x <= 30 and -30 <= y <= 30:
                diag_pts.append((x, y))
        if len(diag_pts) == 30:
            candidates.append(diag_pts)
    
    # 6. Parabolic patterns
    for a in [1, 2, -1, -2]:
        parab_pts = []
        for i in range(-15, 15):
            x = i
            y = a * i * i // 10  # Scale to stay in bounds
            if -30 <= x <= 30 and -30 <= y <= 30:
                parab_pts.append((x, y))
        if len(parab_pts) == 30:
            candidates.append(parab_pts)
    
    # 7. Random Clusters
    for _ in range(10):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-5, 5)
            y = random.randint(-5, 5)
            pts.add((x, y))
        candidates.append(list(pts))
    
    # 8. More structured random: pick from small grid with some randomness
    for _ in range(10):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-4, 4)
            y = random.randint(-4, 4)
            pts.add((x, y))
        candidates.append(list(pts))
    
    # 9. Random from larger grid
    for _ in range(10):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-8, 8)
            y = random.randint(-8, 8)
            pts.add((x, y))
        candidates.append(list(pts))

    return candidates
>>>>>>> REPLACE

