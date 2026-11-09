def construct():
    points = set()
    
    # Hexagon of radius 3: 19 points
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a+b)) <= 3:
                points.add((a, b))
    
    # Ring 4: 24 points, need 13 more
    ring4 = [(a, b) for a in range(-4, 5) for b in range(-4, 5) 
             if max(abs(a), abs(b), abs(a+b)) == 4]
    
    def count_adj(p, pts):
        a, b = p
        return sum(1 for da, db in [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)] 
                   if (a+da, b+db) in pts)
    
    while len(points) < 32:
        candidates = [p for p in ring4 if p not in points]
        best = max(candidates, key=lambda p: count_adj(p, points))
        points.add(best)
    
    return [[a, b] for a, b in points]
