def _greedy_construct():
    pts = []
    pset = set()
    for _ in range(32):
        best_p = None
        best_score = -1
        best_coord = (1000, 1000)
        for a in range(-10, 11):
            for b in range(-10, 11):
                p = (a, b)
                if p in pset:
                    continue
                valid = True
                for q in pts:
                    if _dist_sq(p, q) < 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                neighbors = 0
                for q in pts:
                    if _dist_sq(p, q) == 1:
                        neighbors += 1
                
                # Tie break: prefer smaller a+b, then smaller a
                coord_key = (a + b, a)
                if neighbors > best_score or (neighbors == best_score and coord_key < best_coord):
                    best_score = neighbors
                    best_p = p
                    best_coord = coord_key
        
        if best_p is None:
            # Fallback
            return pts
        
        pts.append(best_p)
        pset.add(best_p)
    return pts
