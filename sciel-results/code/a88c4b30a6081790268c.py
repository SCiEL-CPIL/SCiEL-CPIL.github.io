<<<<<<< SEARCH
def construct():
    if not _deadline_ok(15):
        return [(x, y) for x in range(6) for y in range(5)]
    
    best_pts = None
    best_score = 0
    
    # Phase 1: score seeds, polish each
    seeds = _generate_seeds()
    for seed in seeds:
        if not _deadline_ok(15): break
        if not _valid(seed): continue
        s = _score(seed)
        if s >= best_score:
            counter = _full_counter(seed)
            p, _, c = _local_search(seed, counter, max_iter=80, margin=15)
            ps = max(c.values(), default=0)
            if ps > best_score:
                best_score = ps
                best_pts = list(p)
    
    # Phase 2: model-and-optimize for various target areas A
    rng = random.Random(42)
    target_areas = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80]
    rng.shuffle(target_areas)
    
    for A in target_areas:
        if not _deadline_ok(10): break
        p, s = _construct_for_A(A, rng)
        if p is None: continue
        if s >= best_score - 3:
            counter = _full_counter(p)
            pp, _, c = _local_search(p, counter, max_iter=120, margin=10)
            ps = max(c.values(), default=0)
            if ps > best_score:
                best_score = ps
                best_pts = list(pp)
    
    # Phase 3: random restarts
    for _ in range(25):
        if not _deadline_ok(8): break
        pts = []
        used = set()
        for _ in range(30):
            x = rng.randint(-25, 25)
            y = rng.randint(-25, 25)
            if (x, y) not in used:
                pts.append((x, y))
                used.add((x, y))
        s = _score(pts)
        if s >= best_score - 5:
            counter = _full_counter(pts)
            p, _, c = _local_search(pts, counter, max_iter=80, margin=8)
            ps = max(c.values(), default=0)
            if ps > best_score:
                best_score = ps
                best_pts = list(p)
    
    # Final polish
    if best_pts is not None and _deadline_ok(6):
        counter = _full_counter(best_pts)
        p, _, c = _local_search(best_pts, counter, max_iter=200, margin=6)
        ps = max(c.values(), default=0)
        if ps >= best_score:
            best_pts = list(p)
            best_score = ps
    
    if best_pts is None:
        best_pts = [(x, y) for x in range(6) for y in range(5)]
    
    return best_pts
=======
def construct():
    if not _deadline_ok(15):
        return [(x, y) for x in range(6) for y in range(5)]
    
    best_pts = None
    best_score = 0
    
    # Phase 1: score seeds, polish each
    seeds = _generate_seeds()
    for seed in seeds:
        if not _deadline_ok(15): break
        if not _valid(seed): continue
        s = _score(seed)
        if s >= best_score:
            counter = _full_counter(seed)
            p, _, c = _local_search(seed, counter, max_iter=100, margin=15)
            ps = max(c.values(), default=0)
            if ps > best_score:
                best_score = ps
                best_pts = list(p)
    
    # Phase 2: model-and-optimize for various target areas A
    rng = random.Random(42)
    target_areas = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 200, 220, 240, 260, 280]
    rng.shuffle(target_areas)
    
    for A in target_areas:
        if not _deadline_ok(8): break
        p, s = _construct_for_A(A, rng)
        if p is None: continue
        if s >= best_score - 3:
            counter = _full_counter(p)
            pp, _, c = _local_search(p, counter, max_iter=150, margin=8)
            ps = max(c.values(), default=0)
            if ps > best_score:
                best_score = ps
                best_pts = list(pp)
    
    # Phase 3: random restarts
    for _ in range(30):
        if not _deadline_ok(6): break
        pts = []
        used = set()
        for _ in range(30):
            x = rng.randint(-28, 28)
            y = rng.randint(-28, 28)
            if (x, y) not in used:
                pts.append((x, y))
                used.add((x, y))
        s = _score(pts)
        if s >= best_score - 5:
            counter = _full_counter(pts)
            p, _, c = _local_search(pts, counter, max_iter=100, margin=6)
            ps = max(c.values(), default=0)
            if ps > best_score:
                best_score = ps
                best_pts = list(p)
    
    # Final polish
    if best_pts is not None and _deadline_ok(5):
        counter = _full_counter(best_pts)
        p, _, c = _local_search(best_pts, counter, max_iter=300, margin=5)
        ps = max(c.values(), default=0)
        if ps >= best_score:
            best_pts = list(p)
            best_score = ps
    
    if best_pts is None:
        best_pts = [(x, y) for x in range(6) for y in range(5)]
    
    return best_pts
>>>>>>> REPLACE
