<<<<<<< SEARCH
def construct():
    global _START
    _START = time.time()
    
    best_points = None
    best_score = 999
    
    # 1. Deterministic baseline: Parabola
    pts_parab = [[x, x * x] for x in range(-10, 10)]
    if _valid(pts_parab):
        s = _max_no_three_collinear(pts_parab)
        if s < best_score:
            best_score = s
            best_points = pts_parab
    
    # 2. Random search
    # We want to find configurations with many collinear triples but no 4-collinear.
    # Random points usually have few collinear triples, so score is high (near 20).
    # We need a strategy that *encourages* collinearity.
    
    # Strategy: Pick a set of lines, and place points on them.
    # However, ensuring no 4 collinear is tricky.
    
    # Alternative: Local search starting from random or parabola.
    # Let's try more random seeds, but also try the perturbed parabola.
    
    for seed in range(10000):
        if _remaining() < 10.0:
            break
        
        # Mix strategies
        if seed % 2 == 0:
            pts = _construct_random(seed)
        else:
            pts = _construct_perturbed_parabola(seed)
        
        if pts is None:
            continue
        
        # Quick check: if it's not valid, skip
        if not _valid(pts):
            continue
        
        # Compute score
        s = _max_no_three_collinear(pts)
        if s < best_score:
            best_score = s
            best_points = pts
            if best_score <= 12:  # Good enough, stop early
                break
        
        # Safety check
        if _remaining() < 5.0:
            break
    
    if best_points is None:
        best_points = pts_parab
    
    return best_points
=======
def construct():
    global _START
    _START = time.time()
    
    best_points = None
    best_score = 999
    
    # 1. Deterministic baseline: Parabola
    pts_parab = [[x, x * x] for x in range(-10, 10)]
    if _valid(pts_parab):
        s = _max_no_three_collinear(pts_parab)
        if s < best_score:
            best_score = s
            best_points = pts_parab
    
    # 2. Try line-based construction
    for seed in range(500):
        if _remaining() < 15.0:
            break
        
        pts = _construct_from_lines(seed)
        if pts is None:
            continue
        
        if not _valid(pts):
            continue
        
        s = _max_no_three_collinear(pts)
        if s < best_score:
            best_score = s
            best_points = pts
            if best_score <= 12:
                break
    
    # 3. Local search on best configuration
    if best_points is not None and _remaining() > 20.0:
        improved_pts, improved_score = _local_search(best_points, max_time=min(30.0, _remaining() - 10.0))
        if improved_score < best_score:
            best_score = improved_score
            best_points = improved_pts
    
    # 4. More random search with local search
    for seed in range(5000):
        if _remaining() < 10.0:
            break
        
        if seed % 3 == 0:
            pts = _construct_random(seed)
        elif seed % 3 == 1:
            pts = _construct_from_lines(seed + 10000)
        else:
            pts = _construct_perturbed_parabola(seed)
        
        if pts is None:
            continue
        
        if not _valid(pts):
            continue
        
        s = _max_no_three_collinear(pts)
        
        # If this is promising, do local search
        if s <= best_score + 2 and _remaining() > 15.0:
            improved_pts, improved_score = _local_search(pts, max_time=min(5.0, _remaining() - 5.0))
            if improved_score < best_score:
                best_score = improved_score
                best_points = improved_pts
        
        if s < best_score:
            best_score = s
            best_points = pts
        
        if best_score <= 12:
            break
    
    if best_points is None:
        best_points = pts_parab
    
    return best_points
>>>>>>> REPLACE
