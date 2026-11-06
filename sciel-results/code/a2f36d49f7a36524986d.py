<<<<<<< SEARCH
def _search_incremental_int(D, start_points, max_coord, deadline):
    """Incremental search: try to add points one at a time."""
    current = list(start_points)
    best = current[:]
    best_len = len(current)
    
    # Try adding points in a systematic manner
    for attempt in range(1000000):
        if time.time() > deadline:
            break
        if len(current) >= 12:
            break
        
        # Generate candidate point
        if attempt < 50000:
            # Systematic grid search
            x = (attempt % (2*max_coord+1)) - max_coord
            y = ((attempt // (2*max_coord+1)) % (2*max_coord+1)) - max_coord
        else:
            # Random search
            x = random.randint(-max_coord, max_coord)
            y = random.randint(-max_coord, max_coord)
        
        new_point = (x, y)
        
        # Check if point is already in set
        if new_point in current:
            continue
        
        if _try_add_point_int(current, new_point, D):
            current.append(new_point)
            best = current[:]
            best_len = len(current)
            if best_len >= 8:
                break
    
    return best
=======
def _generate_pythagorean_candidates(center, D, max_coord, num_candidates):
    """Generate candidate points at rational distances from center using Pythagorean triples."""
    candidates = set()
    cx, cy = center
    # Generate Pythagorean triples
    for m in range(2, 50):
        for n in range(1, m):
            if (m - n) % 2 == 0:
                continue
            if gcd(m, n) != 1:
                continue
            a = m * m - n * n
            b = 2 * m * n
            c = m * m + n * n
            # Scale and offset
            for scale in range(1, 10):
                ax, ay = a * scale, b * scale
                bx, by = b * scale, a * scale
                # Try all 8 directions
                for dx, dy in [(ax, ay), (-ax, ay), (ax, -ay), (-ax, -ay),
                               (ay, ax), (-ay, ax), (ay, -ax), (-ay, -ax)]:
                    nx, ny = cx + dx, cy + dy
                    if abs(nx) <= max_coord and abs(ny) <= max_coord:
                        candidates.add((nx, ny))
    return list(candidates)

def _search_incremental_int(D, start_points, max_coord, deadline):
    """Incremental search: try to add points one at a time."""
    current = list(start_points)
    best = current[:]
    best_len = len(current)
    
    # Precompute distances from existing points
    def _check_new_point(pt):
        if pt in current:
            return False
        return _try_add_point_int(current, pt, D)
    
    # Try adding points in a systematic manner
    attempt = 0
    while time.time() < deadline:
        if len(current) >= 12:
            break
        
        # Generate candidate points
        if attempt < 10000:
            # Systematic grid search with bias toward Pythagorean-friendly positions
            x = (attempt % (2*max_coord+1)) - max_coord
            y = ((attempt // (2*max_coord+1)) % (2*max_coord+1)) - max_coord
        elif attempt < 30000:
            # Try points at Pythagorean distances from existing points
            existing = random.choice(current)
            cands = _generate_pythagorean_candidates(existing, D, max_coord, 100)
            if not cands:
                x = random.randint(-max_coord, max_coord)
                y = random.randint(-max_coord, max_coord)
            else:
                x, y = random.choice(cands)
        else:
            # Random search
            x = random.randint(-max_coord, max_coord)
            y = random.randint(-max_coord, max_coord)
        
        new_point = (x, y)
        
        if _check_new_point(new_point):
            current.append(new_point)
            best = current[:]
            best_len = len(current)
            if best_len >= 8:
                break
            # Reset attempt counter for fresh search
            attempt = 0
        else:
            attempt += 1
    
    return best
>>>>>>> REPLACE

