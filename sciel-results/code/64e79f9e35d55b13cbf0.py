import time
import random

def construct():
    start_time = time.time()
    timeout = 140  # Safety margin before 150s kill

    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def has_4_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                count = 0
                for k in range(n):
                    if k != i and k != j and are_collinear(points[i], points[j], points[k]):
                        count += 1
                if count >= 2:
                    return True
        return False

    def get_collinear_triples(points):
        n = len(points)
        triples = set()
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        triples.add((i, j, k))
        return triples

    def max_no_3_collinear_exact(points):
        n = len(points)
        if n <= 2:
            return n

        triples = get_collinear_triples(points)
        if not triples:
            return n

        triple_masks = []
        for u, v, w in triples:
            triple_masks.append((1 << u) | (1 << v) | (1 << w))

        best_size = 0

        def solve(available_mask, current_size):
            nonlocal best_size

            remaining = bin(available_mask).count('1')
            if current_size + remaining <= best_size:
                return

            if available_mask == 0:
                if current_size > best_size:
                    best_size = current_size
                return

            # Find a triple contained in available_mask
            found_triple = None
            for tm in triple_masks:
                if (available_mask & tm) == tm:
                    found_triple = tm
                    break

            if found_triple is None:
                size = bin(available_mask).count('1')
                if current_size + size > best_size:
                    best_size = current_size + size
                return

            # Extract indices
            bits = []
            for i in range(n):
                if found_triple & (1 << i):
                    bits.append(i)

            # Branch: exclude one of the three
            solve(available_mask & ~(1 << bits[0]), current_size)
            solve(available_mask & ~(1 << bits[1]), current_size)
            solve(available_mask & ~(1 << bits[2]), current_size)

        solve((1 << n) - 1, 0)
        return best_size

    def score_config(points):
        return max_no_3_collinear_exact(points)

    best_score = 20
    best_points = None

    # Strategy: Build configurations from lines with exactly 3 points each
    # This maximizes collinear triples, forcing the no-3-collinear subset to be smaller

    def try_line_based_config(seed):
        """
        Build a configuration by placing points on lines with exactly 3 points each.
        We need 20 points. If we use 6 lines of 3 points with no overlap, that's 18 points.
        Then add 2 more points that don't create 4-collinear lines.
        
        Alternative: Use lines that share points. If two lines share one point, 
        we use 3+3-1 = 5 points for 2 lines.
        With k lines of 3, sharing s points total, we use 3k - s points.
        For 20 points: 3k - s = 20.
        k=7, s=1: 21-1=20. 7 lines, 1 shared point.
        k=8, s=4: 24-4=20. 8 lines, 4 shared points.
        """
        random.seed(seed)
        
        # Try a specific structured approach:
        # Use a "pencil" of lines through a common point, but with different slopes
        # and add points not on these lines.
        
        # Actually, let's try a different approach: 
        # Place points in a way that creates many 3-collinear triples.
        
        # Simple approach: Try to create 6 disjoint lines of 3 points + 2 singles
        # But we need to ensure no 4 collinear.
        
        # Let's try: 6 lines of 3 points, each line is y = mx + b with integer coords
        # Lines must not share more than 1 point (otherwise 4 could be collinear on intersection)
        
        lines_data = []
        used_points = set()
        
        # Define 6 lines with 3 points each
        # Line 1: y = 0, x = -3, 0, 3 -> (-3,0), (0,0), (3,0)
        # Line 2: y = 1, x = -2, 1, 4 -> (-2,1), (1,1), (4,1)
        # Line 3: y = 2, x = -1, 2, 5 -> (-1,2), (2,2), (5,2)
        # Line 4: y = 3, x = 0, 3, 6 -> (0,3), (3,3), (6,3)
        # Line 5: y = 4, x = -4, -1, 2 -> (-4,4), (-1,4), (2,4)
        # Line 6: y = 5, x = -3, 0, 3 -> (-3,5), (0,5), (3,5)
        
        # Check for 4-collinear:
        # Vertical lines: x=0 has (0,0),(0,3),(0,5) - 3 points. x=3 has (3,0),(3,3),(3,5) - 3 points.
        # x=-1 has (-1,2),(-1,4) - 2 points. x=-2 has (-2,1) - 1. x=1 has (1,1) - 1.
        # x=-3 has (-3,0),(-3,5) - 2. x=4 has (4,1) - 1. x=5 has (5,2) - 1. x=6 has (6,3) - 1.
        # x=-4 has (-4,4) - 1.
        # Diagonals: Need to check all slopes.
        
        # This is getting complex. Let's just try random configurations and use local search.
        
        pass

    # Better strategy: Use a known good structure and refine with local search
    # Start with configurations that have many 3-collinear triples
    
    def generate_initial_config(seed):
        """Generate an initial configuration with many 3-collinear triples."""
        random.seed(seed)
        points = []
        used = set()
        
        # Try to create 6 lines of 3 points
        # Use lines with small integer slopes and intercepts
        
        # Line parameters: (m, b) where y = mx + b
        # For each line, pick 3 x-values and compute y
        
        line_candidates = []
        for m in [-2, -1, 0, 1, 2]:
            for b in range(-10, 11):
                line_candidates.append((m, b))
        
        random.shuffle(line_candidates)
        
        lines_chosen = 0
        for m, b in line_candidates:
            if lines_chosen >= 6:
                break
            if len(points) >= 20:
                break
            
            # Pick 3 x-values for this line
            xs = random.sample(range(-15, 16), 3)
            pts_on_line = []
            valid = True
            for x in xs:
                y = m * x + b
                if -100 <= x <= 100 and -100 <= y <= 100:
                    if (x, y) not in used:
                        pts_on_line.append((x, y))
                    else:
                        # Point already used, skip this line
                        valid = False
                        break
                else:
                    valid = False
                    break
            
            if valid and len(pts_on_line) == 3:
                for p in pts_on_line:
                    points.append([p[0], p[1]])
                    used.add(p)
                lines_chosen += 1
        
        # Add remaining points randomly
        while len(points) < 20:
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            if (x, y) not in used:
                points.append([x, y])
                used.add((x, y))
        
        return points[:20]

    def local_search(points, max_time=10):
        """Local search to minimize the score."""
        current_points = [p[:] for p in points]
        current_score = score_config(current_points)
        start_ls = time.time()
        improved = True
        
        while improved and (time.time() - start_ls) < max_time and (time.time() - start_time) < timeout - 5:
            improved = False
            indices = list(range(20))
            random.shuffle(indices)
            
            for i in indices:
                if (time.time() - start_time) > timeout - 5:
                    return current_points, current_score
                
                old_x, old_y = current_points[i]
                
                # Try moving point i to a nearby position
                for attempt in range(15):
                    dx = random.randint(-15, 15)
                    dy = random.randint(-15, 15)
                    new_x = old_x + dx
                    new_y = old_y + dy
                    
                    if new_x < -100 or new_x > 100 or new_y < -100 or new_y > 100:
                        continue
                    if (new_x, new_y) == (old_x, old_y):
                        continue
                    
                    # Check collision
                    collision = False
                    for j, p in enumerate(current_points):
                        if j != i and p[0] == new_x and p[1] == new_y:
                            collision = True
                            break
                    if collision:
                        continue
                    
                    # Temporarily move
                    current_points[i] = [new_x, new_y]
                    
                    if has_4_collinear(current_points):
                        current_points[i] = [old_x, old_y]
                        continue
                    
                    new_score = score_config(current_points)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        current_points[i] = [old_x, old_y]
                
                # Early exit check
                if (time.time() - start_time) > timeout - 5:
                    return current_points, current_score
        
        return current_points, current_score

    # Generate and evaluate multiple candidates
    candidates = []
    
    # Strategy 1: Line-based initial configs
    for seed in range(50):
        pts = generate_initial_config(seed)
        if len(pts) == 20 and not has_4_collinear(pts):
            candidates.append(pts)
    
    # Strategy 2: Random dense configs
    for seed in range(20):
        random.seed(1000 + seed)
        pts = []
        used = set()
        while len(pts) < 20:
            x = random.randint(-40, 40)
            y = random.randint(-40, 40)
            if (x, y) not in used:
                pts.append([x, y])
                used.add((x, y))
        if not has_4_collinear(pts):
            candidates.append(pts)
    
    # Strategy 3: Grid perturbation
    for seed in range(10):
        random.seed(2000 + seed)
        pts = []
        used = set()
        for y in range(0, 12, 2):
            for x in range(-20, 21, 4):
                if len(pts) >= 20:
                    break
                nx = x + random.randint(-2, 2)
                ny = y + random.randint(-1, 1)
                if -100 <= nx <= 100 and -100 <= ny <= 100 and (nx, ny) not in used:
                    pts.append([nx, ny])
                    used.add((nx, ny))
        while len(pts) < 20:
            x = random.randint(-60, 60)
            y = random.randint(-60, 60)
            if (x, y) not in used:
                pts.append([x, y])
                used.add((x, y))
        if len(pts) == 20 and not has_4_collinear(pts):
            candidates.append(pts)
    
    # Run local search on candidates, prioritizing those with more triples
    scored_candidates = []
    for cand in candidates:
        if time.time() - start_time > timeout - 20:
            break
        s = score_config(cand)
        scored_candidates.append((s, cand))
    
    # Sort by score (lower is better) and run local search on top candidates
    scored_candidates.sort(key=lambda x: x[0])
    
    for s, cand in scored_candidates[:30]:
        if time.time() - start_time > timeout - 10:
            break
        if s >= best_score:
            continue
        
        pts, new_s = local_search(cand, max_time=3)
        if new_s < best_score:
            best_score = new_s
            best_points = pts
    
    # If no improvement, try more aggressive search
    if best_score >= 18:
        for i in range(20):
            if time.time() - start_time > timeout - 5:
                break
            random.seed(5000 + i)
            pts = []
            used = set()
            while len(pts) < 20:
                x = random.randint(-80, 80)
                y = random.randint(-80, 80)
                if (x, y) not in used:
                    pts.append([x, y])
                    used.add((x, y))
            if not has_4_collinear(pts):
                pts, s = local_search(pts, max_time=2)
                if s < best_score:
                    best_score = s
                    best_points = pts
    
    # Fallback: simple valid config
    if best_points is None:
        pts = []
        for x in range(-5, 6):
            pts.append([x, x * x])
        for x in range(-5, 6):
            pts.append([x, x * x + 25])
        best_points = pts
        best_score = score_config(best_points)
    
    return best_points
